# weatherPipeline.py
#!/usr/bin/env python3
import warnings; warnings.filterwarnings("ignore")
import os, sys, time, signal, logging, logging.handlers, datetime as dt
from contextlib import contextmanager

from downloadFunctions import (
    ensure_dir, read_cycle_marker, archive_folder_contents
)
import regionalDownload as rdl
import globalDownload as gdl
import analysisRegional as ar
import analysisGlobal as ag

# ───── config ───────────────────────────────────────────────────
DATA_REG = "./data/regional"
DATA_GLB = "./data/global"
FIG_REG  = "./figures/regional"
FIG_GLB  = "./figures/global"

LOG_FILE      = os.getenv("WX_LOG_FILE", "./weather_pipeline.log")
CHECK_EVERY   = int(os.getenv("WX_CHECK_EVERY", "900"))   # 15 min
MAX_BACKOFF   = int(os.getenv("WX_MAX_BACKOFF", "3600"))  # 1 hr
LOCKFILE_PATH = os.getenv("WX_LOCKFILE", "./weather_pipeline.lock")
FORCE_RUN     = os.getenv("WX_FORCE_RUN", "0") == "1"     # optional override

for d in (DATA_REG, DATA_GLB, FIG_REG, FIG_GLB):
    ensure_dir(d)

# ───── logging ──────────────────────────────────────────────────
logger = logging.getLogger("pipeline")
logger.setLevel(logging.INFO)
handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=10_000_000, backupCount=3, encoding="utf-8")
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
handler.setFormatter(fmt)
logger.addHandler(handler)
console = logging.StreamHandler(sys.stdout); console.setFormatter(fmt); logger.addHandler(console)

# ───── graceful shutdown ────────────────────────────────────────
_shutdown = False
def _signal_handler(signum, frame):
    global _shutdown
    _shutdown = True
    logger.info(f"Received signal {signum}, will exit after this cycle.")
signal.signal(signal.SIGINT, _signal_handler)
try: signal.signal(signal.SIGTERM, _signal_handler)
except Exception: pass

# ───── single-instance lock ─────────────────────────────────────
@contextmanager
def single_instance(lock_path: str):
    import errno, uuid
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    lock_fd = None
    try:
        lock_fd = os.open(lock_path, flags)
        os.write(lock_fd, str(uuid.uuid4()).encode("utf-8"))
        os.close(lock_fd); lock_fd = None
        yield
    except OSError as e:
        if e.errno == errno.EEXIST:
            raise RuntimeError(f"Another instance is running (lock: {lock_path})") from e
        raise
    finally:
        try:
            if os.path.exists(lock_path):
                os.remove(lock_path)
        except Exception:
            pass

# ───── latest remote cycle detectors ────────────────────────────
def latest_fmi_harmonie_cycle(bbox=(-28.0,44.0,37.0,74.0)):
    """
    Query FMI WFS for Harmonie surface grid around 'now' and return newest (ymd, cyc).
    Falls back to previous hours if needed. If detection fails, returns (None, None).
    """
    try:
        from fmiopendata.wfs import download_stored_query
    except Exception:
        return (None, None)

    def _fmt(ts): return ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    now = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)

    for off in (0, -1, -2, -3, -4, -6):
        t = now + dt.timedelta(hours=off)
        args = [
            f"starttime={_fmt(t)}",
            f"endtime={_fmt(t)}",
            f"bbox={bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
            "timestep=60",
            "parameters=Temperature",
            "format=grib2",
        ]
        try:
            md = download_stored_query("fmi::forecast::harmonie::surface::grid", args=args)
        except Exception:
            continue
        keys = list(getattr(md, "data", {}).keys())
        if not keys:
            continue
        latest = max(keys)
        if isinstance(latest, dt.datetime):
            latest = latest.astimezone(dt.timezone.utc)
            return latest.strftime("%Y%m%d"), latest.strftime("%H")
        # fallback parse
        s = str(latest)
        try:
            tt = dt.datetime.fromisoformat(s.replace("Z","+00:00")).astimezone(dt.timezone.utc)
            return tt.strftime("%Y%m%d"), tt.strftime("%H")
        except Exception:
            continue
    return (None, None)

def latest_gfs_cycle():
    import requests
    GFS_BASE  = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod"
    CYCLES    = ["18","12","06","00"]

    def _exists(url, timeout=6):
        try:
            r = requests.head(url, timeout=timeout)
            return r.status_code == 200
        except Exception:
            return False

    now = dt.datetime.now(dt.timezone.utc)
    for day in (now.date(), (now - dt.timedelta(days=1)).date()):
        ymd = day.strftime("%Y%m%d")
        for cyc in CYCLES:
            f000 = f"{GFS_BASE}/gfs.{ymd}/{cyc}/atmos/gfs.t{cyc}z.pgrb2.0p25.f000"
            if _exists(f000):
                return ymd, cyc
    return (None, None)

# ───── helpers ──────────────────────────────────────────────────
def archive_figures(fig_root: str):
    """Move current figures (all subfolders) into fig_root/older/<timestamp>/."""
    archive_folder_contents(fig_root, older_sub="older")

def process_regional():
    """
    Regional pipeline: check new, then archive old (data+figs), then download+stack, then analyze.
    """
    local = read_cycle_marker(DATA_REG)  # (ymd, cyc) or None
    ymd_remote, cyc_remote = latest_fmi_harmonie_cycle()
    if FORCE_RUN:
        logger.info("Regional: FORCE_RUN=1 skipping new-run guard.")
    elif ymd_remote and cyc_remote and local == (ymd_remote, cyc_remote):
        logger.info(f"Regional: no new run (remote={ymd_remote} {cyc_remote}Z == local). Skipping.")
        return

    logger.info("Regional: archiving previous FIGURES (if any)…")
    archive_figures(FIG_REG)
    if local:
        tag = f"{local[0]}_{local[1]}"
        logger.info("Regional: archiving previous DATA …")
        archive_folder_contents(DATA_REG, older_sub="older", tag=tag, keep_exts={".nc",".txt"})

    ymd, cyc, nc_path = rdl.download_and_build_stack()
    logger.info(f"Regional: got {ymd} {cyc}Z → {nc_path}")
    ar.run_analysis_from_nc(nc_path)

def process_global():
    """
    Global pipeline: check new, then archive old (data+figs), then download+stack, then analyze.
    """
    local = read_cycle_marker(DATA_GLB)
    ymd_remote, cyc_remote = latest_gfs_cycle()
    if FORCE_RUN:
        logger.info("Global: FORCE_RUN=1  skipping new-run guard.")
    elif ymd_remote and cyc_remote and local == (ymd_remote, cyc_remote):
        logger.info(f"Global: no new run (remote={ymd_remote} {cyc_remote}Z == local). Skipping.")
        return

    logger.info("Global: archiving previous FIGURES (if any)…")
    archive_figures(FIG_GLB)
    if local:
        tag = f"{local[0]}_{local[1]}"
        logger.info("Global: archiving previous DATA …")
        archive_folder_contents(DATA_GLB, older_sub="older", tag=tag, keep_exts={".nc",".txt"})

    ymd, cyc, nc_path = gdl.download_and_build_stack()
    logger.info(f"Global: got {ymd} {cyc}Z → {nc_path}")
    ag.run_analysis_from_nc(nc_path)

# ───── main loop ────────────────────────────────────────────────
def main():
    logger.info("Weather pipeline starting (regional, then global)…")
    backoff = CHECK_EVERY
    while not _shutdown:
        try:
            process_regional()
            process_global()
            backoff = CHECK_EVERY
        except Exception as e:
            logger.exception(f"Cycle failed: {e}")
            backoff = min(max(int(backoff * 1.5), CHECK_EVERY), MAX_BACKOFF)
        slept = 0
        while slept < backoff and not _shutdown:
            time.sleep(1); slept += 1
    logger.info("Pipeline exiting cleanly.")

if __name__ == "__main__":
    try:
        with single_instance(LOCKFILE_PATH):
            main()
    except RuntimeError as e:
        logger.error(str(e)); sys.exit(1)
