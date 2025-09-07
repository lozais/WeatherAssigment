# downloadFunctions.py
import warnings; warnings.filterwarnings("ignore")
import os, re, shutil, datetime as dt, logging
from typing import Optional, Tuple, Iterable, List

LOG = logging.getLogger("pipeline.download")

CYCLE_MARKER_RE = re.compile(r"^cycle_(\d{8})_(\d{2})\.txt$")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def read_cycle_marker(dirpath: str) -> Optional[Tuple[str,str]]:
    """Return (ymd, cyc) from the newest cycle_YYYYMMDD_HH.txt in dirpath, if any."""
    if not os.path.isdir(dirpath):
        return None
    files = [f for f in os.listdir(dirpath) if CYCLE_MARKER_RE.match(f)]
    if not files:
        return None
    files.sort()
    name = files[-1]
    m = CYCLE_MARKER_RE.match(name)
    return (m.group(1), m.group(2)) if m else None

def write_cycle_marker(dirpath: str, ymd: str, cyc: str):
    ensure_dir(dirpath)
    with open(os.path.join(dirpath, f"cycle_{ymd}_{cyc}.txt"), "w") as f:
        f.write(f"{ymd} {cyc}\n")

def archive_folder_contents(src_dir: str, older_sub="older", tag: Optional[str]=None, keep_exts: Optional[Iterable[str]]=None):
    """
    Move all non-directory files and subfolders from src_dir into src_dir/older/<tag>/.
    If tag not given, generate UTC timestamp.
    Optionally keep files with extensions in keep_exts (lowercased, dot-included).
    """
    ensure_dir(src_dir)
    if tag is None:
        tag = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    dst_root = os.path.join(src_dir, older_sub, tag)
    ensure_dir(dst_root)

    for name in os.listdir(src_dir):
        if name == older_sub:
            continue
        src = os.path.join(src_dir, name)
        if os.path.isdir(src):
            shutil.move(src, os.path.join(dst_root, name))
            continue
        if keep_exts:
            ext = os.path.splitext(name)[1].lower()
            if ext in keep_exts:
                continue
        shutil.move(src, os.path.join(dst_root, name))

    LOG.info(f"Archived contents of {src_dir} → {dst_root}")
    return dst_root

def safe_remove_files(paths: List[str]):
    for p in paths:
        try:
            if os.path.isfile(p):
                os.remove(p)
        except Exception:
            pass

def latest_synoptic_cycle(now_utc: Optional[dt.datetime]=None) -> Tuple[str,str]:
    """
    Return the most recent cycle (00/06/12/18) not in the future, as (ymd, cyc).
    """
    if now_utc is None:
        now_utc = dt.datetime.now(dt.UTC)
    for hour in (18, 12, 6, 0):
        candidate = dt.datetime(now_utc.year, now_utc.month, now_utc.day, hour, tzinfo=dt.UTC)
        if candidate <= now_utc:
            return candidate.strftime("%Y%m%d"), f"{hour:02d}"
    y = now_utc - dt.timedelta(days=1)
    return y.strftime("%Y%m%d"), "18"

def list_files_with_exts(dirpath: str, exts: Iterable[str]) -> List[str]:
    exts = set(e.lower() for e in exts)
    out = []
    for name in os.listdir(dirpath):
        p = os.path.join(dirpath, name)
        if os.path.isfile(p) and os.path.splitext(name)[1].lower() in exts:
            out.append(p)
    return out
