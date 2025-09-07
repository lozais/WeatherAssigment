# regionalDownload.py
import os, datetime as dt, requests, logging, numpy as np, pandas as pd, xarray as xr
import warnings
from typing import List, Tuple, Optional

from fmiopendata.wfs import download_stored_query
from downloadFunctions import ensure_dir, write_cycle_marker, latest_synoptic_cycle, safe_remove_files

warnings.filterwarnings("ignore")
LOG = logging.getLogger("pipeline.regional")

DATA_DIR = "./data/regional"
ensure_dir(DATA_DIR)

# Scandinavia + Baltics bbox [lon1,lat1,lon2,lat2]
BBOX = (-28.0, 44.0, 37.0, 74.0)

# Ask for cumulative + 1h precip, and other vars.
PARAMS = [
    "Temperature",
    "Humidity",
    "WindUMS", "WindVMS",
    "PrecipitationAmount",     # cumulative since model start (preferred)
    "Precipitation1h",         # one-hour amount (sometimes present)
    "TotalCloudCover",
    "Visibility",
    "WindGust",
]

TIMESTEP_MIN = 60   # 1h
MAX_LEAD_HOURS = 66
FORECAST_HOURS = list(range(0, MAX_LEAD_HOURS + 1, 1))

def _fmt_iso(ymd: str, cyc: str, fhr: int) -> str:
    base = dt.datetime.strptime(f"{ymd}{cyc}", "%Y%m%d%H").replace(tzinfo=dt.UTC)
    t = base + dt.timedelta(hours=fhr)
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")

def _as_aware_utc(x):
    try:
        if isinstance(x, dt.datetime):
            return x if x.tzinfo else x.replace(tzinfo=dt.UTC)
        s = str(x)
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(dt.UTC)
    except Exception:
        return None

def _choose_init_for_cycle(model_data, ymd, cyc, target_time_iso):
    items = []
    for k in getattr(model_data, "data", {}).keys():
        ak = _as_aware_utc(k)
        if ak is not None:
            items.append((ak, k))
    if not items:
        return None
    items.sort(key=lambda t: t[0])
    wanted = dt.datetime.strptime(f"{ymd}{cyc}", "%Y%m%d%H").replace(tzinfo=dt.UTC)
    for ak, ok in items:
        if ak == wanted:
            return ok
    target = dt.datetime.fromisoformat(target_time_iso.replace("Z", "+00:00")).astimezone(dt.UTC)
    le = [(ak, ok) for ak, ok in items if ak <= target]
    return le[-1][1] if le else items[-1][1]

def _ext_from_response(url: str, ctype: str) -> str:
    u = (url or "").lower()
    c = (ctype or "").lower()
    if "grib" in c or ".grib" in u: return ".grib2"
    if "netcdf" in c or ".nc" in u or "hdf" in c: return ".nc"
    return ".bin"

def _save_binary(href: str, out_path: str, timeout=300) -> Optional[str]:
    with requests.get(href, stream=True, timeout=timeout) as r2:
        ctype = (r2.headers.get("Content-Type","") or "").lower()
        if r2.status_code != 200 or any(s in ctype for s in ("xml","html","text/")):
            try: snippet = r2.text[:280]
            except Exception: snippet = ""
            open(out_path + ".err.txt", "w", encoding="utf-8").write(
                f"BIN HTTP {r2.status_code} {ctype}\nURL: {href}\n{snippet}"
            )
            return None
        ext = _ext_from_response(href, ctype)
        final = os.path.splitext(out_path)[0] + ext
        tmp = final + ".tmp"
        with open(tmp, "wb") as f:
            for ch in r2.iter_content(1024*512):
                f.write(ch)
        if os.path.getsize(tmp) == 0:
            os.remove(tmp)
            open(final + ".err.txt", "w", encoding="utf-8").write("Empty binary")
            return None
        if os.path.exists(final): os.remove(final)
        os.rename(tmp, final)
        return final

def _download_hour(ymd, cyc, fhr) -> Optional[str]:
    start_iso = _fmt_iso(ymd, cyc, fhr)
    out_stub = os.path.join(DATA_DIR, f"harmonie.t{cyc}z.surface.f{fhr:03d}.grib2")  # ext may change
    base, _ = os.path.splitext(out_stub)
    for ext in (".grib2", ".nc", ".bin"):
        cand = base + ext
        if os.path.exists(cand) and os.path.getsize(cand) > 0:
            return cand

    args = [
        "starttime=" + start_iso,
        "endtime="   + start_iso,  # 1-hour window
        f"bbox={BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]}",
        "timestep=" + str(TIMESTEP_MIN),
        "parameters=" + ",".join(PARAMS),
        "format=grib2",
    ]
    try:
        md = download_stored_query("fmi::forecast::harmonie::surface::grid", args=args)
    except Exception as e:
        open(out_stub + ".err.txt", "w", encoding="utf-8").write(f"WFS call failed: {e}")
        return None
    if not getattr(md, "data", None):
        open(out_stub + ".err.txt", "w", encoding="utf-8").write("WFS returned no data.")
        return None
    init_key = _choose_init_for_cycle(md, ymd, cyc, start_iso)
    if not init_key:
        open(out_stub + ".err.txt", "w", encoding="utf-8").write("No suitable init time in WFS result.")
        return None
    grid = md.data[init_key]
    href = getattr(grid, "url", None)
    if not href:
        open(out_stub + ".err.txt", "w", encoding="utf-8").write("Grid object had no url.")
        return None
    return _save_binary(href, out_stub)

def _sniff_magic(path: str) -> str:
    with open(path, "rb") as f:
        head = f.read(8)
    if head.startswith(b"GRIB"): return "grib"
    if head.startswith(b"\x89HDF\r\n\x1a\n"): return "hdf5"
    if head.startswith(b"CDF"): return "nc3"
    return "unknown"

def _open_any(path: str) -> xr.Dataset:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".nc", ".nc4", ".cdf"):
        for eng in (None, "netcdf4", "h5netcdf", "scipy"):
            try:
                return xr.open_dataset(path) if eng is None else xr.open_dataset(path, engine=eng)
            except Exception:
                pass
        raise RuntimeError(f"Could not open NetCDF: {os.path.basename(path)}")
    if ext in (".grb", ".grib", ".grib2") or (ext == ".bin" and _sniff_magic(path) == "grib"):
        import cfgrib
        dsets = cfgrib.open_datasets(path, backend_kwargs={"indexpath": ""})
        return xr.merge(dsets, compat="override", join="outer")
    if ext == ".bin":
        magic = _sniff_magic(path)
        if magic in ("hdf5", "nc3"):
            for eng in ("netcdf4", "h5netcdf", None, "scipy"):
                try:
                    return xr.open_dataset(path) if eng is None else xr.open_dataset(path, engine=eng)
                except Exception:
                    pass
    raise RuntimeError(f"Unsupported file: {os.path.basename(path)}")

def _promote_latlon(ds: xr.Dataset) -> xr.Dataset:
    if "latitude" in ds and "lat" not in ds: ds = ds.rename({"latitude": "lat"})
    if "longitude" in ds and "lon" not in ds: ds = ds.rename({"longitude": "lon"})
    for name in ("lat", "lon"):
        if name in ds.variables and name not in ds.coords:
            ds = ds.set_coords(name)
    return ds

def _fhr_from_name(name: str) -> int:
    import re
    m = re.search(r"\.f(\d{3})", name)
    return int(m.group(1)) if m else 0

def _stack_and_harmonize(files: List[str], ymd: str, cyc: str) -> xr.Dataset:
    import numpy as np
    import pandas as pd

    per_t = []
    for p in sorted(files):
        try:
            ds = _open_any(p)
        except Exception as e:
            LOG.warning(f"open fail {os.path.basename(p)}: {e}")
            continue
        ds = _promote_latlon(ds)

        # valid time selection
        vt = None
        try:
            if "valid_time" in ds.coords and ds.coords["valid_time"].size >= 1:
                vt = np.atleast_1d(pd.to_datetime(ds.coords["valid_time"].values))[0]
            elif "time" in ds.coords and "step" in ds.coords:
                vt = np.atleast_1d(pd.to_datetime((ds["time"] + ds["step"]).values))[0]
            elif "forecast_time" in ds.coords and ds.coords["forecast_time"].size >= 1:
                vt = np.atleast_1d(pd.to_datetime(ds.coords["forecast_time"].values))[0]
        except Exception:
            vt = None
        if vt is None:
            fhr = _fhr_from_name(os.path.basename(p))
            base = pd.Timestamp(f"{ymd} {cyc}:00", tz="UTC")
            vt = base + pd.Timedelta(hours=fhr)

        if getattr(vt, "tzinfo", None) is None:
            vt = pd.Timestamp(vt, tz="UTC")
        vt64 = np.datetime64(vt.tz_convert("UTC").tz_localize(None).to_datetime64())
        per_t.append(ds.expand_dims(time=[vt64]))

    if not per_t:
        raise RuntimeError("No valid regional files to stack.")
    merged = xr.concat(per_t, dim="time")

    # If times are duplicated or not strictly increasing, rebuild from filenames
    tvals = pd.to_datetime(merged["time"].values)
    if pd.Series(tvals).duplicated().any():
        LOG.warning("Regional stack had duplicate times; rebuilding times from filenames.")
        new_times = []
        for p in sorted(files):
            fhr = _fhr_from_name(os.path.basename(p))
            base = pd.Timestamp(f"{ymd} {cyc}:00", tz="UTC")
            vt = base + pd.Timedelta(hours=fhr)
            new_times.append(np.datetime64(vt.tz_convert("UTC").tz_localize(None).to_datetime64()))
        merged = merged.assign_coords(time=("time", new_times))

    merged = merged.sortby("time")

    # ---- Harmonize names ----
    rename_map = {
        "u10": "10u", "v10": "10v", "r2": "2r",
        "WindUMS": "10u", "WindVMS": "10v",
        "Humidity": "2r", "RelativeHumidity": "2r",
        "Temperature": "t2m",
        "Visibility": "vis", "WindGust": "gust",
        "TotalCloudCover": "tcc",
        # precip aliases
        "Precipitation1h": "precipitation1h",
        "rr": "tp",  # some exports use rr for accumulated precip
    }
    for src, dst in rename_map.items():
        if src in merged and dst not in merged:
            merged = merged.rename({src: dst})

    # ---- Derivations ----
    if "10u" in merged and "10v" in merged:
        merged["wind10"] = np.hypot(merged["10u"], merged["10v"]).astype("float32")
        merged["wind10"].attrs.update(units="m s-1", long_name="10 m wind speed")

    if "t2m" in merged:
        t2 = merged["t2m"]; u = (t2.attrs.get("units", "") or "").lower()
        if "k" in u or (float(t2.max().item()) if t2.size else 300) > 200:
            merged["t2m"] = (t2 - 273.15).astype("float32")
        merged["t2m"].attrs.update(units="degC", long_name="2 m temperature")

    def _ensure_pct(da):
        arr = da.values
        if np.isfinite(arr).any() and np.nanmax(arr) <= 1.001:
            da = (da * 100.0).astype("float32"); da.attrs["units"] = "%"
        return da

    if "2r" in merged: merged["2r"] = _ensure_pct(merged["2r"].astype("float32"))
    if "tcc" in merged: merged["tcc"] = _ensure_pct(merged["tcc"].astype("float32"))
    if "vis" in merged: merged["vis"] = merged["vis"].astype("float32")
    if "gust" in merged: merged["gust"] = merged["gust"].astype("float32")

    # Precipitation, if precip_step (mm per step)
    def _cum_to_step(cum_da: xr.DataArray) -> xr.DataArray:
        u = (cum_da.attrs.get("units", "") or "").lower()
        scale = 1000.0 if u.strip() in ("m", "meter", "metre", "m of water", "m w.e.") else 1.0  # kg m-2 already mm
        cum_mm = (cum_da * scale).astype("float32")
        step = cum_mm.diff("time")
        zero0 = cum_mm.isel(time=0) * 0.0
        step = xr.concat([zero0, step], dim="time")
        step = xr.where(step < 0, 0.0, step)  # guard against resets
        step.name = "precip_step"
        step.attrs.update(units="mm", long_name="total precipitation per step (from cumulative)")
        return step

    used_source = "none"
    if "tp" in merged:
        merged["precip_step"] = _cum_to_step(merged["tp"])
        used_source = "tp(cumulative)"
    elif "PrecipitationAmount" in merged:
        merged["precip_step"] = _cum_to_step(merged["PrecipitationAmount"])
        used_source = "PrecipitationAmount(cumulative)"
    elif "precipitation1h" in merged:
        merged["precip_step"] = merged["precipitation1h"].astype("float32")
        merged["precip_step"].attrs.update(units="mm", long_name="total precipitation per step (1h field)")
        used_source = "Precipitation1h"
    elif "prate" in merged:
        t = pd.to_datetime(merged["time"].values)
        step_sec = int(np.median(np.diff(t).astype("timedelta64[s]").astype(int))) if t.size >= 2 else 3600
        merged["precip_step"] = (merged["prate"].astype("float32") * step_sec).astype("float32")
        merged["precip_step"].attrs.update(units="mm", long_name="total precipitation per step (from PRATE)")
        used_source = f"prate*{step_sec}s"

    try:
        if "precip_step" in merged:
            pmin = float(merged["precip_step"].min(skipna=True).item())
            pmax = float(merged["precip_step"].max(skipna=True).item())
            LOG.info(f"Regional precip: source={used_source}, min={pmin:.3f}, max={pmax:.3f}")
        else:
            LOG.warning("Regional precip: no usable source found – precip_step missing.")
    except Exception:
        pass

    keep = [c for c in ("time", "lat", "lon") if c in merged.coords] + \
           [v for v in ("10u","10v","wind10","t2m","2r","precip_step","tcc","vis","gust") if v in merged]
    return merged[keep]

def download_and_build_stack(ymd: Optional[str] = None, cyc: Optional[str] = None) -> Tuple[str, str, str]:
    """Download all hours, build stack, save NetCDF, delete raw files. Returns (ymd, cyc, nc_path)."""
    if ymd is None or cyc is None:
        ymd, cyc = latest_synoptic_cycle()
    LOG.info(f"Regional: downloading HARMONIE {ymd} {cyc}Z …")

    files = []
    for fhr in FORECAST_HOURS:
        p = _download_hour(ymd, cyc, fhr)
        if p: files.append(p)
    if not files:
        raise RuntimeError("Regional download produced no files.")

    ds = _stack_and_harmonize(files, ymd, cyc)
    ds.attrs["harmonie_cycle_date"] = ymd
    ds.attrs["harmonie_cycle_hour"] = cyc

    out_nc = os.path.join(DATA_DIR, f"harmonie_stack_{ymd}_{cyc}.nc")
    tmp_nc = out_nc + ".tmp"
    if os.path.exists(tmp_nc): os.remove(tmp_nc)
    ds.to_netcdf(tmp_nc)
    if os.path.exists(out_nc): os.remove(out_nc)
    os.rename(tmp_nc, out_nc)
    write_cycle_marker(DATA_DIR, ymd, cyc)
    LOG.info(f"Regional: saved {out_nc}")

    safe_remove_files(files)
    return ymd, cyc, out_nc
