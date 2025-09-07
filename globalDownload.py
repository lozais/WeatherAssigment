# globalDownload.py
import warnings; warnings.filterwarnings("ignore")
import os, re, tempfile, requests, logging, numpy as np, pandas as pd, xarray as xr
from typing import Tuple, Optional, List
from downloadFunctions import ensure_dir, latest_synoptic_cycle, write_cycle_marker, safe_remove_files

LOG = logging.getLogger("pipeline.global")

DATA_DIR = "./data/global"
ensure_dir(DATA_DIR)

FILTER_BASE    = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
GFS_PROD_BASE  = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod"
CYCLES         = ["18","12","06","00"]
FORECAST_HOURS = list(range(0, 169, 3))

GROUPS = [
    ("sfc",   ["APCP","PRATE","CPRAT","VIS","GUST"],          ["surface"]),
    ("ag2",   ["TMP","RH"],                                   ["2_m_above_ground"]),
    ("ag10",  ["UGRD","VGRD"],                                ["10_m_above_ground"]),
    ("pl850", ["TMP","RH","UGRD","VGRD","VVEL"],              ["850_mb"]),
    ("pwat",  ["PWAT"],                                       ["entire_atmosphere_(considered_as_a_single_layer)"]),
]

TCDC_LEVEL_PRIMARY   = "entire_atmosphere_(considered_as_a_single_layer)"
TCDC_LEVEL_FALLBACK  = "850_mb"
_FHR_RE = re.compile(r"\.f(\d{3})")

def fhour_filename(cyc, fhr):
    return f"gfs.t{cyc}z.pgrb2.0p25.f{fhr:03d}"

def _url_exists(url, timeout=8):
    try:
        r = requests.head(url, timeout=timeout)
        if r.status_code == 200: return True
    except Exception: pass
    try:
        r = requests.get(url, headers={"Range":"bytes=0-0"}, timeout=timeout)
        return r.status_code in (200,206)
    except Exception:
        return False

def find_latest_online_cycle() -> Tuple[str,str]:
    for day in (pd.Timestamp.utcnow().date(), (pd.Timestamp.utcnow()-pd.Timedelta(days=1)).date()):
        ymd = pd.Timestamp(day).strftime("%Y%m%d")
        for cyc in CYCLES:
            f000 = f"{GFS_PROD_BASE}/gfs.{ymd}/{cyc}/atmos/{fhour_filename(cyc,0)}"
            if _url_exists(f000):
                return ymd, cyc
    # fallback if NOMADS slow → synoptic guess
    return latest_synoptic_cycle()

def _build_filter_url(ymd, cyc, fhr, vars_list, levels_list):
    file = fhour_filename(cyc, fhr)
    q = {"file": file, "dir": f"/gfs.{ymd}/{cyc}/atmos"}
    for v in vars_list: q[f"var_{v}"] = "on"
    for L in levels_list: q[f"lev_{L}"] = "on"
    parts = [f"{k}={requests.utils.quote(v) if isinstance(v,str) else v}" for k, v in q.items()]
    return f"{FILTER_BASE}?" + "&".join(parts)

def _download_group_to_temp(url, timeout=240):
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            ctype = (r.headers.get("Content-Type","") or "").lower()
            if r.status_code != 200 or "text" in ctype or "html" in ctype or "xml" in ctype:
                return None, f"http-{r.status_code}:{ctype}"
            fd, tmp = tempfile.mkstemp(suffix=".grb"); os.close(fd)
            size = 0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1024*512):
                    if chunk: f.write(chunk); size += len(chunk)
            if size == 0: os.remove(tmp); return None, "empty"
            with open(tmp, "rb") as f:
                if not f.read(4).startswith(b"GRIB"):
                    os.remove(tmp); return None, "not-grib"
            return tmp, size
    except Exception as e:
        return None, f"exc:{e}"

def _append_file(dst_path, src_path, first=False):
    mode = "wb" if first else "ab"
    with open(dst_path, mode) as out, open(src_path, "rb") as src:
        while True:
            chunk = src.read(1024*512)
            if not chunk: break
            out.write(chunk)

def download_general_hour(ymd, cyc, fhr) -> Optional[str]:
    out = os.path.join(DATA_DIR, fhour_filename(cyc, fhr).replace(".pgrb2.0p25",".subset"))
    if os.path.exists(out) and os.path.getsize(out) > 4096:
        return out
    if os.path.exists(out):
        try: os.remove(out)
        except Exception: pass
    appended = 0; first = True
    for _, gvars, glevels in GROUPS:
        url = _build_filter_url(ymd, cyc, fhr, gvars, glevels)
        tmp, status = _download_group_to_temp(url)
        if tmp is None:
            continue
        try:
            _append_file(out, tmp, first=first)
            appended += 1; first = False
        finally:
            try: os.remove(tmp)
            except Exception: pass
    if appended == 0:
        if os.path.exists(out):
            try: os.remove(out)
            except Exception: pass
        return None
    return out

def download_tcdc_hour(ymd, cyc, fhr) -> Optional[str]:
    out1 = os.path.join(DATA_DIR, f"gfs.t{cyc}z.tcdc.entire_atmosphere.f{fhr:03d}.subset")
    if os.path.exists(out1) and os.path.getsize(out1) > 0: return out1
    url1 = _build_filter_url(ymd, cyc, fhr, ["TCDC"], [TCDC_LEVEL_PRIMARY])
    tmp, _ = _download_group_to_temp(url1)
    if tmp:
        _append_file(out1, tmp, first=True); os.remove(tmp); return out1
    out2 = os.path.join(DATA_DIR, f"gfs.t{cyc}z.tcdc.850_mb.f{fhr:03d}.subset")
    if os.path.exists(out2) and os.path.getsize(out2) > 0: return out2
    url2 = _build_filter_url(ymd, cyc, fhr, ["TCDC"], [TCDC_LEVEL_FALLBACK])
    tmp, _ = _download_group_to_temp(url2)
    if tmp:
        _append_file(out2, tmp, first=True); os.remove(tmp); return out2
    return None

def _open_grib_all_groups(path: str) -> xr.Dataset:
    import cfgrib
    dsets = cfgrib.open_datasets(path, backend_kwargs={"indexpath": ""})
    fixed = []
    for ds in dsets:
        if "valid_time" not in ds and "time" in ds and "step" in ds:
            try: ds = ds.assign_coords(valid_time=ds["time"] + ds["step"])
            except Exception: pass
        if "latitude" in ds and "lat" not in ds: ds = ds.rename({"latitude":"lat"})
        if "longitude" in ds and "lon" not in ds: ds = ds.rename({"longitude":"lon"})
        fixed.append(ds)
    return xr.merge(fixed, compat="override", join="outer")

def _stack_from_files(files: List[str], base_ts_utc: pd.Timestamp) -> xr.Dataset:
    per_time = []
    for path in files:
        try:
            ds = _open_grib_all_groups(path)
        except Exception as e:
            LOG.warning(f"open fail {os.path.basename(path)}: {e}")
            continue
        if "valid_time" in ds:
            vt = pd.to_datetime(ds["valid_time"].values)
            vt = np.atleast_1d(vt)[0]
            if pd.isnull(vt):
                fhr = int(_FHR_RE.search(os.path.basename(path)).group(1))
                vt = base_ts_utc + pd.Timedelta(hours=fhr)
        else:
            fhr = int(_FHR_RE.search(os.path.basename(path)).group(1))
            vt = base_ts_utc + pd.Timedelta(hours=fhr)
        if getattr(vt,"tzinfo",None) is None: vt = pd.Timestamp(vt, tz="UTC")
        vt64 = np.datetime64(vt.tz_convert("UTC").tz_localize(None).to_datetime64())
        keep_coords = [c for c in ("lat","lon","time","step","valid_time") if c in ds.coords]
        sub = ds[list(ds.data_vars) + keep_coords].copy()
        sub = sub.expand_dims(time=[vt64])
        per_time.append(sub)
    if not per_time:
        raise RuntimeError("No valid global files to stack.")
    return xr.concat(per_time, dim="time").sortby("time")

def _normalize_and_derive(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.copy()
    if all(k in ds for k in ("10u","10v")):
        ds["wind10"] = np.hypot(ds["10u"], ds["10v"]).astype("float32")
        ds["wind10"].attrs.update(units="m s-1", long_name="10 m wind speed")
    if "2t" in ds:
        ds["t2m"] = (ds["2t"] - 273.15).astype("float32")
        ds["t2m"].attrs.update(units="degC", long_name="2 m temperature")
    if all(k in ds for k in ("u","v")):
        ds["wind850"] = np.hypot(ds["u"], ds["v"]).astype("float32")
        ds["wind850"].attrs.update(units="m s-1", long_name="850 hPa wind speed")
    if "time" in ds:
        t = pd.to_datetime(ds["time"].values)
        step_sec = int(np.median(np.diff(t).astype("timedelta64[s]").astype(int))) if t.size>=2 else 3*3600
        if "apcp" in ds:
            ds["precip_step"] = ds["apcp"].astype("float32")
            ds["precip_step"].attrs.update(units="mm", long_name="total precipitation per step (APCP)")
        elif "prate" in ds:
            ds["precip_step"] = (ds["prate"] * step_sec).astype("float32")
            ds["precip_step"].attrs.update(units="mm", long_name="total precipitation per step (from PRATE)")
        if "cprat" in ds:
            ds["cprecip_step"] = (ds["cprat"] * step_sec).astype("float32")
            ds["cprecip_step"].attrs.update(units="mm", long_name="convective precipitation per step (from CPRAT)")
    return ds

def _normalize_tcdc(ds_tcc: xr.Dataset) -> xr.Dataset:
    if ds_tcc is None: return None
    tcc_names = [v for v in ds_tcc.data_vars if v.lower()=="tcc"]
    if not tcc_names:
        tcc_names = [v for v in ds_tcc.data_vars if "cloud" in ds_tcc[v].attrs.get("long_name","").lower()]
    if not tcc_names:
        if len(ds_tcc.data_vars)==1: tcc_names=[list(ds_tcc.data_vars)[0]]
        else: return ds_tcc
    da = ds_tcc[tcc_names[0]]
    arr = da.values
    if np.isfinite(arr).any() and np.nanmax(arr) <= 1.001:
        da = (da * 100.0).astype("float32"); da.attrs["units"]="%"
    da.name = "tcc"
    return xr.Dataset({"tcc": da}, coords=ds_tcc.coords, attrs=ds_tcc.attrs)

def download_and_build_stack(ymd: Optional[str]=None, cyc: Optional[str]=None) -> Tuple[str,str,str]:
    """Download all hours (general+tcdc), build stack, save NetCDF, delete raw files."""
    if ymd is None or cyc is None:
        ymd, cyc = find_latest_online_cycle()
    LOG.info(f"Global: downloading GFS {ymd} {cyc}Z …")
    base_ts = pd.Timestamp(f"{ymd} {cyc}:00", tz="UTC")

    gen_files, tcc_files = [], []
    for fhr in FORECAST_HOURS:
        p = download_general_hour(ymd, cyc, fhr)
        if p: gen_files.append(p)
        q = download_tcdc_hour(ymd, cyc, fhr)
        if q: tcc_files.append(q)
    if not gen_files:
        raise RuntimeError("Global download produced no general files.")
    gen = _stack_from_files(sorted(gen_files), base_ts)
    gen = _normalize_and_derive(gen)

    tcc_ds = None
    if tcc_files:
        tcc_raw = _stack_from_files(sorted(tcc_files), base_ts)
        tcc_ds  = _normalize_tcdc(tcc_raw)
        if tcc_ds is not None:
            keep = ["tcc"] + [c for c in ("time","lat","lon") if c in tcc_ds.coords]
            tcc_ds = tcc_ds[keep]

    merged = gen if tcc_ds is None else xr.merge([gen, tcc_ds], compat="override", join="outer")
    merged.attrs["gfs_cycle_date"] = ymd
    merged.attrs["gfs_cycle_hour"] = cyc

    out_nc = os.path.join(DATA_DIR, f"gfs_stack_{ymd}_{cyc}.nc")
    tmp_nc = out_nc + ".tmp"
    if os.path.exists(tmp_nc): os.remove(tmp_nc)
    merged.to_netcdf(tmp_nc)
    if os.path.exists(out_nc): os.remove(out_nc)
    os.rename(tmp_nc, out_nc)
    write_cycle_marker(DATA_DIR, ymd, cyc)
    LOG.info(f"Global: saved {out_nc}")

    safe_remove_files(gen_files + tcc_files)
    return ymd, cyc, out_nc
