# analysisFunctions.py
import os, re, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cartopy.crs as ccrs, cartopy.feature as cfeature

# ─────────────────────────────── helpers ───────────────────────────────

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

# ─────────────────────────────── plotting ──────────────────────────────

def plot_maps(ds: xr.Dataset, out_dir: str, title_prefix: str, extent=None, stride=3, var_plan=None):
    """
    Generic pcolormesh plotter.
    extent=[leftlon,rightlon,bottomlat,toplat] or None for global.
    var_plan: list of (var, title, vmax)
    """
    ensure_dirs(out_dir)
    if "lat" not in ds or "lon" not in ds:
        print("lat/lon missing; skip maps.")
        return
    if var_plan is None:
        return

    times = range(0, ds.sizes.get("time", 1), max(1, stride))
    lon = ds["lon"]; lat = ds["lat"]

    for var, title, vmax in var_plan:
        if var not in ds:
            continue
        for ti in times:
            da = ds[var].isel(time=ti) if "time" in ds[var].dims else ds[var]
            fig = plt.figure(figsize=(10, 6))
            ax = plt.axes(projection=ccrs.PlateCarree())
            if extent:
                ax.set_extent([extent[0],extent[1],extent[2],extent[3]], crs=ccrs.PlateCarree())
            else:
                ax.set_global()
            ax.coastlines(linewidth=0.6)
            ax.add_feature(cfeature.BORDERS, linewidth=0.4)
            im = ax.pcolormesh(lon, lat, da, transform=ccrs.PlateCarree(), shading="auto", vmax=vmax)
            cb = plt.colorbar(im, orientation="horizontal", pad=0.05, shrink=0.9)
            cb.set_label(title)
            ts = str(pd.to_datetime(ds.time.values[ti])) if "time" in ds.dims else ""
            plt.title(f"{title_prefix}: {title} — {ts} UTC")
            out = os.path.join(out_dir, f"map_{var}_{ti:03d}.png")
            fig.savefig(out, dpi=130); plt.close(fig)

def plot_means(ds: xr.Dataset, out_dir: str, title_prefix: str, vars_of_interest):
    ensure_dirs(out_dir)
    if "time" not in ds:
        return
    t = pd.to_datetime(ds["time"].values)
    for v in vars_of_interest:
        if v not in ds:
            continue
        mean_ts = ds[v].mean(dim=("lat","lon"), skipna=True)
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(t, mean_ts.values)
        ax.set_title(f"{title_prefix} mean {v}")
        ax.set_xlabel("time (UTC)")
        ax.set_ylabel(ds[v].attrs.get("units",""))
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"ts_mean_{v}.png"), dpi=130)
        plt.close(fig)

# json manifest generator

_MANIFEST_PATH = "./web/manifest.json"
_MANIFEST_BASEDIR = os.path.dirname(_MANIFEST_PATH) or "."

_TI_RE = re.compile(r"_(\d{3})\.png$")

def _rel(path: str) -> str:
    """Make a path relative to ./web so the browser can load it on GitHub Pages or local http-server."""
    return os.path.relpath(path, start=_MANIFEST_BASEDIR).replace("\\", "/")

def _sorted_frames_for_var(map_dir: str, var: str):
    """Return a sorted list of frame paths for var, based on map_{var}_NNN.png."""
    import glob
    patt = os.path.join(map_dir, f"map_{var}_*.png")
    files = []
    for p in glob.glob(patt):
        m = _TI_RE.search(p)
        if not m:
            continue
        idx = int(m.group(1))
        files.append((idx, p))
    files.sort(key=lambda t: t[0])
    return [ _rel(p) for _, p in files ]

def _collect_var_frames(map_dir: str, candidate_vars):
    out = {}
    for v in candidate_vars:
        frames = _sorted_frames_for_var(map_dir, v)
        if frames:
            out[v] = frames
    return out

def _collect_times_list_from_ds(ds, stride: int = 1) -> list[str]:
    """Convert ds.time to ISO strings (UTC), optionally sub-sampled by 'stride'."""
    if "time" not in ds:
        return []
    t = pd.to_datetime(ds["time"].values)
    t = t[::max(1, stride)]
    return [pd.Timestamp(tt).tz_localize("UTC", nonexistent="NaT", ambiguous="NaT").isoformat().replace("+00:00","Z")
            for tt in t]

def _collect_timeseries_images(ts_dir: str, candidate_vars):
    out = {}
    for v in candidate_vars:
        for stem in (f"ts_global_mean_{v}.png", f"ts_regional_mean_{v}.png", f"ts_mean_{v}.png"):
            p = os.path.join(ts_dir, stem)
            if os.path.exists(p):
                out[v] = _rel(p)
                break
    return out

def _write_manifest_safe(man: dict):
    os.makedirs(_MANIFEST_BASEDIR, exist_ok=True)
    man["generated_at"] = pd.Timestamp.utcnow().isoformat(timespec="seconds") + "Z"
    tmp = _MANIFEST_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(man, f, indent=2)
    os.replace(tmp, _MANIFEST_PATH)

def _load_manifest_safe() -> dict:
    try:
        with open(_MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _collect_per_country_assets(hist_dir: str, stats_dir: str):
    """
    Discover:
      histograms/<Country>/hist_<var>.png
      stats/descriptive_<Country>.csv
    Return {Country: {"histograms": {var: relpath}, "stats": {var: relpath-to-csv}}}
    """
    import glob
    out = {}
    # histograms
    if os.path.isdir(hist_dir):
        for country in os.listdir(hist_dir):
            cdir = os.path.join(hist_dir, country)
            if not os.path.isdir(cdir):
                continue
            for p in glob.glob(os.path.join(cdir, "hist_*.png")):
                var = os.path.splitext(os.path.basename(p))[0].replace("hist_","")
                out.setdefault(country, {}).setdefault("histograms", {})[var] = _rel(p)
    # stats (one CSV per country; map it to all vars so UI can pick the same file then filter)
    if os.path.isdir(stats_dir):
        for p in glob.glob(os.path.join(stats_dir, "descriptive_*.csv")):
            country = os.path.splitext(os.path.basename(p))[0].replace("descriptive_","")
            stats_map = out.setdefault(country, {}).setdefault("stats", {})
            # We don't know the list of vars here; leave map empty and let caller fill per-var,
            # or bind a generic handle—in practice we duplicate the file per var during emit.
            stats_map["__CSV__"] = _rel(p)
    return out

# expose for importers
collect_per_country_assets = _collect_per_country_assets

def update_manifest_for_domain(
    domain: str,
    cycle_date: str, cycle_hour: str,
    ds_for_times,                # xarray dataset (for time axis)
    map_dir: str,                # e.g., ./figures/global/maps_global
    candidate_vars: list[str],   # variables you attempted to plot
    ts_dir: str | None = None,   # timeseries dir (optional)
    per_country: dict | None = None,
    time_stride: int = 1         # must match plot_maps stride
):
    """
    Update ./web/manifest.json under key 'global' or 'regional'.

    - Frames are discovered in map_dir using filenames map_{var}_NNN.png.
    - Times come from ds_for_times['time'], sub-sampled by time_stride.
    - If ts_dir provided, adds timeseries[var] = PNG (if present).
    - If per_country provided, embeds it; if stats given as one CSV per country,
      we map that same CSV under every variable key so the web can filter.
    """
    frames = _collect_var_frames(map_dir, candidate_vars)
    times  = _collect_times_list_from_ds(ds_for_times, stride=time_stride)

    payload = {
        "cycle": {"date": cycle_date, "hour": cycle_hour},
        "times": times,
        "variables": frames,
    }
    if ts_dir:
        payload["timeseries"] = _collect_timeseries_images(ts_dir, candidate_vars)

    # per-country
    if per_country:
        # normalize paths & replicate stats csv per var
        pc_out = {}
        for country, obj in per_country.items():
            h = obj.get("histograms", {})
            s = obj.get("stats", {})
            csv_one = s.get("__CSV__")
            s_out = {}
            if csv_one:
                # put the same CSV for each candidate var present in histograms
                for v in candidate_vars:
                    s_out[v] = csv_one
            else:
                # if already keyed per-var, make paths relative
                for v, p in s.items():
                    s_out[v] = _rel(p)
            h_out = {v: _rel(p) for v, p in h.items()}
            pc_out[country] = {"histograms": h_out, "stats": s_out}
        payload["per_country"] = pc_out

    man = _load_manifest_safe()
    man[domain] = payload
    _write_manifest_safe(man)
