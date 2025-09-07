# analysisRegional.py
import os, warnings, logging
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
from typing import Dict, List

from analysisFunctions import (
    ensure_dirs, plot_maps, plot_means,
    update_manifest_for_domain, collect_per_country_assets
)

import cartopy.io.shapereader as shpreader
from shapely.prepared import prep as shapely_prep
from shapely.geometry import Point

LOG = logging.getLogger("pipeline.analysis.regional")

DATA_DIR = "./data/regional"
FIG_DIR  = "./figures/regional"
MAP_DIR  = os.path.join(FIG_DIR, "maps_regional")
TS_DIR   = os.path.join(FIG_DIR, "timeseries_regional")
HIST_DIR = os.path.join(FIG_DIR, "histograms")
STAT_DIR = os.path.join(FIG_DIR, "stats")

# Same extent used for download (left,right,bottom,top)
REGION_EXTENT = (-28.0, 37.0, 44.0, 74.0)

# Countries (Natural Earth Admin-0 names)
COUNTRIES = ["Finland","Sweden","Norway","Denmark","Estonia","Latvia","Lithuania","Iceland"]

def _load_country_polys(names: List[str]):
    shp = shpreader.natural_earth(resolution="50m", category="cultural", name="admin_0_countries")
    reader = shpreader.Reader(shp)
    out: Dict[str, List] = {n: [] for n in names}
    for rec in reader.records():
        name = rec.attributes.get("NAME_LONG") or rec.attributes.get("ADMIN") or rec.attributes.get("NAME")
        if name in out:
            geom = rec.geometry
            if geom:
                out[name].append(geom)
    return {k: v for k, v in out.items() if v}

def _country_masks(lat: np.ndarray, lon: np.ndarray, polys: Dict[str, List]):
    Ny, Nx = lat.shape
    lons = lon.ravel(); lats = lat.ravel()
    points = [Point(float(lons[i]), float(lats[i])) for i in range(lons.size)]
    masks = {}
    for name, geoms in polys.items():
        from shapely.ops import unary_union
        u = unary_union(geoms)
        pu = shapely_prep(u)
        m_flat = np.fromiter((pu.contains(pt) for pt in points), dtype=bool, count=len(points))
        masks[name] = m_flat.reshape(Ny, Nx)
    return masks

def _desc_stats(arr: np.ndarray) -> Dict[str, float]:
    a = arr[np.isfinite(arr)]
    if a.size == 0:
        return {"count": 0}
    return {
        "count": int(a.size),
        "mean": float(np.mean(a)),
        "std":  float(np.std(a)),
        "min":  float(np.min(a)),
        "p25":  float(np.percentile(a, 25)),
        "median": float(np.percentile(a, 50)),
        "p75":  float(np.percentile(a, 75)),
        "max":  float(np.max(a)),
        "frac_zeros": float(np.count_nonzero(a==0)/a.size),
    }

def _read_cycle_marker(data_dir: str):
    import glob
    markers = sorted(glob.glob(os.path.join(data_dir, "cycle_*.txt")))
    if not markers:
        return None, None
    txt = open(markers[-1], "r", encoding="utf-8").read().strip()
    parts = txt.split()
    if len(parts) >= 2:
        return parts[0], parts[1]
    if len(parts) == 1 and "_" in parts[0]:
        y, c = parts[0].split("_")
        return y, c
    return None, None

def run_analysis_from_nc(nc_path: str):
    LOG.info(f"Regional analysis from {nc_path}")
    ds = xr.open_dataset(nc_path)

    # Re-create output dirs (they may have been archived minutes ago)
    ensure_dirs(FIG_DIR, MAP_DIR, TS_DIR, HIST_DIR, STAT_DIR)

    # ----------------- Maps & time series -----------------
    plan = []
    if "tcc" in ds:          plan.append(("tcc",          "Total Cloud Cover (%)", None))
    if "wind10" in ds:       plan.append(("wind10",       "10 m Wind Speed (m s-1)", None))
    if "t2m" in ds:          plan.append(("t2m",          "2 m Temperature (°C)", None))
    if "2r" in ds:           plan.append(("2r",           "2 m Relative Humidity (%)", None))
    if "vis" in ds:          plan.append(("vis",          "Visibility (m)", None))
    if "gust" in ds:         plan.append(("gust",         "10 m Wind Gust (m s-1)", None))
    if "precip_step" in ds:  plan.append(("precip_step",  "Precip per Step (mm)", 10.0))

    # IMPORTANT: stride = 3 → frames and manifest times will match every 3rd timestep
    stride = 3
    plot_maps(ds, MAP_DIR, "Regional", extent=REGION_EXTENT, stride=stride, var_plan=plan)
    plot_means(ds, TS_DIR, "Regional", vars_of_interest=("t2m","tcc","wind10","precip_step"))

    # ----------------- Per-country histograms & stats -----------------
    if not ({"lat","lon"} <= set(ds.coords)):
        LOG.warning("lat/lon missing – cannot compute per-country stats.")
    else:
        lat = ds["lat"].values; lon = ds["lon"].values
        if lat.ndim == 1 and lon.ndim == 1:
            lon, lat = np.meshgrid(lon, lat)

        polys = _load_country_polys(COUNTRIES)
        if not polys:
            LOG.warning("No country polygons loaded; skipping per-country stats.")
        else:
            masks = _country_masks(lat, lon, polys)
            vars_for_stats = [v for v in ("t2m","2r","wind10","precip_step","tcc","vis","gust") if v in ds]
            ti = 0  # hist/desc from first time slice
            for country, mask in masks.items():
                cdir = os.path.join(HIST_DIR, country); ensure_dirs(cdir)
                rows = []
                for v in vars_for_stats:
                    da = ds[v].isel(time=ti) if "time" in ds[v].dims else ds[v]
                    arr = np.asarray(da.values, dtype="float32")
                    vals = arr[mask]
                    good = vals[np.isfinite(vals)]
                    if good.size == 0:
                        continue
                    # histogram
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(6,4))
                    ax.hist(good, bins=40)
                    ax.set_title(f"{country} – {v}")
                    ax.set_xlabel(f"{v} [{da.attrs.get('units','')}]")
                    ax.set_ylabel("count")
                    fig.tight_layout()
                    fig.savefig(os.path.join(cdir, f"hist_{v}.png"), dpi=120)
                    plt.close(fig)
                    # stats
                    rows.append((v, _desc_stats(good)))
                # save one CSV per country (tidy)
                if rows:
                    recs = []
                    for v, st in rows:
                        st2 = {"variable": v}; st2.update(st); recs.append(st2)
                    df = pd.DataFrame(recs)
                    ensure_dirs(STAT_DIR)
                    out_csv = os.path.join(STAT_DIR, f"descriptive_{country}.csv")
                    df.to_csv(out_csv, index=False, float_format="%.4f")
                    LOG.info(f"Wrote stats for {country}: {out_csv}")

    # ----------------- Manifest emit -----------------
    ymd, cyc = _read_cycle_marker(DATA_DIR)
    if not ymd or not cyc:
        LOG.warning("No regional cycle marker found; skipping manifest update.")
        return

    # collect assets and update manifest (the collector returns one CSV per country)
    assets = collect_per_country_assets(HIST_DIR, STAT_DIR)
    candidate_vars = ["tcc","wind10","t2m","2r","vis","gust","precip_step"]

    try:
        update_manifest_for_domain(
            domain="regional",
            cycle_date=ymd,
            cycle_hour=cyc,
            ds_for_times=ds,
            map_dir=MAP_DIR,
            candidate_vars=candidate_vars,
            ts_dir=TS_DIR,
            per_country=assets if assets else None,
            time_stride=stride,  # match plot stride
        )
        LOG.info("web/manifest.json updated for REGIONAL.")
    except Exception as e:
        LOG.warning(f"Could not update manifest for REGIONAL: {e}")
