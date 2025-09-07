# analysisGlobal.py
import os, warnings, logging
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
from analysisFunctions import ensure_dirs, plot_maps, plot_means, update_manifest_for_domain

LOG = logging.getLogger("pipeline.analysis.global")

DATA_DIR = "./data/global"
FIG_DIR  = "./figures/global"
MAP_DIR  = os.path.join(FIG_DIR, "maps_global")
TS_DIR   = os.path.join(FIG_DIR, "timeseries_global")
ensure_dirs(FIG_DIR, MAP_DIR, TS_DIR)


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
    LOG.info(f"Global analysis from {nc_path}")
    ds = xr.open_dataset(nc_path)

    plan = []
    if "tcc" in ds:          plan.append(("tcc",          "Total Cloud Cover (%)", None))
    if "wind10" in ds:       plan.append(("wind10",       "10 m Wind Speed (m s-1)", None))
    if "wind850" in ds:      plan.append(("wind850",      "850 hPa Wind Speed (m s-1)", None))
    if "w" in ds:            plan.append(("w",            "850 hPa Vertical Velocity (Pa s-1)", None))
    if "t" in ds:            plan.append(("t",            "850 hPa Temperature (K)", None))
    if "r" in ds:            plan.append(("r",            "850 hPa Relative Humidity (%)", None))
    if "pwat" in ds:         plan.append(("pwat",         "Precipitable Water (kg m-2)", None))
    if "t2m" in ds:          plan.append(("t2m",          "2 m Temperature (°C)", None))
    if "vis" in ds:          plan.append(("vis",          "Visibility (m)", None))
    if "gust" in ds:         plan.append(("gust",         "10 m Wind Gust (m s-1)", None))
    if "precip_step" in ds:  plan.append(("precip_step",  "Precip per Step (mm)", 20.0))
    if "cprecip_step" in ds: plan.append(("cprecip_step", "Convective Precip per Step (mm)", 20.0))

    # IMPORTANT: stride = 4 here → frames and manifest times will match every 4th timestep
    plot_maps(ds, MAP_DIR, "Global", extent=None, stride=4, var_plan=plan)
    plot_means(ds, TS_DIR, "Global", vars_of_interest=("t2m","pwat","tcc","wind10","precip_step"))

    # Manifest
    ymd, cyc = _read_cycle_marker(DATA_DIR)
    if not ymd or not cyc:
        LOG.warning("No global cycle marker found; skipping manifest update.")
        return

    candidate_vars = ["tcc","wind10","wind850","w","t","r","pwat","t2m","vis","gust","precip_step","cprecip_step"]

    try:
        update_manifest_for_domain(
            domain="global",
            cycle_date=ymd,
            cycle_hour=cyc,
            ds_for_times=ds,
            map_dir=MAP_DIR,
            candidate_vars=candidate_vars,
            ts_dir=TS_DIR,
            per_country=None,
            time_stride=4,  # match plot_maps stride
        )
        LOG.info("web/manifest.json updated for GLOBAL.")
    except Exception as e:
        LOG.warning(f"Could not update manifest for GLOBAL: {e}")


if __name__ == "__main__":
    import glob
    pat = os.path.join(DATA_DIR, "gfs_stack_*.nc")
    nc_files = sorted(glob.glob(pat))
    if not nc_files:
        LOG.error(f"No NetCDFs found matching {pat}")
    else:
        run_analysis_from_nc(nc_files[-1])
