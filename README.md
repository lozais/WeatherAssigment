								#########################
								###### README FILE ######
								#########################

###### Summary
1. Requirements
2. How to use
3. Outputs
4. Variable selection, level and justification
5. Files & Folders
6. Troubleshooting

########################
#1. Requirements

Make sure you have the following libraries before running the pipeline, with its main function in the pipeline
- requests — HTTP downloads 
- fmiopendata — FMI Open Data WFS client 
- numpy — arrays 
- pandas — timestamps / time axes
- xarray — open/stack/merge datasets 
- cfgrib — GRIB engine for xarray
- eccodes — backend for cfgrib 
- netCDF4 — NetCDF-4/HDF5 engine for xarray
- h5netcdf — alternative NetCDF-4 engine for xarray
- h5py — HDF5 support used by the above
- matplotlib — plotting
- cartopy — map projections and coastlines/borders
- shapely — geometry ops
- pyproj — PROJ projections
- pillow — image writer backend used by Matplotlib for PNG
- tqdm — for progress bars (optional)

########################
#2. How to use

a) Run weatherPipeline.py 
b) type "python -m http.server 8000" in the same folder where weatherPipeline.py is (root)
	Then open:
	http://localhost:8000/web/
	
Please take into consideration that the pipeline requieres time to download the data.
So in the console log you would see when the regional first, and then the global data
is downloaded, and analysis performed. Once these ends, you could see the outputs by
opening the web in localhost as showed above.

This light web viewer does not create maps or figures itself, is just a viewer for the
produced maps and figures by the analysis functions of the pipeline

########################################
#3. Outputs

The pipeline produce two sets of figures and also a .json file for the web viewer

a) Gobal scale GFS NOAA data
	The pipeline will download the data every single new cycle of data is released.
	Then in ./data/figures/global two folders will be created, one for maps and 
	one for histograms. Histograms are averaged data of some variables, interesting 
	to see global scale trends only.
	
b) Regional scale FMI Harmonie data
	Similar to Global scale, in ./data/figures/regional we have stored in folders
	maps for variables and regional scale histograms for the same variables than in 
	global. Additionally we can find folder for other histogram analysis at country 
	level, this allow us to comprehend better the regional scale maps.
	
These two sets of outputs can be seen through the webpage viewer.

c) Frame density (stride)
	Global: stride=4 in analysisGlobal.py
	Regional: stride=3 in analysisRegional.py
	(Maps are rendered every N time steps to keep figures reasonable.)

d) pipeline tree:

Output structure & naming
./data/
  ├── global/         # raw global files + cycle markers
  └── regional/       # raw regional files + cycle markers

./figures/
  ├── global/
  │   ├── maps_global/           # map_<var>_<NNN>.png (NNN = 000, 004, 008, ...)
  │   └── timeseries_global/     # ts_mean_<var>.png
  └── regional/
      ├── maps_regional/         # map_<var>_<NNN>.png (NNN = 000, 003, 006, ...)
      ├── timeseries_regional/   # ts_mean_<var>.png
      ├── histograms/<Country>/  # hist_<var>.png
      └── stats/                 # descriptive_<Country>.csv

./web/
  ├── index.html
  └── manifest.json   # generated registry of all the above

########################################
#4. Variable selection, level and justification
- t2m	2 m temperature	°C	Surface thermal conditions & extremes
- tcc	Total cloud cover	%	Sky state / radiation impact
- wind10	10 m wind speed	m s⁻¹	Surface wind for impacts
- gust	10 m wind gust	m s⁻¹	Short-duration wind hazards
- 2r	2 m relative humidity	%	Fog/comfort & visibility proxy
- vis	Visibility	m	Operational low-vis conditions
- precip_step	Precipitation per model time step	mm	Instantaneous rain/snow signal
- wind850*	Wind speed at 850 hPa	m s⁻¹	LLJ, advection & synoptic setup
- w*	Vertical velocity at 850 hPa	Pa s⁻¹	Ascent/descent proxy (forcing/convection)
- t*	Temperature at 850 hPa	K	Air mass / advection
- r*	Relative humidity at 850 hPa	%	Moisture availability aloft
- pwat*	Precipitable water	kg m⁻²	Column moisture; heavy rain potential

* global-only in the default setup.

Note: precip_step is per-step, not “since T0”. 

########################################
#5. Files & Folders

a) weatherPipeline.py – Orchestrates everything.
	Archives previous figures/data snapshots into older/.
	Runs regional then global: download, then stack, then analyze.
	Uses cycle markers to skip redundant downloads.

b) regionalDownload.py – Downloads FMI HARMONIE for the configured bbox.
	Produces harmonie_stack_YYYYMMDD_HH.nc and writes the cycle marker.
	
c) globalDownload.py – Downloads GFS (0.25°) filtered by variable groups.
	Produces gfs_stack_YYYYMMDD_HH.nc and writes the cycle marker.

d) analysisRegional.py – Builds regional maps/time-series and per-country histograms/stats.
	REGION_EXTENT, COUNTRIES, and plotting “plans” live here.
	Updates the regional block in web/manifest.json. Per country histrograms and stats are not shown in the web file.
	
e) analysisGlobal.py – Builds global maps/time-series.
	Updates the global block in web/manifest.json.

f) analysisFunctions.py – Shared helpers.
	plot_maps(...), plot_means(...), and all manifest collection/writing utilities.
	/web/index.html – Minimal viewer (vanilla JS).
	Domain/variable selectors, animation, per-country UI, simple CSV table rendering.

########################################
#6. Troubleshooting

a) Viewer says “Failed to load manifest.json (Failed to fetch)”
	Serve from the project root and open http://localhost:8000/web/ (not the file directly).

b) Slider moves but the map doesn’t change
	Check that multiple frames exist for the selected variable.
	The play button auto-disables if only one frame is available.

c) Cartopy/PROJ/GEOS errors
	Use conda-forge builds (see Quickstart). If Natural Earth downloads are slow, try again with a stable connection.

d) ecCodes/cfgrib errors
	Ensure eccodes is installed (via conda-forge) and cfgrib via pip.

e) Force a refresh
	Delete data/<domain>/cycle_YYYYMMDD_HH.txt for the domain you want to rerun from scratch.

f) Where are old outputs?
	Each figures folder keeps snapshots under figures/<domain>/older/<timestamp>/.
