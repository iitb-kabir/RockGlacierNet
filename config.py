"""
Central configuration for the Sikkim rock-glacier feature-extraction project.

Edit GEE_PROJECT to your registered Google Earth Engine Cloud project ID,
then run `python auth_setup.py` once to authenticate.
"""

# ---------------------------------------------------------------------------
# Google Earth Engine
# ---------------------------------------------------------------------------
# Your registered Earth Engine / Google Cloud project ID.
# e.g. "ee-ajaykumar" or a custom GCP project id.
GEE_PROJECT = "ee-ajaykumarc2021"

# ---------------------------------------------------------------------------
# Region of interest
# ---------------------------------------------------------------------------
# Whole Sikkim state, taken from the FAO GAUL 2015 level-1 admin dataset.
GAUL_DATASET = "FAO/GAUL_SIMPLIFIED_500m/2015/level1"
ADMIN_NAME_FIELD = "ADM1_NAME"
SIKKIM_NAME = "Sikkim"

# Fallback bounding box for Sikkim (used only if the GAUL lookup ever fails).
# [west, south, east, north]
SIKKIM_BBOX = [88.0, 27.0, 88.95, 28.15]

# ---------------------------------------------------------------------------
# Time windows
# ---------------------------------------------------------------------------
# Analysis year range for compositing / time-series statistics.
START_YEAR = 2020
END_YEAR = 2024

# Seasonal month windows (inclusive). In the eastern Himalaya:
#   - "summer" (ablation / warm season) ~ Jun-Sep (monsoon, but warmest LST)
#   - "winter" (accumulation / cold season) ~ Dec-Feb
# Adjust if you prefer pre-monsoon clear-sky windows for optical work.
SUMMER_MONTHS = [6, 7, 8, 9]
WINTER_MONTHS = [12, 1, 2]

# A clear-sky optical window (pre/post-monsoon) gives the least cloudy S2 data
# in the eastern Himalaya. Used for the main optical composite.
OPTICAL_MONTHS = [10, 11, 12, 1, 2, 3]  # Oct-Mar dry season

# ---------------------------------------------------------------------------
# Sentinel-2 (optical)
# ---------------------------------------------------------------------------
S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
S2_CLOUD_PROB = "COPERNICUS/S2_CLOUD_PROBABILITY"
S2_CLOUD_PROB_MAX = 40       # % cloud probability threshold for masking
S2_MAX_SCENE_CLOUD = 60      # % scene-level cloud cover pre-filter

# ---------------------------------------------------------------------------
# Landsat (thermal)
# ---------------------------------------------------------------------------
L8_COLLECTION = "LANDSAT/LC08/C02/T1_L2"
L9_COLLECTION = "LANDSAT/LC09/C02/T1_L2"
LANDSAT_MAX_CLOUD = 60       # % scene cloud cover pre-filter

# ---------------------------------------------------------------------------
# Digital Elevation Models (terrain)
# ---------------------------------------------------------------------------
# Copernicus GLO-30 is the primary DEM (derivatives computed on it).
DEM_COP = "COPERNICUS/DEM/GLO30"      # ImageCollection, band "DEM"
DEM_SRTM = "USGS/SRTMGL1_003"          # Image, band "elevation"
DEM_NASADEM = "NASA/NASADEM_HGT/001"   # Image, band "elevation"
DEM_ALOS = "JAXA/ALOS/AW3D30/V4_1"     # ImageCollection, band "DSM"

# Neighbourhood window radii for terrain derivatives.
TPI_RADIUS_M = 300       # Topographic Position Index window (metres)
RELIEF_RADIUS_M = 500    # Relative-relief (max-min) window (metres)
# Roughness / TRI use a 3x3 pixel window (set in code).

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
EXPORT_SCALE = 30            # metres; Landsat-native, fine for S2 too here
EXPORT_CRS = "EPSG:32645"    # UTM 45N covers Sikkim
DRIVE_FOLDER = "GEE_Sikkim_Dl"
EXPORT_MAX_PIXELS = 1e13
