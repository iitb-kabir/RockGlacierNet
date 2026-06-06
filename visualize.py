"""
Build an interactive HTML map of every optical + thermal parameter.

Produces a single self-contained `sikkim_parameters.html` with:
  - basemaps (Google Satellite + OpenStreetMap, switchable),
  - a true-colour + false-colour Sentinel-2 reference layer,
  - every spectral index and LST band as its own toggleable layer,
  - the Sikkim outline,
  - a layer-control widget so you can switch parameters on/off.

Built directly on `folium` + Earth Engine tile layers (no geemap.foliumap,
which has a basemap-init bug on current versions).

Run:
    python visualize.py                 # writes sikkim_parameters.html
    python visualize.py --open          # also open it in your browser

The Earth Engine tile URLs baked into the HTML stay valid for viewing without
the viewer needing their own EE login.
"""

import argparse
import os
import webbrowser

import ee
import folium

from auth_setup import init_ee
from roi import get_sikkim
from optical_params import optical_features
from thermal_params import thermal_features
from terrain_params import terrain_features

HTML_OUT = os.path.join(os.path.dirname(__file__), "sikkim_parameters.html")

# Palettes -------------------------------------------------------------------
NDVI_PAL = ["#a50026", "#d73027", "#f46d43", "#fee08b",
            "#d9ef8b", "#66bd63", "#1a9850", "#006837"]
SNOW_PAL = ["#08306b", "#4292c6", "#9ecae1", "#deebf7", "#ffffff"]
WATER_PAL = ["#ffffff", "#9ecae1", "#4292c6", "#08306b"]
MOIST_PAL = ["#8c510a", "#d8b365", "#f6e8c3", "#c7eae5", "#5ab4ac", "#01665e"]
BARE_PAL = ["#006837", "#ffffbf", "#a6611a", "#7f3b08"]
GRAY_PAL = ["#000000", "#ffffff"]
LST_PAL = ["#000080", "#0000ff", "#00ffff", "#00ff00",
           "#ffff00", "#ff8000", "#ff0000", "#800000"]
ANOM_PAL = ["#2166ac", "#67a9cf", "#d1e5f0", "#f7f7f7",
            "#fddbc7", "#ef8a62", "#b2182b"]
ELEV_PAL = ["#006837", "#ffffbf", "#a6611a", "#7f3b08", "#ffffff"]
SLOPE_PAL = ["#1a9850", "#ffffbf", "#d73027", "#7f0000"]
ASPECT_PAL = ["#ff0000", "#ffff00", "#00ff00", "#00ffff",
              "#0000ff", "#ff00ff", "#ff0000"]
CURV_PAL = ["#5e3c99", "#b2abd2", "#f7f7f7", "#fdb863", "#e66101"]
RUG_PAL = ["#ffffcc", "#a1dab4", "#41b6c4", "#225ea8", "#0c2c84"]
DIFF_PAL = ["#2166ac", "#67a9cf", "#f7f7f7", "#ef8a62", "#b2182b"]

# (band, vis-params, display name)
OPTICAL_LAYERS = [
    ("NDVI",   {"min": -0.2, "max": 0.8, "palette": NDVI_PAL},  "NDVI (vegetation)"),
    ("NDSI",   {"min": -0.5, "max": 1.0, "palette": SNOW_PAL},  "NDSI (snow)"),
    ("NDWI",   {"min": -0.3, "max": 0.5, "palette": WATER_PAL}, "NDWI (water)"),
    ("NDMI",   {"min": -0.4, "max": 0.4, "palette": MOIST_PAL}, "NDMI (moisture)"),
    ("BSI",    {"min": -0.3, "max": 0.4, "palette": BARE_PAL},  "BSI (bare/debris)"),
    ("ALBEDO", {"min": 0.0,  "max": 0.8, "palette": GRAY_PAL},  "Albedo"),
]

THERMAL_LAYERS = [
    ("LST_summer",  {"min": -10, "max": 30, "palette": LST_PAL},  "LST summer (degC)"),
    ("LST_winter",  {"min": -25, "max": 15, "palette": LST_PAL},  "LST winter (degC)"),
    ("LST_annual",  {"min": -15, "max": 25, "palette": LST_PAL},  "LST annual (degC)"),
    ("LST_anomaly", {"min": -5,  "max": 5,  "palette": ANOM_PAL}, "LST anomaly (degC)"),
]

TERRAIN_LAYERS = [
    ("ELEV_COP",      {"min": 200, "max": 8500, "palette": ELEV_PAL}, "Elevation (Copernicus, m)"),
    ("DIFF_SRTM_COP", {"min": -30, "max": 30, "palette": DIFF_PAL},   "DEM diff: SRTM - COP (m)"),
    ("DIFF_NASA_COP", {"min": -30, "max": 30, "palette": DIFF_PAL},   "DEM diff: NASADEM - COP (m)"),
    ("DIFF_ALOS_COP", {"min": -30, "max": 30, "palette": DIFF_PAL},   "DEM diff: ALOS - COP (m)"),
    ("SLOPE",         {"min": 0,  "max": 60, "palette": SLOPE_PAL},   "Slope (deg)"),
    ("ASPECT",        {"min": 0,  "max": 360, "palette": ASPECT_PAL}, "Aspect (deg)"),
    ("HILLSHADE",     {"min": 0,  "max": 255},                        "Hillshade"),
    ("CURVATURE",     {"min": -5, "max": 5, "palette": CURV_PAL},     "Curvature"),
    ("TRI",           {"min": 0,  "max": 30, "palette": RUG_PAL},     "TRI (ruggedness)"),
    ("ROUGHNESS",     {"min": 0,  "max": 80, "palette": RUG_PAL},     "Roughness (max-min, m)"),
    ("TPI",           {"min": -30, "max": 30, "palette": DIFF_PAL},   "TPI (position index)"),
    ("REL_RELIEF",    {"min": 0,  "max": 300, "palette": RUG_PAL},    "Relative relief (m)"),
]


def add_ee_layer(fmap, ee_image, vis_params, name, show=False):
    """Attach an Earth Engine image as a folium tile layer."""
    map_id = ee.Image(ee_image).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id["tile_fetcher"].url_format,
        attr="Google Earth Engine",
        name=name,
        overlay=True,
        control=True,
        show=show,
    ).add_to(fmap)


def build_map():
    init_ee()
    geom, fc = get_sikkim()

    optical = optical_features(geom)
    thermal = thermal_features(geom)
    terrain = terrain_features(geom)

    m = folium.Map(location=[27.6, 88.5], zoom_start=9, control_scale=True)

    # Basemaps.
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google", name="Google Satellite", overlay=False, control=True,
    ).add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap",
                     overlay=False, control=True).add_to(m)

    # Sentinel-2 references.
    add_ee_layer(m, optical, {"bands": ["B4", "B3", "B2"], "min": 0.0, "max": 0.3},
                 "Sentinel-2 true colour", show=True)
    add_ee_layer(m, optical, {"bands": ["B11", "B8", "B4"], "min": 0.0, "max": 0.4},
                 "Sentinel-2 false colour (SWIR)", show=False)

    # Parameter layers.
    for band, vis, label in OPTICAL_LAYERS:
        add_ee_layer(m, optical.select(band), vis, label, show=False)
    for band, vis, label in THERMAL_LAYERS:
        add_ee_layer(m, thermal.select(band), vis, label, show=False)
    for band, vis, label in TERRAIN_LAYERS:
        add_ee_layer(m, terrain.select(band), vis, label, show=False)

    # Sikkim outline.
    outline = ee.Image().byte().paint(featureCollection=fc, color=1, width=2)
    add_ee_layer(m, outline, {"palette": ["#ff00ff"]}, "Sikkim boundary", show=True)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open", action="store_true",
                        help="Open the HTML in the default browser when done.")
    args = parser.parse_args()

    m = build_map()
    m.save(HTML_OUT)
    print(f"Wrote interactive map -> {HTML_OUT}")
    if args.open:
        webbrowser.open("file://" + os.path.abspath(HTML_OUT))


if __name__ == "__main__":
    main()
