# GEE Sikkim — Optical & Thermal Parameters

Earth Engine pipeline that builds an **optical + thermal feature stack** over
the State of Sikkim, as the first stage of a rock-glacier detection /
activity-classification workflow.

This is **Phase 1** (optical + thermal). The wider feature stack — multi-DEM
terrain, Sentinel-1 SAR + texture, ERA5-Land climate, MODIS snow persistence,
Dynamic World, JRC water, and external InSAR velocity — is planned as
additional modules (see *Roadmap*).

## Setup

```bash
pip install -r requirements.txt
```

1. Edit [config.py](config.py) and set `GEE_PROJECT` to your registered
   Earth Engine Cloud project ID (e.g. `ee-yourname`).
2. Authenticate once:
   ```bash
   python auth_setup.py
   ```
   This opens a browser / token flow and confirms `1 + 1 = 2` through the
   Earth Engine servers.

## Run

```bash
python main.py                          # print bands + quick ROI statistics
python main.py --export                 # ONE GeoTIFF per output band (default)
python main.py --export --mode grouped  # optical + thermal multiband TIFFs
python main.py --export --mode stack    # single combined multiband TIFF
python main.py --export --mode all      # per-band + grouped + stack
```

All exports are **float32 GeoTIFFs** written to the Google Drive folder set by
`DRIVE_FOLDER` in [config.py](config.py), at `EXPORT_SCALE` (30 m) /
`EXPORT_CRS` (EPSG:32645). Default `per_band` mode produces one `.tif` per
output, e.g. `sikkim_optical_NDVI_2020_2024.tif`,
`sikkim_thermal_LST_summer_2020_2024.tif`. Monitor running tasks at
<https://code.earthengine.google.com/tasks>.

### Interactive HTML map

```bash
python visualize.py          # writes sikkim_parameters.html
python visualize.py --open   # also open it in your browser
```

Produces a standalone **`sikkim_parameters.html`** with a satellite / OSM
basemap, a Sentinel-2 true-colour and false-colour reference, and every index
+ LST band as a separately toggleable, colour-mapped layer with the Sikkim
outline. Open the file in any browser — no Earth Engine login needed to view.

Individual modules are runnable for testing:

```bash
python roi.py             # ROI area sanity check (~7,096 km^2)
python optical_params.py  # list optical bands
python thermal_params.py  # list thermal bands
```

## Files

| File | Purpose |
|------|---------|
| [config.py](config.py) | Project ID, ROI, date windows, dataset IDs, export settings |
| [auth_setup.py](auth_setup.py) | `init_ee()` — authenticate + initialize Earth Engine |
| [roi.py](roi.py) | Sikkim boundary (FAO GAUL, bbox fallback) |
| [optical_params.py](optical_params.py) | Sentinel-2 composite + NDVI/NDSI/NDWI/NDMI/BSI/albedo |
| [thermal_params.py](thermal_params.py) | Landsat 8/9 LST: summer/winter/annual + local anomaly |
| [terrain_params.py](terrain_params.py) | Multi-DEM elevation + differences + slope/aspect/curvature/TRI/TPI/roughness/relief |
| [exports.py](exports.py) | GeoTIFF export helpers (per-band / grouped / stack) |
| [visualize.py](visualize.py) | Build interactive `sikkim_parameters.html` map |
| [main.py](main.py) | Combine stack, print stats, export |

## Feature stack (Phase 1)

**Optical (Sentinel-2 SR, dry-season median composite)**
- Reflectance: B2 B3 B4 B5 B6 B7 B8 B8A B11 B12
- NDVI, NDSI, NDWI, NDMI, BSI, broadband albedo

**Thermal (Landsat 8/9 C2 L2 surface temperature)**
- `LST_summer`, `LST_winter`, `LST_annual`, `LST_anomaly`

**Terrain (multi-DEM: Copernicus GLO-30, SRTM, NASADEM, ALOS AW3D30)**
- Elevation: `ELEV_COP`, `ELEV_SRTM`, `ELEV_NASA`, `ELEV_ALOS`
- DEM differences vs Copernicus: `DIFF_SRTM_COP`, `DIFF_NASA_COP`, `DIFF_ALOS_COP`
- Derivatives (on Copernicus): `SLOPE`, `ASPECT`, `HILLSHADE`, `CURVATURE`,
  `TRI`, `ROUGHNESS`, `TPI`, `REL_RELIEF`

## Roadmap (next modules)

- `sar_params.py` — Sentinel-1 seasonal VV/VH + GLCM texture
- `climate_params.py` — ERA5-Land MAAT, snow depth/fall, radiation
- `snow_params.py` — MODIS snow persistence / duration / onset / melt
- `landcover_params.py` — Dynamic World probabilities, JRC surface water
- `insar_params.py` — ingest external InSAR velocity / coherence assets
