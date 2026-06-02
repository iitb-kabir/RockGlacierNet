# Feature Stack Specifications

This document outlines the internal structure and metadata of the multi-band GeoTIFF files extracted from Google Earth Engine for the Rock Glacier Detection project.

## File Format & Metadata
- **Format:** Multi-band GeoTIFF (`.tif`)
- **Coordinate Reference System (CRS):** `EPSG:32645` (WGS 84 / UTM zone 45N)
- **Spatial Resolution:** `10.0` meters per pixel
- **Spatial Extent:** Variable per rock glacier. The bounds strictly cover the bounding box of the rock glacier polygon plus a precise `200m` buffer on all sides.

## Band Mapping (12-Band Stack)
Every exported feature file (e.g., `feature_0000.tif`) contains exactly 12 aligned bands. Because GeoTIFFs often strip band names during Earth Engine export, this index mapping is required for loading the data into PyTorch/TensorFlow.

| Band Index | Feature Name | Data Source | Description |
| :---: | :--- | :--- | :--- |
| **1** | `B2` | Sentinel-2 (SR) | Blue Band |
| **2** | `B3` | Sentinel-2 (SR) | Green Band |
| **3** | `B4` | Sentinel-2 (SR) | Red Band |
| **4** | `B8` | Sentinel-2 (SR) | Near Infrared (NIR) |
| **5** | `B11` | Sentinel-2 (SR) | Shortwave Infrared (SWIR) 1 |
| **6** | `B12` | Sentinel-2 (SR) | Shortwave Infrared (SWIR) 2 |
| **7** | `NDVI` | Derived (S2) | Normalized Difference Vegetation Index `(B8 - B4) / (B8 + B4)` |
| **8** | `NDWI` | Derived (S2) | Normalized Difference Water Index `(B3 - B8) / (B3 + B8)` |
| **9** | `NDSI` | Derived (S2) | Normalized Difference Snow Index `(B3 - B11) / (B3 + B11)` |
| **10** | `Elevation` | Copernicus DEM | Elevation in meters (GLO-30) |
| **11** | `Slope` | Derived (DEM) | Terrain slope in degrees |
| **12** | `Aspect` | Derived (DEM) | Terrain aspect in degrees |

## Temporal Processing
- **Timeframe:** 3-year aggregate (`2021-01-01` to `2024-01-01`).
- **Cloud Masking:** Strict pixel-level masking using the Sentinel-2 `QA60` band (removing both opaque and cirrus clouds).
- **Compositing:** A `.median()` reducer is applied across the 3-year stack to ensure a perfectly clear, snow-free (summer-biased), and cloud-free representation of the surface.

## Actual Dataset Values (Sample from `feature_0000.tif`)

Below are the actual numbers (Min, Max, and Mean) calculated directly from the extracted pixels inside `feature_0000.tif`:

| Band Index | Feature Name | Min Value | Max Value | Mean Value |
| :---: | :--- | :--- | :--- | :--- |
| **1** | `B2` | 0.1843 | 0.7763 | 0.5603 |
| **2** | `B3` | 0.1761 | 0.7384 | 0.5447 |
| **3** | `B4` | 0.1900 | 0.7294 | 0.5364 |
| **4** | `B8` | 0.1899 | 0.6998 | 0.5119 |
| **5** | `B11` | 0.1028 | 0.4263 | 0.2215 |
| **6** | `B12` | 0.0954 | 0.3540 | 0.1999 |
| **7** | `NDVI` | -0.0783 | 0.1143 | -0.0220 |
| **8** | `NDWI` | -0.1570 | 0.0973 | 0.0292 |
| **9** | `NDSI` | -0.1862 | 0.6793 | 0.4111 |
| **10** | `Elevation` | 4643.9253 | 5007.4976 | 4790.8872 |
| **11** | `Slope` | 0.1379 | 57.5960 | 22.2821 |
| **12** | `Aspect` | 0.0065 | 359.9010 | 165.6444 |
