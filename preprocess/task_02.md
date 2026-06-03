# TASK_02_Feature_Extraction

**Assignee:** Google Earth Engine Specialist  
**Role:** Feature Extraction & Stack Generation  

---

## 1. Project Overview

We are building a Rock Glacier Segmentation Dataset for Himalayan rock glacier mapping. The segmentation labels have already been generated from manually delineated rock glacier polygons.

**Current Project Structure:**
```text
RockGlacier_Project/
│
├── EDA/
│   ├── individual_polygons/
│   │   ├── polygon_0.geojson
│   │   ├── polygon_1.geojson
│   │   └── ...
│   └── individual_plots/
│
├── masks/
│   ├── mask_0000.tif
│   ├── mask_0001.tif
│   └── ...
│
├── features/
└── dataset/
```

- **Polygons:** Located at `EDA/individual_polygons/` (e.g., `polygon_0.geojson`)
- **Masks:** Already generated and located at `masks/` (e.g., `mask_0000.tif`)
- **Mask Values:** `0` = Background, `1` = Rock Glacier

---

## 2. Objective

Generate georeferenced multi-band feature stacks from Google Earth Engine for every rock glacier polygon. The generated features **must align perfectly** with their corresponding masks.

> [!WARNING]
> **Important Notes**
> This is a semantic segmentation project.
> - **Do NOT** create polygon-level statistics.
> - **Do NOT** calculate mean NDVI, mean slope, mean elevation, etc.
> - **Do NOT** generate CSV tables.
> 
> We explicitly need pixel-level feature images (rasters).

---

## 3. Data Sources & Required Features

Use **Google Earth Engine** to extract the following:

**Primary Imagery:** Sentinel-2 Surface Reflectance  
**Terrain:** Copernicus DEM (preferred)

### Required Bands/Features:
**Sentinel-2 Bands:**
- Blue (B2)
- Green (B3)
- Red (B4)
- NIR (B8)
- SWIR1 (B11)
- SWIR2 (B12)

**Spectral Indices:**
- NDVI
- NDWI
- NDSI

**Terrain Features:**
- Elevation
- Slope
- Aspect

*If feasible, also include:*
- Land Surface Temperature (LST)
- Surface Velocity

---

## 4. Technical Specifications

### Spatial Resolution
All outputs must use exactly **10 meter spatial resolution**.
- **Reason:** Sentinel-2 native resolution is 10m. The masks and features must have absolutely identical spatial resolutions to prevent dimension mismatch during U-Net training.

### Extraction Strategy
For each polygon, perform the following sequence:
1. Load `polygon_i.geojson`
2. Compute bounding box
3. Add **200 meter padding** around the bounding box. *(Note: This padding allows future segmentation models to see surrounding terrain context, not just the glacier).*
4. Download all required features for this bounds area.
5. Clip features to the padded region.
6. Resample all layers to exactly 10m resolution.
7. Stack all features into a single multi-band GeoTIFF.

### Output Location
Save all generated feature stacks inside the `features/` directory:
```text
features/
├── feature_0000.tif
├── feature_0001.tif
├── feature_0002.tif
└── ...
```

---

## 5. Band Order Specification

Your final GeoTIFF stacks must strictly follow this band ordering:
- **Band 1** = Blue
- **Band 2** = Green
- **Band 3** = Red
- **Band 4** = NIR
- **Band 5** = SWIR1
- **Band 6** = SWIR2
- **Band 7** = NDVI
- **Band 8** = NDWI
- **Band 9** = NDSI
- **Band 10** = Elevation
- **Band 11** = Slope
- **Band 12** = Aspect

*Optional (if acquired):*
- **Band 13** = LST
- **Band 14** = Velocity

---

## 6. Alignment Validation

You must physically verify that the features perfectly overlap the masks:
- `feature_0000.tif` aligns perfectly with `mask_0000.tif`
- `feature_0001.tif` aligns perfectly with `mask_0001.tif`

**Checklist for alignment:**
- [ ] Matching CRS (Coordinate Reference System)
- [ ] Matching Resolution (10m)
- [ ] Matching Bounding Box Coordinates
- [ ] Exact Pixel-to-Pixel alignment (dimensions must be identical)

---

## 7. Deliverables & Current Scope

**What you must deliver:**
1. Google Earth Engine extraction script.
2. Feature stack generation pipeline.
3. Validation script (to verify alignment checklist above).
4. Documentation explaining each generated feature.
5. Feature extraction report.

> [!IMPORTANT]
> **Strict Scope Limitations**
> - Only perform feature extraction and feature stack generation.
> - **Do NOT** create train/validation/test splits.
> - **Do NOT** generate patches.
> - **Do NOT** train any machine learning model.
> 
> The sole goal of this task is to produce correctly aligned multi-band feature stacks for all rock glacier masks.
