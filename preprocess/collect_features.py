"""
RockGlacierNet — Stage 1: feature/mask COLLECTION (Option A: enlarged window).

For every rock-glacier polygon in the inventory shapefile:
  1. compute its bbox in UTM and pad by CONTEXT_BUFFER_M on every side,
  2. download a 12-band Sentinel-2 + terrain feature stack over that window,
  3. rasterize the polygon to the EXACT grid of the downloaded feature tile,
     so feature and mask are co-registered by construction (no harmonize step).

Outputs (and nothing outside patch_dataset/):
  patch_dataset/raw/features/feature_XXXX.tif   (12-band, float32, 10 m, UTM45N)
  patch_dataset/raw/masks/mask_XXXX.tif         (1-band uint8, same grid)

ENVIRONMENT: needs `earthengine-api`, `geemap`, `geopandas`, `rasterio`.
Run on the machine that has them (your Windows box):
    python preprocess/collect_features.py
    python preprocess/collect_features.py --limit 5     # smoke test
    python preprocess/collect_features.py --shapefile /path/to/your.shp

All tunables live in patch_dataset/config.py.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from patch_dataset import config as C

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import ee
import geemap
from shapely.geometry import box


# ─────────────────────────────────────────────────────────────────────────────
# Earth Engine image builders  (band order MUST match C.BAND_NAMES)
# ─────────────────────────────────────────────────────────────────────────────

def get_s2_image(roi):
    coll = (ee.ImageCollection(C.S2_COLLECTION)
            .filterBounds(roi)
            .filterDate(C.S2_DATE_START, C.S2_DATE_END))

    def mask_clouds(image):
        qa = image.select("QA60")
        cloud  = 1 << 10
        cirrus = 1 << 11
        m = qa.bitwiseAnd(cloud).eq(0).And(qa.bitwiseAnd(cirrus).eq(0))
        return image.updateMask(m).divide(C.S2_SR_DIVISOR)

    s2 = coll.map(mask_clouds).median().clip(roi)
    ndvi = s2.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndwi = s2.normalizedDifference(["B3", "B8"]).rename("NDWI")
    ndsi = s2.normalizedDifference(["B3", "B11"]).rename("NDSI")
    # B2,B3,B4,B8,B11,B12,NDVI,NDWI,NDSI
    return s2.select(["B2", "B3", "B4", "B8", "B11", "B12"]).addBands([ndvi, ndwi, ndsi])


def get_terrain_image(roi):
    # Compute slope/aspect BEFORE mosaicking so the native 30 m grid is preserved.
    def add_terrain(img):
        slope  = ee.Terrain.slope(img).rename("slope")
        aspect = ee.Terrain.aspect(img).rename("aspect")
        return img.addBands([slope, aspect])

    terrain = (ee.ImageCollection(C.DEM_COLLECTION).select("DEM")
               .map(add_terrain).mosaic().clip(roi))
    elevation = terrain.select("DEM").rename("elevation")   # Elev
    slope     = terrain.select("slope")                     # Slope
    aspect    = terrain.select("aspect")                    # Aspect
    return elevation.addBands([slope, aspect])


def build_feature_image(roi):
    """12-band stack in the contractual order C.BAND_NAMES."""
    return get_s2_image(roi).addBands(get_terrain_image(roi))


# ─────────────────────────────────────────────────────────────────────────────
# Per-polygon processing
# ─────────────────────────────────────────────────────────────────────────────

def padded_roi(geom_utm):
    """Return (ee_roi_4326, bounds_utm) for a single UTM geometry + buffer."""
    minx, miny, maxx, maxy = geom_utm.bounds
    b = C.CONTEXT_BUFFER_M
    minx, miny, maxx, maxy = minx - b, miny - b, maxx + b, maxy + b
    box_utm   = gpd.GeoDataFrame(geometry=[box(minx, miny, maxx, maxy)], crs=C.TARGET_CRS)
    box_4326  = box_utm.to_crs("EPSG:4326").total_bounds
    roi = ee.Geometry.BBox(box_4326[0], box_4326[1], box_4326[2], box_4326[3])
    return roi, (minx, miny, maxx, maxy)


def rasterize_aligned_mask(geom_utm, feat_tif_path, out_mask_path):
    """Rasterize geom onto the EXACT grid of the downloaded feature tile."""
    with rasterio.open(feat_tif_path) as src:
        transform, height, width, crs = src.transform, src.height, src.width, src.crs
    mask = rasterize(
        [(geom_utm, 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
    )
    with rasterio.open(
        out_mask_path, "w", driver="GTiff",
        height=height, width=width, count=1, dtype="uint8",
        crs=crs, transform=transform, nodata=0,
    ) as dst:
        dst.write(mask, 1)
    return int(mask.sum())


def main():
    ap = argparse.ArgumentParser(description="Collect enlarged-window feature/mask tiles.")
    ap.add_argument("--shapefile", default=str(C.SHAPEFILE_PATH))
    ap.add_argument("--limit", type=int, default=None, help="process only first N polygons")
    args = ap.parse_args()

    # EE init
    try:
        ee.Initialize(project=C.GEE_PROJECT)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=C.GEE_PROJECT)

    C.RAW_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    C.RAW_MASKS_DIR.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(args.shapefile).to_crs(C.TARGET_CRS)
    n = len(gdf) if args.limit is None else min(args.limit, len(gdf))
    print(f"Loaded {len(gdf)} polygons from {args.shapefile} | processing {n} "
          f"| buffer={C.CONTEXT_BUFFER_M:.0f} m | scale={C.SCALE_M:.0f} m")

    n_done = n_skip = n_fail = 0
    for idx in range(n):
        geom = gdf.geometry.iloc[idx]
        if geom is None or geom.is_empty:
            print(f"  [{idx:04d}] empty geometry — skipped")
            n_skip += 1
            continue

        feat_path = C.RAW_FEATURES_DIR / f"feature_{idx:04d}.tif"
        mask_path = C.RAW_MASKS_DIR / f"mask_{idx:04d}.tif"
        if feat_path.exists() and mask_path.exists():
            n_skip += 1
            continue

        try:
            roi, _ = padded_roi(geom)
            img = build_feature_image(roi)
            geemap.ee_export_image(
                img, filename=str(feat_path), scale=C.SCALE_M,
                region=roi, crs=C.TARGET_CRS, file_per_band=False,
            )
            fg = rasterize_aligned_mask(geom, str(feat_path), str(mask_path))
            n_done += 1
            if n_done % 25 == 0 or idx == n - 1:
                print(f"  [{idx:04d}] done={n_done} skip={n_skip} fail={n_fail} "
                      f"(last fg px={fg})")
        except Exception as e:
            n_fail += 1
            print(f"  [{idx:04d}] FAILED: {e}")

    print(f"\nCollection complete. done={n_done} skipped={n_skip} failed={n_fail}")
    print(f"Features -> {C.RAW_FEATURES_DIR}")
    print(f"Masks    -> {C.RAW_MASKS_DIR}")


if __name__ == "__main__":
    main()
