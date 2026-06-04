"""
RockGlacierNet — single source of truth for DATA COLLECTION + PATCH BUILDING.

Everything the collection/patching pipeline writes lives under `patch_dataset/`.
Edit values here; no other file needs touching to re-collect at a different
patch size or with a different context window.

Pipeline (Option A — enlarged per-glacier window):
    1. preprocess/collect_features.py   shapefile -> patch_dataset/raw/{features,masks}   (run on Windows: needs ee+geopandas)
    2. preprocess/build_patches.py      raw/      -> patch_dataset/patches_{size}/{split}  (run anywhere: needs rasterio)

Why Option A: the old 200 m buffer produced tiles whose median size is 83 px, so
86% were smaller than a 128 px patch — a 128 patch on them would be mostly
reflect-padding. We instead download a generous context window ONCE (big enough
for the largest target patch) and tile locally to any size in PATCH_SIZES.
"""

from pathlib import Path

# ── Roots ─────────────────────────────────────────────────────────────────────
# config.py lives in <project>/patch_dataset/ , so:
PATCH_DATASET_DIR = Path(__file__).resolve().parent          # .../patch_dataset
PROJECT_ROOT      = PATCH_DATASET_DIR.parent                 # .../RockGlacierNet

# Input shapefile (rock-glacier inventory polygons). Override on Windows.
SHAPEFILE_PATH = PROJECT_ROOT / "Final_merged01" / "Final_merged01.shp"

# All COLLECTED + BUILT artifacts stay inside patch_dataset/ (and nowhere else).
RAW_DIR          = PATCH_DATASET_DIR / "raw"
RAW_FEATURES_DIR = RAW_DIR / "features"      # feature_XXXX.tif  (12-band)
RAW_MASKS_DIR    = RAW_DIR / "masks"         # mask_XXXX.tif     (1-band, pixel-aligned)

# ── Geometry / projection ─────────────────────────────────────────────────────
TARGET_CRS = "EPSG:32645"   # UTM Zone 45N (metres) — Sikkim
SCALE_M    = 10.0           # output pixel size (metres). Sentinel-2 native 10 m.

# ── Patch sizes ───────────────────────────────────────────────────────────────
PATCH_SIZES   = [64, 128, 256]   # sizes you may build from the SAME raw download
DEFAULT_PATCH = 128              # used when build_patches.py is run with no --patch-size

# Context window downloaded per glacier (metres added on every side of the
# polygon bbox). Must be >= (max(PATCH_SIZES)/2)*SCALE_M so even a tiny glacier
# yields a tile large enough to hold the biggest patch. For 256 px that is 1280 m.
CONTEXT_BUFFER_M = (max(PATCH_SIZES) // 2) * SCALE_M + 50.0   # = 1330 m

# ── Patch extraction ──────────────────────────────────────────────────────────
STRIDE_FRACTION         = 0.5    # patch overlap; stride = round(patch_size * this)
BACKGROUND_KEEP_RATIO   = 0.25   # fraction of pure-background patches to keep
MIN_GLACIER_PERCENTAGE  = 0.0    # keep any patch with > this % glacier pixels
PADDING_MODE            = "reflect"   # "reflect" | "constant" (0-fill)

# ── Train / val / test split (by SOURCE GLACIER — prevents patch leakage) ─────
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED  = 42

# ── Bands ─────────────────────────────────────────────────────────────────────
# ORDER IS CONTRACTUAL: must match data_generator.GLOBAL_MEAN / GLOBAL_STD.
NUM_BANDS  = 12
BAND_NAMES = ["B2", "B3", "B4", "B8", "B11", "B12",
              "NDVI", "NDWI", "NDSI", "Elev", "Slope", "Aspect"]

# ── Earth Engine collection settings (used by collect_features.py) ────────────
GEE_PROJECT       = "aerobic-mile-484705-s8"
S2_COLLECTION     = "COPERNICUS/S2_SR_HARMONIZED"
DEM_COLLECTION    = "COPERNICUS/DEM/GLO30"
S2_DATE_START     = "2021-01-01"
S2_DATE_END       = "2024-01-01"
S2_SR_DIVISOR     = 10000.0      # scale Sentinel-2 SR DN -> reflectance
NODATA_VALUE      = -9999


# ── Derived helpers ───────────────────────────────────────────────────────────
def stride_for(patch_size: int) -> int:
    """Step between patch origins for a given patch size."""
    return max(1, round(patch_size * STRIDE_FRACTION))


def patches_dir(patch_size: int) -> Path:
    """Output root for a given patch size, e.g. patch_dataset/patches_128/."""
    return PATCH_DATASET_DIR / f"patches_{patch_size}"


def as_dict(patch_size: int) -> dict:
    """Flat config snapshot written into each build's metadata/config.json."""
    return {
        "patch_size":             patch_size,
        "stride":                 stride_for(patch_size),
        "stride_fraction":        STRIDE_FRACTION,
        "scale_m":                SCALE_M,
        "context_buffer_m":       CONTEXT_BUFFER_M,
        "background_keep_ratio":  BACKGROUND_KEEP_RATIO,
        "minimum_glacier_pct":    MIN_GLACIER_PERCENTAGE,
        "padding_mode":           PADDING_MODE,
        "split_ratios":           SPLIT_RATIOS,
        "random_seed":            RANDOM_SEED,
        "target_crs":             TARGET_CRS,
        "num_bands":              NUM_BANDS,
        "band_names":             BAND_NAMES,
        "raw_features_dir":       str(RAW_FEATURES_DIR),
        "raw_masks_dir":          str(RAW_MASKS_DIR),
        "output_dir":             str(patches_dir(patch_size)),
    }
