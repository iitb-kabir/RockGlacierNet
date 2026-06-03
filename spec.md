# RockGlacierNet — Data Preprocessing Specification (Shapefile → Training-Ready Dataset)

This is a **complete, step-by-step recipe** for turning one rock-glacier Shapefile (`.shp`)
into a machine-learning-ready dataset of aligned 12-band feature tensors and binary
segmentation masks.

It is written so that **someone with no prior knowledge of the project** can follow it
end to end. Every stage has:

1. **What it does** (objective)
2. **Inputs / Outputs** (exact files and folders)
3. **How to run it** (the reference script)
4. **✅ TEST GATE** — a *mandatory* check that must pass before moving on.

> **Golden rule:** Never skip a TEST GATE. Each gate exists to catch a specific failure
> that would otherwise silently corrupt model training. If a gate fails, **stop and fix
> the previous stage** — do not proceed.

---

## 0. Setup (do this first)

### 0.1 Inputs you must have
- **One Shapefile** of rock-glacier polygons. A `.shp` never travels alone — make sure the
  sidecar files are in the same folder: `.shx`, `.dbf`, and ideally `.prj` and `.cpg`.
  - Example used to build the original dataset: `Final_merged01.shp` (626 polygons,
    Sikkim Himalayas, CRS `EPSG:4326`).
- **A Google Earth Engine (GEE) account** with a cloud project ID (needed in Phase 2 to
  download satellite + terrain features).

### 0.2 Software environment
Install these Python packages (the original project used a conda env; any env works):

```bash
pip install geopandas matplotlib rasterio earthengine-api geemap shapely numpy pandas
```

| Package | Used for |
| :--- | :--- |
| `geopandas`, `shapely` | reading/splitting the shapefile, reprojection |
| `rasterio` | rasterizing polygons → masks, reading/writing GeoTIFFs |
| `earthengine-api`, `geemap` | downloading Sentinel-2 + DEM features from GEE |
| `numpy`, `pandas` | array math, patch extraction, reports |
| `matplotlib` | EDA plots and patch visualizations |

### 0.3 ⚠️ Fix the hard-coded paths first (critical)
The reference scripts were written on Windows and contain **hard-coded paths** like
`r"D:\RockGlacier_Project\..."`. Before running anything, open each script and change the
path constants at the top to **your** project root.

On this machine the project root is:
```
/data1/nasiruddink/rockglacier/RockGlacierNet
```

| Script | Constants to update |
| :--- | :--- |
| `EDA/explore_shapefile.py` | `SHAPEFILE_PATH`, `OUTPUT_DIR` |
| `EDA/separate_polygons.py` | `SHAPEFILE_PATH`, `OUTPUT_DIR`, `PLOTS_DIR` |
| `preprocess/generate_masks.py` | `INPUT_DIR`, `OUTPUT_MASK_DIR` |
| `preprocess/gee_feature_extraction.py` | `INPUT_DIR`, `OUTPUT_DIR`, GEE `project=` |
| `preprocess/harmonize_dataset.py` | `FEATURE_DIR`, `MASK_DIR`, `H_FEATURE_DIR`, `H_MASK_DIR` |
| `preprocess/split_dataset.py` | `H_FEATURE_DIR`, `H_MASK_DIR`, `DATASET_DIR` |
| `check_separability.py` | `H_FEATURE_DIR`, `H_MASK_DIR` |

> Tip: pick ONE root folder and keep the sub-folder names exactly as below so the scripts
> chain together without surprises.

### 0.4 Folder layout produced by this pipeline
```
RockGlacierNet/
├── EDA/
│   ├── individual_polygons/      # polygon_0.geojson … polygon_625.geojson
│   └── individual_plots/         # first-5 sanity plots
├── masks/                        # mask_0000.tif … (rasterized polygons) + mask_report.csv
├── features/                     # feature_0000.tif … (12-band GEE stacks)
├── preprocess/
│   ├── harmonized_features/      # cropped to match masks pixel-for-pixel
│   └── harmonized_masks/
├── dataset/                      # train/val/test split of harmonized .tif pairs
│   ├── train/{features,masks}/
│   ├── val/{features,masks}/
│   └── test/{features,masks}/
└── patches/                      # final 64×64 .npy tensors fed to the model
    ├── train/{features,masks}/
    ├── val/{features,masks}/
    ├── test/{features,masks}/
    └── metadata/                 # config.json, patch_metadata.csv, dataset_summary.csv, diagnostics_report.txt
```

### 0.5 The 12 features (memorize this band order — everything depends on it)
Every `feature_*.tif` and every feature `.npy` has **exactly 12 channels in this order**:

| Band | Name | Source | Notes |
| :---: | :--- | :--- | :--- |
| 1 | `B2` Blue | Sentinel-2 SR | reflectance 0–1 |
| 2 | `B3` Green | Sentinel-2 SR | reflectance 0–1 |
| 3 | `B4` Red | Sentinel-2 SR | reflectance 0–1 |
| 4 | `B8` NIR | Sentinel-2 SR | reflectance 0–1 |
| 5 | `B11` SWIR-1 | Sentinel-2 SR | reflectance 0–1 |
| 6 | `B12` SWIR-2 | Sentinel-2 SR | reflectance 0–1 |
| 7 | `NDVI` | derived | `(B8−B4)/(B8+B4)`, range −1…1 |
| 8 | `NDWI` | derived | `(B3−B8)/(B3+B8)`, range −1…1 |
| 9 | `NDSI` | derived | `(B3−B11)/(B3+B11)`, range −1…1 |
| 10 | `Elevation` | Copernicus GLO-30 DEM | metres |
| 11 | `Slope` | derived from DEM | degrees 0–90 |
| 12 | `Aspect` | derived from DEM | degrees 0–360 |

**Mask convention:** `1 = rock glacier`, `0 = background`.

---

## Phase 1 — EDA & Polygon Separation

### Step 1. Inspect the Shapefile
- **Objective:** Confirm the file loads, see its CRS, feature count, and attributes before
  trusting it.
- **Run:** `EDA/explore_shapefile.py`
- **Outputs:** console summary + `rock_glaciers_plot.png` (overview map).

#### ✅ TEST GATE 1 — Shapefile sanity
Run this quick check (paste into a Python shell, edit the path):

```python
import geopandas as gpd
gdf = gpd.read_file("PATH/TO/Final_merged01.shp")
print("features:", len(gdf))
print("CRS:", gdf.crs)
print("geom types:", gdf.geom_type.value_counts().to_dict())
print("empty geoms:", gdf.geometry.is_empty.sum())
print("null geoms:", gdf.geometry.isna().sum())
print("invalid geoms:", (~gdf.is_valid).sum())
print("bbox:", gdf.total_bounds)
```

**PASS criteria:**
- [ ] `features` > 0 (original dataset: **626**).
- [ ] `CRS` is defined (original: `EPSG:4326`). If `None`, the file is unusable — get a
      version with a `.prj`.
- [ ] Geometry type is `Polygon` / `MultiPolygon`.
- [ ] `bbox` looks geographically plausible for your study area (original Sikkim bbox:
      `[88.07, 27.32] → [88.93, 28.08]`).
- [ ] Note any empty/null/invalid geometries — these will be **skipped** later (expected and OK).

---

### Step 2. Split into individual polygons
- **Objective:** Break the one big shapefile into one GeoJSON per polygon so each can be
  processed independently. GeoJSON is used (not `.shp`) to avoid creating thousands of
  sidecar files.
- **Run:** `EDA/separate_polygons.py`
- **Output:** `EDA/individual_polygons/polygon_{idx}.geojson` (e.g. `polygon_0.geojson`).
  The numeric `idx` is the row index and becomes the **ID that links masks ↔ features**
  for the rest of the pipeline.

#### ✅ TEST GATE 2 — One GeoJSON per polygon
```python
import glob, geopandas as gpd
files = glob.glob("EDA/individual_polygons/polygon_*.geojson")
print("geojson files:", len(files))
g = gpd.read_file(files[0])
print("rows in one file:", len(g), "| CRS:", g.crs)   # expect 1 row, same CRS as source
```
**PASS criteria:**
- [ ] Number of GeoJSON files == number of features from Gate 1 (original: 626).
- [ ] Each file contains exactly **1** polygon and keeps the source CRS.

---

## Phase 2 — Mask & Feature Generation

### Step 3. Generate ground-truth masks
- **Objective:** Rasterize each polygon into a binary mask aligned to a fixed 10 m grid.
- **Run:** `preprocess/generate_masks.py`
- **Key parameters (do not change unless you also change them in Step 4):**
  - `TARGET_CRS = EPSG:32645` (UTM 45N — a metre-based CRS so padding is in real metres).
  - `RESOLUTION = 10.0` m/pixel (matches Sentinel-2 native resolution).
  - `PADDING = 200.0` m added on every side of the polygon's bounding box (gives the model
    surrounding background context, not just the glacier).
- **Outputs:** `masks/mask_{idx:04d}.tif` (single-band `uint8`, values {0,1}) and
  `masks/mask_report.csv` (per-mask width/height, foreground/background pixel counts, area).
- **Behavior to expect:** polygons with empty/NaN geometry are **skipped** (logged). This is
  why the final pair count (≈607) can be lower than the polygon count (626).

#### ✅ TEST GATE 3 — Masks are valid binary rasters
```python
import glob, rasterio, numpy as np
bad = []
for f in sorted(glob.glob("masks/mask_*.tif")):
    with rasterio.open(f) as src:
        m = src.read(1)
        vals = set(np.unique(m).tolist())
        if not vals.issubset({0, 1}):        bad.append((f, "values", vals))
        if str(src.crs) != "EPSG:32645":     bad.append((f, "crs", str(src.crs)))
        if m.sum() == 0:                      bad.append((f, "empty-foreground", 0))
print("masks:", len(glob.glob('masks/mask_*.tif')), "| problems:", len(bad))
for b in bad[:10]: print("  ✗", b)
```
**PASS criteria:**
- [ ] Every mask contains **only** values `{0, 1}`.
- [ ] Every mask CRS is `EPSG:32645`.
- [ ] No mask is entirely background (`m.sum() == 0`) — a fully-empty mask means the
      rasterization missed the polygon (CRS or geometry problem).
- [ ] Open `masks/mask_report.csv`: `Foreground_Pixels` > 0 for every row, and
      `Rasterized_Area_m2` should be in the same ballpark as `Polygon_Area_m2`.

---

### Step 4. Extract the 12-band feature stacks from Google Earth Engine
- **Objective:** For each polygon, download Sentinel-2 + terrain features over the **same
  padded bounding box** used for the mask, stacked into one 12-band GeoTIFF.
- **Run:** `preprocess/gee_feature_extraction.py` (set your GEE `project=` ID first).
- **How the features are built (must match the band table in §0.5):**
  - **Sentinel-2:** collection `COPERNICUS/S2_SR_HARMONIZED`, date range
    `2021-01-01 → 2024-01-01`, pixel-level cloud mask via the `QA60` band (drops opaque +
    cirrus), reflectance scaled by `÷10000`, then a **median** composite (clear, summer-biased,
    cloud-free surface). Bands taken: `B2,B3,B4,B8,B11,B12`; indices computed: `NDVI,NDWI,NDSI`.
  - **Terrain:** `COPERNICUS/DEM/GLO30`. Slope & aspect are computed **before** mosaicking so
    the native 30 m projection is preserved, then `Elevation, Slope, Aspect` are added.
  - **Export:** `scale=10`, `crs=EPSG:32645`, `file_per_band=False` → one 12-band
    `features/feature_{idx:04d}.tif`.
- **Behavior to expect:** the script **skips** indices whose `.tif` already exists (safe to
  re-run after interruptions) and **skips** empty geometries. GEE downloads can fail
  transiently — failures are logged; re-run to fill gaps.

#### ✅ TEST GATE 4 — Feature stacks are complete & well-formed
```python
import glob, rasterio, numpy as np
bad = []
for f in sorted(glob.glob("features/feature_*.tif")):
    with rasterio.open(f) as src:
        if src.count != 12:                 bad.append((f, "bands", src.count))
        if str(src.crs) != "EPSG:32645":    bad.append((f, "crs", str(src.crs)))
        if round(src.res[0]) != 10:         bad.append((f, "res", src.res))
        arr = src.read()
        # a band that is entirely no-data means GEE returned nothing here
        if np.all(~np.isfinite(arr)) or np.all(arr == 0): bad.append((f, "all-nodata", None))
print("features:", len(glob.glob('features/feature_*.tif')), "| problems:", len(bad))
for b in bad[:10]: print("  ✗", b)
```
**PASS criteria:**
- [ ] Each feature file has **exactly 12 bands**, CRS `EPSG:32645`, ~10 m resolution.
- [ ] No feature stack is entirely no-data/zero.
- [ ] **Every mask has a matching feature** (same `idx`). Run the coverage check:
```python
import glob, os
m = {os.path.basename(p).split('_')[1].split('.')[0] for p in glob.glob('masks/mask_*.tif')}
f = {os.path.basename(p).split('_')[1].split('.')[0] for p in glob.glob('features/feature_*.tif')}
print("masks without features:", sorted(m - f)[:20])   # must be empty (or re-run GEE for these)
```
  - [ ] `masks - features` is empty (re-run the GEE script for any missing IDs).

> **Note on no-data:** GEE may write a sentinel value of `-9999` (or `NaN`) outside valid
> coverage. That is tolerated here — it is cleaned to `0` automatically during patch
> extraction (Step 7). The gate above only fails when a **whole** stack is no-data.

---

## Phase 3 — Harmonize, Split, and Patch

### Step 5. Harmonize feature/mask pixel grids
- **Objective:** GEE export and `rasterio` rasterization can differ by **one pixel** in
  height/width. This step crops each feature and its mask to their **common (min H, min W)**
  and copies the feature's geo-transform onto the mask so the pair is **pixel-perfect aligned**.
- **Run:** `preprocess/harmonize_dataset.py`
- **Outputs:** `preprocess/harmonized_features/feature_{idx}.tif`,
  `preprocess/harmonized_masks/mask_{idx}.tif`, and `harmonization_summary.csv`
  (records original shapes, harmonized shape, and CRS-match per pair).

#### ✅ TEST GATE 5 — Every pair is identically shaped & georeferenced
```python
import glob, os, rasterio
fdir, mdir = "preprocess/harmonized_features", "preprocess/harmonized_masks"
bad = []
for fp in sorted(glob.glob(f"{fdir}/feature_*.tif")):
    idx = os.path.basename(fp).split('_')[1].split('.')[0]
    mp = f"{mdir}/mask_{idx}.tif"
    if not os.path.exists(mp): bad.append((idx, "no-mask")); continue
    with rasterio.open(fp) as fs, rasterio.open(mp) as ms:
        if fs.shape != ms.shape:           bad.append((idx, "shape", fs.shape, ms.shape))
        if fs.crs != ms.crs:               bad.append((idx, "crs"))
        if fs.transform != ms.transform:   bad.append((idx, "transform"))
print("pairs:", len(glob.glob(f'{fdir}/feature_*.tif')), "| problems:", len(bad))
for b in bad[:10]: print("  ✗", b)
```
**PASS criteria:**
- [ ] For **every** pair: identical `shape`, identical `crs`, identical `transform`.
- [ ] `0` problems printed. (Original harmonized set: **607** valid pairs.)

---

### Step 6. Split into train / val / test
- **Objective:** Randomly partition the harmonized pairs into **70 % train / 15 % val /
  15 % test**, copying matched feature+mask pairs into split folders.
- **Run:** `preprocess/split_dataset.py`
- **Reproducibility:** uses `random.seed(42)` — the split is deterministic. Re-running gives
  the same partition.
- **Outputs:** `dataset/{train,val,test}/{features,masks}/`.
  (Original split: train **424**, val **91**, test **92** = 607.)

#### ✅ TEST GATE 6 — Splits are matched and leak-free
```python
import glob, os
def ids(split, kind):
    return {os.path.basename(p).split('_')[1].split('.')[0]
            for p in glob.glob(f"dataset/{split}/{kind}/*.tif")}
sets = {s: (ids(s,'features'), ids(s,'masks')) for s in ['train','val','test']}
for s,(f,m) in sets.items():
    print(f"{s}: feat={len(f)} mask={len(m)} | feat-mask mismatch={f ^ m}")
tr, va, te = (sets['train'][0], sets['val'][0], sets['test'][0])
print("train∩val:", tr & va, "| train∩test:", tr & te, "| val∩test:", va & te)
```
**PASS criteria:**
- [ ] In each split, the feature ID set **equals** the mask ID set (no orphans).
- [ ] **No overlap** between any two splits (`train∩val`, `train∩test`, `val∩test` all empty).
      Leakage here inflates validation/test scores and is the most dangerous silent bug.
- [ ] `train + val + test` counts add up to the harmonized pair total.

---

### Step 7. Extract fixed-size patches
- **Objective:** Convert variable-sized `.tif` images into fixed **64×64** `.npy` tensors the
  model can batch, while controlling class imbalance.
- **Config:** `preprocess/patch_config.py` (all knobs live here):
  - `patch_size = 64`, `stride = 32` (50 % overlap).
  - `background_keep_ratio = 0.25` — keep only ~25 % of pure-background patches (the rest are
    discarded to fight the ~4.5 : 1 background:glacier imbalance).
  - `minimum_glacier_percentage = 0.0` — keep **any** patch that contains at least one glacier
    pixel.
  - `padding_mode = "reflect"` — features are reflect-padded at the right/bottom edge so the
    sliding window covers the whole image; **masks are always zero-padded** (padding is
    background, never glacier).
  - `num_bands = 12`, `random_seed = 42`.
- **Run:** `python preprocess/extract_patches.py`
- **What it does internally:** reads each `.tif`, reorders to `(H, W, 12)` `float32`,
  **cleans no-data** (`-9999` and any non-finite → `0.0`), slides the window, filters
  background patches by the keep-ratio, and saves kept patches as `.npy`.
- **Outputs:**
  - `patches/{split}/features/{id}_r{row}_c{col}.npy` — shape `(64, 64, 12)` `float32`.
  - `patches/{split}/masks/{id}_r{row}_c{col}.npy` — shape `(64, 64, 1)` `float32`.
  - `patches/metadata/config.json` (exact config used),
    `patch_metadata.csv` (per-patch stats + per-band mean/std),
    `dataset_summary.csv` (per-image stats),
    `diagnostics_report.txt` (human-readable summary: counts, class balance, coverage
    histograms, padding stats, expansion factor).
  - (Original result: train **2081**, val **454**, test **510** kept patches.)

#### ✅ TEST GATE 7 — Read the diagnostics before validating
Open `patches/metadata/diagnostics_report.txt` and confirm:
- [ ] **Image counts** per split match Step 6.
- [ ] **Pixel class balance** is sane (original ≈ 4.5 : 1 background:glacier). If it is wildly
      imbalanced (e.g. 50 : 1), revisit `background_keep_ratio`.
- [ ] **Padding stats** are small — only tiny images should need padding. A high
      "patches with padding" fraction hints at many sub-64 px images (expected for small
      glaciers, but worth a glance).
- [ ] Kept-patch counts are non-zero for **all three** splits.

---

## Phase 4 — MANDATORY Dataset Validation (the safety net before training)

This is the most important phase for "nothing bad during training." Run **both** scripts
below and require a clean result. Each guards against a concrete failure mode.

### Step 8. Patch alignment & integrity validation
- **Run:** `python preprocess/validate_alignment.py`
- **What it checks (per split, every patch):**

| # | Check | Why it matters |
| :-: | :--- | :--- |
| 1 | **File-count match** features vs masks | a missing mask = an unlabeled training sample |
| 2 | **Name match** every `feature/<id>.npy` has `mask/<id>.npy` | mispaired tensors teach the wrong label |
| 3 | **Shape** features `(64,64,12)`, masks `(64,64,1)`, H/W equal | shape mismatch crashes the model or mislabels pixels |
| 4 | **Dtype** features `float32`; masks `float32/uint8/int32/int64` | wrong dtype breaks loss/normalization |
| 5 | **No NaN / Inf** in features | a single NaN turns the whole loss into NaN |
| 6 | **Mask values ⊆ {0, 1}** | any other value corrupts BCE/Dice loss & metrics |
| 7 | **Zero-bleed** glacier pixels are **not** all-zero across all 12 bands | catches padding/alignment errors where labels point at no-data |
| 8 | **Spot-check** prints shape/glacier%/min/max for 5 random patches/split | human eyeball sanity |

#### ✅ TEST GATE 8 — Alignment must be perfectly clean
**PASS criteria:** the script ends with
```
OVERALL: ALL CHECKS PASSED
```
- [ ] `Count match : OK` and `Name match : OK` for every split.
- [ ] `Shape errors : 0`, `NaN errors : 0`, `Inf errors : 0`.
- [ ] `Mask value : 0 errors`, `Zero-bleed : 0 patches`.
- [ ] Spot-check rows look reasonable: feature `min/max` finite, `glacier%` between 0–100.

> If **any** count is non-zero, the script prints the first 10 offending patch IDs. Fix the
> root cause (usually Step 5 harmonization or Step 7 extraction) and re-run — **do not**
> hand-delete patches to make the gate pass.

### Step 9. Trainability / separability report (recommended)
- **Run:** `python check_separability.py` (operates on the harmonized pairs).
- **Output:** `is_trainable.md` — global class balance + per-band mean/std for glacier vs
  background and **Cohen's d** effect size (how discriminative each feature is).
- **Use it to confirm the signal is real**, not just that files are well-formed:
  - [ ] Class imbalance is manageable (original ≈ 4.5 : 1 → fine for BCE + Dice loss).
  - [ ] At least a few bands show meaningful separation (original strongest: **Slope**
        d≈0.41, **NDSI**, **NDVI**, **SWIR-1**). If *every* effect size ≈ 0, the features and
        labels may be misaligned — go back to Phase 2/3.

### Step 10. Visual spot-check (recommended)
- **Run:** `python preprocess/visualize_patches.py`
- **Outputs** in `outputs/patch_viz/`: `sample_grid_{split}.png` (RGB + red glacier overlay),
  `class_distribution.png`, `patch_size_distribution.png`, `band_histograms.png`.
- [ ] On the sample grids, the **red overlay should land on plausible glacier terrain**
      (debris tongues), not on random background — the ultimate human confirmation that
      features and masks line up.

---

## End-to-End Run Order (cheat sheet)

```bash
# Phase 1 — EDA & split
python EDA/explore_shapefile.py          # → Gate 1
python EDA/separate_polygons.py          # → Gate 2

# Phase 2 — masks & features
python preprocess/generate_masks.py      # → Gate 3
python preprocess/gee_feature_extraction.py   # GEE auth needed → Gate 4

# Phase 3 — harmonize, split, patch
python preprocess/harmonize_dataset.py   # → Gate 5
python preprocess/split_dataset.py       # → Gate 6
python preprocess/extract_patches.py     # → Gate 7

# Phase 4 — MANDATORY validation
python preprocess/validate_alignment.py  # → Gate 8 (must say ALL CHECKS PASSED)
python check_separability.py             # → Gate 9 (is_trainable.md)
python preprocess/visualize_patches.py   # → Gate 10 (eyeball outputs/patch_viz/)
```

**Only start model training after Gate 8 prints `ALL CHECKS PASSED`.**

---

## Troubleshooting — failure → likely cause → fix

| Symptom at gate | Likely cause | Fix |
| :--- | :--- | :--- |
| Gate 1: `CRS: None` | shapefile missing `.prj` | obtain the `.prj` sidecar or set CRS explicitly |
| Gate 3: mask all-background | polygon didn't rasterize (CRS/geometry) | check the polygon's GeoJSON validity; confirm reprojection to `EPSG:32645` |
| Gate 4: missing features for some masks | GEE download failed/skipped | re-run `gee_feature_extraction.py` (it skips existing, retries gaps) |
| Gate 4: band count ≠ 12 | edited band list or export error | restore the band/index list in `get_s2_image` / `get_terrain_image` |
| Gate 5: shape/transform mismatch | ran split before harmonize | (re)run `harmonize_dataset.py`, then split |
| Gate 6: split overlap | re-split with different seed / mixed old files | clear `dataset/` and re-run `split_dataset.py` |
| Gate 8: NaN/Inf errors | no-data not cleaned | confirm Step 7 ran (it converts `-9999`/non-finite → 0); re-extract |
| Gate 8: mask value errors | mask resampled with interpolation | masks must be nearest-neighbor / untouched `uint8` {0,1} |
| Gate 8: zero-bleed > 0 | feature/mask misaligned or padding over real labels | re-check Gate 5; ensure masks are zero-padded, never reflect-padded |
| Gate 9: all effect sizes ≈ 0 | features ↔ masks mismatched | verify `idx` linkage from Step 2 onward |

---

## Notes for Automated Agents

- **Follow the phases linearly.** Each `python …` command corresponds to exactly one
  reference script; do not reorder.
- **Authenticate GEE before Phase 2** (`ee.Authenticate()` / set the cloud `project=`).
- **Treat every TEST GATE as a hard precondition** for the next stage. On failure, fix the
  *upstream* stage; never silently delete data to pass a gate.
- **The ID is sacred.** The integer `idx` assigned in Step 2 links `polygon_{idx}` →
  `mask_{idx}` → `feature_{idx}` → `dataset` → `patches`. Never renumber or sort-rename files
  in a way that breaks this correspondence.
- **All geometry/config constants** (CRS `EPSG:32645`, 10 m, 200 m pad, 64-px patch, stride 32,
  seed 42, 12 bands) must stay identical between mask generation, feature extraction, and
  patching, or alignment breaks.
- **Definition of done:** `validate_alignment.py` prints `ALL CHECKS PASSED` and
  `is_trainable.md` reports a manageable class balance with non-trivial feature separability.
```
