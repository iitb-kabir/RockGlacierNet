# patch_dataset/ — RockGlacierNet data collection & patches

Self-contained dataset pipeline. **Everything generated lives under this folder.**
All tunables are in [`config.py`](config.py) — change one value, re-run, done.

## Method (why this design)

Reference: Liu, Xing & Yao, *Remote Sens.* 2025 (Hunza Basin, U-Net/DeepLabV3+/HRnet).

The old object-centric collection used a 200 m buffer → tiles with **median 83 px**,
**86 % smaller than 128 px**. A 128 px patch on those is mostly reflect-padding, not
terrain — the cause of the 64 px IoU regression. Fix = **Option A: enlarge the
per-glacier window** to `CONTEXT_BUFFER_M` (default 1330 m, sized for the largest
patch in `PATCH_SIZES`), download **once**, then tile locally to any size.

## Two stages

| Stage | Script | Needs | Output |
|---|---|---|---|
| 1. Collect | `preprocess/collect_features.py` | `ee`, `geemap`, `geopandas`, `rasterio` | `patch_dataset/raw/{features,masks}/` |
| 2. Build patches | `preprocess/build_patches.py` | `rasterio` only | `patch_dataset/patches_{size}/{split}/{features,masks}/` |

> No env on the Linux box has `earthengine`/`geopandas` — **run Stage 1 on Windows**
> (where you run the shapefile). Stage 2 runs anywhere (e.g. conda env `brats`).

### Stage 1 — collect (run on Windows)
```bash
python preprocess/collect_features.py                 # all polygons
python preprocess/collect_features.py --limit 5       # smoke test
python preprocess/collect_features.py --shapefile path/to/your.shp
```
For each polygon: pads the bbox by `CONTEXT_BUFFER_M`, downloads the 12-band stack
(`B2 B3 B4 B8 B11 B12 NDVI NDWI NDSI Elev Slope Aspect`), and rasterizes the polygon
onto the **exact grid** of the downloaded tile → feature & mask are co-registered by
construction (the old `harmonize_dataset.py` step is gone).

### Stage 2 — build patches at any size
```bash
python preprocess/build_patches.py                    # DEFAULT_PATCH = 128
python preprocess/build_patches.py --patch-size 64
python preprocess/build_patches.py --patch-size 256
```
Splits **by source glacier** (no overlapping-patch leakage), tiles with 50 % overlap,
keeps every glacier patch + `BACKGROUND_KEEP_RATIO` of background patches, and writes
**georeferenced GeoTIFF** patches `feature_*.tif` / `mask_*.tif` plus `metadata/`
(`config.json`, `patch_metadata.csv`, `dataset_summary.csv`, `diagnostics_report.txt`).

Each size lands in its own folder, so they coexist:
```
patch_dataset/
  raw/{features,masks}/
  patches_64/   {train,val,test}/{features,masks}/  metadata/
  patches_128/  {train,val,test}/{features,masks}/  metadata/
  patches_256/  {train,val,test}/{features,masks}/  metadata/
```

## Training on a patch dataset (drop-in)

Patches are named `feature_*.tif` / `mask_*.tif`, so `RockGlacierDataGenerator`
trains on them unchanged — only repoint the paths in `train.py`:
```python
TRAIN_FEAT = "patch_dataset/patches_128/train/features"
TRAIN_MASK = "patch_dataset/patches_128/train/masks"
VAL_FEAT   = "patch_dataset/patches_128/val/features"
VAL_MASK   = "patch_dataset/patches_128/val/masks"
INPUT_SHAPE = (128, 128, 12)   # match the patch size you built
```
Patches already equal `INPUT_SHAPE`, so the generator's crop/pad is a no-op; flips/
rotations still augment. (Band order matches `data_generator.GLOBAL_MEAN/STD`.)

## Changing patch size later
Edit `DEFAULT_PATCH` / `PATCH_SIZES` in `config.py`, or pass `--patch-size`. No
re-download needed as long as the size ≤ what the raw context window supports
(`CONTEXT_BUFFER_M` is sized for `max(PATCH_SIZES)`). To go larger than 256, bump
`PATCH_SIZES` and **re-run Stage 1**.
```
