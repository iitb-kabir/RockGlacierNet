# RockGlacierNet — Project Intelligence File

> **Purpose of this file**: Complete project context for any agent picking up this work.
> Auto-updated after every session. Read this before touching anything.

---

## 1. WHAT THIS PROJECT IS

**Task**: Binary semantic segmentation of rock glaciers in the Sikkim Himalayas using multi-modal satellite imagery.

**Input**: 12-band GeoTIFF feature stacks per glacier polygon
- Bands 0–5: Sentinel-2 SR (B2, B3, B4, B8, B11, B12) — reflectance, ~0.1–0.8 range
- Bands 6–8: Spectral indices (NDVI, NDWI, NDSI)
- Bands 9–11: Copernicus DEM terrain (Elevation m, Slope °, Aspect °)
- Resolution: 10m/pixel, CRS: UTM Zone 45N (EPSG:32645)
- Temporal: 2021–2024 median composite, cloud-masked

**Output**: Binary mask — 1 = rock glacier, 0 = background

**Dataset**:
- 626 raw rock glacier polygons (Sikkim Himalayas, shapefile)
- 607 harmonized feature+mask pairs after QC
- Split: 424 train / 91 val / 92 test (70/15/15, seed=42, done before patching)
- Class balance: ~18% glacier / 82% background (4.5:1 imbalance)
- Image sizes: min=46px, max=325px, **mean=83px, median=75px** — most images are SMALL

---

## 2. PROJECT FILE MAP

```
/data1/nasiruddink/rockglacier/RockGlacierNet/
│
├── dataset/                        ← Original harmonized GeoTIFFs (607 pairs)
│   ├── train/features/ + masks/    ← 424 pairs
│   ├── val/features/ + masks/      ← 91 pairs
│   └── test/features/ + masks/     ← 92 pairs
│
├── patches/                        ← Pre-extracted 64×64 .npy patches (generated)
│   ├── train/features/ + masks/
│   ├── val/features/ + masks/
│   ├── test/features/ + masks/     ← 510 patches used in latest evaluation
│   └── metadata/
│       ├── patch_metadata.csv      ← Per-patch stats (glacier%, padding, band means)
│       ├── dataset_summary.csv     ← Per-image patch counts
│       └── diagnostics_report.txt  ← Full extraction report
│
├── models/
│   └── swin_unet.py                ← ACTIVE MODEL (see §3)
│
├── data_generator.py               ← Loads patches/.npy files, z-score normalizes
├── train.py                        ← Training entry point
├── evaluate.py                     ← Evaluation with threshold sweep 0.30–0.70
│
├── preprocess/
│   ├── patch_config.py             ← All tunable params (patch_size, stride, etc.)
│   ├── extract_patches.py          ← One-time patch extraction → patches/
│   ├── validate_alignment.py       ← Sanity checks on extracted patches
│   └── visualize_patches.py        ← 4 diagnostic figures → outputs/patch_viz/
│
├── outputs/
│   ├── checkpoints/
│   │   ├── best_model.weights.h5   ← Current best checkpoint (new architecture)
│   │   └── training_history.csv    ← 58 epochs of metrics
│   └── predictions/trained_model_output/
│       ├── evaluation_metrics.csv  ← Full threshold sweep results
│       ├── per_patch_metrics.csv   ← Per-patch TP/FP/FN/IoU
│       ├── confusion_matrix.csv
│       └── evaluation_metrics_best.txt
│
└── EDA/
    ├── is_trainable.md             ← Cohen's d separability analysis
    └── feature.md                  ← Band specification document
```

---

## 3. CURRENT MODEL ARCHITECTURE

**File**: `models/swin_unet.py`
**Name**: `ResidualSwinUNET_64x64`
**Input**: `(64, 64, 12)` → **Output**: `(64, 64, 1)` sigmoid

```
Encoder:
  Input (64×64×12)
  enc1 → Conv block, 32 filters   → 64×64×32
  pool1 → MaxPool2D               → 32×32×32
  enc2 → Conv block, 64 filters   → 32×32×64
  pool2 → MaxPool2D               → 16×16×64
  enc3 → Conv block, 128 filters  → 16×16×128
  pool3 → MaxPool2D               → 8×8×128

Bottleneck:
  btn  → Conv block, 256 filters  → 8×8×256
  swin_block × 2 (num_heads=8)    → 8×8×256  [64 tokens — global attention]

Decoder:
  up3  → ConvTranspose 128        → 16×16×128 + skip c3
  dec3 → Conv block, 128 filters
  up2  → ConvTranspose 64         → 32×32×64  + skip c2
  dec2 → Conv block, 64 filters
  up1  → ConvTranspose 32         → 64×64×32  + skip c1
  dec1 → Conv block, 32 filters
  output → Conv2D(1, sigmoid)     → 64×64×1
```

Conv block = Conv3×3 → BN → ReLU → Dropout(0.2) → Conv3×3 → BN + residual (Conv1×1) → ReLU

**Training config**:
- Loss: Focal(α=0.75, γ=2.0) × 2.0 + Dice
- Optimizer: Adam, lr=1e-4
- Batch: 16, max Epochs: 100
- Callbacks: EarlyStopping(val_iou_m, patience=15), ReduceLROnPlateau(val_iou_m, factor=0.5, patience=5), ModelCheckpoint(val_iou_m)

---

## 4. RESULTS — BASELINE vs CURRENT (REGRESSION)

### Baseline — Old model (128×128 input, 4 pool layers, filters 16→32→64→128→256)
Trained: 70 epochs | Checkpoint: from commit `c3c1eb4`

| Metric | Value |
|---|---|
| Val Accuracy | 95.38% |
| Val Dice | 0.7618 |
| Val Loss | 0.3034 |
| **Test Precision** | **0.6233** |
| **Test Recall** | **0.8990** |
| **Test F1** | **0.7362** |
| **Test IoU** | **0.5825** |

### Current — New model (64×64 input, 3 pool layers, filters 32→64→128→256)
Trained: 58 epochs | Stopped by EarlyStopping | LR decayed to 2e-6

| Metric | Value |
|---|---|
| Val IoU (best epoch) | 0.3308 |
| Val Dice (best epoch) | 0.4802 |
| Val Loss (best epoch) | 0.846 |
| **Test Precision (t=0.45)** | **0.3746** |
| **Test Recall (t=0.45)** | **0.7065** |
| **Test F1 (t=0.45)** | **0.4896** |
| **Test IoU (t=0.45)** | **0.3241** |

**Full threshold sweep (new model):**
```
thresh  precision  recall    f1        iou       dice
0.30    0.3393     0.8277    0.4813    0.3169    0.4813
0.35    0.3496     0.7944    0.4855    0.3206    0.4855
0.40    0.3607     0.7543    0.4881    0.3228    0.4881
0.45    0.3746     0.7065    0.4896    0.3241    0.4896  ← best IoU
0.50    0.3898     0.6472    0.4866    0.3215    0.4866
0.55    0.4072     0.5821    0.4792    0.3151    0.4792
0.60    0.4278     0.5189    0.4690    0.3063    0.4690
0.65    0.4508     0.4553    0.4530    0.2928    0.4530
0.70    0.4743     0.3857    0.4254    0.2702    0.4254
```

**REGRESSION**: IoU dropped from **0.5825 → 0.3241** (−44%). This is significantly worse.

---

## 5. REGRESSION ANALYSIS — WHY DID IT GET WORSE?

### Confirmed facts:
- New model trained for 58 epochs (stopped by EarlyStopping, patience=15)
- LR decayed to 2e-6 → model is fully converged/stuck
- Train IoU at epoch 57: **0.499** vs Val IoU: **0.331** → **large train/val gap → overfitting**
- Test has 510 patches (92 images → ~5.5 patches/image) — extraction worked correctly

### Likely causes (in priority order):

**1. The 64×64 patch size is too small for spatial context**
- At 10m resolution, 64×64 px = 640×640m field of view
- Rock glaciers in this dataset have mean image size ~83px including 200m buffer
- The glacier body itself is ~40–60px wide = 400–600m
- A 64×64 patch often contains only a fragment of a glacier, not the whole shape
- The old 128×128 model saw the full glacier + surrounding context in most images
- **Without spatial context, the model cannot distinguish glacier from similar terrain**

**2. Patch fragmentation increases false positives**
- With stride=32 (50% overlap), adjacent patches share 50% of pixels
- The model sees the same glacier from many partial views → may not generalize
- background_keep_ratio=0.25 means 75% of background patches are dropped
- Remaining patches have higher glacier% → artificially inflated positive signal
- Consequence: model learns "predict glacier more aggressively" → low precision (0.37)

**3. Train/val split after patching creates spatial overlap**
- Even though image-level split was done before patch extraction, with stride=32 overlap,
  patches from the same 83×85 image cover the full image — no truly held-out spatial region
- Val images are different images but patches look very similar to train patches
- Could explain the train/val gap

**4. Architecture mismatch with new scale**
- Old model: 128×128 input → 4 pools → 8×8 bottleneck = 64 tokens ✓ 
- New model: 64×64 input → 3 pools → 8×8 bottleneck = 64 tokens ✓ (same)
- But encoder filters 32→64→128 may not be deep enough to learn 12-band signatures
- The Swin transformer at 8×8 with key_dim=256 and 8 heads = massive attention for 64 tokens
- 70% of parameters still wasted in the transformer at very coarse resolution

---

## 6. WHAT NEEDS TO BE FIXED (FOR THE NEXT AGENT)

### Option A — Quick fix: go back to full-image training
Restore the original approach but fix the actual bugs:
- Use `target_size=None` or per-image variable size (no padding)
- Pad to nearest multiple of 32 only (minimal padding)
- Remove transformer blocks (they waste 70% of params at 8×8)
- Use pure UNet with filters 64→128→256→512
- This should recover the baseline or beat it

### Option B — Fix patch approach properly
If sticking with patches:
- **Increase patch_size to 128** — covers most images fully, no fragmentation
- Use stride=64 (non-overlapping or 50% overlap on large images)
- Fix background_keep_ratio=0.50 (currently too aggressive at 0.25)
- Keep current 64×64 images as-is (they're small enough)
- Larger patches = more spatial context per sample

### Option C — Hybrid (recommended)
- For images < 96px: use full image padded to nearest 32 (minimal reflect padding)
- For images >= 96px: extract 96×96 patches with stride=48
- This gives the model full context for small glaciers and patches for large ones

### Other fixes regardless of approach:
- **Remove transformer blocks** — they contribute 70% of params but 8×8 attention adds nothing
- Use pure residual UNet (just the CNN part) with larger filters
- Fix EarlyStopping + ModelCheckpoint monitoring inconsistency (both should watch same metric)
- Add class-weighted loss (current focal alpha=0.75 may not be enough for 4.5:1 imbalance)

---

## 7. KEY CODE LOCATIONS FOR FIXING

| What to change | File | Line / Function |
|---|---|---|
| Patch size / stride | `preprocess/patch_config.py` | `PATCH_CONFIG` dict |
| Input shape | `models/swin_unet.py` | `build_model(input_shape=...)` |
| Remove transformer | `models/swin_unet.py` | Remove `_swin_block` calls in `build_model` |
| Filter sizes | `models/swin_unet.py` | `_conv_block` calls in encoder |
| Number of pool layers | `models/swin_unet.py` | `pool1..pool4` in `build_model` |
| Loss function weights | `train.py` | `combined_loss`, `focal_loss(alpha=...)` |
| bg keep ratio | `preprocess/patch_config.py` | `background_keep_ratio` |
| LR / batch size | `train.py` | `LR`, `BATCH_SIZE` |
| Data loading | `data_generator.py` | `PatchDataGenerator.__init__` |

---

## 8. ENVIRONMENT

```bash
# Correct conda environment:
conda run -n brats python <script>

# All scripts should be run from project root:
cd /data1/nasiruddink/rockglacier/RockGlacierNet

# Run order:
python preprocess/extract_patches.py   # regenerate patches if config changes
python preprocess/validate_alignment.py
python train.py
python evaluate.py
```

**Python packages**: tensorflow==2.13.0, rasterio, numpy, pandas, matplotlib, sklearn

---

## 9. SESSION LOG

---

### Session — 2026-06-03 (Session 1)

**Topics discussed:**
- Set up project-level `/update-context` skill and Stop hook for automatic context updates
- Read full codebase: data pipeline, model architecture (ResWinUNET), training config, evaluation

**Key decisions / findings:**
- Diagnosed low model performance: IoU=0.583, Precision=0.623, Recall=0.899
- Found CNN encoder only 307K params (4.5% of model) — severely underpowered
- Transformer uses 70% of params but operates at 8×8=64 tokens — mostly wasted

---

### Session — 2026-06-03 (Session 2 — Pipeline Redesign)

**Topics discussed:**
- Dataset quality check: confirmed 607 pairs, 0 corrupted files, class balance 17.8–19.3% across splits
- Found 88% of images smaller than 128×128 → heavy zero-padding in old pipeline
- Image size distribution: <64px=89, 64–128px=444, 128–192px=63, >192px=11
- Complete pipeline redesign: 64×64 patch extraction with configurable stride/filtering

**Key decisions / findings:**
- Patch size 64px chosen — fits 88% of images without padding
- stride=32 (50% overlap), background_keep_ratio=0.25
- Model updated: 3 pooling layers (not 4), filters doubled to 32→64→128→256
- Training CSV now logs 8 metrics per epoch: loss, accuracy, dice, precision, recall, F1, IoU + val_*
- Evaluation now does full threshold sweep 0.30–0.70, outputs 3 CSVs + per-patch metrics

**Files changed:**
- `preprocess/patch_config.py` — NEW
- `preprocess/extract_patches.py` — NEW
- `preprocess/validate_alignment.py` — NEW
- `preprocess/visualize_patches.py` — NEW
- `data_generator.py` — REWRITTEN (loads .npy patches)
- `models/swin_unet.py` — UPDATED (64×64 input, 3 pool layers, doubled filters)
- `train.py` — UPDATED (6 Keras metrics, monitors val_iou_m)
- `evaluate.py` — REWRITTEN (threshold sweep, 3 CSVs)

**Next steps:**
- Run new pipeline and compare IoU vs baseline 0.5825

---

### Session — 2026-06-03 (Session 3 — REGRESSION DETECTED)

**Topics discussed:**
- New 64×64 pipeline was run and produced WORSE results than baseline
- New model: IoU=0.324, F1=0.490, Precision=0.375, Recall=0.707 (58 epochs, LR decayed to 2e-6)
- Baseline was: IoU=0.5825, F1=0.7362, Precision=0.6233, Recall=0.8990 (70 epochs)
- Updated agent.md with full project index for handoff to next agent

**Key decisions / findings:**
- Regression confirmed: IoU dropped 44% (0.5825 → 0.3241)
- Train IoU=0.499 vs Val IoU=0.331 → overfitting / patch fragmentation problem
- 64×64 patches too small to capture full glacier shape + surrounding context
- Transformer blocks still wasteful (70% of params, 8×8 resolution)
- Recommended fix: increase patch_size to 128 OR use full-image variable-size training

**Files changed:**
- `agent.md` — UPDATED with full project index, regression analysis, fix recommendations
- `.claude/settings.json` — UPDATED (added write permissions for Stop hook agent)

**Next steps:**
- Decide on fix approach (see §6 above)
- Either restore full-image approach with minimal padding OR increase patch_size to 128
- Remove transformer blocks and use pure residual UNet with 64→128→256→512 filters
- Retrain and verify IoU > 0.58 baseline

---

## Session — 2026-06-03 10:34 UTC

*(Auto-recorded by Stop hook — run `/update-context` for full intelligent summary)*

**Git state:**
- c3c1eb4 trained model
- e7421db reswinUnetR
- ea0223e Initial project upload
- Modified: M __pycache__/data_generator.cpython-310.pyc,  D checkpoints/best_model.weights.h5,  D checkpoints/training_history.csv,  M data_generator.py,  D debug_0.py

**Training state:**
- Checkpoint: exists (best_model.weights.h5)
- Last epoch: 57  |  val_iou=0.33081600069999695  val_dice=0.48022589087486267  val_loss=0.8464699387550354
- val_precision=0.38591960072517395  val_recall=0.7173172235488892

**Patches:**
- train: 2081 patches
- val: 454 patches
- test: 510 patches

**Next steps:** *(fill in manually or run /update-context)*

---

## Session — 2026-06-03 10:34 UTC

*(Auto-recorded by Stop hook — run `/update-context` for full intelligent summary)*

**Git state:**
- c3c1eb4 trained model
- e7421db reswinUnetR
- ea0223e Initial project upload
- Modified: M __pycache__/data_generator.cpython-310.pyc,  D checkpoints/best_model.weights.h5,  D checkpoints/training_history.csv,  M data_generator.py,  D debug_0.py

**Training state:**
- Checkpoint: exists (best_model.weights.h5)
- Last epoch: 57  |  val_iou=0.33081600069999695  val_dice=0.48022589087486267  val_loss=0.8464699387550354
- val_precision=0.38591960072517395  val_recall=0.7173172235488892

**Patches:**
- train: 2081 patches
- val: 454 patches
- test: 510 patches

**Next steps:** *(fill in manually or run /update-context)*

---

## Session — 2026-06-03 11:03 UTC

*(Auto-recorded by Stop hook — run `/update-context` for full intelligent summary)*

**Git state:**
- c3c1eb4 trained model
- e7421db reswinUnetR
- ea0223e Initial project upload
- Modified: M __pycache__/data_generator.cpython-310.pyc,  M checkpoints/best_model.weights.h5,  M checkpoints/training_history.csv,  M data_generator.py,  D debug_0.py

**Training state:**
- Checkpoint: NOT FOUND (best_model.weights.h5)
- No training history found

**Patches:**
- train: 2081 patches
- val: 454 patches
- test: 510 patches

**Next steps:** *(fill in manually or run /update-context)*

---

## Session — 2026-06-03 11:11 UTC

*(Auto-recorded by Stop hook — run `/update-context` for full intelligent summary)*

**Git state:**
- c3c1eb4 trained model
- e7421db reswinUnetR
- ea0223e Initial project upload
- Modified: M __pycache__/data_generator.cpython-310.pyc,  M checkpoints/best_model.weights.h5,  M checkpoints/training_history.csv,  M data_generator.py,  D debug_0.py

**Training state:**
- Checkpoint: NOT FOUND (best_model.weights.h5)
- No training history found

**Patches:**
- train: 2081 patches
- val: 454 patches
- test: 510 patches

**Next steps:** *(fill in manually or run /update-context)*

---

## Session — 2026-06-03 11:38 UTC

*(Auto-recorded by Stop hook — run `/update-context` for full intelligent summary)*

**Git state:**
- e8e4467 make all the files structed
- c3c1eb4 trained model
- e7421db reswinUnetR
- ea0223e Initial project upload
- Modified: D checkpoints/best_model.weights.h5,  D checkpoints/training_history.csv,  D outputs/predictions/.gitkeep,  D outputs/predictions/baseline_model_output/confusion_matrix.csv,  D outputs/predictions/baseline_model_output/evaluation_metrics.csv

**Training state:**
- Checkpoint: NOT FOUND (best_model.weights.h5)
- No training history found

**Patches:**
- train: 2081 patches
- val: 454 patches
- test: 510 patches

**Next steps:** *(fill in manually or run /update-context)*

---

## Session — 2026-06-03 14:40 UTC

*(Auto-recorded by Stop hook — run `/update-context` for full intelligent summary)*

**Git state:**
- e8e4467 make all the files structed
- c3c1eb4 trained model
- e7421db reswinUnetR
- ea0223e Initial project upload
- Modified: M agent.md,  D checkpoints/best_model.weights.h5,  D checkpoints/training_history.csv,  D outputs/predictions/.gitkeep,  D outputs/predictions/baseline_model_output/confusion_matrix.csv

**Training state:**
- Checkpoint: NOT FOUND (best_model.weights.h5)
- No training history found

**Patches:**
- train: 2081 patches
- val: 454 patches
- test: 510 patches

**Next steps:** *(fill in manually or run /update-context)*

---

## Session — 2026-06-04 05:18 UTC

*(Auto-recorded by Stop hook — run `/update-context` for full intelligent summary)*

**Git state:**
- 07e67b7 attention unet
- e8e4467 make all the files structed
- c3c1eb4 trained model
- e7421db reswinUnetR
- ea0223e Initial project upload
- Modified: M evaluate.py, ?? Final_merged01/, ?? patch_dataset/, ?? preprocess/build_patches.py, ?? preprocess/collect_features.py

**Training state:**
- Checkpoint: NOT FOUND (best_model.weights.h5)
- No training history found

**Patches:**
- train: 2081 patches
- val: 454 patches
- test: 510 patches

**Next steps:** *(fill in manually or run /update-context)*
