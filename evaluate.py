"""
Evaluation script for RockGlacierNet — baseline 128x128 full-image model.

Each test GeoTIFF is normalized and padded/cropped to 128x128 exactly like
training, then predicted. Metrics are computed on the VALID (non-padded) region
of each image so zero-padding does not inflate the scores.

Outputs (outputs/predictions/baseline_model_output/)
-----------------------------------------------------
    per_image_metrics.csv      — ONE ROW PER TEST IMAGE, sorted worst IoU first
                                  (this is the file to inspect for problem images)
    evaluation_metrics.csv     — global threshold sweep 0.30–0.70
    confusion_matrix.csv       — aggregate CM at best IoU threshold
    evaluation_summary.txt     — human-readable summary
    pred_*.png                 — RGB / GT / prediction comparison figures
"""

import os
import glob
import numpy as np
np.Inf = np.inf
import pandas as pd
import rasterio
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from data_generator import GLOBAL_MEAN, GLOBAL_STD
from models.swin_unet import build_model

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
TEST_FEAT   = "dataset/test/features"
TEST_MASK   = "dataset/test/masks"
CKPT_PATH   = "checkpoints/best_model.weights.h5"
OUT_DIR     = "outputs/predictions/baseline_model_output"
INPUT_SHAPE = (128, 128, 12)
TARGET      = INPUT_SHAPE[:2]
BATCH_SIZE  = 8
THRESHOLDS  = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Loss / metric helpers (needed for model.compile)
# ─────────────────────────────────────────────────────────────────────────────
def dice_loss(y_true, y_pred):
    smooth = 1e-6
    yt = tf.cast(y_true, tf.float32)
    i = tf.reduce_sum(yt * y_pred, axis=[1, 2, 3])
    u = tf.reduce_sum(yt, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    return 1.0 - tf.reduce_mean((2.0 * i + smooth) / (u + smooth))


def combined_loss(y_true, y_pred):
    bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred)) * 2.0
    return bce + dice_loss(y_true, y_pred)


def dice_coef(y_true, y_pred):
    yp = tf.cast(y_pred > 0.5, tf.float32); yt = tf.cast(y_true, tf.float32)
    return (2 * tf.reduce_sum(yt * yp) + 1e-6) / (tf.reduce_sum(yt) + tf.reduce_sum(yp) + 1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocess one image exactly like the (non-augment) data generator.
# Returns the padded/cropped 128x128 feature + mask AND the valid region size.
# ─────────────────────────────────────────────────────────────────────────────
def load_and_prep(f_path, m_path):
    with rasterio.open(f_path) as s:
        feat = np.moveaxis(s.read(), 0, -1)              # (H, W, 12)
    with rasterio.open(m_path) as s:
        mask = np.expand_dims(s.read(1), axis=-1)        # (H, W, 1)

    H, W = feat.shape[:2]
    feat = np.where((feat == -9999) | ~np.isfinite(feat), 0, feat).astype(np.float32)
    feat = (feat - GLOBAL_MEAN) / (GLOBAL_STD + 1e-7)

    th, tw = TARGET
    pad_h, pad_w = max(0, th - H), max(0, tw - W)
    if pad_h or pad_w:
        feat = np.pad(feat, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
        mask = np.pad(mask, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')

    h, w = feat.shape[:2]
    if h > th or w > tw:                                  # center-crop large images
        hs = (h - th) // 2 if h > th else 0
        ws = (w - tw) // 2 if w > tw else 0
        feat = feat[hs:hs + th, ws:ws + tw, :]
        mask = mask[hs:hs + th, ws:ws + tw, :]

    # valid (real, non-padded) region inside the 128x128 canvas
    valid_h, valid_w = min(H, th), min(W, tw)
    return (feat.astype(np.float32), mask.astype(np.float32),
            H, W, valid_h, valid_w, pad_h > 0 or pad_w > 0, H > th or W > tw)


# ─────────────────────────────────────────────────────────────────────────────
# Build model + load weights
# ─────────────────────────────────────────────────────────────────────────────
print("Building model…")
model = build_model(input_shape=INPUT_SHAPE, num_classes=1, num_heads=8)
model.compile(loss=combined_loss, metrics=['accuracy', dice_coef])

if not os.path.exists(CKPT_PATH):
    raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}. Train first (python train.py).")
model.load_weights(CKPT_PATH)
print("Weights loaded.")

# ─────────────────────────────────────────────────────────────────────────────
# Load + preprocess every test image
# ─────────────────────────────────────────────────────────────────────────────
feat_files = sorted(glob.glob(os.path.join(TEST_FEAT, 'feature_*.tif')))
mask_files = sorted(glob.glob(os.path.join(TEST_MASK, 'mask_*.tif')))
assert len(feat_files) == len(mask_files) and len(feat_files) > 0, "Test feature/mask mismatch."

print(f"\nLoading {len(feat_files)} test images…")
X, Y, meta, ids = [], [], [], []
for fp, mp in zip(feat_files, mask_files):
    feat, mask, H, W, vh, vw, padded, cropped = load_and_prep(fp, mp)
    X.append(feat); Y.append(mask)
    meta.append((H, W, vh, vw, padded, cropped))
    ids.append(os.path.basename(fp).replace('feature_', '').replace('.tif', ''))

X = np.array(X, dtype=np.float32)       # (N, 128, 128, 12)
Y = np.array(Y, dtype=np.float32)       # (N, 128, 128, 1)
print("Predicting…")
Y_prob = model.predict(X, batch_size=BATCH_SIZE, verbose=1)   # (N, 128, 128, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Collect VALID-region pixels (drop zero-padding) for global + per-image metrics
# ─────────────────────────────────────────────────────────────────────────────
def metrics_from_cm(tp, fp, fn, tn):
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    iou  = tp / (tp + fp + fn + 1e-9)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-9)
    acc  = (tp + tn) / (tp + tn + fp + fn + 1e-9)
    spec = tn / (tn + fp + 1e-9)
    return prec, rec, f1, iou, dice, acc, spec


# ─────────────────────────────────────────────────────────────────────────────
# Per-image metrics @ t=0.5  (worst IoU first)
# ─────────────────────────────────────────────────────────────────────────────
all_true, all_prob = [], []
per_image = []
for i, iid in enumerate(ids):
    H, W, vh, vw, padded, cropped = meta[i]
    yt = Y[i, :vh, :vw, 0].flatten().astype(np.int32)
    yprob = Y_prob[i, :vh, :vw, 0].flatten()
    all_true.append(yt); all_prob.append(yprob)

    yp = (yprob >= 0.5).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    prec, rec, f1, iou, dice, acc, spec = metrics_from_cm(tp, fp, fn, tn)
    n_px = yt.size
    per_image.append({
        'image_id':       iid,
        'orig_H':         H,
        'orig_W':         W,
        'valid_px':       int(n_px),
        'was_padded':     bool(padded),
        'was_cropped':    bool(cropped),
        'glacier_px_gt':  int(yt.sum()),
        'glacier_frac_gt': round(float(yt.mean()), 4),
        'glacier_px_pred': int(yp.sum()),
        'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn),
        'precision':      round(prec, 4),
        'recall':         round(rec, 4),
        'f1_score':       round(f1, 4),
        'iou':            round(iou, 4),
        'dice':           round(dice, 4),
        'accuracy':       round(acc, 4),
        'mean_pred_prob': round(float(yprob.mean()), 4),
        'max_pred_prob':  round(float(yprob.max()), 4),
    })

per_image_df = pd.DataFrame(per_image).sort_values('iou', ascending=True).reset_index(drop=True)
per_image_df.to_csv(os.path.join(OUT_DIR, 'per_image_metrics.csv'), index=False)

# ─────────────────────────────────────────────────────────────────────────────
# Global threshold sweep (valid pixels only)
# ─────────────────────────────────────────────────────────────────────────────
y_true_flat = np.concatenate(all_true)
y_prob_flat = np.concatenate(all_prob)

rows = []
for t in THRESHOLDS:
    yp = (y_prob_flat >= t).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(y_true_flat, yp, labels=[0, 1]).ravel()
    prec, rec, f1, iou, dice, acc, spec = metrics_from_cm(tp, fp, fn, tn)
    rows.append({
        'threshold': t,
        'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn),
        'precision': round(prec, 6), 'recall': round(rec, 6),
        'specificity': round(spec, 6), 'f1_score': round(f1, 6),
        'iou': round(iou, 6), 'dice': round(dice, 6), 'accuracy': round(acc, 6),
    })
thresh_df = pd.DataFrame(rows)
thresh_df.to_csv(os.path.join(OUT_DIR, 'evaluation_metrics.csv'), index=False)

best_row = thresh_df.loc[thresh_df['iou'].idxmax()]
best_t = float(best_row['threshold'])

cm = confusion_matrix(y_true_flat, (y_prob_flat >= best_t).astype(int), labels=[0, 1])
pd.DataFrame(cm,
             index=['actual_background', 'actual_glacier'],
             columns=['pred_background', 'pred_glacier']
             ).to_csv(os.path.join(OUT_DIR, 'confusion_matrix.csv'))

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
mean_iou = per_image_df['iou'].mean()
median_iou = per_image_df['iou'].median()
worst = per_image_df.head(10)[['image_id', 'orig_H', 'orig_W', 'glacier_frac_gt', 'iou', 'precision', 'recall']]

lines = [
    "=" * 60,
    "  ROCKGLACIERNET — EVALUATION SUMMARY (baseline 128x128)",
    "=" * 60,
    "",
    f"Test images           : {len(ids)}",
    f"Best threshold by IoU : {best_t}",
    "",
    "GLOBAL (pixel-pooled) metrics @ best threshold:",
    f"  precision : {best_row['precision']:.4f}",
    f"  recall    : {best_row['recall']:.4f}",
    f"  f1        : {best_row['f1_score']:.4f}",
    f"  iou       : {best_row['iou']:.4f}",
    f"  dice      : {best_row['dice']:.4f}",
    "",
    "PER-IMAGE IoU (@ t=0.5):",
    f"  mean   : {mean_iou:.4f}",
    f"  median : {median_iou:.4f}",
    "",
    "Threshold sweep:",
    "  {:<8} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
        "thresh", "precision", "recall", "f1", "iou", "dice"),
    "  " + "-" * 60,
]
for _, r in thresh_df.iterrows():
    tag = " <- best IoU" if r['threshold'] == best_t else ""
    lines.append("  {:<8} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}{}".format(
        r['threshold'], r['precision'], r['recall'], r['f1_score'], r['iou'], r['dice'], tag))

lines += ["", "10 WORST images by IoU (@ t=0.5):", worst.to_string(index=False)]
lines += [
    "",
    f"Output dir: {OUT_DIR}",
    "Files: per_image_metrics.csv, evaluation_metrics.csv, confusion_matrix.csv",
]
summary = "\n".join(lines)
print("\n" + summary)
with open(os.path.join(OUT_DIR, 'evaluation_summary.txt'), 'w') as f:
    f.write(summary)

# ─────────────────────────────────────────────────────────────────────────────
# Prediction comparison PNGs (10 worst + 10 best)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    id_to_idx = {iid: i for i, iid in enumerate(ids)}
    to_plot = list(per_image_df.head(10)['image_id']) + list(per_image_df.tail(10)['image_id'])
    for iid in to_plot:
        i = id_to_idx[iid]
        H, W, vh, vw, padded, cropped = meta[i]
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        rgb = X[i, :vh, :vw, :][:, :, [2, 1, 0]]
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        iou_val = float(per_image_df.loc[per_image_df['image_id'] == iid, 'iou'].iloc[0])
        axes[0].imshow(rgb);                                     axes[0].set_title(f'{iid} RGB'); axes[0].axis('off')
        axes[1].imshow(Y[i, :vh, :vw, 0], cmap='gray');          axes[1].set_title('GT');         axes[1].axis('off')
        axes[2].imshow((Y_prob[i, :vh, :vw, 0] >= best_t), cmap='gray')
        axes[2].set_title(f'Pred IoU={iou_val:.2f}');            axes[2].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f'pred_{iid}.png'), dpi=80)
        plt.close()
    print(f"Saved {len(to_plot)} comparison PNGs (10 worst + 10 best).")
except Exception as e:
    print(f"PNG save skipped: {e}")

print("\nEvaluation complete.")
