"""
Evaluation script — Attention U-Net.

Loads each test GeoTIFF, normalizes and pads/crops to 128×128 (same as
training), predicts, then evaluates only on the VALID (non-padded) pixels
so zero-padding does not inflate scores.

All outputs go directly into outputs/ — no subdirectories.

Outputs
-------
outputs/
    attn_evaluation_metrics.csv       — global threshold sweep 0.30–0.70
    attn_per_image_metrics.csv        — per-image TP/FP/FN/TN, IoU, F1, etc.
    attn_confusion_matrix.csv         — aggregate CM at best IoU threshold
    attn_evaluation_summary.txt       — human-readable summary
    attn_pred_*.png                   — RGB / GT / prediction side-by-side
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
from models.attention_unet import build_attention_unet

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
TEST_FEAT   = "dataset/test/features"
TEST_MASK   = "dataset/test/masks"
CKPT_PATH   = "outputs/checkpoints/attention_unet_best.weights.h5"
OUT_DIR     = "outputs"
INPUT_SHAPE = (128, 128, 12)
TARGET      = INPUT_SHAPE[:2]
BATCH_SIZE  = 8
THRESHOLDS  = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Loss / metric stubs (needed for model.compile)
# ─────────────────────────────────────────────────────────────────────────────
def dice_loss(y_true, y_pred):
    smooth = 1e-6
    yt = tf.cast(y_true, tf.float32)
    i  = tf.reduce_sum(yt * y_pred, axis=[1, 2, 3])
    u  = tf.reduce_sum(yt, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    return 1.0 - tf.reduce_mean((2.0 * i + smooth) / (u + smooth))

def focal_loss(alpha=0.75, gamma=2.0):
    def _l(y_true, y_pred):
        yt = tf.cast(y_true, tf.float32)
        yp = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1-tf.keras.backend.epsilon())
        at = yt*alpha + (1-yt)*(1-alpha)
        pt = yt*yp + (1-yt)*(1-yp)
        return tf.reduce_mean(-at * tf.pow(1-pt, gamma) * tf.math.log(pt))
    return _l

def combined_loss(y_true, y_pred):
    return focal_loss()(y_true, y_pred)*2.0 + dice_loss(y_true, y_pred)

def dice_coef(y_true, y_pred):
    yp = tf.cast(y_pred > 0.5, tf.float32); yt = tf.cast(y_true, tf.float32)
    return (2*tf.reduce_sum(yt*yp)+1e-6)/(tf.reduce_sum(yt)+tf.reduce_sum(yp)+1e-6)

def precision_m(y_true, y_pred):
    yp = tf.cast(y_pred > 0.5, tf.float32); yt = tf.cast(y_true, tf.float32)
    tp = tf.reduce_sum(yt*yp); fp = tf.reduce_sum(yp*(1-yt))
    return (tp+1e-6)/(tp+fp+1e-6)

def recall_m(y_true, y_pred):
    yp = tf.cast(y_pred > 0.5, tf.float32); yt = tf.cast(y_true, tf.float32)
    tp = tf.reduce_sum(yt*yp); fn = tf.reduce_sum(yt*(1-yp))
    return (tp+1e-6)/(tp+fn+1e-6)

def f1_m(y_true, y_pred):
    p = precision_m(y_true, y_pred); r = recall_m(y_true, y_pred)
    return (2*p*r+1e-6)/(p+r+1e-6)

def iou_m(y_true, y_pred):
    yp = tf.cast(y_pred > 0.5, tf.float32); yt = tf.cast(y_true, tf.float32)
    i = tf.reduce_sum(yt*yp); u = tf.reduce_sum(yt)+tf.reduce_sum(yp)-i
    return (i+1e-6)/(u+1e-6)

# ─────────────────────────────────────────────────────────────────────────────
# Image preprocessing (identical to training — no data leakage)
# ─────────────────────────────────────────────────────────────────────────────
def load_and_prep(f_path, m_path):
    with rasterio.open(f_path) as s:
        feat = np.moveaxis(s.read(), 0, -1).astype(np.float32)
    with rasterio.open(m_path) as s:
        mask = np.expand_dims(s.read(1), -1).astype(np.float32)

    H, W = feat.shape[:2]
    feat = np.where((feat == -9999) | ~np.isfinite(feat), 0, feat)
    feat = (feat - GLOBAL_MEAN) / (GLOBAL_STD + 1e-7)

    th, tw = TARGET
    pad_h, pad_w = max(0, th - H), max(0, tw - W)
    if pad_h or pad_w:
        feat = np.pad(feat, ((0, pad_h), (0, pad_w), (0, 0)))
        mask = np.pad(mask, ((0, pad_h), (0, pad_w), (0, 0)))

    h, w = feat.shape[:2]
    if h > th or w > tw:
        hs = (h - th) // 2 if h > th else 0
        ws = (w - tw) // 2 if w > tw else 0
        feat = feat[hs:hs+th, ws:ws+tw, :]
        mask = mask[hs:hs+th, ws:ws+tw, :]

    valid_h = min(H, th)
    valid_w = min(W, tw)
    padded  = pad_h > 0 or pad_w > 0
    cropped = H > th or W > tw
    return feat, mask, H, W, valid_h, valid_w, padded, cropped

# ─────────────────────────────────────────────────────────────────────────────
# Build + load weights
# ─────────────────────────────────────────────────────────────────────────────
print("Building Attention U-Net…")
model = build_attention_unet(input_shape=INPUT_SHAPE)
model.compile(loss=combined_loss,
              metrics=['accuracy', dice_coef, precision_m, recall_m, f1_m, iou_m])

if not os.path.exists(CKPT_PATH):
    raise FileNotFoundError(
        f"Checkpoint not found: {CKPT_PATH}\n"
        "Run: conda run -n brats python train_attention.py")
model.load_weights(CKPT_PATH)
print("Weights loaded.")

# ─────────────────────────────────────────────────────────────────────────────
# Load all test images
# ─────────────────────────────────────────────────────────────────────────────
feat_files = sorted(glob.glob(os.path.join(TEST_FEAT, 'feature_*.tif')))
mask_files = sorted(glob.glob(os.path.join(TEST_MASK, 'mask_*.tif')))
assert len(feat_files) > 0, f"No test images found in {TEST_FEAT}"
assert len(feat_files) == len(mask_files)

print(f"\nLoading {len(feat_files)} test images…")
X, Y, meta, ids = [], [], [], []
for fp, mp in zip(feat_files, mask_files):
    feat, mask, H, W, vh, vw, padded, cropped = load_and_prep(fp, mp)
    X.append(feat); Y.append(mask)
    meta.append((H, W, vh, vw, padded, cropped))
    ids.append(os.path.basename(fp).replace('feature_', '').replace('.tif', ''))

X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.float32)

print("Predicting…")
Y_prob = model.predict(X, batch_size=BATCH_SIZE, verbose=1)

# Keras evaluate at t=0.5
print("\nmodel.evaluate (t=0.5):")
eval_vals  = model.evaluate(X, Y, batch_size=BATCH_SIZE, verbose=1)
eval_names = model.metrics_names

# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────
def cm_metrics(tp, fp, fn, tn):
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    iou  = tp / (tp + fp + fn + 1e-9)
    dice = 2 * tp / (2*tp + fp + fn + 1e-9)
    acc  = (tp + tn) / (tp + tn + fp + fn + 1e-9)
    spec = tn / (tn + fp + 1e-9)
    return prec, rec, f1, iou, dice, acc, spec

# ─────────────────────────────────────────────────────────────────────────────
# Per-image metrics @ t=0.5  (valid region only — no padding pixels)
# ─────────────────────────────────────────────────────────────────────────────
all_true, all_prob = [], []
per_image = []

for i, iid in enumerate(ids):
    H, W, vh, vw, padded, cropped = meta[i]
    yt    = Y[i,     :vh, :vw, 0].flatten().astype(np.int32)
    yprob = Y_prob[i,:vh, :vw, 0].flatten()
    all_true.append(yt); all_prob.append(yprob)

    yp = (yprob >= 0.5).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0,1]).ravel()
    prec, rec, f1, iou, dice, acc, spec = cm_metrics(tp, fp, fn, tn)

    per_image.append({
        'image_id':        iid,
        'orig_H': H, 'orig_W': W,
        'valid_px':        int(yt.size),
        'was_padded':      bool(padded),
        'was_cropped':     bool(cropped),
        'glacier_px_gt':   int(yt.sum()),
        'glacier_frac_gt': round(float(yt.mean()), 4),
        'glacier_px_pred': int(yp.sum()),
        'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn),
        'precision':  round(prec, 4),
        'recall':     round(rec,  4),
        'f1_score':   round(f1,   4),
        'iou':        round(iou,  4),
        'dice':       round(dice, 4),
        'accuracy':   round(acc,  4),
        'specificity':round(spec, 4),
    })

per_img_df = pd.DataFrame(per_image).sort_values('iou')
per_img_df.to_csv(os.path.join(OUT_DIR, 'attn_per_image_metrics.csv'), index=False)

# ─────────────────────────────────────────────────────────────────────────────
# Global threshold sweep
# ─────────────────────────────────────────────────────────────────────────────
y_true_all = np.concatenate(all_true)
y_prob_all = np.concatenate(all_prob)

thresh_rows = []
for t in THRESHOLDS:
    yp = (y_prob_all >= t).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(y_true_all, yp, labels=[0,1]).ravel()
    prec, rec, f1, iou, dice, acc, spec = cm_metrics(tp, fp, fn, tn)
    thresh_rows.append({
        'threshold': t,
        'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn),
        'precision':   round(prec, 6),
        'recall':      round(rec,  6),
        'specificity': round(spec, 6),
        'f1_score':    round(f1,   6),
        'iou':         round(iou,  6),
        'dice':        round(dice, 6),
        'accuracy':    round(acc,  6),
    })

thresh_df = pd.DataFrame(thresh_rows)
thresh_df.to_csv(os.path.join(OUT_DIR, 'attn_evaluation_metrics.csv'), index=False)

# Best threshold by IoU
best_row = thresh_df.loc[thresh_df['iou'].idxmax()]
best_t   = float(best_row['threshold'])

# Aggregate confusion matrix at best threshold
cm = confusion_matrix(y_true_all, (y_prob_all >= best_t).astype(int), labels=[0,1])
pd.DataFrame(cm,
             index=['actual_background', 'actual_glacier'],
             columns=['pred_background',  'pred_glacier']
             ).to_csv(os.path.join(OUT_DIR, 'attn_confusion_matrix.csv'))

# ─────────────────────────────────────────────────────────────────────────────
# Human-readable summary
# ─────────────────────────────────────────────────────────────────────────────
lines = [
    "=" * 58,
    "  ATTENTION U-NET — EVALUATION SUMMARY",
    "=" * 58,
    "",
    "model.evaluate (threshold = 0.5):",
]
for n, v in zip(eval_names, eval_vals):
    lines.append(f"  {n:<22}: {v:.6f}")

lines += [
    "",
    f"Best threshold by IoU: {best_t}",
    "",
    "  {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
        "thresh", "precision", "recall", "f1", "iou", "dice"),
    "  " + "-" * 60,
]
for _, r in thresh_df.iterrows():
    tag = " ◄ best" if r['threshold'] == best_t else ""
    lines.append("  {:<10} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}{}".format(
        r['threshold'], r['precision'], r['recall'],
        r['f1_score'], r['iou'], r['dice'], tag))

lines += [
    "",
    "Per-image stats (valid pixels only, t=0.5):",
    f"  Mean IoU : {per_img_df['iou'].mean():.4f}",
    f"  Median IoU: {per_img_df['iou'].median():.4f}",
    f"  Worst 5  :",
]
for _, r in per_img_df.head(5).iterrows():
    lines.append(f"    {r['image_id']} — IoU={r['iou']:.4f}  F1={r['f1_score']:.4f}"
                 f"  glacier={r['glacier_frac_gt']*100:.1f}%  padded={r['was_padded']}")

lines += [
    "",
    f"Total test images : {len(feat_files)}",
    f"Output dir        : {OUT_DIR}/",
    "",
    "Files written:",
    "  attn_evaluation_metrics.csv    — full threshold sweep",
    "  attn_per_image_metrics.csv     — per-image metrics (sorted worst IoU first)",
    "  attn_confusion_matrix.csv      — aggregate CM at best threshold",
    "  attn_evaluation_summary.txt",
    "  attn_pred_*.png                — comparison figures",
    "",
    "─" * 58,
    "Baseline (SwinUNet 128×128, 70 epochs):",
    "  Precision=0.6233  Recall=0.8990  F1=0.7362  IoU=0.5825",
    "─" * 58,
]
summary = "\n".join(lines)
print("\n" + summary)
with open(os.path.join(OUT_DIR, 'attn_evaluation_summary.txt'), 'w') as f:
    f.write(summary)

# ─────────────────────────────────────────────────────────────────────────────
# Save comparison PNGs for top 20 worst + 5 best images
# ─────────────────────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    y_pred_bin = (Y_prob >= best_t).astype(np.float32)

    save_indices = list(per_img_df.head(10).index) + list(per_img_df.tail(5).index)
    save_indices = list(dict.fromkeys(save_indices))[:20]

    for i in save_indices:
        iid  = ids[i]
        H, W, vh, vw, _, _ = meta[i]
        rgb  = X[i, :, :, [2, 1, 0]]
        lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
        rgb  = np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)

        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        axes[0].imshow(rgb);                             axes[0].set_title('RGB');          axes[0].axis('off')
        axes[1].imshow(Y[i,:,:,0],       cmap='gray');  axes[1].set_title('Ground Truth'); axes[1].axis('off')
        axes[2].imshow(y_pred_bin[i,:,:,0], cmap='gray')
        iou_val = per_img_df.loc[per_img_df['image_id']==iid, 'iou'].values
        axes[2].set_title(f'Pred  IoU={iou_val[0]:.3f}' if len(iou_val) else 'Pred')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f'attn_pred_{iid}.png'), dpi=80)
        plt.close()

    print(f"Saved {len(save_indices)} PNGs to {OUT_DIR}/")
except Exception as e:
    print(f"PNG save skipped: {e}")

print("\nEvaluation complete.")
