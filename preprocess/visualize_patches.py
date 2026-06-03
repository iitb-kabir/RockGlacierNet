"""
Patch visualization utilities for RockGlacierNet.

Generates:
  outputs/patch_viz/
      sample_grid_{split}.png       — 4×8 grid of RGB+mask overlays
      class_distribution.png        — glacier % histogram per split
      patch_size_distribution.png   — original image H/W scatter
      band_histograms.png           — per-band value distributions (raw)

Usage (from project root):
    conda run -n brats python preprocess/visualize_patches.py
"""

import os
import sys
import glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess.patch_config import PATCH_CONFIG

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PATCH_DIR  = PATCH_CONFIG['output_dir']
SPLITS     = PATCH_CONFIG['splits']
VIZ_DIR    = "outputs/patch_viz"
META_CSV   = os.path.join(PATCH_DIR, "metadata", "patch_metadata.csv")
IMG_CSV    = os.path.join(PATCH_DIR, "metadata", "dataset_summary.csv")
BAND_NAMES = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12',
              'NDVI', 'NDWI', 'NDSI', 'Elev', 'Slope', 'Aspect']

os.makedirs(VIZ_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Sample grid: RGB + glacier mask overlay
# ─────────────────────────────────────────────────────────────────────────────

def _rgb(feat: np.ndarray) -> np.ndarray:
    """Extract normalized RGB from raw (64,64,12) patch."""
    rgb = feat[:, :, [2, 1, 0]].astype(np.float32)
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    return np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)


def plot_sample_grid(split: str, n_cols: int = 8, n_rows: int = 4):
    feat_dir   = os.path.join(PATCH_DIR, split, 'features')
    feat_files = sorted(glob.glob(os.path.join(feat_dir, '*.npy')))
    if not feat_files:
        print(f"  No patches for {split}, skipping grid.")
        return

    rng     = np.random.RandomState(42)
    n_show  = min(n_cols * n_rows, len(feat_files))
    picks   = rng.choice(feat_files, size=n_show, replace=False)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2, n_rows * 2.2))
    fig.suptitle(f"Sample Patches — {split}  (RGB + glacier overlay)",
                 fontsize=12, y=1.01)

    for ax_row in axes:
        for ax in ax_row:
            ax.axis('off')

    for i, fp in enumerate(sorted(picks)):
        row, col = divmod(i, n_cols)
        ax = axes[row][col]

        feat = np.load(fp)
        mask = np.load(fp.replace('features', 'masks'))

        rgb = _rgb(feat)
        ax.imshow(rgb)

        # Semi-transparent glacier overlay in red
        overlay = np.zeros((*mask.shape[:2], 4), dtype=np.float32)
        overlay[mask[:, :, 0] > 0.5] = [1, 0, 0, 0.45]
        ax.imshow(overlay)

        g_pct = 100.0 * (mask > 0.5).sum() / mask.size
        ax.set_title(f"{g_pct:.1f}%", fontsize=7, pad=2)

    red_patch = mpatches.Patch(color='red', alpha=0.45, label='Glacier')
    fig.legend(handles=[red_patch], loc='lower center', ncol=1, fontsize=9)
    plt.tight_layout()
    out = os.path.join(VIZ_DIR, f'sample_grid_{split}.png')
    plt.savefig(out, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Class distribution histograms
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_distribution():
    if not os.path.exists(META_CSV):
        print(f"  Metadata not found: {META_CSV}")
        return

    df   = pd.read_csv(META_CSV)
    kept = df[df['kept_or_filtered'] == 'kept']

    colors = {'train': '#2196F3', 'val': '#4CAF50', 'test': '#FF9800'}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    fig.suptitle("Glacier Coverage Distribution in Kept Patches", fontsize=12)

    for ax, split in zip(axes, SPLITS):
        sub = kept[kept['split'] == split]['glacier_percentage']
        if sub.empty:
            ax.set_title(split); continue
        ax.hist(sub, bins=30, color=colors.get(split, 'gray'),
                edgecolor='white', linewidth=0.5)
        ax.set_xlabel('Glacier coverage (%)')
        ax.set_ylabel('Patch count')
        ax.set_title(f"{split}  (n={len(sub)})\nmean={sub.mean():.1f}%  median={sub.median():.1f}%")
        ax.axvline(sub.mean(), color='black', linestyle='--', linewidth=1, label='mean')
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(VIZ_DIR, 'class_distribution.png')
    plt.savefig(out, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Image size scatter
# ─────────────────────────────────────────────────────────────────────────────

def plot_image_sizes():
    if not os.path.exists(IMG_CSV):
        print(f"  Image summary not found: {IMG_CSV}")
        return

    df     = pd.read_csv(IMG_CSV)
    colors = {'train': '#2196F3', 'val': '#4CAF50', 'test': '#FF9800'}

    fig, ax = plt.subplots(figsize=(7, 6))
    for split in SPLITS:
        sub = df[df['split'] == split]
        ax.scatter(sub['original_width'], sub['original_height'],
                   alpha=0.5, s=20, label=split, color=colors.get(split, 'gray'))

    patch_size = PATCH_CONFIG['patch_size']
    ax.axvline(patch_size, color='red', linestyle='--', linewidth=1,
               label=f'patch_size={patch_size}')
    ax.axhline(patch_size, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Original image width (px)')
    ax.set_ylabel('Original image height (px)')
    ax.set_title('Original Image Size Distribution')
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = os.path.join(VIZ_DIR, 'patch_size_distribution.png')
    plt.savefig(out, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Per-band histograms from a random sample of patches
# ─────────────────────────────────────────────────────────────────────────────

def plot_band_histograms(n_samples: int = 200):
    feat_files = []
    for split in SPLITS:
        feat_files += glob.glob(os.path.join(PATCH_DIR, split, 'features', '*.npy'))

    if not feat_files:
        print("  No patches found for band histograms.")
        return

    rng   = np.random.RandomState(0)
    picks = rng.choice(feat_files, size=min(n_samples, len(feat_files)), replace=False)

    # Collect band data
    band_data = [[] for _ in range(12)]
    for fp in picks:
        feat = np.load(fp)               # (64, 64, 12) raw
        for b in range(12):
            band_data[b].append(feat[:, :, b].flatten())

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    fig.suptitle(f"Per-Band Value Distributions (raw, {len(picks)} random patches)", fontsize=12)

    for b, ax in enumerate(axes.flatten()):
        vals = np.concatenate(band_data[b])
        # Clip extreme outliers for readability
        lo, hi = np.percentile(vals, 1), np.percentile(vals, 99)
        vals_clip = vals[(vals >= lo) & (vals <= hi)]
        ax.hist(vals_clip, bins=60, color='#1976D2', edgecolor='none', alpha=0.8)
        ax.set_title(BAND_NAMES[b], fontsize=9)
        ax.set_xlabel('Value', fontsize=7)
        ax.set_ylabel('Count', fontsize=7)
        ax.tick_params(labelsize=7)
        ax.text(0.97, 0.93, f"μ={vals.mean():.3f}\nσ={vals.std():.3f}",
                transform=ax.transAxes, ha='right', va='top', fontsize=7,
                bbox=dict(boxstyle='round', fc='white', alpha=0.7))

    plt.tight_layout()
    out = os.path.join(VIZ_DIR, 'band_histograms.png')
    plt.savefig(out, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Generating patch visualizations…")

    print("\n1. Sample grids…")
    for split in SPLITS:
        plot_sample_grid(split)

    print("\n2. Class distribution…")
    plot_class_distribution()

    print("\n3. Image size scatter…")
    plot_image_sizes()

    print("\n4. Band histograms…")
    plot_band_histograms()

    print(f"\nAll figures saved to {VIZ_DIR}/")


if __name__ == '__main__':
    main()
