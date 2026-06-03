"""
Patch extraction pipeline for RockGlacierNet.

Reads original harmonized GeoTIFFs from dataset/{split}/,
extracts 64×64 patches with configurable stride, filters pure-background
patches, saves .npy arrays, and writes rich metadata + diagnostics.

Usage (from project root):
    conda run -n brats python preprocess/extract_patches.py

Configuration: preprocess/patch_config.py
"""

import os
import sys
import glob
import json
import math
import numpy as np
import rasterio
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess.patch_config import PATCH_CONFIG

BAND_NAMES = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12',
              'NDVI', 'NDWI', 'NDSI', 'Elev', 'Slope', 'Aspect']


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def padded_size(dim: int, patch: int, stride: int) -> int:
    """Minimum padded dimension that gives full patch coverage with stride."""
    if dim <= patch:
        return patch
    n = math.ceil((dim - patch) / stride)
    return n * stride + patch


def pad_image(arr: np.ndarray, pad_h: int, pad_w: int, mode: str) -> np.ndarray:
    """Pad (H, W, C) array on right and bottom."""
    if arr.ndim == 3:
        return np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)),
                      mode=mode if arr.shape[2] > 0 else 'constant')
    return np.pad(arr, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)


def extract_patches(feat: np.ndarray, mask: np.ndarray,
                    patch: int, stride: int, padding_mode: str):
    """
    Yield (feat_patch, mask_patch, row, col, padding_applied, pad_h, pad_w)
    for every patch position covering the entire image.

    feat: (H, W, 12)  float32
    mask: (H, W, 1)   float32
    """
    H, W = feat.shape[:2]
    pH = padded_size(H, patch, stride)
    pW = padded_size(W, patch, stride)
    dh, dw = pH - H, pW - W

    applied = dh > 0 or dw > 0
    if applied:
        feat = pad_image(feat, dh, dw, padding_mode)
        mask = np.pad(mask, ((0, dh), (0, dw), (0, 0)),
                      mode='constant', constant_values=0)

    for r in range(0, pH - patch + 1, stride):
        for c in range(0, pW - patch + 1, stride):
            yield (feat[r:r+patch, c:c+patch, :],
                   mask[r:r+patch, c:c+patch, :],
                   r, c, applied, dh, dw)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(cfg: dict):
    rng = np.random.RandomState(cfg['random_seed'])
    P      = cfg['patch_size']
    S      = cfg['stride']
    bg_r   = cfg['background_keep_ratio']
    min_g  = cfg['minimum_glacier_percentage']
    mode   = cfg['padding_mode']
    outdir = cfg['output_dir']
    srcdir = cfg['dataset_dir']

    os.makedirs(f"{outdir}/metadata", exist_ok=True)
    with open(f"{outdir}/metadata/config.json", 'w') as f:
        json.dump(cfg, f, indent=2)

    patch_records = []
    image_records = []

    for split in cfg['splits']:
        feat_dir = f"{srcdir}/{split}/features"
        mask_dir = f"{srcdir}/{split}/masks"
        os.makedirs(f"{outdir}/{split}/features", exist_ok=True)
        os.makedirs(f"{outdir}/{split}/masks", exist_ok=True)

        feat_files = sorted(glob.glob(f"{feat_dir}/feature_*.tif"))
        total = len(feat_files)
        print(f"\n[{split.upper()}] {total} images")

        n_gen = n_kept = 0

        for idx, fp in enumerate(feat_files):
            img_id = os.path.basename(fp).replace('feature_', '').replace('.tif', '')
            mp = f"{mask_dir}/mask_{img_id}.tif"

            with rasterio.open(fp) as fs:
                feat = np.moveaxis(fs.read().astype(np.float32), 0, -1)
            with rasterio.open(mp) as ms:
                mask = ms.read(1).astype(np.float32)[:, :, np.newaxis]

            feat = np.where((feat == -9999) | ~np.isfinite(feat), 0.0, feat)

            oH, oW = feat.shape[:2]
            img_fg  = int((mask > 0.5).sum())
            img_bg  = mask.size - img_fg
            img_pct = 100.0 * img_fg / mask.size

            n_img_gen = n_img_kept = 0

            for fp_p, mp_p, r, c, padded, pad_h, pad_w in \
                    extract_patches(feat, mask, P, S, mode):

                fg_px  = int((mp_p > 0.5).sum())
                bg_px  = mp_p.size - fg_px
                g_pct  = 100.0 * fg_px / mp_p.size

                has_glacier = g_pct > min_g
                keep = has_glacier or (rng.rand() < bg_r)

                n_gen  += 1
                n_img_gen += 1

                b_means = [round(float(fp_p[:, :, b].mean()), 6) for b in range(fp_p.shape[2])]
                b_stds  = [round(float(fp_p[:, :, b].std()),  6) for b in range(fp_p.shape[2])]

                patch_id = f"{img_id}_r{r:04d}_c{c:04d}"
                pad_px   = int(pad_h * (oW + pad_w) + pad_w * oH) if padded else 0

                rec = {
                    'patch_id':              patch_id,
                    'source_image_id':       img_id,
                    'split':                 split,
                    'original_image_height': oH,
                    'original_image_width':  oW,
                    'patch_row':             r,
                    'patch_col':             c,
                    'patch_height':          fp_p.shape[0],
                    'patch_width':           fp_p.shape[1],
                    'glacier_pixel_count':   fg_px,
                    'background_pixel_count': bg_px,
                    'glacier_percentage':    round(g_pct, 4),
                    'kept_or_filtered':      'kept' if keep else 'filtered',
                    'padding_applied':       padded,
                    'padding_bottom':        pad_h,
                    'padding_right':         pad_w,
                    'padding_pixels':        pad_px,
                }
                for b, name in enumerate(BAND_NAMES):
                    rec[f'mean_{name}'] = b_means[b]
                    rec[f'std_{name}']  = b_stds[b]

                patch_records.append(rec)

                if keep:
                    np.save(f"{outdir}/{split}/features/{patch_id}.npy", fp_p)
                    np.save(f"{outdir}/{split}/masks/{patch_id}.npy",    mp_p)
                    n_kept     += 1
                    n_img_kept += 1

            image_records.append({
                'image_id':                  img_id,
                'split':                     split,
                'original_height':           oH,
                'original_width':            oW,
                'glacier_pixels':            img_fg,
                'background_pixels':         img_bg,
                'glacier_percentage':        round(img_pct, 4),
                'number_of_generated_patches': n_img_gen,
                'number_of_kept_patches':    n_img_kept,
            })

            if (idx + 1) % 50 == 0 or (idx + 1) == total:
                print(f"  {idx+1}/{total}  gen={n_gen}  kept={n_kept}")

    # ── Save metadata ─────────────────────────────────────────────
    patch_df = pd.DataFrame(patch_records)
    image_df = pd.DataFrame(image_records)
    patch_df.to_csv(f"{outdir}/metadata/patch_metadata.csv", index=False)
    image_df.to_csv(f"{outdir}/metadata/dataset_summary.csv", index=False)
    print("\nMetadata CSV saved.")

    _write_diagnostics(patch_df, image_df, outdir, cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def _write_diagnostics(patch_df: pd.DataFrame, image_df: pd.DataFrame,
                        outdir: str, cfg: dict):
    kept = patch_df[patch_df['kept_or_filtered'] == 'kept']
    lines = []

    def h(title):
        lines.append("")
        lines.append("─" * 58)
        lines.append(f"  {title}")
        lines.append("─" * 58)

    lines.append("=" * 58)
    lines.append("  ROCKGLACIERENET — PATCH EXTRACTION DIAGNOSTICS")
    lines.append("=" * 58)
    lines.append(f"  patch_size           : {cfg['patch_size']}")
    lines.append(f"  stride               : {cfg['stride']}")
    lines.append(f"  background_keep_ratio: {cfg['background_keep_ratio']}")
    lines.append(f"  minimum_glacier_pct  : {cfg['minimum_glacier_percentage']}")
    lines.append(f"  padding_mode         : {cfg['padding_mode']}")

    # 1–4: image counts
    h("1–4  IMAGE COUNTS")
    lines.append(f"  Total original images: {len(image_df)}")
    for sp in cfg['splits']:
        lines.append(f"    {sp:5s}: {len(image_df[image_df['split']==sp])}")

    # 5–6: patch counts
    h("5–6  PATCH COUNTS")
    for sp in cfg['splits']:
        g = len(patch_df[patch_df['split'] == sp])
        k = len(kept[kept['split'] == sp])
        lines.append(f"  {sp:5s}: generated={g:5d}  kept={k:5d}  filtered={g-k:5d}")
    lines.append(f"  TOTAL: generated={len(patch_df):5d}  kept={len(kept):5d}")

    # 7: pixel class balance
    h("7  PIXEL CLASS BALANCE (kept patches)")
    tfg = kept['glacier_pixel_count'].sum()
    tbg = kept['background_pixel_count'].sum()
    tpx = tfg + tbg
    lines.append(f"  Glacier   : {int(tfg):>10,}  ({100*tfg/tpx:.2f}%)")
    lines.append(f"  Background: {int(tbg):>10,}  ({100*tbg/tpx:.2f}%)")
    lines.append(f"  Imbalance : {tbg/tfg:.2f}:1")

    # 8: glacier coverage distribution in kept patches
    h("8  GLACIER COVERAGE DISTRIBUTION (kept patches, %)")
    bins = [(0, 0.1), (0.1, 5), (5, 20), (20, 50), (50, 80), (80, 100.01)]
    for lo, hi in bins:
        n = int(((kept['glacier_percentage'] >= lo) & (kept['glacier_percentage'] < hi)).sum())
        bar = '█' * min(n // 5, 40)
        lines.append(f"  {lo:5.1f}–{hi:5.1f}%: {n:5d}  {bar}")

    # 9: patches per image
    h("9  KEPT PATCHES PER IMAGE")
    for sp in cfg['splits']:
        sub = image_df[image_df['split'] == sp]['number_of_kept_patches']
        lines.append(f"  {sp:5s}: min={sub.min()}"
                     f"  max={sub.max()}"
                     f"  mean={sub.mean():.1f}"
                     f"  median={sub.median():.0f}")

    # 10: image size histogram
    h("10  IMAGE SIZE HISTOGRAM (max(H,W))")
    mx = np.maximum(image_df['original_height'].values,
                    image_df['original_width'].values)
    for lo, hi, label in [(0,64,'<64'), (64,128,'64–128'),
                           (128,192,'128–192'), (192,256,'192–256'),
                           (256,9999,'>=256')]:
        n = int(((mx >= lo) & (mx < hi)).sum())
        bar = '█' * min(n // 5, 40)
        lines.append(f"  {label:>8}: {n:4d}  {bar}")

    # 11: histogram of image glacier %
    h("11  IMAGE-LEVEL GLACIER % HISTOGRAM")
    for lo, hi in [(0,5),(5,10),(10,20),(20,40),(40,60),(60,100.01)]:
        n = int(((image_df['glacier_percentage'] >= lo) &
                 (image_df['glacier_percentage'] < hi)).sum())
        bar = '█' * min(n // 5, 40)
        lines.append(f"  {lo:3.0f}–{hi:3.0f}%: {n:4d}  {bar}")

    # 12: histogram of patch glacier %
    h("12  PATCH-LEVEL GLACIER % HISTOGRAM (all generated)")
    for lo, hi in [(0,0.1),(0.1,5),(5,20),(20,50),(50,100.01)]:
        n = int(((patch_df['glacier_percentage'] >= lo) &
                 (patch_df['glacier_percentage'] < hi)).sum())
        bar = '█' * min(n // 10, 40)
        lines.append(f"  {lo:5.1f}–{hi:5.1f}%: {n:5d}  {bar}")

    # 13–14: padding stats
    h("13–14  PADDING STATS")
    need_pad = ((image_df['original_height'] < cfg['patch_size']) |
                (image_df['original_width']  < cfg['patch_size'])).sum()
    pat_pad  = patch_df['padding_applied'].sum()
    lines.append(f"  Images needing padding : {need_pad}/{len(image_df)}"
                 f"  ({100*need_pad/len(image_df):.1f}%)")
    lines.append(f"  Patches with padding   : {pat_pad}/{len(patch_df)}"
                 f"  ({100*pat_pad/len(patch_df):.1f}%)")

    # 15: expansion factor
    h("15  DATASET EXPANSION FACTOR")
    for sp in cfg['splits']:
        n_img = len(image_df[image_df['split'] == sp])
        n_kpt = len(kept[kept['split'] == sp])
        factor = n_kpt / n_img if n_img else 0
        lines.append(f"  {sp:5s}: {n_img} images → {n_kpt} patches  (×{factor:.1f})")

    report = "\n".join(lines)
    print("\n" + report)

    out_path = f"{outdir}/metadata/diagnostics_report.txt"
    with open(out_path, 'w') as f:
        f.write(report)
    print(f"\nDiagnostics → {out_path}")


if __name__ == '__main__':
    run(PATCH_CONFIG)
