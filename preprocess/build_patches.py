"""
RockGlacierNet — Stage 2: PATCH BUILDING (config-driven, any size).

raw/{features,masks}  ->  patch_dataset/patches_{size}/{split}/{features,masks}/

What it does (replaces harmonize_dataset + split_dataset + extract_patches):
  * pairs feature_XXXX.tif with mask_XXXX.tif, crops both to common shape
    (safety net; the collector already aligns them),
  * assigns each SOURCE GLACIER to train/val/test  (split by image -> no
    overlapping-patch leakage across splits),
  * tiles every image into patch_size x patch_size GeoTIFF patches with overlap,
    keeping all glacier patches + a sampled fraction of background patches,
  * writes georeferenced GeoTIFF patches named feature_*.tif / mask_*.tif so the
    existing RockGlacierDataGenerator can train on them UNCHANGED,
  * writes metadata/ (config.json, patch_metadata.csv, dataset_summary.csv,
    diagnostics_report.txt).

Usage (rasterio only — runs anywhere, e.g. conda env `brats`):
    python preprocess/build_patches.py                 # uses DEFAULT_PATCH (128)
    python preprocess/build_patches.py --patch-size 64
    python preprocess/build_patches.py --patch-size 256
    # test against the OLD per-glacier tiles before re-collecting:
    python preprocess/build_patches.py --patch-size 64 \
        --features-dir features --masks-dir masks --limit 40

All tunables live in patch_dataset/config.py.
"""

import os
import sys
import glob
import json
import math
import argparse
import numpy as np
import rasterio
import pandas as pd
from rasterio.windows import Window, transform as window_transform

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from patch_dataset import config as C


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def padded_size(dim: int, patch: int, stride: int) -> int:
    if dim <= patch:
        return patch
    n = math.ceil((dim - patch) / stride)
    return n * stride + patch


def iter_patches(feat, mask, patch, stride, mode):
    """Yield (feat_patch, mask_patch, row, col, padded) covering the whole image."""
    H, W = feat.shape[:2]
    pH, pW = padded_size(H, patch, stride), padded_size(W, patch, stride)
    dh, dw = pH - H, pW - W
    padded = dh > 0 or dw > 0
    if padded:
        feat = np.pad(feat, ((0, dh), (0, dw), (0, 0)),
                      mode=mode if mode == "reflect" else "constant")
        mask = np.pad(mask, ((0, dh), (0, dw), (0, 0)),
                      mode="constant", constant_values=0)
    for r in range(0, pH - patch + 1, stride):
        for c in range(0, pW - patch + 1, stride):
            yield feat[r:r+patch, c:c+patch, :], mask[r:r+patch, c:c+patch, :], r, c, padded


# ─────────────────────────────────────────────────────────────────────────────
# Split assignment (by source glacier id)
# ─────────────────────────────────────────────────────────────────────────────

def assign_splits(image_ids, ratios, seed):
    rng = np.random.RandomState(seed)
    ids = sorted(image_ids)
    rng.shuffle(ids)
    n = len(ids)
    n_tr = int(n * ratios["train"])
    n_va = int(n * (ratios["train"] + ratios["val"]))
    out = {}
    for i, iid in enumerate(ids):
        out[iid] = "train" if i < n_tr else ("val" if i < n_va else "test")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(patch, feat_dir, mask_dir, limit):
    stride = C.stride_for(patch)
    outdir = C.patches_dir(patch)
    cfg = C.as_dict(patch)
    cfg["features_dir_used"], cfg["masks_dir_used"] = str(feat_dir), str(mask_dir)

    (outdir / "metadata").mkdir(parents=True, exist_ok=True)
    for sp in C.SPLIT_RATIOS:
        (outdir / sp / "features").mkdir(parents=True, exist_ok=True)
        (outdir / sp / "masks").mkdir(parents=True, exist_ok=True)
    with open(outdir / "metadata" / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    feat_files = sorted(glob.glob(os.path.join(str(feat_dir), "feature_*.tif")))
    if limit:
        feat_files = feat_files[:limit]

    def img_id(fp):
        return os.path.basename(fp).replace("feature_", "").replace(".tif", "")

    ids = [img_id(fp) for fp in feat_files]
    split_of = assign_splits(ids, C.SPLIT_RATIOS, C.RANDOM_SEED)

    print(f"patch_size={patch} stride={stride} | {len(feat_files)} source tiles "
          f"-> {outdir.name}/")

    rng = np.random.RandomState(C.RANDOM_SEED)
    patch_records, image_records = [], []
    n_gen = n_kept = 0

    for fp in feat_files:
        iid = img_id(fp)
        split = split_of[iid]
        mp = os.path.join(str(mask_dir), f"mask_{iid}.tif")
        if not os.path.exists(mp):
            print(f"  {iid}: no matching mask — skipped")
            continue

        with rasterio.open(fp) as fs:
            feat = np.moveaxis(fs.read().astype(np.float32), 0, -1)   # (H,W,C)
            base_transform, crs = fs.transform, fs.crs
        with rasterio.open(mp) as ms:
            mask = ms.read(1).astype(np.uint8)[:, :, np.newaxis]      # (H,W,1)

        # Safety net: crop both to common shape (collector already aligns them).
        ch, cw = min(feat.shape[0], mask.shape[0]), min(feat.shape[1], mask.shape[1])
        feat, mask = feat[:ch, :cw, :], mask[:ch, :cw, :]
        feat = np.where((feat == C.NODATA_VALUE) | ~np.isfinite(feat), 0.0, feat)

        oH, oW = feat.shape[:2]
        img_fg = int((mask > 0).sum())
        n_img_kept = n_img_gen = 0

        for fpat, mpat, r, c, padded in iter_patches(feat, mask, patch, stride, C.PADDING_MODE):
            g_pct = 100.0 * int((mpat > 0).sum()) / mpat.size
            keep = (g_pct > C.MIN_GLACIER_PERCENTAGE) or (rng.rand() < C.BACKGROUND_KEEP_RATIO)
            n_gen += 1
            n_img_gen += 1

            patch_id = f"{iid}_r{r:04d}_c{c:04d}"
            rec = {
                "patch_id": patch_id, "source_image_id": iid, "split": split,
                "patch_row": r, "patch_col": c,
                "glacier_pixel_count": int((mpat > 0).sum()),
                "glacier_percentage": round(g_pct, 4),
                "kept_or_filtered": "kept" if keep else "filtered",
                "padding_applied": bool(padded),
            }
            for b, name in enumerate(C.BAND_NAMES):
                rec[f"mean_{name}"] = round(float(fpat[:, :, b].mean()), 6)
            patch_records.append(rec)

            if not keep:
                continue

            ptransform = window_transform(Window(c, r, patch, patch), base_transform)
            fout = outdir / split / "features" / f"feature_{patch_id}.tif"
            mout = outdir / split / "masks" / f"mask_{patch_id}.tif"
            with rasterio.open(fout, "w", driver="GTiff", height=patch, width=patch,
                               count=C.NUM_BANDS, dtype="float32",
                               crs=crs, transform=ptransform) as dst:
                dst.write(np.moveaxis(fpat.astype(np.float32), -1, 0))
            with rasterio.open(mout, "w", driver="GTiff", height=patch, width=patch,
                               count=1, dtype="uint8",
                               crs=crs, transform=ptransform, nodata=0) as dst:
                dst.write(mpat[:, :, 0].astype(np.uint8), 1)
            n_kept += 1
            n_img_kept += 1

        image_records.append({
            "image_id": iid, "split": split,
            "original_height": oH, "original_width": oW,
            "glacier_pixels": img_fg,
            "glacier_percentage": round(100.0 * img_fg / mask.size, 4),
            "generated_patches": n_img_gen, "kept_patches": n_img_kept,
        })

    patch_df = pd.DataFrame(patch_records)
    image_df = pd.DataFrame(image_records)
    patch_df.to_csv(outdir / "metadata" / "patch_metadata.csv", index=False)
    image_df.to_csv(outdir / "metadata" / "dataset_summary.csv", index=False)
    _diagnostics(patch_df, image_df, outdir, cfg)
    print(f"\nDone. generated={n_gen} kept={n_kept} -> {outdir}")


def _diagnostics(patch_df, image_df, outdir, cfg):
    kept = patch_df[patch_df["kept_or_filtered"] == "kept"] if len(patch_df) else patch_df
    L = []
    def h(t): L.extend(["", "─" * 56, f"  {t}", "─" * 56])

    L += ["=" * 56, "  ROCKGLACIERENET — PATCH BUILD DIAGNOSTICS", "=" * 56]
    L += [f"  patch_size {cfg['patch_size']} | stride {cfg['stride']} | "
          f"bg_keep {cfg['background_keep_ratio']} | pad {cfg['padding_mode']}"]

    h("PATCH COUNTS")
    for sp in C.SPLIT_RATIOS:
        g = len(patch_df[patch_df["split"] == sp]) if len(patch_df) else 0
        k = len(kept[kept["split"] == sp]) if len(kept) else 0
        L.append(f"  {sp:5s}: generated={g:6d}  kept={k:6d}  filtered={g-k:6d}")
    L.append(f"  TOTAL: generated={len(patch_df):6d}  kept={len(kept):6d}")

    h("IMAGE COUNTS PER SPLIT")
    for sp in C.SPLIT_RATIOS:
        L.append(f"  {sp:5s}: {len(image_df[image_df['split']==sp]) if len(image_df) else 0} glaciers")

    if len(kept):
        h("PIXEL CLASS BALANCE (kept patches)")
        tfg = int(kept["glacier_pixel_count"].sum())
        tpx = len(kept) * cfg["patch_size"] ** 2
        tbg = tpx - tfg
        L.append(f"  Glacier   : {tfg:>12,}  ({100*tfg/tpx:.2f}%)")
        L.append(f"  Background: {tbg:>12,}  ({100*tbg/tpx:.2f}%)")
        if tfg:
            L.append(f"  Imbalance : {tbg/tfg:.1f} : 1")

        h("GLACIER COVERAGE OF KEPT PATCHES (%)")
        for lo, hi in [(0, 0.1), (0.1, 5), (5, 20), (20, 50), (50, 80), (80, 100.01)]:
            n = int(((kept["glacier_percentage"] >= lo) & (kept["glacier_percentage"] < hi)).sum())
            L.append(f"  {lo:5.1f}-{hi:5.1f}%: {n:6d}  {'█' * min(n // 5, 40)}")

    if len(image_df):
        h("DATASET EXPANSION (glaciers -> kept patches)")
        for sp in C.SPLIT_RATIOS:
            sub = image_df[image_df["split"] == sp]
            ni, nk = len(sub), int(sub["kept_patches"].sum()) if len(sub) else 0
            L.append(f"  {sp:5s}: {ni:4d} glaciers -> {nk:6d} patches  (×{nk/ni:.1f})"
                     if ni else f"  {sp:5s}: 0 glaciers")

    report = "\n".join(L)
    print("\n" + report)
    with open(outdir / "metadata" / "diagnostics_report.txt", "w") as f:
        f.write(report)


def main():
    ap = argparse.ArgumentParser(description="Build GeoTIFF patches from raw tiles.")
    ap.add_argument("--patch-size", type=int, default=C.DEFAULT_PATCH, choices=None)
    ap.add_argument("--features-dir", default=str(C.RAW_FEATURES_DIR))
    ap.add_argument("--masks-dir", default=str(C.RAW_MASKS_DIR))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run(args.patch_size, args.features_dir, args.masks_dir, args.limit)


if __name__ == "__main__":
    main()
