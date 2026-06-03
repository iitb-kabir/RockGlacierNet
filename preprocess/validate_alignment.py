"""
Validate feature-mask alignment for extracted patches.

Checks:
  1. Every feature patch has a corresponding mask patch (filename match)
  2. Shape consistency: both are (64, 64, *) with correct dtype
  3. No NaN / Inf in features
  4. Mask contains only {0, 1} values
  5. Pixel-level sanity: glacier pixels in mask must NOT be all-zero in features
     (if every band is zero at a glacier pixel, that is suspicious)
  6. Random spot-check: print stats for 5 random patches per split

Usage (from project root):
    conda run -n brats python preprocess/validate_alignment.py
"""

import os
import sys
import glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess.patch_config import PATCH_CONFIG

PATCH_DIR = PATCH_CONFIG['output_dir']
SPLITS    = PATCH_CONFIG['splits']


def check_split(split: str) -> dict:
    feat_dir = os.path.join(PATCH_DIR, split, 'features')
    mask_dir = os.path.join(PATCH_DIR, split, 'masks')

    feat_files = sorted(glob.glob(os.path.join(feat_dir, '*.npy')))
    mask_files = sorted(glob.glob(os.path.join(mask_dir, '*.npy')))

    results = {
        'split': split,
        'n_feat': len(feat_files),
        'n_mask': len(mask_files),
        'count_mismatch': 0,
        'name_mismatch': 0,
        'shape_errors': 0,
        'dtype_errors': 0,
        'nan_errors': 0,
        'inf_errors': 0,
        'mask_value_errors': 0,
        'all_zero_glacier_errors': 0,
        'errors': [],
    }

    # 1. Count match
    if len(feat_files) != len(mask_files):
        results['count_mismatch'] = abs(len(feat_files) - len(mask_files))
        results['errors'].append(
            f"Count mismatch: {len(feat_files)} features vs {len(mask_files)} masks")

    # 2. Per-file checks
    for fp in feat_files:
        pid = os.path.basename(fp)
        mp  = os.path.join(mask_dir, pid)

        if not os.path.exists(mp):
            results['name_mismatch'] += 1
            results['errors'].append(f"No mask for {pid}")
            continue

        feat = np.load(fp)
        mask = np.load(mp)

        # Shape
        if feat.ndim != 3 or feat.shape[2] != PATCH_CONFIG['num_bands']:
            results['shape_errors'] += 1
            results['errors'].append(f"{pid}: feat shape {feat.shape}")
        if mask.ndim != 3 or mask.shape[2] != 1:
            results['shape_errors'] += 1
            results['errors'].append(f"{pid}: mask shape {mask.shape}")
        if feat.shape[:2] != mask.shape[:2]:
            results['shape_errors'] += 1
            results['errors'].append(f"{pid}: H/W mismatch feat{feat.shape} mask{mask.shape}")

        # Dtype
        if feat.dtype != np.float32:
            results['dtype_errors'] += 1
        if mask.dtype not in (np.float32, np.uint8, np.int32, np.int64):
            results['dtype_errors'] += 1

        # NaN / Inf
        if np.isnan(feat).any():
            results['nan_errors'] += 1
            results['errors'].append(f"{pid}: NaN in feature")
        if np.isinf(feat).any():
            results['inf_errors'] += 1
            results['errors'].append(f"{pid}: Inf in feature")

        # Mask values
        unique_vals = np.unique(mask)
        if not set(unique_vals.tolist()).issubset({0, 1, 0.0, 1.0}):
            results['mask_value_errors'] += 1
            results['errors'].append(f"{pid}: unexpected mask values {unique_vals}")

        # Pixel-level sanity: glacier pixels should not be all-zero across all bands
        glacier_mask = mask[:, :, 0] > 0.5
        if glacier_mask.sum() > 0:
            glacier_feat = feat[glacier_mask]          # (N_glacier, 12)
            all_zero_rows = (glacier_feat == 0).all(axis=1)
            frac_zero = all_zero_rows.mean()
            if frac_zero > 0.5:                        # >50% glacier pixels all-zero
                results['all_zero_glacier_errors'] += 1
                results['errors'].append(
                    f"{pid}: {frac_zero:.1%} of glacier pixels have all-zero features (padding bleed?)")

    return results


def spot_check(split: str, n: int = 5):
    feat_dir  = os.path.join(PATCH_DIR, split, 'features')
    feat_files = sorted(glob.glob(os.path.join(feat_dir, '*.npy')))
    if not feat_files:
        return
    rng   = np.random.RandomState(0)
    picks = rng.choice(feat_files, size=min(n, len(feat_files)), replace=False)

    print(f"\n  Spot-check ({split}, {len(picks)} random patches):")
    print(f"  {'patch_id':<35} {'shape':<14} {'glacier%':>9} {'feat_min':>9} {'feat_max':>9}")
    print("  " + "-" * 82)

    for fp in sorted(picks):
        pid  = os.path.splitext(os.path.basename(fp))[0]
        mask_path = fp.replace('features', 'masks')
        feat = np.load(fp)
        mask = np.load(mask_path)
        g_pct = 100.0 * (mask > 0.5).sum() / mask.size
        print(f"  {pid:<35} {str(feat.shape):<14} {g_pct:>8.2f}% "
              f"{feat.min():>9.3f} {feat.max():>9.3f}")


def main():
    print("=" * 58)
    print("  PATCH ALIGNMENT VALIDATION")
    print("=" * 58)

    all_ok = True
    for split in SPLITS:
        feat_dir = os.path.join(PATCH_DIR, split, 'features')
        if not os.path.isdir(feat_dir):
            print(f"\n[{split.upper()}] SKIPPED — directory not found: {feat_dir}")
            continue

        print(f"\n[{split.upper()}] Checking…")
        r = check_split(split)

        status = "OK" if not r['errors'] else f"ISSUES FOUND"
        print(f"  Files       : {r['n_feat']} features, {r['n_mask']} masks")
        print(f"  Count match : {'OK' if r['count_mismatch'] == 0 else f'FAIL ({r[chr(99)+chr(111)+chr(117)+chr(110)+chr(116)+chr(95)+chr(109)+chr(105)+chr(115)+chr(109)+chr(97)+chr(116)+chr(99)+chr(104)]} mismatch)'}")
        print(f"  Name match  : {'OK' if r['name_mismatch'] == 0 else f'FAIL ({r[chr(110)+chr(97)+chr(109)+chr(101)+chr(95)+chr(109)+chr(105)+chr(115)+chr(109)+chr(97)+chr(116)+chr(99)+chr(104)]} unmatched)'}")
        print(f"  Shape errors: {r['shape_errors']}")
        print(f"  NaN errors  : {r['nan_errors']}")
        print(f"  Inf errors  : {r['inf_errors']}")
        print(f"  Mask value  : {r['mask_value_errors']} errors")
        print(f"  Zero-bleed  : {r['all_zero_glacier_errors']} patches")
        print(f"  Status      : {status}")

        if r['errors']:
            all_ok = False
            print("  First 10 errors:")
            for e in r['errors'][:10]:
                print(f"    ✗ {e}")

        spot_check(split)

    print("\n" + "=" * 58)
    print(f"  OVERALL: {'ALL CHECKS PASSED' if all_ok else 'ISSUES DETECTED — see above'}")
    print("=" * 58)


if __name__ == '__main__':
    main()
