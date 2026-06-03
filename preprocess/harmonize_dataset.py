# D:\RockGlacier_Project\harmonize_dataset.py
import os
import glob
import rasterio
from rasterio.windows import Window
import numpy as np
import pandas as pd

FEATURE_DIR = r"D:\RockGlacier_Project\features"
MASK_DIR = r"D:\RockGlacier_Project\masks"
H_FEATURE_DIR = r"D:\RockGlacier_Project\harmonized_features"
H_MASK_DIR = r"D:\RockGlacier_Project\harmonized_masks"

os.makedirs(H_FEATURE_DIR, exist_ok=True)
os.makedirs(H_MASK_DIR, exist_ok=True)

feature_files = sorted(glob.glob(os.path.join(FEATURE_DIR, 'feature_*.tif')))
mask_files    = sorted(glob.glob(os.path.join(MASK_DIR,    'mask_*.tif')))

def get_idx(filepath):
    return os.path.basename(filepath).split('_')[1].split('.')[0]

mask_lookup = {get_idx(p): p for p in mask_files}

results = []

print(f"Found {len(feature_files)} feature files and {len(mask_files)} mask files.")
print("Starting harmonization process...")

for f_path in feature_files:
    idx = get_idx(f_path)
    m_path = mask_lookup.get(idx)
    
    if not m_path:
        continue
        
    with rasterio.open(f_path) as f_src, rasterio.open(m_path) as m_src:
        # Determine the minimum common shape
        min_h = min(f_src.shape[0], m_src.shape[0])
        min_w = min(f_src.shape[1], m_src.shape[1])
        
        f_window = Window(0, 0, min_w, min_h)
        m_window = Window(0, 0, min_w, min_h)
        
        f_data = f_src.read(window=f_window)
        m_data = m_src.read(1, window=m_window)
        
        f_meta = f_src.meta.copy()
        m_meta = m_src.meta.copy()
        
        # Calculate new transform for the window
        f_transform = f_src.window_transform(f_window)
        
        f_meta.update({
            'height': min_h,
            'width': min_w,
            'transform': f_transform
        })
        # Enforce exact match for mask meta based on feature
        m_meta.update({
            'height': min_h,
            'width': min_w,
            'transform': f_transform
        })
        
        out_f = os.path.join(H_FEATURE_DIR, f"feature_{idx}.tif")
        out_m = os.path.join(H_MASK_DIR, f"mask_{idx}.tif")
        
        with rasterio.open(out_f, 'w', **f_meta) as dest:
            dest.write(f_data)
            
        with rasterio.open(out_m, 'w', **m_meta) as dest:
            dest.write(m_data, 1)
            
        results.append({
            'idx': idx,
            'orig_f_shape': f_src.shape,
            'orig_m_shape': m_src.shape,
            'harmonized_shape': (min_h, min_w),
            'CRS_match': f_src.crs == m_src.crs
        })

df = pd.DataFrame(results)
print("\n--- Harmonization Complete ---")
print(f"Total processed pairs: {len(df)}")
mismatch_count = len(df[df['orig_f_shape'] != df['orig_m_shape']])
print(f"Pairs that required cropping: {mismatch_count}")
print("\nSample metadata summary:")
print(df.head())

summary_csv = os.path.join("D:\\RockGlacier_Project", "harmonization_summary.csv")
df.to_csv(summary_csv, index=False)
print(f"\nFull metadata summary saved to {summary_csv}")
