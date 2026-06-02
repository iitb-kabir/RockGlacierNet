import rasterio
import numpy as np
f = r'D:\RockGlacier_Project\features\feature_0000.tif'
import os
if not os.path.exists(f):
    f = r'D:\RockGlacier_Project\features\feature_0100.tif'

bands = ['B2','B3','B4','B8','B11','B12','NDVI','NDWI','NDSI','Elevation','Slope','Aspect']

with rasterio.open(f) as src:
    data = src.read()

print('\n## Actual Dataset Values (Sample)')
print(f'Below are the actual numbers (Min, Max, and Mean) calculated directly from the pixels inside `{os.path.basename(f)}`:')
print('\n| Band Index | Feature Name | Min Value | Max Value | Mean Value |')
print('| :---: | :--- | :--- | :--- | :--- |')

for i in range(12):
    b = data[i]
    # Mask out completely invalid data (often 0 or extremely low negative values from GEE)
    b_valid = b[np.isfinite(b)]
    
    min_val = b_valid.min()
    max_val = b_valid.max()
    mean_val = b_valid.mean()
    
    print(f'| **{i+1}** | `{bands[i]}` | {min_val:.4f} | {max_val:.4f} | {mean_val:.4f} |')
