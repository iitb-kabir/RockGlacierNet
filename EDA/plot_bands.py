import rasterio
import matplotlib.pyplot as plt
import numpy as np
import os

f = r'D:\RockGlacier_Project\features\feature_0000.tif'
if not os.path.exists(f):
    f = r'D:\RockGlacier_Project\features\feature_0100.tif'

out_dir = r'D:\RockGlacier_Project\EDA\plots'
os.makedirs(out_dir, exist_ok=True)

bands = ['B2','B3','B4','B8','B11','B12','NDVI','NDWI','NDSI','Elevation','Slope','Aspect']

with rasterio.open(f) as src:
    data = src.read()

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for i in range(12):
    ax = axes[i]
    band_data = data[i]
    # Mask out completely invalid data (-9999 or nan)
    band_data = np.ma.masked_where((band_data == -9999) | (~np.isfinite(band_data)), band_data)
    
    cmap = 'gray'
    if bands[i] in ['NDVI', 'NDWI', 'NDSI']:
        cmap = 'RdYlGn' if bands[i] == 'NDVI' else 'Blues' if bands[i] == 'NDWI' else 'coolwarm'
    elif bands[i] == 'Elevation':
        cmap = 'terrain'
        
    im = ax.imshow(band_data, cmap=cmap)
    ax.set_title(bands[i])
    ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
out_file = os.path.join(out_dir, 'all_bands_preview.png')
plt.savefig(out_file, dpi=150, bbox_inches='tight')
print(f'Saved plot to {out_file}')
