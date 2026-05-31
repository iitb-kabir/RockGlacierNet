import os
import glob
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

INPUT_DIR = r'D:\RockGlacier_Project\EDA\individual_polygons'
OUTPUT_MASK_DIR = r'D:\RockGlacier_Project\masks'
REPORT_PATH = os.path.join(OUTPUT_MASK_DIR, 'mask_report.csv')

TARGET_CRS = 'EPSG:32645'  # UTM Zone 45N for Sikkim
RESOLUTION = 10.0  # 10 meters per pixel
PADDING = 200.0  # 200 meters padding around the polygon

def generate_masks():
    os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)
    
    geojson_files = sorted(glob.glob(os.path.join(INPUT_DIR, '*.geojson')))
    if not geojson_files:
        print(f"No GeoJSON files found in {INPUT_DIR}")
        return

    print(f"Found {len(geojson_files)} polygons. Starting rasterization at {RESOLUTION}m/px...")
    
    report_data = []
    
    for filepath in geojson_files:
        filename = os.path.basename(filepath)
        # Extracts index from 'polygon_0.geojson'
        idx_str = filename.split('_')[1].split('.')[0]
        idx = int(idx_str)
        
        # 1. Read polygon
        gdf = gpd.read_file(filepath)
        
        # 2. Reproject to UTM Zone 45N for meter-based math
        gdf_proj = gdf.to_crs(TARGET_CRS)
        
        # 3. Compute Bounds with Padding
        bounds = gdf_proj.total_bounds
        import numpy as np
        if np.isnan(bounds).any() or gdf_proj.empty or gdf_proj.geometry.is_empty.all() or gdf_proj.geometry.isna().all():
            print(f"Skipping invalid/empty geometry for polygon {idx}")
            continue
            
        minx, miny, maxx, maxy = bounds
        
        minx -= PADDING
        miny -= PADDING
        maxx += PADDING
        maxy += PADDING
        
        # 4. Calculate Raster Dimensions
        width = int((maxx - minx) / RESOLUTION)
        height = int((maxy - miny) / RESOLUTION)
        
        # Adjust maxx and miny slightly so resolution is exactly 10.0
        maxx = minx + (width * RESOLUTION)
        miny = maxy - (height * RESOLUTION)
        
        # 5. Create Affine Transform
        # from_origin(west, north, xsize, ysize)
        transform = from_origin(minx, maxy, RESOLUTION, RESOLUTION)
        
        # 6. Rasterize
        # shapes requires an iterable of (geometry, value) pairs
        shapes = ((geom, 1) for geom in gdf_proj.geometry)
        mask = rasterize(
            shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype='uint8'
        )
        
        # 7. Compute Stats
        fg_pixels = int(mask.sum())
        bg_pixels = int((width * height) - fg_pixels)
        rasterized_area = fg_pixels * (RESOLUTION ** 2)
        polygon_area = gdf_proj.geometry.area.sum()
        
        # 8. Save GeoTIFF
        out_tif = os.path.join(OUTPUT_MASK_DIR, f"mask_{idx:04d}.tif")
        with rasterio.open(
            out_tif,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=mask.dtype,
            crs=TARGET_CRS,
            transform=transform,
        ) as dst:
            dst.write(mask, 1)
            
        # 9. Plot first 5 masks
        if idx < 5:
            plt.figure(figsize=(6, 6))
            plt.imshow(mask, cmap='gray')
            plt.title(f"Mask {idx:04d} ({width}x{height} pixels)")
            plt.axis('off')
            plt.savefig(os.path.join(OUTPUT_MASK_DIR, f"plot_mask_{idx:04d}.png"), dpi=150, bbox_inches='tight')
            plt.close()
            
        report_data.append({
            'Polygon_ID': idx,
            'Width': width,
            'Height': height,
            'Foreground_Pixels': fg_pixels,
            'Background_Pixels': bg_pixels,
            'Polygon_Area_m2': round(polygon_area, 2),
            'Rasterized_Area_m2': round(rasterized_area, 2)
        })
        
        if idx % 50 == 0:
            print(f"Processed {idx+1}/{len(geojson_files)} masks...")

    # Save Report
    df_report = pd.DataFrame(report_data)
    df_report.to_csv(REPORT_PATH, index=False)
    print(f"\nSuccess! Generated {len(geojson_files)} masks.")
    print(f"Validation report saved to {REPORT_PATH}")

if __name__ == '__main__':
    generate_masks()
