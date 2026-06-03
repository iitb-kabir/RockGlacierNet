import ee
import geemap
import geopandas as gpd
import os
import glob
from shapely.geometry import box

# ==========================================
# Authentication & Initialization
# ==========================================
try:
    ee.Initialize(project='aerobic-mile-484705-s8')
except Exception:
    print("\n--- Google Earth Engine Authentication Required ---")
    ee.Authenticate()
    ee.Initialize(project='aerobic-mile-484705-s8')
    print("Authentication successful!\n")

INPUT_DIR = r'D:\RockGlacier_Project\EDA\individual_polygons'
OUTPUT_DIR = r'D:\RockGlacier_Project\features'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_CRS_UTM = 'EPSG:32645'  # UTM Zone 45N for meter calculations

def get_s2_image(roi):
    # Expanded date range for the Himalayas (Monsoon clouds often block strict filters)
    # We rely on the pixel-level QA60 mask instead of scene-level filtering.
    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(roi) \
        .filterDate('2021-01-01', '2024-01-01')
    
    # Cloud masking function
    def mask_s2_clouds(image):
        qa = image.select('QA60')
        cloudBitMask = 1 << 10
        cirrusBitMask = 1 << 11
        mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
        return image.updateMask(mask).divide(10000)
    
    # Apply cloud mask and get median
    s2_median = collection.map(mask_s2_clouds).median().clip(roi)
    
    # Calculate indices
    ndvi = s2_median.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndwi = s2_median.normalizedDifference(['B3', 'B8']).rename('NDWI')
    ndsi = s2_median.normalizedDifference(['B3', 'B11']).rename('NDSI')
    
    # Return selected bands + indices
    return s2_median.select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12']).addBands([ndvi, ndwi, ndsi])

def get_terrain_image(roi):
    # COPERNICUS DEM is a collection of tiles. We must calculate slope/aspect BEFORE mosaicking
    # so that the native 30m projection is preserved for the terrain algorithms.
    def add_terrain(img):
        slope = ee.Terrain.slope(img).rename('slope')
        aspect = ee.Terrain.aspect(img).rename('aspect')
        return img.addBands([slope, aspect])
        
    terrain = ee.ImageCollection('COPERNICUS/DEM/GLO30').select('DEM').map(add_terrain).mosaic().clip(roi)
    elevation = terrain.select('DEM').rename('elevation')
    slope = terrain.select('slope')
    aspect = terrain.select('aspect')
    return elevation.addBands([slope, aspect])

def main():
    geojson_files = sorted(glob.glob(os.path.join(INPUT_DIR, '*.geojson')))
    print(f"Found {len(geojson_files)} polygons to process.")
    
    # Process all 626 polygons
    for filepath in geojson_files:
        filename = os.path.basename(filepath)
        idx_str = filename.split('_')[1].split('.')[0]
        idx = int(idx_str)
        
        out_tif = os.path.join(OUTPUT_DIR, f"feature_{idx:04d}.tif")
        if os.path.exists(out_tif):
            print(f"Skipping {idx}, already exists.")
            continue
            
        # 1. Read polygon
        gdf = gpd.read_file(filepath)
        if gdf.empty or gdf.geometry.is_empty.all() or gdf.geometry.isna().all():
            print(f"Skipping invalid/empty geometry for polygon {idx}")
            continue
            
        # 2. Add 200m Buffer in UTM
        gdf_utm = gdf.to_crs(TARGET_CRS_UTM)
        bounds = gdf_utm.total_bounds
        minx, miny, maxx, maxy = bounds
        minx -= 200
        miny -= 200
        maxx += 200
        maxy += 200
        
        # 3. Convert padded bounds back to EPSG:4326 for Earth Engine Bounding Box
        padded_box_utm = gpd.GeoDataFrame(geometry=[box(minx, miny, maxx, maxy)], crs=TARGET_CRS_UTM)
        padded_box_4326 = padded_box_utm.to_crs('EPSG:4326')
        bounds_4326 = padded_box_4326.total_bounds
        
        # Earth Engine ROI
        roi = ee.Geometry.BBox(bounds_4326[0], bounds_4326[1], bounds_4326[2], bounds_4326[3])
        
        # 4. Generate Feature Stacks
        try:
            s2_img = get_s2_image(roi)
            terrain_img = get_terrain_image(roi)
            
            # Combine into single 12-band image
            final_img = s2_img.addBands(terrain_img)
            
            # 5. Download Image
            print(f"Downloading feature stack {idx:04d}...")
            geemap.ee_export_image(
                final_img, 
                filename=out_tif, 
                scale=10, 
                region=roi, 
                crs=TARGET_CRS_UTM,
                file_per_band=False
            )
        except Exception as e:
            print(f"Failed to process {idx:04d}: {e}")

if __name__ == '__main__':
    main()
