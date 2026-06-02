import ee
import geopandas as gpd
from shapely.geometry import box

# Initialize
ee.Initialize(project='aerobic-mile-484705-s8')

# ROI
TARGET_CRS_UTM = 'EPSG:32645'
gdf = gpd.read_file(r'D:\RockGlacier_Project\EDA\individual_polygons\polygon_0.geojson')
gdf_utm = gdf.to_crs(TARGET_CRS_UTM)
bounds = gdf_utm.total_bounds
minx, miny, maxx, maxy = bounds
minx -= 200
miny -= 200
maxx += 200
maxy += 200
padded_box_utm = gpd.GeoDataFrame(geometry=[box(minx, miny, maxx, maxy)], crs=TARGET_CRS_UTM)
padded_box_4326 = padded_box_utm.to_crs('EPSG:4326')
bounds_4326 = padded_box_4326.total_bounds
roi = ee.Geometry.BBox(bounds_4326[0], bounds_4326[1], bounds_4326[2], bounds_4326[3])

# Collection
collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
    .filterBounds(roi) \
    .filterDate('2021-01-01', '2024-01-01')

print('Collection size:', collection.size().getInfo())

def mask_s2_clouds(image):
    qa = image.select('QA60')
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
    return image.updateMask(mask).divide(10000)

s2_median = collection.map(mask_s2_clouds).median().clip(roi)

try:
    bands = s2_median.bandNames().getInfo()
    print('Bands:', bands)
except Exception as e:
    print('Error getting bands:', e)

try:
    test_select = s2_median.select(['B2', 'B3'])
    print('Select worked:', test_select.bandNames().getInfo())
except Exception as e:
    print('Error selecting bands:', e)
