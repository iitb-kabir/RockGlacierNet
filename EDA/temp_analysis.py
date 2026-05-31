import geopandas as gpd
import os

SHAPEFILE_PATH = r'D:\RockGlacier_Project\raw_data\Final_merged01\Final_merged01.shp'
OUT_PATH = r'D:\RockGlacier_Project\EDA\dataset_analysis.md'

def analyze():
    gdf = gpd.read_file(SHAPEFILE_PATH)
    with open(OUT_PATH, 'w') as f:
        f.write('# Rock Glacier Dataset Analysis\n\n')
        f.write('## Basic Info\n')
        f.write(f'- Total Features: {len(gdf)}\n')
        f.write(f'- CRS: {gdf.crs}\n')
        f.write(f'- Geometry Types: {gdf.geom_type.unique()}\n')
        f.write(f'- Bounds (minx, miny, maxx, maxy): {gdf.total_bounds}\n\n')
        
        f.write('## Columns\n```text\n')
        f.write(gdf.dtypes.to_string() + '\n```\n\n')
        
        f.write('## Data Preview (First 5 Rows)\n```text\n')
        if 'geometry' in gdf.columns:
            f.write(gdf.drop(columns=['geometry']).head(5).to_string() + '\n```\n\n')
        else:
            f.write(gdf.head(5).to_string() + '\n```\n\n')
            
        f.write('## Numeric Summary\n```text\n')
        f.write(gdf.describe().to_string() + '\n```\n')

if __name__ == '__main__':
    analyze()
