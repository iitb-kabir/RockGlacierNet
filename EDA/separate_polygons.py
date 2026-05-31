import matplotlib
matplotlib.use('Agg')
import geopandas as gpd
import matplotlib.pyplot as plt
import os

SHAPEFILE_PATH = r"D:\RockGlacier_Project\raw_data\Final_merged01\Final_merged01.shp"
OUTPUT_DIR = r"D:\RockGlacier_Project\EDA\individual_polygons"
PLOTS_DIR = r"D:\RockGlacier_Project\EDA\individual_plots"

def separate_and_plot():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    print("Loading shapefile...")
    gdf = gpd.read_file(SHAPEFILE_PATH)
    
    total = len(gdf)
    print(f"Separating {total} polygons...")
    
    for idx, row in gdf.iterrows():
        # Create a single-row GeoDataFrame for this specific polygon
        single_gdf = gpd.GeoDataFrame([row], crs=gdf.crs)
        
        # 1. Save Individual Polygon Data
        # We save as GeoJSON because saving 626 Shapefiles would create over 2,500 files!
        out_json = os.path.join(OUTPUT_DIR, f"polygon_{idx}.geojson")
        single_gdf.to_file(out_json, driver="GeoJSON")
        
        # 2. Plot "with fill" for the first 5 polygons
        if idx < 5:
            fig, ax = plt.subplots(figsize=(6, 6))
            # facecolor='cyan' is the "fill"
            single_gdf.plot(ax=ax, edgecolor='red', facecolor='cyan', linewidth=2, alpha=0.7)
            ax.set_title(f"Rock Glacier Index: {idx}")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.grid(True, linestyle='--', alpha=0.5)
            
            plot_path = os.path.join(PLOTS_DIR, f"plot_{idx}.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig) # close to free up memory
            
    print(f"Done! Saved {total} GeoJSON files to {OUTPUT_DIR}")
    print(f"Saved first 5 filled plots to {PLOTS_DIR}")

if __name__ == "__main__":
    separate_and_plot()
