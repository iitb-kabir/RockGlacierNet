import matplotlib
matplotlib.use('Agg')
import geopandas as gpd
import matplotlib.pyplot as plt
import os

SHAPEFILE_PATH = r'D:\RockGlacier_Project\raw_data\Final_merged01\Final_merged01.shp'
PLOT_PATH = r'D:\RockGlacier_Project\EDA\rock_glaciers_plot.png'

def plot_dataset():
    gdf = gpd.read_file(SHAPEFILE_PATH)
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(ax=ax, edgecolor='red', facecolor='cyan', linewidth=0.8, alpha=0.6)
    ax.set_title('Rock Glacier Polygons (Sikkim Region)', fontsize=15)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(PLOT_PATH, dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    plot_dataset()
