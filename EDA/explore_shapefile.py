# %%
import geopandas as gpd
import matplotlib.pyplot as plt
import os

# %%
# Define paths
SHAPEFILE_PATH = r"D:\RockGlacier_Project\raw_data\Final_merged01\Final_merged01.shp"
OUTPUT_DIR = r"D:\RockGlacier_Project\EDA"

# %%
print(f"--- Exploring Shapefile: {SHAPEFILE_PATH} ---")

# Check if file exists
if not os.path.exists(SHAPEFILE_PATH):
    print(f"Error: Shapefile not found at {SHAPEFILE_PATH}")
else:
    # Load shapefile
    print("Loading data...")
    gdf = gpd.read_file(SHAPEFILE_PATH)

# %%
# 1. Basic Information
if 'gdf' in locals():
    print("\n[1] Basic Information:")
    print(f"Total number of features (polygons): {len(gdf)}")
    print(f"Coordinate Reference System (CRS): {gdf.crs}")
    
# %%
# 2. Data Attributes
if 'gdf' in locals():
    print("\n[2] Data Attributes (Columns):")
    print(gdf.columns.tolist())
    
    print("\n[3] First 5 Rows:")
    print(gdf.head())

# %%
# 3. Basic Statistics for numeric columns
if 'gdf' in locals():
    print("\n[4] Basic Statistics:")
    print(gdf.describe())
    
# %%
# 4. Plotting the Shapefile
if 'gdf' in locals():
    print("\n[5] Generating Plot...")
    fig, ax = plt.subplots(figsize=(10, 10))
    # Plotting the geometries
    gdf.plot(ax=ax, edgecolor='blue', facecolor='lightblue', linewidth=0.5, alpha=0.7)
    
    ax.set_title("Rock Glacier Polygons - Final_merged01", fontsize=15)
    ax.set_xlabel("Easting / Longitude")
    ax.set_ylabel("Northing / Latitude")
    
    # Save the plot
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_path = os.path.join(OUTPUT_DIR, "rock_glaciers_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to: {plot_path}")
    
    # Show the plot
    plt.show()
