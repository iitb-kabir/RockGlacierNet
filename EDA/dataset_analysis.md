# Exploratory Data Analysis: Rock Glacier Dataset

## 1. Basic Dataset Information

| Property | Value | Description |
| :--- | :--- | :--- |
| **Total Features** | `626` | The total number of rock glacier polygons in the dataset. |
| **Geometry Type** | `Polygon` | The geometric shape format of the data. |
| **CRS** | `EPSG:4326` | Coordinate Reference System (Standard Latitude/Longitude, WGS 84). |
| **Bounding Box** | `[88.0747, 27.3212]` to `[88.9281, 28.0849]` | The geographic extent (Longitude / Latitude). This falls roughly in the Sikkim Himalayas region. |

---

## 2. Attribute Table Analysis

The shapefile contains **20 columns**, many of which are empty or artifacts from a previous format conversion (likely KML/KMZ to Shapefile). 

### Useful Columns
These columns contain data that may be relevant for identifying or tracking the source of the polygons:

| Column Name | Data Type | Description & Preview |
| :--- | :--- | :--- |
| `Name` | String | IDs or names (e.g., `"1"`, `"2"`, `"NULL01"`). |
| `id` | String | A secondary ID field present in some rows. |
| `layer` | String | Source layer name (e.g., `finl_sikm_shp`). |
| `path` | String | The original file path of the source data. |
| `geometry` | Geometry | The actual polygon boundary coordinates. **(Crucial for deep learning masks)** |

### Unused / Empty Columns
The following columns are mostly filled with `None`, `NaN`, or default values (`1`, `0`, `-1`) and can be safely ignored or dropped during the data preprocessing phase:

- **Empty Data**: `descriptio`, `timestamp`, `begin`, `end`, `altitudeMo`, `shp_05`, `descript_1`, `altitude_1`, `RG_AJ`, `lohanak`
- **KML Artifacts**: `tessellate`, `extrude`, `visibility`, `drawOrder`, `icon`

---

## 3. Data Preview (First 5 Rows)

Below is a cleaned snippet of the first 5 entries showing only the most relevant columns, instead of the messy raw pandas output:

| Index (Hidden ID) | Name | layer | path | id | geometry |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | 1 | *NaN* | *NaN* | *NaN* | POLYGON Z ((88.0865... |
| 1 | 2 | *NaN* | *NaN* | *NaN* | POLYGON Z ((88.0864... |
| 2 | 3 | *NaN* | *NaN* | *NaN* | POLYGON Z ((88.0863... |
| 3 | 3 | *NaN* | *NaN* | *NaN* | POLYGON Z ((88.0863... |
| 4 | 4 | *NaN* | *NaN* | *NaN* | POLYGON Z ((88.0862... |

---

## 4. Recommendations for Deep Learning Pipeline

1. **Geometry Rasterization**: For your deep learning models (U-Net, SegFormer), you only need the `geometry` column. These polygons will be rasterized to create binary masks where `1 = Rock Glacier` and `0 = Background`.
2. **Coordinate Projection**: Since the dataset is in degrees (`EPSG:4326`), ensure that your input rasters (Sentinel-2, DEM) match this CRS. If you need to extract patches of a specific size in meters (e.g., 256x256 pixels at 10m/pixel), you should project this shapefile to a local UTM zone (in meters) before patching.
3. **Data Cleaning**: You can completely drop the 15+ unused KML columns when loading this shapefile with `geopandas` to save memory and keep the pipeline clean.
