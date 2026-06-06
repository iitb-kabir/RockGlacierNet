"""
Terrain feature parameters from multiple Digital Elevation Models.

Loads four free global 30 m DEMs (Copernicus GLO-30, SRTM, NASADEM, ALOS
AW3D30), keeps each elevation band, computes their differences (DEM
disagreement is itself a useful feature / uncertainty proxy), and derives the
standard geomorphometric terrain variables on the primary DEM (Copernicus).

Bands produced
--------------
Elevation:
    ELEV_COP, ELEV_SRTM, ELEV_NASA, ELEV_ALOS
DEM differences (vs Copernicus):
    DIFF_SRTM_COP, DIFF_NASA_COP, DIFF_ALOS_COP
Derivatives (on Copernicus):
    SLOPE        degrees
    ASPECT       degrees (0-360, clockwise from north)
    HILLSHADE    0-255 (az 315, elev 45)
    CURVATURE    Laplacian of elevation (convex +, concave -)
    TRI          Terrain Ruggedness Index (mean |center-neighbour|, 3x3)
    ROUGHNESS    max-min elevation in 3x3 window (m)
    TPI          Topographic Position Index (elev - focal mean, ~300 m)
    REL_RELIEF   relative relief (focal max-min, ~500 m)
"""

import ee

from config import (
    DEM_COP,
    DEM_SRTM,
    DEM_NASADEM,
    DEM_ALOS,
    TPI_RADIUS_M,
    RELIEF_RADIUS_M,
)


def _mosaic_with_projection(coll_id, band):
    """Mosaic an ImageCollection but keep a valid 30 m projection.

    ImageCollection.mosaic() resets the result to the default 1-degree
    geographic projection, which makes ee.Terrain.slope/aspect return masked
    output. Restore the native tile projection so terrain ops work.
    """
    coll = ee.ImageCollection(coll_id).select(band)
    proj = coll.first().projection()
    return coll.mosaic().setDefaultProjection(proj)


def get_dems(geom):
    """Return individual elevation images (renamed, clipped) for all 4 DEMs."""
    cop = _mosaic_with_projection(DEM_COP, "DEM").rename("ELEV_COP")
    srtm = ee.Image(DEM_SRTM).select("elevation").rename("ELEV_SRTM")
    nasa = ee.Image(DEM_NASADEM).select("elevation").rename("ELEV_NASA")
    alos = _mosaic_with_projection(DEM_ALOS, "DSM").rename("ELEV_ALOS")
    return {
        "cop": cop.clip(geom),
        "srtm": srtm.clip(geom),
        "nasa": nasa.clip(geom),
        "alos": alos.clip(geom),
    }


def _derivatives(dem):
    """Geomorphometric derivatives computed on a single elevation image."""
    terrain = ee.Terrain.products(dem.rename("elevation"))
    slope = terrain.select("slope").rename("SLOPE")
    aspect = terrain.select("aspect").rename("ASPECT")
    hillshade = terrain.select("hillshade").rename("HILLSHADE")

    # Curvature: Laplacian (second derivative) of elevation.
    laplacian = ee.Kernel.fixed(
        3, 3,
        [[0, 1, 0],
         [1, -4, 1],
         [0, 1, 0]],
    )
    curvature = dem.convolve(laplacian).rename("CURVATURE")

    # 3x3 pixel neighbourhood stats.
    px = ee.Kernel.square(radius=1, units="pixels")
    neigh = dem.neighborhoodToBands(px)              # 9 bands (incl. center)
    tri = (
        neigh.subtract(dem)
        .abs()
        .reduce(ee.Reducer.sum())
        .divide(8)
        .rename("TRI")
    )
    roughness = (
        dem.reduceNeighborhood(ee.Reducer.max(), px)
        .subtract(dem.reduceNeighborhood(ee.Reducer.min(), px))
        .rename("ROUGHNESS")
    )

    # Metre-based windows for TPI and relative relief.
    tpi_k = ee.Kernel.circle(radius=TPI_RADIUS_M, units="meters")
    tpi = dem.subtract(
        dem.reduceNeighborhood(ee.Reducer.mean(), tpi_k)
    ).rename("TPI")

    relief_k = ee.Kernel.circle(radius=RELIEF_RADIUS_M, units="meters")
    rel_relief = (
        dem.reduceNeighborhood(ee.Reducer.max(), relief_k)
        .subtract(dem.reduceNeighborhood(ee.Reducer.min(), relief_k))
        .rename("REL_RELIEF")
    )

    return ee.Image.cat([
        slope, aspect, hillshade, curvature,
        tri, roughness, tpi, rel_relief,
    ])


def terrain_features(geom):
    """Full multi-DEM terrain feature stack, clipped to geom."""
    dems = get_dems(geom)
    cop = dems["cop"]

    # Elevation differences vs Copernicus (resampled to its grid implicitly).
    diff_srtm = dems["srtm"].subtract(cop).rename("DIFF_SRTM_COP")
    diff_nasa = dems["nasa"].subtract(cop).rename("DIFF_NASA_COP")
    diff_alos = dems["alos"].subtract(cop).rename("DIFF_ALOS_COP")

    derivatives = _derivatives(cop)

    return ee.Image.cat([
        dems["cop"], dems["srtm"], dems["nasa"], dems["alos"],
        diff_srtm, diff_nasa, diff_alos,
        derivatives,
    ]).clip(geom)


if __name__ == "__main__":
    from auth_setup import init_ee
    from roi import get_sikkim

    init_ee()
    geom, _ = get_sikkim()
    feats = terrain_features(geom)
    print("Terrain feature bands:")
    print(feats.bandNames().getInfo())
