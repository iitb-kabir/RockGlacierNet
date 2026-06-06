"""
Optical feature parameters from Sentinel-2 Surface Reflectance.

Builds a cloud-masked dry-season composite over Sikkim and derives the
spectral indices most useful for discriminating ice-rich / debris-covered
rock-glacier surfaces from vegetation, snow, water and bare talus.

Bands produced
--------------
Reflectance (scaled 0-1):  B2 B3 B4 B5 B6 B7 B8 B8A B11 B12
Indices:
    NDVI   vegetation                 (B8-B4)/(B8+B4)
    NDSI   snow                        (B3-B11)/(B3+B11)
    NDWI   open water (McFeeters)      (B3-B8)/(B3+B8)
    NDMI   moisture / debris ice       (B8-B11)/(B8+B11)
    BSI    bare soil / debris          ((B11+B4)-(B8+B2))/((B11+B4)+(B8+B2))
    ALBEDO broadband shortwave albedo  (Liang 2001 narrow-to-broadband)
"""

import ee

from config import (
    S2_COLLECTION,
    S2_CLOUD_PROB,
    S2_CLOUD_PROB_MAX,
    S2_MAX_SCENE_CLOUD,
    OPTICAL_MONTHS,
    START_YEAR,
    END_YEAR,
)

# Reflectance bands kept in the output (native S2 scaling is x10000).
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]


def _mask_s2_clouds(img):
    """Mask clouds using the S2 cloud-probability product joined per scene."""
    cloud_prob = ee.Image(img.get("cloud_mask")).select("probability")
    is_clear = cloud_prob.lt(S2_CLOUD_PROB_MAX)
    # Also drop cirrus / opaque clouds flagged in the SCL scene-class layer.
    scl = img.select("SCL")
    scl_mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
    return img.updateMask(is_clear).updateMask(scl_mask)


def _scale_reflectance(img):
    """Scale S2 SR DN to 0-1 reflectance, preserving metadata."""
    optical = img.select(S2_BANDS).multiply(0.0001)
    return img.addBands(optical, overwrite=True)


def build_s2_composite(geom, start_year=START_YEAR, end_year=END_YEAR):
    """Median dry-season Sentinel-2 composite, cloud-masked, scaled to 0-1."""
    month_filter = ee.Filter.inList("month", OPTICAL_MONTHS)

    s2 = (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(geom)
        .filterDate(f"{start_year}-01-01", f"{end_year}-12-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", S2_MAX_SCENE_CLOUD))
        .map(lambda i: i.set("month", ee.Date(i.get("system:time_start")).get("month")))
        .filter(month_filter)
    )

    clouds = (
        ee.ImageCollection(S2_CLOUD_PROB)
        .filterBounds(geom)
        .filterDate(f"{start_year}-01-01", f"{end_year}-12-31")
    )

    joined = ee.Join.saveFirst("cloud_mask").apply(
        primary=s2,
        secondary=clouds,
        condition=ee.Filter.equals(leftField="system:index", rightField="system:index"),
    )

    composite = (
        ee.ImageCollection(joined)
        .map(_mask_s2_clouds)
        .map(_scale_reflectance)
        .select(S2_BANDS)
        .median()
        .clip(geom)
    )
    return composite


def add_optical_indices(img):
    """Append spectral indices to a scaled (0-1) S2 composite."""
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndsi = img.normalizedDifference(["B3", "B11"]).rename("NDSI")
    ndwi = img.normalizedDifference(["B3", "B8"]).rename("NDWI")
    ndmi = img.normalizedDifference(["B8", "B11"]).rename("NDMI")

    bsi = img.expression(
        "((SWIR + RED) - (NIR + BLUE)) / ((SWIR + RED) + (NIR + BLUE))",
        {
            "SWIR": img.select("B11"),
            "RED": img.select("B4"),
            "NIR": img.select("B8"),
            "BLUE": img.select("B2"),
        },
    ).rename("BSI")

    # Liang (2001) Sentinel-2 narrow-to-broadband shortwave albedo.
    albedo = img.expression(
        "0.356*B2 + 0.130*B4 + 0.373*B8 + 0.085*B11 + 0.072*B12 - 0.0018",
        {
            "B2": img.select("B2"),
            "B4": img.select("B4"),
            "B8": img.select("B8"),
            "B11": img.select("B11"),
            "B12": img.select("B12"),
        },
    ).rename("ALBEDO")

    return img.addBands([ndvi, ndsi, ndwi, ndmi, bsi, albedo])


def optical_features(geom, start_year=START_YEAR, end_year=END_YEAR):
    """Full optical feature stack (reflectance bands + indices), clipped."""
    composite = build_s2_composite(geom, start_year, end_year)
    return add_optical_indices(composite).clip(geom)


if __name__ == "__main__":
    from auth_setup import init_ee
    from roi import get_sikkim

    init_ee()
    geom, _ = get_sikkim()
    feats = optical_features(geom)
    print("Optical feature bands:")
    print(feats.bandNames().getInfo())
