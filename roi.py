"""
Region of interest: the State of Sikkim.

Pulls the administrative boundary from FAO GAUL; falls back to a bounding box
if the named feature cannot be found.
"""

import ee

from config import (
    GAUL_DATASET,
    ADMIN_NAME_FIELD,
    SIKKIM_NAME,
    SIKKIM_BBOX,
)


def get_sikkim():
    """Return (geometry, feature_collection) for Sikkim state."""
    gaul = ee.FeatureCollection(GAUL_DATASET)
    sikkim = gaul.filter(ee.Filter.eq(ADMIN_NAME_FIELD, SIKKIM_NAME))
    geom = ee.Algorithms.If(
        sikkim.size().gt(0),
        sikkim.geometry(),
        ee.Geometry.Rectangle(SIKKIM_BBOX),
    )
    return ee.Geometry(geom), sikkim


if __name__ == "__main__":
    from auth_setup import init_ee

    init_ee()
    geom, fc = get_sikkim()
    area_km2 = geom.area(maxError=100).divide(1e6).getInfo()
    print(f"Sikkim ROI area ~ {area_km2:,.0f} km^2 "
          f"(reference: ~7,096 km^2)")
