"""
Thermal feature parameters from Landsat 8/9 Collection-2 Level-2.

Land Surface Temperature (LST) is taken directly from the C2 L2 ST_B10 band
(already atmospherically corrected and emissivity-adjusted), converted to
degrees Celsius. Ice-rich rock glaciers tend to stay cooler than the
surrounding debris-covered / talus slopes, so summer LST and a local thermal
anomaly are the most diagnostic.

Bands produced
--------------
    LST_summer    mean warm-season LST (degC)
    LST_winter    mean cold-season LST (degC)
    LST_annual    mean annual LST (degC)
    LST_anomaly   LST_summer minus its local neighbourhood mean (degC)
"""

import ee

from config import (
    L8_COLLECTION,
    L9_COLLECTION,
    LANDSAT_MAX_CLOUD,
    SUMMER_MONTHS,
    WINTER_MONTHS,
    START_YEAR,
    END_YEAR,
)


def _mask_landsat_l2(img):
    """Apply the QA_PIXEL bit mask (cloud, shadow, cirrus, snow-dilated)."""
    qa = img.select("QA_PIXEL")
    # Bits: 1 dilated cloud, 2 cirrus, 3 cloud, 4 cloud shadow.
    mask = (
        qa.bitwiseAnd(1 << 1).eq(0)
        .And(qa.bitwiseAnd(1 << 2).eq(0))
        .And(qa.bitwiseAnd(1 << 3).eq(0))
        .And(qa.bitwiseAnd(1 << 4).eq(0))
    )
    return img.updateMask(mask)


def _to_lst_celsius(img):
    """Convert C2 L2 ST_B10 (scaled Kelvin) to degrees Celsius."""
    # Scale factors per USGS C2 L2: ST = DN*0.00341802 + 149.0  (Kelvin)
    lst_k = img.select("ST_B10").multiply(0.00341802).add(149.0)
    lst_c = lst_k.subtract(273.15).rename("LST")
    return img.addBands(lst_c)


def _landsat_collection(geom, start_year, end_year):
    """Merged, masked, LST-converted Landsat 8 + 9 collection."""
    def prep(coll_id):
        return (
            ee.ImageCollection(coll_id)
            .filterBounds(geom)
            .filterDate(f"{start_year}-01-01", f"{end_year}-12-31")
            .filter(ee.Filter.lt("CLOUD_COVER", LANDSAT_MAX_CLOUD))
            .map(_mask_landsat_l2)
            .map(_to_lst_celsius)
            .map(lambda i: i.set(
                "month", ee.Date(i.get("system:time_start")).get("month")))
        )

    return prep(L8_COLLECTION).merge(prep(L9_COLLECTION))


def _seasonal_mean(coll, months, name):
    season = coll.filter(ee.Filter.inList("month", months))
    return season.select("LST").mean().rename(name)


def thermal_features(geom, start_year=START_YEAR, end_year=END_YEAR):
    """Full thermal feature stack, clipped to geom."""
    coll = _landsat_collection(geom, start_year, end_year)

    lst_summer = _seasonal_mean(coll, SUMMER_MONTHS, "LST_summer")
    lst_winter = _seasonal_mean(coll, WINTER_MONTHS, "LST_winter")
    lst_annual = coll.select("LST").mean().rename("LST_annual")

    # Local thermal anomaly: summer LST minus a 1 km neighbourhood mean.
    neighbourhood_mean = lst_summer.reduceNeighborhood(
        reducer=ee.Reducer.mean(),
        kernel=ee.Kernel.circle(radius=1000, units="meters"),
    )
    lst_anomaly = lst_summer.subtract(neighbourhood_mean).rename("LST_anomaly")

    return (
        ee.Image.cat([lst_summer, lst_winter, lst_annual, lst_anomaly])
        .clip(geom)
    )


if __name__ == "__main__":
    from auth_setup import init_ee
    from roi import get_sikkim

    init_ee()
    geom, _ = get_sikkim()
    feats = thermal_features(geom)
    print("Thermal feature bands:")
    print(feats.bandNames().getInfo())
