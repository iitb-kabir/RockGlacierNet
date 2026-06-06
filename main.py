"""
Build and (optionally) export the Sikkim optical + thermal feature stack.

Run:
    python main.py                       # print band list + quick stats
    python main.py --export              # export ONE GeoTIFF per output band
    python main.py --export --mode all   # per-band + grouped + full stack
    python main.py --export --mode grouped   # optical + thermal multiband
    python main.py --export --mode stack     # single combined multiband
"""

import argparse

import ee

from auth_setup import init_ee
from roi import get_sikkim
from optical_params import optical_features
from thermal_params import thermal_features
from terrain_params import terrain_features
from exports import export_all_outputs, report


def quick_stats(stack, geom):
    """Print min/mean/max of a few diagnostic bands over the ROI."""
    diag = ["NDVI", "NDSI", "ALBEDO", "LST_summer", "LST_anomaly",
            "ELEV_COP", "SLOPE", "TPI", "REL_RELIEF"]
    stats = stack.select(diag).reduceRegion(
        reducer=ee.Reducer.minMax().combine(ee.Reducer.mean(), sharedInputs=True),
        geometry=geom,
        scale=500,            # coarse for a fast sanity check
        maxPixels=1e12,
        bestEffort=True,
        tileScale=4,          # split work to stay under the memory limit
    ).getInfo()
    print("\nQuick ROI statistics (scale=500 m, sanity check):")
    for b in diag:
        lo = stats.get(f"{b}_min")
        me = stats.get(f"{b}_mean")
        hi = stats.get(f"{b}_max")
        def fmt(v):
            return f"{v:8.3f}" if isinstance(v, (int, float)) else f"{str(v):>8}"
        print(f"  {b:12s} min={fmt(lo)} mean={fmt(me)} max={fmt(hi)}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--export", action="store_true",
                        help="Start GeoTIFF export task(s) to Google Drive.")
    parser.add_argument("--mode", default="per_band",
                        choices=["per_band", "grouped", "stack", "all"],
                        help="Export layout (default: per_band = one TIFF per output).")
    args = parser.parse_args()

    init_ee()
    geom, _ = get_sikkim()

    optical = optical_features(geom)
    thermal = thermal_features(geom)
    terrain = terrain_features(geom)
    stack = ee.Image.cat([optical, thermal, terrain]).clip(geom)

    print("Sikkim optical + thermal + terrain feature stack")
    print("Bands:", stack.bandNames().getInfo())

    quick_stats(stack, geom)

    if args.export:
        tasks = export_all_outputs(optical, thermal, terrain, geom, mode=args.mode)
        report(tasks)


if __name__ == "__main__":
    main()
