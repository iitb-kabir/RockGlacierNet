"""
GeoTIFF export helpers.

Every feature is exportable three ways:
  1. one multiband GeoTIFF for the full optical+thermal stack,
  2. one GeoTIFF per group  (optical, thermal),
  3. one GeoTIFF per individual band/index  ("all outputs" mode).

All exports go to Google Drive (folder set in config.DRIVE_FOLDER) as float32
GeoTIFFs at config.EXPORT_SCALE / config.EXPORT_CRS. Each call returns the
started ee.batch.Task so the caller can report task ids.
"""

import ee

from config import (
    EXPORT_SCALE,
    EXPORT_CRS,
    DRIVE_FOLDER,
    EXPORT_MAX_PIXELS,
    START_YEAR,
    END_YEAR,
)

_TAG = f"{START_YEAR}_{END_YEAR}"


def _export_image(image, geom, name):
    """Start a single GeoTIFF export task to Drive and return the task."""
    task = ee.batch.Export.image.toDrive(
        image=image.toFloat(),
        description=f"sikkim_{name}_{_TAG}",
        folder=DRIVE_FOLDER,
        fileNamePrefix=f"sikkim_{name}_{_TAG}",
        region=geom,
        scale=EXPORT_SCALE,
        crs=EXPORT_CRS,
        maxPixels=EXPORT_MAX_PIXELS,
        fileFormat="GeoTIFF",
    )
    task.start()
    return task


def export_multiband(image, geom, name):
    """Export the whole image as one multiband GeoTIFF."""
    return [_export_image(image, geom, name)]


def export_per_band(image, geom, prefix=""):
    """Export every band of `image` as its own single-band GeoTIFF.

    `prefix` (e.g. 'optical_' / 'thermal_') is prepended to each file name.
    Band names are resolved client-side so each task can be named distinctly.
    """
    band_names = image.bandNames().getInfo()
    tasks = []
    for band in band_names:
        single = image.select([band])
        tasks.append(_export_image(single, geom, f"{prefix}{band}"))
    return tasks


def export_all_outputs(optical, thermal, terrain, geom, mode="per_band"):
    """Kick off the requested set of exports.

    mode:
      'per_band' (default) - one GeoTIFF per individual output band,
      'grouped'            - one optical + thermal + terrain multiband GeoTIFF,
      'stack'              - one combined multiband GeoTIFF,
      'all'                - all of the above.
    """
    tasks = []
    stack = ee.Image.cat([optical, thermal, terrain]).clip(geom)

    if mode in ("per_band", "all"):
        tasks += export_per_band(optical, geom, prefix="optical_")
        tasks += export_per_band(thermal, geom, prefix="thermal_")
        tasks += export_per_band(terrain, geom, prefix="terrain_")

    if mode in ("grouped", "all"):
        tasks += export_multiband(optical, geom, "optical_stack")
        tasks += export_multiband(thermal, geom, "thermal_stack")
        tasks += export_multiband(terrain, geom, "terrain_stack")

    if mode in ("stack", "all"):
        tasks += export_multiband(stack, geom, "opt_thermal_terrain_stack")

    return tasks


def report(tasks):
    """Print a concise summary of started tasks."""
    print(f"\nStarted {len(tasks)} GeoTIFF export task(s) to Drive "
          f"folder '{DRIVE_FOLDER}':")
    for t in tasks:
        cfg = t.config
        desc = cfg.get("description", t.id)
        print(f"  - {desc}  (task id {t.id})")
    print("\nMonitor: https://code.earthengine.google.com/tasks  "
          "or  `earthengine task list`")
    print(f"Output: {EXPORT_SCALE} m, {EXPORT_CRS}, float32 GeoTIFF")
