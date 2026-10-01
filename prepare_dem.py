#!/usr/bin/env python
"""
Crop the province-wide DEMs down to the study-site footprints (R3-3).

The DEMs in ``input/DEM`` are province-wide PALSAR tiles of 200-270 MB each. This
script clips each one to the extent of the matching image in ``input/01 DATA/IMAGE``,
reprojects it into that image's CRS, and writes the result to ``input/01 DATA/DEM``.

    python prepare_dem.py                 # all six sites, with slope/aspect rasters
    python prepare_dem.py --no-derivatives
    python prepare_dem.py --image "input/01 DATA/IMAGE/JKT_ROOF.tif"

Read the caveat this prints for every site: at 12.5 m the DEM resolves *terrain*, not
roof pitch. See ``solar_calculator/dem_slope.py`` for the full explanation.
"""

import argparse
import os
import sys

from solar_calculator.dem_slope import (
    DEFAULT_DEM_DIR,
    DEFAULT_DEM_OUT_DIR,
    crop_dem_for_directory,
    crop_dem_for_image,
    discover_province_dems,
    load_slope_aspect,
    terrain_summary,
)

DEFAULT_IMAGE_DIR = os.path.join("input", "01 DATA", "IMAGE")


def parse_args():
    p = argparse.ArgumentParser(
        description="Crop province DEMs to the study-site image extents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image", "-i", default=None,
                   help="single image to process; default is every GeoTIFF in --image-dir")
    p.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR, help="folder of site orthomosaics")
    p.add_argument("--dem-dir", default=DEFAULT_DEM_DIR, help="folder of province-wide DEMs")
    p.add_argument("--out-dir", "-o", default=DEFAULT_DEM_OUT_DIR, help="where cropped DEMs go")
    p.add_argument("--buffer-m", type=float, default=50.0,
                   help="margin around the image extent, so the 3x3 slope kernel has real neighbours")
    p.add_argument("--no-derivatives", dest="derivatives", action="store_false",
                   help="skip writing the slope and aspect rasters")
    p.add_argument("--stats", action="store_true", help="print terrain statistics for each output")
    p.set_defaults(derivatives=True)
    return p.parse_args()


def main():
    args = parse_args()

    dems = discover_province_dems(args.dem_dir)
    if not dems:
        print("ERROR: no DEM GeoTIFFs found in %s" % args.dem_dir)
        return 1
    print("Province DEMs available (%d):" % len(dems))
    for d in dems:
        print("   %-34s %s  %.1f m" % (d.name, d.crs, d.res_m))
    print()

    if args.image:
        results = [
            crop_dem_for_image(
                args.image,
                dem_dir=args.dem_dir,
                out_dir=args.out_dir,
                buffer_m=args.buffer_m,
                write_derivatives=args.derivatives,
                dems=dems,
            )
        ]
    else:
        results = crop_dem_for_directory(
            args.image_dir,
            dem_dir=args.dem_dir,
            out_dir=args.out_dir,
            buffer_m=args.buffer_m,
            write_derivatives=args.derivatives,
        )

    failures = 0
    for r in results:
        if "error" in r:
            failures += 1
            print("[FAIL] %-12s %s" % (r.get("site", "?"), r["error"]))
            continue
        print("[ OK ] %-12s %s" % (r["site"], os.path.basename(r["dem_path"])))
        print("         source     : %s (%s)" % (r["source_dem"], r["source_crs"]))
        print("         target CRS : %s%s" % (r["target_crs"], "  [reprojected]" if r["reprojected"] else ""))
        print("         native DEM cells over tile: %.0f x %.0f at %.1f m"
              % (r["native_cells_x"], r["native_cells_y"], r["native_res_m"]))
        print("         nodata     : %.1f %%" % (r["nodata_fraction"] * 100.0))
        if args.stats:
            summary = terrain_summary(load_slope_aspect(r["dem_path"]))
            if summary.get("available"):
                print("         slope      : min %.2f / median %.2f / max %.2f deg, relief %.1f m"
                      % (summary["slope_min_deg"], summary["slope_median_deg"],
                         summary["slope_max_deg"], summary["relief_m"]))
        print()

    print("Done: %d succeeded, %d failed." % (len(results) - failures, failures))
    print()
    print("CAVEAT: these rasters describe TERRAIN slope and aspect. At 12.5 m over a")
    print("200 m tile a single roof spans 1-2 cells, so they are not roof pitch.")
    print("Roof pitch is handled as a scenario grid -- see the Slope & Orientation tab.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
