#!/usr/bin/env python
"""
Batch SAM3 inference over the six study sites, both classes (roof and PV).

Produces the predicted polygons that R1-1 (recall recovery) and R1-3 (inner
buffering) both need, and that are currently missing from the repo:
`input/01 DATA/Results/SHP/` turned out to be a copy of the ground truth, and the
`output*/` folders belong to other images.

    python run_batch_inference.py                    # all 6 sites, both classes
    python run_batch_inference.py --classes roof     # roofs only
    python run_batch_inference.py --site JKT_ROOF    # one site
    python run_batch_inference.py --overwrite        # redo finished outputs

For every site and class it writes, into `output_inference/`:

    <SITE>_<class>.geojson         regularized polygons, georeferenced
    <SITE>_<class>_raw.geojson     polygons before regularization
    <SITE>_<class>_mask.tif        binary mask as a georeferenced GeoTIFF
    manifest.json                  per-run timing, object count, coverage

The raw GeoJSON and the mask both matter downstream. R1-3 buffers the raw
polygons, because the orthogonalization in the regularizer would otherwise be
baked into the usable-area measurement. R1-1 works on the mask, because canopy
recovery happens at pixel level before vectorization.

Reproducibility note
--------------------
This reuses `sam3rooffix.py` unchanged so the baseline matches what the paper
reports. Be aware that `--confidence` there is effectively a no-op: the script
sets `config.conf_threshold_large/medium/small`, which `AdvancedConfig` does not
define, so the dataclass defaults (0.20 / 0.22 / 0.28) are what actually run.
`--apply-confidence` below wires the value through to the real field instead, but
leave it off for the baseline or the numbers will not match the manuscript.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# sam3rooffix.py prints arrows and box characters. On a console using a legacy
# code page (cp1252 on Windows) those raise UnicodeEncodeError mid-run, which
# would abort every inference after the checkpoint has already loaded. Force a
# tolerant encoding before importing it.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np

import sam3rooffix as sr
from sam3_advanced.config import AdvancedConfig
from sam3_advanced.inference import segment_multiscale_advanced
from sam3_advanced.postprocess import post_process_mask
from sam3_advanced.utils import clear_memory

DEFAULT_IMAGE_DIR = os.path.join("input", "01 DATA", "IMAGE")
DEFAULT_OUT_DIR = "output_inference"

# Two prompt sets, because the two entry points in this repo do not agree.
#
#   "dashboard" -- what app_solar_calculator.py uses. THIS is what produced the
#                  figures in the manuscript (34 roofs / 17,646.76 m2 / 25 PV /
#                  443.04 m2), so it is the default.
#   "rooffix"   -- what sam3rooffix.py uses: warehouse/industrial-leaning roof
#                  prompts aimed at a different kind of scene. Kept for
#                  comparison; it will NOT reproduce the manuscript numbers.
DASHBOARD_PROMPTS = {
    "roof": {
        "large": ["roof", "large roof", "big roof", "building roof", "rooftop"],
        "medium": ["roof", "medium roof", "rooftop", "house roof"],
        "small": ["roof", "small roof", "house roof", "building top"],
    },
    "pv": {
        "large": ["solar panel", "large solar panel", "photovoltaic array"],
        "medium": ["solar panel", "medium solar panel", "photovoltaic panel", "PV panel"],
        "small": ["solar panel", "small solar panel", "PV panel", "solar cell"],
    },
}

ROOFFIX_PROMPTS = {
    "roof": sr.ROOF_PROMPTS,
    "pv": DASHBOARD_PROMPTS["pv"],  # sam3rooffix has no PV prompts of its own
}

PROMPT_SETS = {"dashboard": DASHBOARD_PROMPTS, "rooffix": ROOFFIX_PROMPTS}

# PV panels are much smaller than roofs, so a roof-sized minimum would drop them.
MIN_AREA_PX = {"roof": 300, "pv": 60}

# Vectorization filters. The dashboard applies NO compactness filter and exports
# shapefiles at min_area_m2 = 1.0, and its reported gross area is simply the raw
# mask area. R1-3 has to start from that same gross area, so the defaults here
# stay permissive; sam3rooffix's stricter 4.0 m2 / 0.15 compactness would silently
# shrink the baseline.
MIN_AREA_M2 = {"roof": 1.0, "pv": 0.25}
MIN_COMPACTNESS = {"roof": 0.0, "pv": 0.0}


def detect(image_pil, processor, cls, confidence, min_area_px, apply_confidence,
           prompt_set="dashboard"):
    """Run multiscale segmentation for one concept class."""
    config = AdvancedConfig()

    # Kept for parity with sam3rooffix.py. These attribute names do not exist on
    # AdvancedConfig, so assigning them creates dead attributes that nothing reads
    # -- which is exactly the behaviour the published results were produced under.
    config.conf_threshold_large = confidence
    config.conf_threshold_medium = max(0.35, confidence - 0.15)
    config.conf_threshold_small = max(0.30, confidence - 0.20)
    config.min_area_pixels = min_area_px

    if apply_confidence:
        # Opt-in: write to the fields inference.py actually reads.
        config.confidence_thresholds = {
            "level1": confidence,
            "level2": max(0.35, confidence - 0.15),
            "level3": max(0.30, confidence - 0.20),
        }

    p = PROMPT_SETS[prompt_set][cls]
    config.prompts_large = p["large"]
    config.prompts_medium = p["medium"]
    config.prompts_small = p["small"]

    processor.set_image(image_pil)
    final_mask, stats = segment_multiscale_advanced(image_pil, processor, config)
    mask = post_process_mask(final_mask, config, verbose=False)
    # Both are returned: post_process_mask runs a geometry filter that discards
    # low-compactness blobs, which is exactly the shape a canopy-occluded roof
    # takes. R1-1 needs to see what that filter removed, so keep the raw mask.
    return mask, final_mask, stats


def save_mask_geotiff(mask, handler, path):
    """Write the binary mask as a georeferenced GeoTIFF so R1-1 can work on pixels."""
    try:
        import rasterio
    except ImportError:
        print("      [WARN] rasterio missing, mask GeoTIFF skipped")
        return None
    if handler is None or not getattr(handler, "is_geotiff", False):
        print("      [WARN] no georeferencing, mask GeoTIFF skipped")
        return None

    profile = {
        "driver": "GTiff",
        "height": mask.shape[0],
        "width": mask.shape[1],
        "count": 1,
        "dtype": "uint8",
        "crs": handler.crs,
        "transform": handler.transform,
        "nodata": 0,
        "compress": "deflate",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write((mask > 0).astype("uint8") * 255, 1)
        dst.set_band_description(1, "binary_mask")
    return path


def run_one(image_path, cls, processor, args, manifest):
    site = os.path.splitext(os.path.basename(image_path))[0]
    key = "%s_%s" % (site, cls)
    out_dir = Path(args.out_dir)
    reg_path = out_dir / ("%s.geojson" % key)

    if reg_path.exists() and not args.overwrite:
        print("  [SKIP] %s already done" % key)
        return

    print("\n  [RUN ] %s" % key)
    t0 = time.time()
    image_pil, handler = sr.load_image(image_path)
    t_load = time.time() - t0

    t0 = time.time()
    mask, mask_prepost, stats = detect(
        image_pil, processor, cls, args.confidence,
        MIN_AREA_PX[cls], args.apply_confidence, args.prompts,
    )
    t_infer = time.time() - t0

    n_obj = sr._count_objects(mask)
    coverage = float((mask > 0).mean() * 100.0)

    # Exactly how app_solar_calculator.py counts and measures, so these numbers are
    # directly comparable with the manuscript rather than approximately so.
    #
    # The area is computed inline rather than via calculate_area_from_mask(): that
    # helper calls handler.calculate_area_m2(), which exists on
    # sam3_advanced.geotiff_utils.GeoTIFFHandler but NOT on the separate
    # GeoTIFFHandler that sam3rooffix.py defines and that sr.load_image() returns.
    # The arithmetic below is identical to that method.
    from solar_calculator.utils import count_objects_in_mask
    n_obj_dash = count_objects_in_mask(mask)
    px_count = int((mask > 0).sum())
    if handler is not None and getattr(handler, "is_geotiff", False):
        tr = handler.transform
        area_dash = px_count * abs(tr.a) * abs(tr.e)
    else:
        area_dash = px_count * (args.pixel_ratio ** 2)
    print("      %d object(s) [dashboard count %d], %.1f m2, %.2f%% coverage, %.1f s"
          % (n_obj, n_obj_dash, area_dash, coverage, t_infer))

    px2m = handler.get_pixel_to_meter_ratio() if handler else args.pixel_ratio

    t0 = time.time()
    raw = sr.mask_to_geojson(
        mask, handler=handler, pixel_to_meter=px2m,
        simplify_tol=args.simplify, min_area_m2=MIN_AREA_M2[cls],
        min_compactness=MIN_COMPACTNESS[cls],
    )
    sr.save_geojson(raw, str(out_dir / ("%s_raw.geojson" % key)))

    reg = raw if args.no_regularize else sr.regularize_geojson(raw, handler=handler)
    sr.save_geojson(reg, str(reg_path))
    t_vec = time.time() - t0

    mask_path = save_mask_geotiff(mask, handler, str(out_dir / ("%s_mask.tif" % key)))
    prepost_path = save_mask_geotiff(
        mask_prepost, handler, str(out_dir / ("%s_mask_prepost.tif" % key)))
    dropped = float((mask_prepost > 0).mean() - (mask > 0).mean()) * 100.0

    manifest[key] = {
        "site": site,
        "class": cls,
        "image": image_path,
        "crs": str(handler.crs) if handler else None,
        "gsd_m": round(px2m, 5) if px2m else None,
        "n_objects": int(n_obj),
        "n_objects_dashboard": int(n_obj_dash),
        "area_m2_dashboard": round(float(area_dash), 2),
        "coverage_pct": round(coverage, 3),
        "n_polygons_raw": len(raw.get("features", [])),
        "n_polygons_regularized": len(reg.get("features", [])),
        "seconds_load": round(t_load, 2),
        "seconds_inference": round(t_infer, 2),
        "seconds_vectorize": round(t_vec, 2),
        "seconds_per_object": round(t_infer / n_obj, 3) if n_obj else None,
        "level_stats": stats,
        "mask_tif": mask_path,
        "mask_prepost_tif": prepost_path,
        "coverage_dropped_by_postprocess_pct": round(dropped, 3),
        "geojson": str(reg_path),
        "geojson_raw": str(out_dir / ("%s_raw.geojson" % key)),
        "confidence_applied": bool(args.apply_confidence),
    }

    clear_memory()


def parse_args():
    ap = argparse.ArgumentParser(
        description="Batch SAM3 inference over the study sites, roof and PV classes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    ap.add_argument("--out-dir", "-o", default=DEFAULT_OUT_DIR)
    ap.add_argument("--checkpoint", "-c", default="sam3.pt")
    ap.add_argument("--site", default=None, help="run one site only, e.g. JKT_ROOF")
    ap.add_argument("--classes", default="roof,pv", help="comma-separated: roof, pv")
    ap.add_argument("--prompts", default="dashboard", choices=sorted(PROMPT_SETS),
                    help="dashboard reproduces the manuscript; rooffix uses "
                         "sam3rooffix.py's warehouse-leaning roof prompts")
    ap.add_argument("--confidence", type=float, default=0.5)
    ap.add_argument("--apply-confidence", action="store_true",
                    help="actually wire --confidence into the model (changes results; "
                         "leave off to reproduce the manuscript baseline)")
    ap.add_argument("--simplify", type=float, default=0.5)
    ap.add_argument("--pixel-ratio", type=float, default=0.5,
                    help="fallback m/pixel when an image has no georeferencing")
    ap.add_argument("--no-regularize", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    classes = [c.strip().lower() for c in args.classes.split(",") if c.strip()]
    for c in classes:
        if c not in DASHBOARD_PROMPTS:
            print("[ERROR] unknown class %r; choose from %s" % (c, list(DASHBOARD_PROMPTS)))
            return 1

    images = sorted(Path(args.image_dir).glob("*.tif")) + sorted(Path(args.image_dir).glob("*.tiff"))
    if args.site:
        images = [p for p in images if p.stem == args.site]
    if not images:
        print("[ERROR] no images found in %s" % args.image_dir)
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

    print("=" * 70)
    print("Batch SAM3 inference")
    print("  images   : %d  (%s)" % (len(images), ", ".join(p.stem for p in images)))
    print("  classes  : %s" % ", ".join(classes))
    print("  prompts  : %s%s" % (args.prompts,
          "  (reproduces the manuscript)" if args.prompts == "dashboard" else
          "  (WILL NOT match the manuscript)"))
    print("  output   : %s" % out_dir)
    print("  runs     : %d" % (len(images) * len(classes)))
    if args.apply_confidence:
        print("  WARNING  : --apply-confidence is ON, results will differ from the manuscript")
    print("=" * 70)

    # sam3rooffix.load_model() returns the processor alone, unlike the
    # dashboard's load_sam3_model() which returns (model, processor).
    processor = sr.load_model(args.checkpoint)

    t_all = time.time()
    failed = []
    for image_path in images:
        print("\n[SITE] %s" % image_path.stem)
        for cls in classes:
            try:
                run_one(str(image_path), cls, processor, args, manifest)
            except Exception as exc:
                failed.append("%s_%s: %s" % (image_path.stem, cls, exc))
                print("      [FAIL] %s" % exc)
            finally:
                # Written after every run so a crash does not lose earlier work.
                manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    # Which site reproduces the manuscript? The published techno-economic figures
    # trace to a dashboard run that was never saved, so identify it here by
    # matching the reported roof/PV counts and areas.
    print("\n" + "=" * 70)
    print("MATCH AGAINST THE PUBLISHED SITE")
    print("  target: 34 roofs / 17,646.76 m2 roof / 25 PV / 443.04 m2 PV")
    print("-" * 70)
    print("%-10s %6s %11s %5s %10s  %8s  %s"
          % ("site", "roofs", "roof m2", "PV", "PV m2", "area err", ""))
    for site in sorted({v.get("site") for v in manifest.values() if v.get("site")}):
        r = manifest.get("%s_roof" % site, {})
        v = manifest.get("%s_pv" % site, {})
        ar = r.get("area_m2_dashboard")
        if ar is None:
            continue
        nr, nv, av = r.get("n_objects_dashboard"), v.get("n_objects_dashboard"), v.get("area_m2_dashboard")
        # Area discriminates better than counts, which shift as blobs merge.
        err = abs(ar - 17646.76) / 17646.76 * 100.0
        tag = "<<< MATCH" if err < 2 else ("close" if err < 10 else "")
        print("%-10s %6s %11.1f %5s %10s  %7.1f%%  %s"
              % (site, nr if nr is not None else "-", ar,
                 nv if nv is not None else "-",
                 ("%.1f" % av) if av is not None else "-", err, tag))
    print()
    print("No match means the published run used different settings or an earlier")
    print("build. Say that in the response letter rather than presenting R1-3 as a")
    print("delta against a baseline that cannot be regenerated.")
    print("\n" + "=" * 70)
    print("Done in %.1f min. %d succeeded, %d failed." % (
        (time.time() - t_all) / 60.0, len(manifest), len(failed)))
    for f in failed:
        print("  FAILED  %s" % f)
    print("Manifest: %s" % manifest_path)
    print()
    print("Next: python evaluate_inference.py   (accuracy vs the ground-truth shapefiles)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
