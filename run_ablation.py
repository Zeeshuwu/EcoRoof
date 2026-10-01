#!/usr/bin/env python
"""
Single-level ablation for R2-1: does hierarchical inference actually help?

    python run_ablation.py                 # all six sites, roof class
    python run_ablation.py --site JOG_PV
    python run_ablation.py --overwrite

`segment_multiscale_advanced` already accepts `return_levels=True`, which hands
back each level's mask before merging. One inference run per site therefore
yields all four configurations, instead of four separate runs:

    level1_binary   full orthomosaic, single scale
    level2_merged   512 x 512 patches only
    level3_merged   256 x 256 patches only
    final_mask      merged three-level output (what the paper reports)

The *_merged masks are used rather than the *_validated ones deliberately. The
validated variants are filtered against level 1 (and level 2), so they are not
independent single-level configurations; comparing the hierarchy against them
would understate the hierarchy's contribution because level 1 is already baked
into them.

Outputs land in `output_ablation/` as one GeoTIFF per site per configuration,
plus a manifest with per-level timings for the reproducibility claim.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import sam3rooffix as sr
from sam3_advanced.config import AdvancedConfig
from sam3_advanced.inference import segment_multiscale_advanced
from sam3_advanced.postprocess import post_process_mask
from sam3_advanced.utils import clear_memory

DEFAULT_IMAGE_DIR = os.path.join("input", "01 DATA", "IMAGE")
DEFAULT_OUT_DIR = "output_ablation"

# Same prompts the dashboard uses, so the ablation sits on the published baseline.
ROOF_PROMPTS = {
    "large": ["roof", "large roof", "big roof", "building roof", "rooftop"],
    "medium": ["roof", "medium roof", "rooftop", "house roof"],
    "small": ["roof", "small roof", "house roof", "building top"],
}

CONFIGS = {
    "full_orthophoto": "level1_binary",
    "patch_512": "level2_merged",
    "patch_256": "level3_merged",
}


def save_mask(mask, handler, path):
    import rasterio

    if handler is None or not getattr(handler, "is_geotiff", False):
        return None
    profile = {
        "driver": "GTiff", "height": mask.shape[0], "width": mask.shape[1],
        "count": 1, "dtype": "uint8", "crs": handler.crs,
        "transform": handler.transform, "nodata": 0, "compress": "deflate",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write((mask > 0).astype("uint8") * 255, 1)
    return path


def run_site(image_path, processor, args, manifest):
    site = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = Path(args.out_dir)
    done = out_dir / ("%s_hierarchical.tif" % site)
    if done.exists() and not args.overwrite:
        print("  [SKIP] %s already done" % site)
        return

    print("\n[SITE] %s" % site)
    image_pil, handler = sr.load_image(image_path)

    config = AdvancedConfig()
    config.prompts_large = ROOF_PROMPTS["large"]
    config.prompts_medium = ROOF_PROMPTS["medium"]
    config.prompts_small = ROOF_PROMPTS["small"]

    t0 = time.time()
    final_mask, stats, levels = segment_multiscale_advanced(
        image_pil, processor, config, return_levels=True)
    elapsed = time.time() - t0

    entry = {"site": site, "image": image_path,
             "crs": str(handler.crs) if handler else None,
             "seconds_total": round(elapsed, 2),
             "level_stats": stats,
             # `timing` is a nested dict of per-level wall times, not a scalar.
             "level_timing": levels.get("timing", {}),
             "level3_skipped": bool(levels.get("level3_skipped", False)),
             "masks": {}}

    # Post-process every configuration identically, so the only thing that
    # differs between them is which levels contributed.
    variants = {"hierarchical": final_mask}
    for name, key in CONFIGS.items():
        val = levels.get(key)
        if val is not None:
            variants[name] = val
        else:
            # Level 3 is skipped when levels 1+2 already exceed
            # early_stop_coverage. That is a real outcome, not an error: record
            # it as an empty mask so the site still contributes to the paired
            # comparison rather than dropping out of it entirely.
            print("      [note] %s unavailable (level 3 early-stopped); using empty mask" % key)
            variants[name] = np.zeros_like(np.asarray(final_mask))

    for name, raw in variants.items():
        m = post_process_mask(np.asarray(raw), config, verbose=False)
        p = save_mask(m, handler, str(out_dir / ("%s_%s.tif" % (site, name))))
        cov = float((m > 0).mean() * 100)
        entry["masks"][name] = {"path": p, "coverage_pct": round(cov, 3),
                                "n_objects": int(sr._count_objects(m))}
        print("      %-16s coverage %5.2f%%  objects %4d" % (name, cov, entry["masks"][name]["n_objects"]))

    inst = levels.get("instances")
    if inst is not None:
        entry["n_instances"] = len(inst)

    manifest[site] = entry
    clear_memory()


def main():
    ap = argparse.ArgumentParser(description="R2-1 single-level ablation.")
    ap.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    ap.add_argument("--out-dir", "-o", default=DEFAULT_OUT_DIR)
    ap.add_argument("--checkpoint", "-c", default="sam3.pt")
    ap.add_argument("--site", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    images = sorted(Path(args.image_dir).glob("*.tif"))
    if args.site:
        images = [p for p in images if p.stem == args.site]
    if not images:
        print("[ERROR] no images in %s" % args.image_dir)
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mpath = out_dir / "manifest.json"
    manifest = {}
    if mpath.exists():
        try:
            manifest = json.loads(mpath.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

    print("=" * 70)
    print("R2-1 ABLATION: hierarchical vs single-level")
    print("  sites   : %d  (%s)" % (len(images), ", ".join(p.stem for p in images)))
    print("  configs : hierarchical, %s" % ", ".join(CONFIGS))
    print("  output  : %s" % out_dir)
    print("=" * 70)

    processor = sr.load_model(args.checkpoint)

    failed = []
    for image_path in images:
        try:
            run_site(str(image_path), processor, args, manifest)
        except Exception as exc:
            failed.append("%s: %s" % (image_path.stem, exc))
            print("      [FAIL] %s" % exc)
        finally:
            mpath.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print("\n%d sites done, %d failed." % (len(manifest), len(failed)))
    for f in failed:
        print("  FAILED %s" % f)
    print("Manifest: %s" % mpath)
    print()
    print("Next: python run_ablation_stats.py")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
