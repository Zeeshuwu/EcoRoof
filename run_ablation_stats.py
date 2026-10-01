#!/usr/bin/env python
"""
R2-1 statistics: fills every placeholder in Raffi's draft.

    python run_ablation_stats.py              # TABLE I + ablation if available
    python run_ablation_stats.py --boot 10000

Two parts, and they have different prerequisites:

* **TABLE I** (per-class mean +/- between-site SD with bootstrap CIs) needs only
  `output_inference/`, which already exists. It runs immediately.
* **The ablation** (hierarchical vs each single level) needs `output_ablation/`,
  produced by `run_ablation.py` on a GPU. It is skipped with a notice if absent.

The building is the unit of analysis throughout, because pixels inside one roof
are spatially dependent.
"""

import argparse
import json
import os
import sys

import numpy as np

try:
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
except ImportError as exc:
    print("[ERROR] needs geopandas + rasterio: %s" % exc)
    sys.exit(1)

from solar_calculator.ablation_stats import (
    attribute_pixels_to_buildings, between_site_summary, building_iou,
    compare_configurations, stratified_bootstrap_ci,
)

GT_MAP = {
    "JKT_PV": {"roof": "JKT_PV_roof", "pv": "JKT_PV_pv"},
    "JKT_ROOF": {"roof": "JKT_ROOF", "pv": None},
    "JOG_PV": {"roof": "JOGJA_PV_roof", "pv": "JOGJA_PV_pv"},
    "JOG_ROOF": {"roof": "JOGJA_ROOF", "pv": None},
    "SBY_PV": {"roof": "SBY_PV_roof", "pv": "SBY_PV_pv"},
    "SBY_ROOF": {"roof": "SBY_ROOF", "pv": None},
}
ABLATION_CONFIGS = ["hierarchical", "patch_512", "patch_256", "full_orthophoto"]
LABEL = {"hierarchical": "Hierarchical (3 levels)", "patch_512": "512 x 512 only",
         "patch_256": "256 x 256 only", "full_orthophoto": "Full orthophoto only"}


def load_gt_buildings(site, cls, gt_dir, img_dir):
    """Each ground-truth polygon rasterized separately, on the image grid."""
    stem = GT_MAP[site][cls]
    if stem is None:
        return None, None
    with rasterio.open(os.path.join(img_dir, "%s.tif" % site)) as ref:
        g = gpd.read_file(os.path.join(gt_dir, "%s.shp" % stem))
        if str(g.crs) != str(ref.crs):
            g = g.to_crs(ref.crs)
        shp = (ref.height, ref.width)
        out = []
        for geom in g.geometry:
            if geom is None or geom.is_empty:
                continue
            out.append(rasterize([(geom, 1)], out_shape=shp, transform=ref.transform,
                                 fill=0, dtype="uint8") > 0)
        return out, shp


def read_mask(path, shape):
    if not os.path.exists(path):
        return None
    with rasterio.open(path) as m:
        a = m.read(1) > 0
    return a if a.shape == shape else None


def table_one(args):
    """Per-class pooled metrics with between-site SD and bootstrap CIs."""
    print()
    print("=" * 78)
    print("TABLE I  — pooled metrics, between-site SD, %d%% bootstrap CI (B=%d)"
          % (args.ci, args.boot))
    print("=" * 78)

    result = {}
    for cls in ("roof", "pv"):
        counts_by_site, per_site = {}, {}
        for site in GT_MAP:
            if GT_MAP[site][cls] is None:
                continue
            buildings, shp = load_gt_buildings(site, cls, args.gt_dir, args.image_dir)
            if not buildings:
                continue
            pred = read_mask(os.path.join(args.pred_dir, "%s_%s_mask.tif" % (site, cls)), shp)
            if pred is None:
                print("  [skip] %s/%s no prediction" % (site, cls))
                continue
            # Pixel-level counts partitioned across buildings, so the pooled
            # metrics equal the true pixel-level figures while the bootstrap
            # still resamples buildings.
            rows = attribute_pixels_to_buildings(pred, buildings)
            counts_by_site[site] = [{"tp": r["tp"], "fp": r["fp"], "fn": r["fn"]} for r in rows]
            tp = sum(r["tp"] for r in rows); fp = sum(r["fp"] for r in rows); fn = sum(r["fn"] for r in rows)
            prec = 100 * tp / (tp + fp) if tp + fp else 0
            rec = 100 * tp / (tp + fn) if tp + fn else 0
            per_site[site] = {"iou": 100 * tp / (tp + fp + fn) if tp + fp + fn else 0,
                              "precision": prec, "recall": rec,
                              "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0}
        if not counts_by_site:
            continue

        boot = stratified_bootstrap_ci(counts_by_site, n_boot=args.boot, seed=args.seed, ci=args.ci)
        bss = between_site_summary(per_site)
        result[cls] = {"bootstrap": boot, "between_site": bss, "per_site": per_site}

        print()
        print("  %s  (%d sites, %d buildings)" % (cls.upper(), bss["n_sites"], boot["n_buildings"]))
        print("  %-12s %10s %8s %22s" % ("metric", "pooled", "SD", "%d%% CI" % args.ci))
        for m in ("iou", "precision", "recall", "f1"):
            print("  %-12s %9.2f%% %7.2f  %10.2f – %.2f" % (
                m, boot[m]["point"], bss[m]["sd"], boot[m]["ci_low"], boot[m]["ci_high"]))
        print()
        print("  Cell format for the table (value ± SD (CI low–CI high)):")
        for m in ("iou", "precision", "recall", "f1"):
            print("    %-10s %.2f ± %.2f (%.2f–%.2f)" % (
                m.upper(), boot[m]["point"], bss[m]["sd"], boot[m]["ci_low"], boot[m]["ci_high"]))
    return result


def ablation(args):
    """Hierarchical vs each single-level configuration, at building level."""
    if not os.path.isdir(args.ablation_dir):
        print()
        print("=" * 78)
        print("ABLATION — skipped: %s not found." % args.ablation_dir)
        print("Run  python run_ablation.py  on the GPU machine first.")
        print("=" * 78)
        return None

    per_config, sites_used = {c: [] for c in ABLATION_CONFIGS}, []
    for site in GT_MAP:
        buildings, shp = load_gt_buildings(site, "roof", args.gt_dir, args.image_dir)
        if not buildings:
            continue
        masks = {c: read_mask(os.path.join(args.ablation_dir, "%s_%s.tif" % (site, c)), shp)
                 for c in ABLATION_CONFIGS}
        if any(v is None for v in masks.values()):
            missing = [c for c, v in masks.items() if v is None]
            print("  [skip] %s missing %s" % (site, ", ".join(missing)))
            continue
        sites_used.append(site)
        allb = np.zeros_like(buildings[0])
        for b in buildings:
            allb |= b
        for c in ABLATION_CONFIGS:
            per_config[c].extend(
                building_iou(masks[c], b, args.pad, other_buildings=allb & ~b)["iou"]
                for b in buildings)

    if not sites_used:
        print("\n[ABLATION] no site had all four configurations.")
        return None

    res = compare_configurations(per_config, reference="hierarchical", alpha=args.alpha)
    n = len(per_config["hierarchical"])

    print()
    print("=" * 78)
    print("ABLATION — hierarchical vs single level, building-level IoU")
    print("  %d buildings across %d sites: %s" % (n, len(sites_used), ", ".join(sites_used)))
    print("  Two-sided Wilcoxon signed-rank, Holm-Bonferroni across %d comparisons, alpha=%.2f"
          % (len(res["comparisons"]), args.alpha))
    print("=" * 78)
    print("  %-24s %8s %9s %9s %10s %7s" % ("configuration", "median", "vs hier.", "p (raw)", "p (Holm)", "r"))
    print("  %-24s %8.2f" % (LABEL["hierarchical"], np.nanmedian(per_config["hierarchical"])))
    for c in res["comparisons"]:
        print("  %-24s %8.2f %+9.2f %9.2e %10.2e %7.3f%s" % (
            LABEL.get(c["configuration"], c["configuration"]),
            c["median_other_iou"], c["median_difference_pp"],
            c["p_raw"], c["p_holm"], c["rank_biserial_r"],
            "  *" if c["significant"] else ""))
    print()
    for c in res["comparisons"]:
        print("  %-24s improved on %d buildings, worse on %d, tied %d" % (
            LABEL.get(c["configuration"], c["configuration"]),
            c["n_improved"], c["n_worsened"], c["n_tied"]))
    return {"n_buildings": n, "sites": sites_used, "result": res}


def main():
    ap = argparse.ArgumentParser(description="R2-1 statistics.")
    ap.add_argument("--pred-dir", default="output_inference")
    ap.add_argument("--ablation-dir", default="output_ablation")
    ap.add_argument("--image-dir", default=os.path.join("input", "01 DATA", "IMAGE"))
    ap.add_argument("--gt-dir", default=os.path.join("input", "01 DATA", "SHP"))
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--ci", type=float, default=95.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pad", type=int, default=50, help="window padding in pixels")
    ap.add_argument("--out", default="output_inference/r2_statistics.json")
    args = ap.parse_args()

    t1 = table_one(args)
    ab = ablation(args)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"table_one": t1, "ablation": ab,
                   "settings": vars(args)}, f, indent=2, default=str)
    print()
    print("Written: %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
