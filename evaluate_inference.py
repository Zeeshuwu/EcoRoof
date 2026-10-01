#!/usr/bin/env python
"""
Score the batch inference against the ground-truth shapefiles.

This is the baseline R1-1 is measured against: the recall-recovery step only means
something if we can state recall before and recall after on the same footing. It
also checks whether the run reproduces the metrics printed in the manuscript
(roof IoU 76.51 %, Precision 96.60 %, Recall 75.23 %, F1 83.89 %, Accuracy 90.96 %).

    python evaluate_inference.py
    python evaluate_inference.py --pred-dir output_inference --csv metrics.csv

Metrics are computed at pixel level on the image grid, which is how segmentation
papers normally report IoU and recall. Object-level counts are reported alongside,
because a reviewer reading "75.23 % recall" may reasonably read it either way.

Ground-truth files do not follow one naming rule, so the mapping below is explicit
rather than guessed. Note JKT_ROOF.shp is stored in EPSG:32749 while its image is
EPSG:32748; everything is reprojected to the image CRS before rasterizing.
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

GT_DIR = os.path.join("input", "01 DATA", "SHP")

# site -> {class: ground-truth shapefile stem}
GT_MAP = {
    "JKT_PV":   {"roof": "JKT_PV_roof",   "pv": "JKT_PV_pv"},
    "JKT_ROOF": {"roof": "JKT_ROOF",      "pv": None},
    "JOG_PV":   {"roof": "JOGJA_PV_roof", "pv": "JOGJA_PV_pv"},
    "JOG_ROOF": {"roof": "JOGJA_ROOF",    "pv": None},
    "SBY_PV":   {"roof": "SBY_PV_roof",   "pv": "SBY_PV_pv"},
    "SBY_ROOF": {"roof": "SBY_ROOF",      "pv": None},
}

PUBLISHED = {
    "roof": {"iou": 76.51, "precision": 96.60, "recall": 75.23, "f1": 83.89, "accuracy": 90.96},
    "pv":   {"iou": 84.71, "precision": 87.06, "recall": 96.95, "f1": 91.69, "accuracy": 99.30},
}


def rasterize_gt(shp_path, ref):
    """Burn ground-truth polygons onto the reference image grid."""
    g = gpd.read_file(shp_path)
    if g.crs is None:
        raise ValueError("%s has no CRS" % shp_path)
    if str(g.crs) != str(ref.crs):
        g = g.to_crs(ref.crs)
    geoms = [geom for geom in g.geometry if geom is not None and not geom.is_empty]
    if not geoms:
        return np.zeros((ref.height, ref.width), dtype=np.uint8), 0
    arr = rasterize(
        [(geom, 1) for geom in geoms],
        out_shape=(ref.height, ref.width),
        transform=ref.transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    )
    return arr, len(geoms)


def pixel_metrics(pred, gt):
    p = pred > 0
    t = gt > 0
    tp = int(np.logical_and(p, t).sum())
    fp = int(np.logical_and(p, ~t).sum())
    fn = int(np.logical_and(~p, t).sum())
    tn = int(np.logical_and(~p, ~t).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn)
    return {
        "tp_px": tp, "fp_px": fp, "fn_px": fn, "tn_px": tn,
        "precision": prec * 100, "recall": rec * 100, "f1": f1 * 100,
        "iou": iou * 100, "accuracy": acc * 100,
    }


def object_recall(pred, gt_gdf, ref, min_overlap=0.5):
    """Fraction of ground-truth objects whose area is at least half covered."""
    hit = 0
    for geom in gt_gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        m = rasterize([(geom, 1)], out_shape=(ref.height, ref.width),
                      transform=ref.transform, fill=0, dtype="uint8")
        area = int(m.sum())
        if area == 0:
            continue
        if int(np.logical_and(m > 0, pred > 0).sum()) / area >= min_overlap:
            hit += 1
    total = sum(1 for g in gt_gdf.geometry if g is not None and not g.is_empty)
    return hit, total, (100.0 * hit / total if total else 0.0)


def main():
    ap = argparse.ArgumentParser(description="Score inference against ground truth.")
    ap.add_argument("--pred-dir", default="output_inference")
    ap.add_argument("--gt-dir", default=GT_DIR)
    ap.add_argument("--image-dir", default=os.path.join("input", "01 DATA", "IMAGE"))
    ap.add_argument("--csv", default=None, help="also write per-site metrics to CSV")
    ap.add_argument("--mask-suffix", default="_mask.tif",
                    help="use _mask_prepost.tif to score before post-processing")
    ap.add_argument("--object-recall", action="store_true",
                    help="also compute per-object recall (slower)")
    args = ap.parse_args()

    if not os.path.isdir(args.pred_dir):
        print("[ERROR] %s not found. Run run_batch_inference.py first." % args.pred_dir)
        return 1

    rows = []
    for site, classes in GT_MAP.items():
        for cls, gt_stem in classes.items():
            if gt_stem is None:
                continue
            mask_path = os.path.join(args.pred_dir, "%s_%s%s" % (site, cls, args.mask_suffix))
            gt_path = os.path.join(args.gt_dir, "%s.shp" % gt_stem)
            img_path = os.path.join(args.image_dir, "%s.tif" % site)
            if not os.path.exists(mask_path):
                print("[skip] %-14s no prediction (%s)" % ("%s/%s" % (site, cls), os.path.basename(mask_path)))
                continue
            if not os.path.exists(gt_path):
                print("[skip] %-14s no ground truth (%s)" % ("%s/%s" % (site, cls), gt_stem))
                continue

            with rasterio.open(img_path) as ref:
                gt, n_gt = rasterize_gt(gt_path, ref)
                with rasterio.open(mask_path) as mp:
                    pred = mp.read(1)
                if pred.shape != gt.shape:
                    print("[skip] %-14s shape mismatch pred%s vs gt%s"
                          % ("%s/%s" % (site, cls), pred.shape, gt.shape))
                    continue
                m = pixel_metrics(pred, gt)
                m.update(site=site, cls=cls, n_gt_objects=n_gt)
                if args.object_recall:
                    g = gpd.read_file(gt_path)
                    if str(g.crs) != str(ref.crs):
                        g = g.to_crs(ref.crs)
                    hit, tot, pct = object_recall(pred, g, ref)
                    m.update(obj_hit=hit, obj_total=tot, obj_recall=pct)
            rows.append(m)

    if not rows:
        print("\n[ERROR] nothing scored.")
        return 1

    # ---- per site ----
    print()
    print("PER-SITE PIXEL METRICS (%s)" % args.mask_suffix)
    print("-" * 86)
    hdr = "%-10s %-5s %5s %8s %8s %8s %8s %8s" % (
        "site", "cls", "nGT", "IoU", "Prec", "Recall", "F1", "Acc")
    print(hdr)
    print("-" * 86)
    for r in rows:
        print("%-10s %-5s %5d %7.2f%% %7.2f%% %7.2f%% %7.2f%% %7.2f%%" % (
            r["site"], r["cls"], r["n_gt_objects"],
            r["iou"], r["precision"], r["recall"], r["f1"], r["accuracy"]))

    # ---- aggregate: pool pixels, do not average percentages ----
    print()
    print("AGGREGATE (pixels pooled across sites) vs PUBLISHED")
    print("-" * 86)
    print("%-6s %-10s %8s %8s %8s %8s %8s" % ("cls", "source", "IoU", "Prec", "Recall", "F1", "Acc"))
    print("-" * 86)
    summary = {}
    for cls in ("roof", "pv"):
        sub = [r for r in rows if r["cls"] == cls]
        if not sub:
            continue
        tp = sum(r["tp_px"] for r in sub)
        fp = sum(r["fp_px"] for r in sub)
        fn = sum(r["fn_px"] for r in sub)
        tn = sum(r["tn_px"] for r in sub)
        prec = 100 * tp / (tp + fp) if tp + fp else 0
        rec = 100 * tp / (tp + fn) if tp + fn else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        iou = 100 * tp / (tp + fp + fn) if tp + fp + fn else 0
        acc = 100 * (tp + tn) / (tp + tn + fp + fn)
        got = dict(iou=iou, precision=prec, recall=rec, f1=f1, accuracy=acc)
        summary[cls] = got
        pub = PUBLISHED[cls]
        print("%-6s %-10s %7.2f%% %7.2f%% %7.2f%% %7.2f%% %7.2f%%" % (
            cls, "measured", iou, prec, rec, f1, acc))
        print("%-6s %-10s %7.2f%% %7.2f%% %7.2f%% %7.2f%% %7.2f%%" % (
            "", "published", pub["iou"], pub["precision"], pub["recall"], pub["f1"], pub["accuracy"]))
        print("%-6s %-10s %+7.2f  %+7.2f  %+7.2f  %+7.2f  %+7.2f" % (
            "", "delta", iou - pub["iou"], prec - pub["precision"],
            rec - pub["recall"], f1 - pub["f1"], acc - pub["accuracy"]))
        print()

    if args.object_recall:
        print("OBJECT-LEVEL RECALL (>=50% of each GT polygon covered)")
        print("-" * 60)
        for r in rows:
            if "obj_recall" in r:
                print("  %-10s %-5s %3d/%3d  %6.2f%%" % (
                    r["site"], r["cls"], r["obj_hit"], r["obj_total"], r["obj_recall"]))
        print()

    out = os.path.join(args.pred_dir, "metrics.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"per_site": rows, "aggregate": summary,
                   "published": PUBLISHED, "mask_suffix": args.mask_suffix},
                  f, indent=2)
    print("Written: %s" % out)

    if args.csv:
        import csv as _csv
        keys = sorted({k for r in rows for k in r})
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print("Written: %s" % args.csv)

    print()
    print("Read the delta rows above before using these as an R1-1 baseline. If the")
    print("measured roof recall is far from 75.23 %, the published figure came from a")
    print("different run or a different definition, and the before/after comparison")
    print("must be stated against the measured baseline, not the published one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
