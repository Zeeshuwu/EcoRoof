#!/usr/bin/env python
"""
Apply the R1-1 recall recovery to every site and report before/after accuracy.

    python run_recall_recovery.py                 # stages 1-2 (recommended)
    python run_recall_recovery.py --dilate        # also stage 3
    python run_recall_recovery.py --decompose     # where the misses come from
    python run_recall_recovery.py --save-masks    # write recovered masks to disk

Writes `output_inference/recall_recovery.json` and, with --save-masks,
`<SITE>_roof_mask_recovered.tif` alongside the originals.
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

from solar_calculator.recall_recovery import (
    DEFAULT_EXG_THRESHOLD,
    decompose_false_negatives,
    pixel_metrics,
    recover_roof_mask,
)

GT_MAP = {
    "JKT_PV": "JKT_PV_roof", "JKT_ROOF": "JKT_ROOF",
    "JOG_PV": "JOGJA_PV_roof", "JOG_ROOF": "JOGJA_ROOF",
    "SBY_PV": "SBY_PV_roof", "SBY_ROOF": "SBY_ROOF",
}
PUBLISHED = {"iou": 76.51, "precision": 96.60, "recall": 75.23, "f1": 83.89}


def load_site(site, pred_dir, img_dir, gt_dir):
    img_p = os.path.join(img_dir, "%s.tif" % site)
    with rasterio.open(img_p) as s:
        rgb = s.read([1, 2, 3])
        tr, crs, shp = s.transform, s.crs, (s.height, s.width)
        profile = s.profile
    g = gpd.read_file(os.path.join(gt_dir, "%s.shp" % GT_MAP[site]))
    if str(g.crs) != str(crs):
        g = g.to_crs(crs)
    gt = rasterize([(x, 1) for x in g.geometry if x is not None and not x.is_empty],
                   out_shape=shp, transform=tr, fill=0, dtype="uint8") > 0
    with rasterio.open(os.path.join(pred_dir, "%s_roof_mask.tif" % site)) as m:
        pred = m.read(1) > 0
    pre_p = os.path.join(pred_dir, "%s_roof_mask_prepost.tif" % site)
    pre = None
    if os.path.exists(pre_p):
        with rasterio.open(pre_p) as m:
            pre = m.read(1) > 0
    return rgb, gt, pred, pre, profile


def main():
    ap = argparse.ArgumentParser(description="R1-1 recall recovery across all sites.")
    ap.add_argument("--pred-dir", default="output_inference")
    ap.add_argument("--image-dir", default=os.path.join("input", "01 DATA", "IMAGE"))
    ap.add_argument("--gt-dir", default=os.path.join("input", "01 DATA", "SHP"))
    ap.add_argument("--exg", type=float, default=DEFAULT_EXG_THRESHOLD)
    ap.add_argument("--dilate", action="store_true", help="enable stage 3")
    ap.add_argument("--dilate-iters", type=int, default=6)
    ap.add_argument("--decompose", action="store_true")
    ap.add_argument("--save-masks", action="store_true")
    args = ap.parse_args()

    rows, pooled = [], {"before": [], "after": []}
    print()
    print("R1-1 RECALL RECOVERY  (ExG threshold %.2f, stages 1-2%s)"
          % (args.exg, " + guided dilation" if args.dilate else ""))
    print("=" * 84)
    print("%-10s %-8s %8s %8s %8s %8s   %s" % ("site", "stage", "Prec", "Recall", "F1", "IoU", "added px"))
    print("-" * 84)

    for site in GT_MAP:
        try:
            rgb, gt, pred, pre, profile = load_site(site, args.pred_dir, args.image_dir, args.gt_dir)
        except Exception as exc:
            print("[skip] %-10s %s" % (site, exc))
            continue

        before = pixel_metrics(pred, gt)
        rec, info = recover_roof_mask(
            pred, rgb, prepost=pre, exg_threshold=args.exg,
            guided_dilation=args.dilate, dilate_iters=args.dilate_iters,
        )
        after = pixel_metrics(rec, gt)

        print("%-10s %-8s %7.2f%% %7.2f%% %7.2f%% %7.2f%%" % (
            site, "before", before["precision"], before["recall"], before["f1"], before["iou"]))
        print("%-10s %-8s %7.2f%% %7.2f%% %7.2f%% %7.2f%%   %s" % (
            "", "after", after["precision"], after["recall"], after["f1"], after["iou"],
            ", ".join("%s +%d" % (k.replace("_", " "), v) for k, v in info["pixels_added"].items())))
        print("%-10s %-8s %+7.2f  %+7.2f  %+7.2f  %+7.2f" % (
            "", "delta", after["precision"] - before["precision"], after["recall"] - before["recall"],
            after["f1"] - before["f1"], after["iou"] - before["iou"]))

        row = {"site": site, "before": before, "after": after, "recovery": info}
        if args.decompose:
            row["false_negative_sources"] = decompose_false_negatives(pred, gt, pre)
        rows.append(row)
        pooled["before"].append(before)
        pooled["after"].append(after)

        if args.save_masks:
            p = dict(profile)
            p.update(count=1, dtype="uint8", nodata=0, compress="deflate")
            out = os.path.join(args.pred_dir, "%s_roof_mask_recovered.tif" % site)
            with rasterio.open(out, "w", **p) as d:
                d.write(rec.astype("uint8") * 255, 1)
        print()

    if not rows:
        print("[ERROR] nothing processed.")
        return 1

    # Pool pixels rather than averaging percentages.
    def agg(key):
        tp = sum(r[key]["tp_px"] for r in rows)
        fp = sum(r[key]["fp_px"] for r in rows)
        fn = sum(r[key]["fn_px"] for r in rows)
        pr = 100 * tp / (tp + fp) if tp + fp else 0
        rc = 100 * tp / (tp + fn) if tp + fn else 0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0
        return {"precision": pr, "recall": rc, "f1": f1,
                "iou": 100 * tp / (tp + fp + fn) if tp + fp + fn else 0}

    a, b = agg("before"), agg("after")
    print("=" * 84)
    print("AGGREGATE (pixels pooled across all six sites)")
    print("-" * 84)
    print("%-22s %8s %8s %8s %8s" % ("", "Prec", "Recall", "F1", "IoU"))
    print("%-22s %7.2f%% %7.2f%% %7.2f%% %7.2f%%" % ("published (manuscript)",
          PUBLISHED["precision"], PUBLISHED["recall"], PUBLISHED["f1"], PUBLISHED["iou"]))
    print("%-22s %7.2f%% %7.2f%% %7.2f%% %7.2f%%" % ("measured baseline",
          a["precision"], a["recall"], a["f1"], a["iou"]))
    print("%-22s %7.2f%% %7.2f%% %7.2f%% %7.2f%%" % ("after recovery",
          b["precision"], b["recall"], b["f1"], b["iou"]))
    print("%-22s %+7.2f  %+7.2f  %+7.2f  %+7.2f" % ("delta",
          b["precision"] - a["precision"], b["recall"] - a["recall"],
          b["f1"] - a["f1"], b["iou"] - a["iou"]))

    out = os.path.join(args.pred_dir, "recall_recovery.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"per_site": rows, "aggregate_before": a, "aggregate_after": b,
                   "published": PUBLISHED, "exg_threshold": args.exg,
                   "guided_dilation": args.dilate}, f, indent=2, default=str)
    print()
    print("Written: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
