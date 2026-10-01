#!/usr/bin/env python
"""
Figures for R2-1: the hierarchical-vs-single-level ablation.

    python make_r2_figures.py

Writes into `figures/`:

    Fig_R2_ablation_boxplot.png   building-level IoU by configuration, with the
                                  Holm-adjusted significance brackets
    Fig_R2_qualitative.png        the same rooftops under all four
                                  configurations, so the statistics have a
                                  visual counterpart

Both are drawn at 300 dpi on a white ground, sized for a single IEEE column.
"""

import argparse
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

try:
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
except ImportError as exc:
    print("[ERROR] needs geopandas + rasterio: %s" % exc)
    sys.exit(1)

from solar_calculator.ablation_stats import building_iou

CONFIGS = ["full_orthophoto", "patch_256", "patch_512", "hierarchical"]
LABEL = {"full_orthophoto": "Full\northophoto", "patch_256": "256 x 256\nonly",
         "patch_512": "512 x 512\nonly", "hierarchical": "Hierarchical\n(3 levels)"}
COLOR = {"full_orthophoto": "#9aa5b1", "patch_256": "#7fa8d4",
         "patch_512": "#4a7fb5", "hierarchical": "#1f4e79"}
GT_MAP = {"JKT_PV": "JKT_PV_roof", "JKT_ROOF": "JKT_ROOF", "JOG_PV": "JOGJA_PV_roof",
          "JOG_ROOF": "JOGJA_ROOF", "SBY_PV": "SBY_PV_roof", "SBY_ROOF": "SBY_ROOF"}


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


def sci(p):
    """p-value as a compact power of ten."""
    if p == 0:
        return "p < 1e-300"
    e = int(np.floor(np.log10(p)))
    return "p = %.1f×10$^{%d}$" % (p / 10 ** e, e)


def collect_iou(args):
    """Per-building IoU for every configuration, in a common building order."""
    per = {c: [] for c in CONFIGS}
    for site, stem in GT_MAP.items():
        img = os.path.join(args.image_dir, "%s.tif" % site)
        if not os.path.exists(img):
            continue
        with rasterio.open(img) as ref:
            g = gpd.read_file(os.path.join(args.gt_dir, "%s.shp" % stem))
            if str(g.crs) != str(ref.crs):
                g = g.to_crs(ref.crs)
            shp = (ref.height, ref.width)
            builds = [rasterize([(x, 1)], out_shape=shp, transform=ref.transform,
                                fill=0, dtype="uint8") > 0
                      for x in g.geometry if x is not None and not x.is_empty]
        masks = {}
        ok = True
        for c in CONFIGS:
            p = os.path.join(args.ablation_dir, "%s_%s.tif" % (site, c))
            if not os.path.exists(p):
                ok = False
                break
            with rasterio.open(p) as m:
                masks[c] = m.read(1) > 0
        if not ok or not builds:
            continue
        allb = np.zeros_like(builds[0])
        for b in builds:
            allb |= b
        for c in CONFIGS:
            per[c].extend(building_iou(masks[c], b, args.pad,
                                       other_buildings=allb & ~b)["iou"] for b in builds)
    return per


def fig_boxplot(per, stats, out):
    data = [np.array([v for v in per[c] if np.isfinite(v)]) for c in CONFIGS]
    n = len(data[-1])

    fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=300)
    bp = ax.boxplot(data, widths=0.58, patch_artist=True, showfliers=False,
                    medianprops=dict(color="white", lw=2.0),
                    whiskerprops=dict(color="#555555"),
                    capprops=dict(color="#555555"),
                    boxprops=dict(lw=0))
    for patch, c in zip(bp["boxes"], CONFIGS):
        patch.set_facecolor(COLOR[c])

    rng = np.random.default_rng(0)
    for i, d in enumerate(data, start=1):
        x = rng.normal(i, 0.055, d.size)
        ax.plot(x, d, ".", ms=2.0, alpha=0.20, color="#1a1a1a", zorder=1)

    for i, d in enumerate(data, start=1):
        ax.annotate("%.1f" % np.median(d), (i, np.median(d)), xytext=(0, -14),
                    textcoords="offset points", ha="center", fontsize=8,
                    color="white", weight="bold")

    # Significance brackets against the hierarchical arm.
    comps = {c["configuration"]: c for c in stats["result"]["comparisons"]}
    order = ["patch_512", "patch_256", "full_orthophoto"]
    top = max(np.percentile(d, 95) for d in data)
    y = top + 4
    for k, name in enumerate(order):
        c = comps[name]
        x1, x2 = CONFIGS.index(name) + 1, CONFIGS.index("hierarchical") + 1
        yy = y + k * 9
        ax.plot([x1, x1, x2, x2], [yy, yy + 2.2, yy + 2.2, yy], lw=1.0, color="#333333")
        ax.text((x1 + x2) / 2, yy + 2.8,
                "%s   %s,  r = %.2f" % (stars(c["p_holm"]), sci(c["p_holm"]), c["rank_biserial_r"]),
                ha="center", va="bottom", fontsize=7.6, color="#222222")

    ax.set_xticks(range(1, len(CONFIGS) + 1))
    ax.set_xticklabels([LABEL[c] for c in CONFIGS], fontsize=9)
    ax.set_ylabel("Building-level IoU (%)", fontsize=10)
    ax.set_ylim(-3, y + 3 * 9 + 10)
    # IoU cannot exceed 100 %, so ticks stop there even though the axis extends
    # to make room for the significance brackets.
    ax.set_yticks(list(range(0, 101, 20)))
    ax.set_title("Hierarchical patch inference against single-scale configurations",
                 fontsize=10.5, pad=10)
    ax.text(0.5, -0.155,
            "n = %d buildings, six sites. Two-sided Wilcoxon signed-rank, "
            "Holm–Bonferroni corrected." % n,
            transform=ax.transAxes, ha="center", fontsize=8, color="#555555")
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out, n


def fig_qualitative(args, out, site="SBY_PV", win=700):
    """The same rooftops under each configuration, over the orthophoto."""
    img_p = os.path.join(args.image_dir, "%s.tif" % site)
    with rasterio.open(img_p) as s:
        rgb = np.dstack([s.read(1), s.read(2), s.read(3)])
        shp = (s.height, s.width)
        tr, crs = s.transform, s.crs
    g = gpd.read_file(os.path.join(args.gt_dir, "%s.shp" % GT_MAP[site]))
    if str(g.crs) != str(crs):
        g = g.to_crs(crs)
    gt = rasterize([(x, 1) for x in g.geometry if x is not None and not x.is_empty],
                   out_shape=shp, transform=tr, fill=0, dtype="uint8") > 0

    # Centre on the densest part of the ground truth, which is where the
    # configurations actually differ.
    rs, cs = np.nonzero(gt)
    r0 = max(0, int(rs.mean()) - win // 2); c0 = max(0, int(cs.mean()) - win // 2)
    r1 = min(shp[0], r0 + win); c1 = min(shp[1], c0 + win)
    sl = (slice(r0, r1), slice(c0, c1))

    fig, axes = plt.subplots(1, 5, figsize=(15.5, 3.4), dpi=300)
    axes[0].imshow(rgb[sl])
    axes[0].set_title("Orthophoto", fontsize=10)
    for ax, c in zip(axes[1:], CONFIGS):
        p = os.path.join(args.ablation_dir, "%s_%s.tif" % (site, c))
        with rasterio.open(p) as m:
            pred = m.read(1)[sl] > 0
        gtw = gt[sl]
        ov = rgb[sl].copy()
        ov[gtw & pred] = (0.45 * ov[gtw & pred] + 0.55 * np.array([40, 190, 90])).astype(np.uint8)
        ov[gtw & ~pred] = (0.45 * ov[gtw & ~pred] + 0.55 * np.array([230, 70, 70])).astype(np.uint8)
        ov[~gtw & pred] = (0.45 * ov[~gtw & pred] + 0.55 * np.array([250, 190, 60])).astype(np.uint8)
        ax.imshow(ov)
        ax.set_title(LABEL[c].replace("\n", " "), fontsize=10)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.legend(handles=[Patch(facecolor="#28be5a", label="True positive"),
                        Patch(facecolor="#e64646", label="False negative (missed roof)"),
                        Patch(facecolor="#fabe3c", label="False positive")],
               loc="lower center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("%s — the same rooftops under each configuration" % site,
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description="R2-1 figures.")
    ap.add_argument("--ablation-dir", default="output_ablation")
    ap.add_argument("--image-dir", default=os.path.join("input", "01 DATA", "IMAGE"))
    ap.add_argument("--gt-dir", default=os.path.join("input", "01 DATA", "SHP"))
    ap.add_argument("--stats", default="output_inference/r2_statistics.json")
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--site", default="SBY_PV")
    ap.add_argument("--pad", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    stats = json.load(open(args.stats, encoding="utf-8"))["ablation"]
    if stats is None:
        print("[ERROR] no ablation in %s; run run_ablation_stats.py first" % args.stats)
        return 1

    print("collecting per-building IoU ...")
    per = collect_iou(args)

    p1, n = fig_boxplot(per, stats, os.path.join(args.out_dir, "Fig_R2_ablation_boxplot.png"))
    print("  written %s  (n = %d buildings)" % (p1, n))
    p2 = fig_qualitative(args, os.path.join(args.out_dir, "Fig_R2_qualitative.png"), args.site)
    print("  written %s  (site %s)" % (p2, args.site))
    return 0


if __name__ == "__main__":
    sys.exit(main())
