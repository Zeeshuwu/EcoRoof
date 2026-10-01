"""Shared helpers for the Reviewer-2 revision (R2-2, R2-3).

Everything here reuses the evaluation logic of ../../evaluate_inference.py
(ground-truth mapping, rasterization with all_touched=False, pixel metrics)
so that numbers stay comparable with the manuscript pipeline.
"""
# torch must be loaded before rasterio/GDAL on Windows (c10.dll init error otherwise)
import torch  # noqa: F401
import os
import sys

import numpy as np
import rasterio

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R2 = os.path.join(REPO, "revisi_r2")
RESULTS = os.path.join(R2, "results")
FIGURES = os.path.join(R2, "figures")
CACHE = os.path.join(R2, "cache")
IMAGE_DIR = os.path.join(REPO, "input", "01 DATA", "IMAGE")
GT_DIR = os.path.join(REPO, "input", "01 DATA", "SHP")
PRED_DIR = os.path.join(REPO, "output_inference")

if REPO not in sys.path:
    sys.path.insert(0, REPO)

from evaluate_inference import GT_MAP, PUBLISHED, rasterize_gt  # noqa: E402

SITES = list(GT_MAP.keys())
CITY = {"JKT_PV": "Jakarta", "JKT_ROOF": "Jakarta",
        "JOG_PV": "Yogyakarta", "JOG_ROOF": "Yogyakarta",
        "SBY_PV": "Surabaya", "SBY_ROOF": "Surabaya"}
CLASSES = ("roof", "pv")
BASE_SEED = 42
BLOCK_M = 25.0


def pairs():
    """(site, cls) pairs that have ground truth: 6 roof + 3 PV."""
    return [(s, c) for s in SITES for c in CLASSES if GT_MAP[s][c] is not None]


def image_path(site):
    return os.path.join(IMAGE_DIR, "%s.tif" % site)


_GT_CACHE = {}


def load_gt(site, cls):
    key = (site, cls)
    if key not in _GT_CACHE:
        with rasterio.open(image_path(site)) as ref:
            gt, n = rasterize_gt(os.path.join(GT_DIR, "%s.shp" % GT_MAP[site][cls]), ref)
        _GT_CACHE[key] = (gt > 0)
    return _GT_CACHE[key]


def load_mask(path):
    with rasterio.open(path) as src:
        return src.read(1) > 0


def ref_profile(site):
    with rasterio.open(image_path(site)) as ref:
        return dict(transform=ref.transform, crs=ref.crs, height=ref.height,
                    width=ref.width, res=abs(ref.transform.a))


def confusion(pred, gt):
    p = pred.astype(bool)
    t = gt.astype(bool)
    tp = int(np.count_nonzero(p & t))
    fp = int(np.count_nonzero(p & ~t))
    fn = int(np.count_nonzero(~p & t))
    tn = int(p.size - tp - fp - fn)
    return tp, fp, fn, tn


def metrics_from_counts(tp, fp, fn, tn):
    """Same formulas as evaluate_inference.pixel_metrics, returned as fractions."""
    tp, fp, fn, tn = (float(x) for x in (tp, fp, fn, tn))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    tot = tp + fp + fn + tn
    acc = (tp + tn) / tot if tot else 0.0
    return {"IoU": iou, "Precision": prec, "Recall": rec, "F1": f1, "Accuracy": acc}


METRICS = ["IoU", "Precision", "Recall", "F1", "Accuracy"]


# ---------------------------------------------------------------- blocks
def block_grid(shape, res, block_m=BLOCK_M):
    """Label image of spatial blocks (block_m x block_m metres).

    Returns (labels, n_rows, n_cols). Edge blocks that are narrower than
    half a block are merged into their neighbour so all blocks are comparable.
    """
    h, w = shape
    bpx = int(round(block_m / res))
    nr = max(1, int(round(h / bpx)))
    nc = max(1, int(round(w / bpx)))
    r_idx = np.minimum(np.arange(h) // bpx, nr - 1)
    c_idx = np.minimum(np.arange(w) // bpx, nc - 1)
    labels = r_idx[:, None] * nc + c_idx[None, :]
    return labels.astype(np.int32), nr, nc


def block_counts(pred, gt, labels, n_blocks):
    """Per-block TP/FP/FN/TN via bincount (fast)."""
    p = pred.ravel().astype(bool)
    t = gt.ravel().astype(bool)
    lab = labels.ravel()
    tp = np.bincount(lab, weights=(p & t), minlength=n_blocks)
    fp = np.bincount(lab, weights=(p & ~t), minlength=n_blocks)
    fn = np.bincount(lab, weights=(~p & t), minlength=n_blocks)
    npx = np.bincount(lab, minlength=n_blocks)
    tn = npx - tp - fp - fn
    return np.stack([tp, fp, fn, tn], 1).astype(np.int64)


def vec_metrics(c):
    """Vectorised metrics on an (..., 4) array of TP, FP, FN, TN counts."""
    c = np.asarray(c, dtype=np.float64)
    tp, fp, fn, tn = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
        iou = np.where(tp + fp + fn > 0, tp / (tp + fp + fn), 0.0)
        acc = (tp + tn) / (tp + fp + fn + tn)
    return {"IoU": iou, "Precision": prec, "Recall": rec, "F1": f1, "Accuracy": acc}


def ensure_dirs():
    for d in (RESULTS, FIGURES, CACHE):
        os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------- dashboard export
# PAPER_CONFIG export step: exactly what app_solar_calculator.py ->
# solar_calculator.export_utils.export_mask_as_shapefile does (connectivity 4,
# drop polygons < 1 m2, Douglas-Peucker simplify 0.5 m with topology preserved),
# rasterized back to the image grid with the same rule as the ground truth.
EXPORT_SIMPLIFY_M = 0.5
EXPORT_MIN_AREA_M2 = 1.0


def export_roundtrip(mask, transform, simplify=EXPORT_SIMPLIFY_M, min_area=EXPORT_MIN_AREA_M2):
    from rasterio.features import rasterize, shapes
    from shapely.geometry import shape
    m = (mask > 0).astype(np.uint8)
    geoms = []
    for gd, v in shapes(m, mask=m, transform=transform, connectivity=4):
        g = shape(gd)
        if g.area < min_area:
            continue
        if simplify > 0:
            g = g.simplify(simplify, preserve_topology=True)
        if g.is_valid and not g.is_empty:
            geoms.append(g)
    if not geoms:
        return np.zeros(mask.shape, bool)
    return rasterize([(g, 1) for g in geoms], out_shape=mask.shape, transform=transform,
                     fill=0, all_touched=False, dtype="uint8") > 0
