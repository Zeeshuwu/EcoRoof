"""
Post-processing recovery of missed rooftops, for R1-1.

Reviewer 1 observed that 75.23 % recall means a lot of roof area is being missed,
and suggested "incorporating a Normalized Difference Vegetation Index (NDVI) mask
from the multispectral bands (if available) could help explicitly separate tree
canopy from rooftop edges, reducing false negatives".

Two things had to change about that suggestion, both for reasons in the data.

**NDVI is not computable here.** All six orthomosaics are RGB. The Jakarta tiles
carry a fourth band, but it is a constant-255 alpha channel, not near-infrared.
The reviewer allowed for this ("if available"). The Excess Green index is used
instead, an established RGB-only vegetation proxy. It separates the classes
cleanly on this imagery: median ExG is -0.089 over ground-truth roof pixels
against +0.130 over non-roof pixels at the study site.

**Canopy is not the main cause of the recall gap.** Decomposing the false
negatives against the ground truth shows where the missed pixels actually are:

    boundary under-segmentation   46-93 %
    discarded by post-processing   6-51 %
    holes enclosed by prediction   1-14 %
    buildings missed entirely     0-2.5 %

and only 12-16 % of missed pixels are vegetation at a sensible ExG threshold. A
purely canopy-driven recovery would therefore address a minority of the gap. The
recovery below targets the causes that dominate, and uses the vegetation mask the
way the reviewer intended: to stop boundary growth leaking into tree canopy.

Stages
------
1. **Restore post-processing discards.** ``post_process_mask`` applies a
   geometry filter that removes low-compactness blobs, which is the shape a
   partially occluded roof takes. Non-vegetation pixels it removed are restored.
   This is the single largest recoverable component (up to 51 % of the gap).
2. **Fill enclosed holes.** A gap completely surrounded by roof is roof, whether
   the occluder is canopy, an air-conditioning unit or shadow.
3. **ExG-guided dilation** (optional, off by default). Grows the boundary only
   into non-vegetation pixels. It helps decisively where boundary error dominates
   and hurts where it does not, so it is a per-site choice rather than a default.

Stages 1-2 raise recall at every site with precision essentially unchanged.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

try:
    from scipy.ndimage import binary_dilation, binary_fill_holes, label

    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    SCIPY_AVAILABLE = False

#: ExG above this counts as vegetation. Calibrated on the study imagery: it sits
#: between the roof and non-roof medians at every site, and is the point where
#: the fraction of genuinely-roof pixels wrongly called vegetation falls below
#: about 15 % while still flagging most canopy.
DEFAULT_EXG_THRESHOLD = 0.05

#: Dilation iterations for stage 3, with a 3x3 structuring element. Six steps is
#: about 0.6 m at the 10 cm ground sampling distance of this imagery.
DEFAULT_DILATE_ITERS = 6


def excess_green(rgb: np.ndarray) -> np.ndarray:
    """Excess Green index from an RGB stack shaped (3, H, W).

    ``ExG = 2g - r - b`` on chromatic coordinates, so it is invariant to overall
    brightness and usable across tiles captured at different exposures.
    """
    a = np.asarray(rgb, dtype=np.float32)
    if a.ndim != 3 or a.shape[0] < 3:
        raise ValueError("expected an array shaped (3, H, W); got %s" % (a.shape,))
    r, g, b = a[0], a[1], a[2]
    total = r + g + b + 1e-6
    return 2.0 * (g / total) - (r / total) - (b / total)


def vegetation_mask(rgb: np.ndarray, threshold: float = DEFAULT_EXG_THRESHOLD) -> np.ndarray:
    """Boolean canopy/vegetation mask from RGB imagery."""
    return excess_green(rgb) > threshold


def recover_roof_mask(
    pred: np.ndarray,
    rgb: np.ndarray,
    prepost: Optional[np.ndarray] = None,
    exg_threshold: float = DEFAULT_EXG_THRESHOLD,
    restore_filtered: bool = True,
    fill_holes: bool = True,
    guided_dilation: bool = False,
    dilate_iters: int = DEFAULT_DILATE_ITERS,
    max_hole_area_px: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Recover missed rooftop area. Returns ``(recovered_mask, breakdown)``.

    ``prepost`` is the mask before ``post_process_mask`` ran; stage 1 is skipped
    without it. ``max_hole_area_px`` caps hole filling so that genuine courtyards
    enclosed by a ring of buildings are not absorbed; ``None`` fills every hole.
    """
    if not SCIPY_AVAILABLE:  # pragma: no cover
        raise ImportError("scipy is required for recall recovery")

    base = np.asarray(pred) > 0
    veg = vegetation_mask(rgb, exg_threshold)
    out = base.copy()
    added: Dict[str, int] = {}

    if restore_filtered and prepost is not None:
        pre = np.asarray(prepost) > 0
        # Only restore what is not vegetation: a blob the filter removed that is
        # green was probably a tree, and restoring it would cost precision.
        gain = pre & ~out & ~veg
        out = out | gain
        added["restored_from_postprocess"] = int(gain.sum())

    if fill_holes:
        filled = binary_fill_holes(out)
        holes = filled & ~out
        if max_hole_area_px is not None and holes.any():
            lab, n = label(holes)
            keep = np.zeros_like(holes)
            for i in range(1, n + 1):
                comp = lab == i
                if comp.sum() <= max_hole_area_px:
                    keep |= comp
            holes = keep
        out = out | holes
        added["filled_holes"] = int(holes.sum())

    if guided_dilation and dilate_iters > 0:
        before = out.copy()
        se = np.ones((3, 3), dtype=bool)
        for _ in range(dilate_iters):
            # Grow one ring at a time, never into vegetation. Doing it stepwise
            # rather than in one large dilation stops growth jumping a canopy
            # strip and landing on the far side.
            out = (binary_dilation(out, se) & ~veg) | out
        added["guided_dilation"] = int((out & ~before).sum())

    total = int(out.sum()) - int(base.sum())
    return out, {
        "pixels_added": added,
        "pixels_added_total": total,
        "coverage_before_pct": round(100.0 * base.mean(), 3),
        "coverage_after_pct": round(100.0 * out.mean(), 3),
        "vegetation_fraction_pct": round(100.0 * veg.mean(), 2),
        "exg_threshold": exg_threshold,
        "stages": {
            "restore_filtered": bool(restore_filtered and prepost is not None),
            "fill_holes": bool(fill_holes),
            "guided_dilation": bool(guided_dilation),
        },
    }


def pixel_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """Precision, recall, F1, IoU and accuracy, in percent."""
    p = np.asarray(pred) > 0
    t = np.asarray(gt) > 0
    tp = int((p & t).sum())
    fp = int((p & ~t).sum())
    fn = int((~p & t).sum())
    tn = int((~p & ~t).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {
        "precision": 100 * prec, "recall": 100 * rec, "f1": 100 * f1,
        "iou": 100 * iou, "accuracy": 100 * (tp + tn) / t.size,
        "tp_px": tp, "fp_px": fp, "fn_px": fn, "tn_px": tn,
    }


def decompose_false_negatives(
    pred: np.ndarray, gt: np.ndarray, prepost: Optional[np.ndarray] = None
) -> Dict[str, object]:
    """Attribute missed pixels to their cause, so recovery targets the real one."""
    if not SCIPY_AVAILABLE:  # pragma: no cover
        raise ImportError("scipy is required")
    p = np.asarray(pred) > 0
    t = np.asarray(gt) > 0
    miss = t & ~p
    total = int(miss.sum())
    if total == 0:
        return {"total_missed_px": 0}

    holes = binary_fill_holes(p) & ~p
    in_holes = int((miss & holes).sum())

    from_filter = 0
    if prepost is not None:
        pre = np.asarray(prepost) > 0
        from_filter = int((miss & pre & ~p & ~holes).sum())

    lab, n = label(t)
    whole = np.zeros_like(t)
    for i in range(1, n + 1):
        comp = lab == i
        if comp.sum() and (comp & p).sum() / comp.sum() < 0.10:
            whole |= comp
    missed_entirely = int((miss & whole & ~holes).sum())

    boundary = total - in_holes - from_filter - missed_entirely
    pct = lambda x: round(100.0 * x / total, 1)
    return {
        "total_missed_px": total,
        "in_holes_px": in_holes, "in_holes_pct": pct(in_holes),
        "discarded_by_postprocess_px": from_filter,
        "discarded_by_postprocess_pct": pct(from_filter),
        "buildings_missed_entirely_px": missed_entirely,
        "buildings_missed_entirely_pct": pct(missed_entirely),
        "boundary_undersegmentation_px": max(boundary, 0),
        "boundary_undersegmentation_pct": pct(max(boundary, 0)),
    }
