"""
Statistical significance analysis for R2-1.

Implements the three levels of analysis described in Raffi's III.C draft:

1. Per-site metrics summarized as mean +/- standard deviation, expressing
   between-site variability.
2. 95 % confidence intervals for the pooled metrics from a **stratified
   non-parametric bootstrap**: buildings are resampled with replacement *within
   each site* (B = 10,000), pixel-level TP/FP/FN are re-aggregated for every
   resample, and the 2.5th and 97.5th percentiles bound the interval.
3. The contribution of hierarchical inference, tested by comparing
   **building-level IoU** of the merged three-level output against each
   single-level configuration with a two-sided **Wilcoxon signed-rank** test,
   effect size as the **matched-pairs rank-biserial correlation**, and p-values
   adjusted across the three comparisons by **Holm-Bonferroni** (alpha = 0.05).

The building, not the pixel, is the unit of analysis, because pixels within one
roof are spatially dependent and treating them as independent would inflate
significance enormously.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    from scipy.stats import wilcoxon

    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    SCIPY_AVAILABLE = False


# ------------------------------------------------------------------ per-building

def building_iou(pred: np.ndarray, gt_building: np.ndarray, pad_px: int = 50,
                 other_buildings: Optional[np.ndarray] = None) -> Dict[str, float]:
    """IoU for one building, evaluated in a window around it.

    Restricting to a padded window is what makes this a *building-level* score:
    a prediction on the far side of the tile should neither help nor hurt this
    building, but it would enter the union if the whole image were used.

    ``other_buildings`` is the union of every *other* ground-truth footprint and
    must be supplied for dense scenes. Without it, a correct detection on the
    neighbouring roof falls inside this building's window and is charged as a
    false positive, which collapses precision: on these sites it reported 48 %
    where the true pooled figure is 96 %.
    """
    rows, cols = np.nonzero(gt_building)
    if rows.size == 0:
        return {"iou": float("nan"), "tp": 0, "fp": 0, "fn": 0, "gt_px": 0}

    r0 = max(0, rows.min() - pad_px)
    r1 = min(gt_building.shape[0], rows.max() + pad_px + 1)
    c0 = max(0, cols.min() - pad_px)
    c1 = min(gt_building.shape[1], cols.max() + pad_px + 1)

    p = pred[r0:r1, c0:c1] > 0
    g = gt_building[r0:r1, c0:c1] > 0
    if other_buildings is not None:
        # Pixels belonging to a different building are not this building's
        # business, in either direction.
        p = p & ~(other_buildings[r0:r1, c0:c1] > 0)

    tp = int((p & g).sum())
    fp = int((p & ~g).sum())
    fn = int((~p & g).sum())
    denom = tp + fp + fn
    return {"iou": (tp / denom * 100.0) if denom else float("nan"),
            "tp": tp, "fp": fp, "fn": fn, "gt_px": int(g.sum())}


def attribute_pixels_to_buildings(pred: np.ndarray, buildings: Sequence[np.ndarray]):
    """Partition every pixel among buildings so the per-building counts sum exactly
    to the image totals.

    This is what Raffi's III.C describes: pooled metrics are pixel-level, but the
    *resampling unit* for the bootstrap is the building. For that to be coherent
    each pixel must belong to exactly one building, otherwise overlapping
    windows double-count false positives and the pooled figures drift away from
    the pixel-level truth.

    True and false negatives are unambiguous -- they sit inside a footprint.
    False positives are assigned to the nearest footprint by Euclidean distance,
    so predicted area that belongs to no building is still charged to the
    building it is nearest to rather than silently dropped.
    """
    from scipy.ndimage import distance_transform_edt

    p = np.asarray(pred) > 0
    if not buildings:
        return []

    labels = np.zeros(p.shape, dtype=np.int32)
    for i, b in enumerate(buildings, start=1):
        labels[b > 0] = i
    gt_any = labels > 0

    # Nearest footprint for every pixel outside a footprint.
    _, (ri, ci) = distance_transform_edt(~gt_any, return_indices=True)
    nearest = labels[ri, ci]

    fp_mask = p & ~gt_any
    fp_owner = np.where(fp_mask, nearest, 0)

    out = []
    for i, b in enumerate(buildings, start=1):
        g = b > 0
        tp = int((p & g).sum())
        fn = int((~p & g).sum())
        fp = int((fp_owner == i).sum())
        denom = tp + fp + fn
        out.append({"tp": tp, "fp": fp, "fn": fn, "gt_px": int(g.sum()),
                    "iou": (tp / denom * 100.0) if denom else float("nan")})
    return out


# ------------------------------------------------------------------ effect size

def rank_biserial(x: Sequence[float], y: Sequence[float]) -> float:
    """Matched-pairs rank-biserial correlation for a Wilcoxon signed-rank test.

    Defined as ``(W+ - W-) / (W+ + W-)``, the signed-rank sums over non-zero
    differences. It ranges from -1 to +1; positive means ``x`` exceeds ``y``.
    Independent of sample size, unlike the test statistic itself.
    """
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    d = d[np.isfinite(d)]
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    from scipy.stats import rankdata

    ranks = rankdata(np.abs(d))
    w_pos = float(ranks[d > 0].sum())
    w_neg = float(ranks[d < 0].sum())
    total = w_pos + w_neg
    return (w_pos - w_neg) / total if total else 0.0


def holm_bonferroni(pvalues: Sequence[float], alpha: float = 0.05) -> Dict[str, object]:
    """Holm-Bonferroni step-down adjustment.

    Controls the family-wise error rate without assuming independence, which
    matters here: the three comparisons share the hierarchical arm.
    """
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for i, idx in enumerate(order):
        val = (m - i) * p[idx]
        running = max(running, val)          # enforce monotonicity
        adjusted[idx] = min(1.0, running)
    return {"p_adjusted": adjusted.tolist(),
            "reject": (adjusted < alpha).tolist(),
            "alpha": alpha, "method": "holm-bonferroni"}


# ------------------------------------------------------------------ comparison

def compare_configurations(
    per_building: Dict[str, List[float]],
    reference: str = "hierarchical",
    alpha: float = 0.05,
) -> Dict[str, object]:
    """Wilcoxon + rank-biserial + Holm across every single-level configuration.

    ``per_building`` maps configuration name to a list of building IoUs, all in
    the same building order.
    """
    if not SCIPY_AVAILABLE:  # pragma: no cover
        raise ImportError("scipy is required")

    ref = np.asarray(per_building[reference], dtype=float)
    others = [k for k in per_building if k != reference]

    rows, pvals = [], []
    for name in others:
        other = np.asarray(per_building[name], dtype=float)
        ok = np.isfinite(ref) & np.isfinite(other)
        a, b = ref[ok], other[ok]
        diff = a - b

        if np.all(diff == 0):
            stat, p = float("nan"), 1.0
        else:
            stat, p = wilcoxon(a, b, alternative="two-sided",
                               zero_method="wilcox")
        rows.append({
            "configuration": name,
            "n_buildings": int(ok.sum()),
            "median_reference_iou": round(float(np.median(a)), 2),
            "median_other_iou": round(float(np.median(b)), 2),
            "median_difference_pp": round(float(np.median(diff)), 2),
            "mean_difference_pp": round(float(diff.mean()), 2),
            "n_improved": int((diff > 0).sum()),
            "n_worsened": int((diff < 0).sum()),
            "n_tied": int((diff == 0).sum()),
            "wilcoxon_W": None if not np.isfinite(stat) else float(stat),
            "p_raw": float(p),
            "rank_biserial_r": round(rank_biserial(a, b), 3),
        })
        pvals.append(p)

    adj = holm_bonferroni(pvals, alpha)
    for row, pa, rej in zip(rows, adj["p_adjusted"], adj["reject"]):
        row["p_holm"] = float(pa)
        row["significant"] = bool(rej)
    return {"reference": reference, "alpha": alpha,
            "correction": "holm-bonferroni", "comparisons": rows}


# ------------------------------------------------------------------ bootstrap

def stratified_bootstrap_ci(
    counts_by_site: Dict[str, List[Dict[str, int]]],
    n_boot: int = 10_000,
    seed: int = 0,
    ci: float = 95.0,
) -> Dict[str, object]:
    """95 % CIs for pooled metrics, resampling buildings within each site.

    ``counts_by_site`` maps site name to a list of per-building
    ``{"tp", "fp", "fn"}`` dicts. Stratifying by site preserves the design: the
    six sites are fixed, the buildings within them are the sampling unit.
    """
    rng = np.random.default_rng(seed)
    sites = list(counts_by_site)
    arrays = {s: np.array([[b["tp"], b["fp"], b["fn"]] for b in counts_by_site[s]],
                          dtype=float) for s in sites}
    arrays = {s: a for s, a in arrays.items() if a.size}
    if not arrays:
        return {}

    def metrics(tp, fp, fn):
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
        return 100 * iou, 100 * prec, 100 * rec, 100 * f1

    draws = np.empty((n_boot, 4), dtype=float)
    for i in range(n_boot):
        tot = np.zeros(3)
        for s, a in arrays.items():
            idx = rng.integers(0, a.shape[0], a.shape[0])
            tot += a[idx].sum(axis=0)
        draws[i] = metrics(*tot)

    obs = np.zeros(3)
    for a in arrays.values():
        obs += a.sum(axis=0)
    point = metrics(*obs)

    lo_q, hi_q = (100 - ci) / 2, 100 - (100 - ci) / 2
    names = ["iou", "precision", "recall", "f1"]
    out = {"n_boot": n_boot, "ci_level": ci, "seed": seed,
           "n_sites": len(arrays),
           "n_buildings": int(sum(a.shape[0] for a in arrays.values()))}
    for j, nm in enumerate(names):
        lo, hi = np.percentile(draws[:, j], [lo_q, hi_q])
        out[nm] = {"point": round(float(point[j]), 2),
                   "ci_low": round(float(lo), 2),
                   "ci_high": round(float(hi), 2),
                   "ci_width": round(float(hi - lo), 2)}
    return out


def between_site_summary(per_site: Dict[str, Dict[str, float]]) -> Dict[str, object]:
    """Mean +/- standard deviation of each metric across sites."""
    keys = ["iou", "precision", "recall", "f1"]
    out: Dict[str, object] = {"n_sites": len(per_site)}
    for k in keys:
        vals = np.array([m[k] for m in per_site.values() if k in m and np.isfinite(m[k])])
        if vals.size:
            # ddof=1: these six sites are a sample of possible urban scenes, so
            # the sample standard deviation is the right estimator.
            out[k] = {"mean": round(float(vals.mean()), 2),
                      "sd": round(float(vals.std(ddof=1)), 2) if vals.size > 1 else 0.0,
                      "min": round(float(vals.min()), 2),
                      "max": round(float(vals.max()), 2)}
    return out
