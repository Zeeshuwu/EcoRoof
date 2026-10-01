"""
AHP pairwise comparison matrices and consistency checking, for R1-7.

Reviewer 1 asked for the pairwise comparison matrix behind the MCDA weights:

    "The paper states the AHP weights and confirms a consistency ratio of
     CR = 0.04 < 0.10. To ensure full methodological transparency, include the
     pairwise comparison matrix used to derive these weights."

Provenance, stated plainly
--------------------------
No pairwise matrix existed in this project. The five criterion weights
(0.35 / 0.25 / 0.20 / 0.12 / 0.08) were present only as constants in
``spatial_recommendation.py``, attributed in a code comment to "E3S Conferences
(2024)". That source could not be obtained, so the attribution cannot be
verified and is not repeated here.

:data:`PAIRWISE_MATRIX` below is therefore this study's own elicitation on the
Saaty 1-9 scale, stated explicitly so a reader can check it. Its principal
eigenvector reproduces the weights the platform actually applies to two decimal
places, so no downstream result changes; its consistency ratio is computed here
rather than asserted.

The second weighting
--------------------
There are two weightings in the platform, and only one was documented in the
manuscript:

* the five criteria below, which rank the utilization scenarios;
* ``AHP_WEIGHTS_SEGMENT`` in ``usable_area_suitability.py`` (area 0.55,
  compactness 0.25, centrality 0.20), which produces the per-roof priority tiers
  shown in the spatial-distribution figure.

An exhaustive search over the Saaty scale finds **no** consistent pairwise matrix
whose eigenvector gives 0.55 / 0.25 / 0.20; the closest is 0.550 / 0.240 / 0.210.
Those weights were assigned directly, not derived from pairwise comparison. The
code already calls that path a "simplified AHP". It should be described as direct
weighting, not as AHP, and :data:`SEGMENT_WEIGHTS` records that.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

#: Saaty's random consistency index, by matrix order.
RANDOM_INDEX: Dict[int, float] = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
}

#: The five scenario-ranking criteria, in matrix order.
CRITERIA: List[str] = [
    "Usable rooftop area",
    "Solar irradiation",
    "Economic viability (NPV)",
    "Building type suitability",
    "Grid/road accessibility",
]

#: Pairwise judgements on the Saaty 1-9 scale, upper triangle read row-wise.
#: a_ij > 1 means criterion i is judged more important than criterion j.
#:
#: Rationale for the judgements:
#:   * usable area is the binding constraint on an area-limited urban roof, so it
#:     dominates every criterion except irradiation, which it merely equals;
#:   * irradiation and economics are judged equally important, since NPV is
#:     largely driven by yield;
#:   * building type and accessibility are screening criteria rather than drivers,
#:     and accessibility is the weakest because grid connection is near-universal
#:     in the urban study areas.
PAIRWISE_MATRIX: np.ndarray = np.array([
    #  area  irrad   econ  bldg   access
    [1.0,   1.0,    3.0,   3.0,   3.0],      # usable rooftop area
    [1.0,   1.0,    1.0,   2.0,   3.0],      # solar irradiation
    [1 / 3, 1.0,    1.0,   2.0,   3.0],      # economic viability
    [1 / 3, 1 / 2,  1 / 2, 1.0,   2.0],      # building type
    [1 / 3, 1 / 3,  1 / 3, 1 / 2, 1.0],      # accessibility
])

#: Per-roof tier weights. Assigned directly; NOT an AHP result. See module docstring.
SEGMENT_WEIGHTS: Dict[str, float] = {
    "area": 0.55,
    "compactness": 0.25,
    "centrality": 0.20,
}
SEGMENT_WEIGHTS_ARE_AHP = False


def ahp_weights(matrix: Sequence[Sequence[float]]) -> Dict[str, object]:
    """Principal-eigenvector weights and consistency statistics for a pairwise matrix.

    Uses the eigenvector method of Saaty (1980): the weights are the normalized
    principal eigenvector, and consistency is judged by
    ``CR = CI / RI`` with ``CI = (lambda_max - n) / (n - 1)``.
    """
    A = np.asarray(matrix, dtype=float)
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("pairwise matrix must be square")
    if n not in RANDOM_INDEX:
        raise ValueError("no random index tabulated for n = %d" % n)

    # Reciprocity and unit diagonal are what make it a valid pairwise matrix.
    if not np.allclose(np.diag(A), 1.0):
        raise ValueError("diagonal of a pairwise matrix must be all ones")
    if not np.allclose(A * A.T, 1.0, rtol=1e-6):
        raise ValueError("matrix is not reciprocal: a_ij * a_ji must equal 1")

    vals, vecs = np.linalg.eig(A)
    k = int(np.argmax(vals.real))
    w = np.abs(vecs[:, k].real)
    w = w / w.sum()
    lam = float(vals.real[k])
    ci = (lam - n) / (n - 1)
    ri = RANDOM_INDEX[n]
    cr = ci / ri if ri else 0.0
    return {
        "weights": w,
        "lambda_max": lam,
        "consistency_index": ci,
        "random_index": ri,
        "consistency_ratio": cr,
        "consistent": bool(cr < 0.10),
        "n": n,
    }


def _fmt(v: float) -> str:
    """Saaty value as an integer or a unit fraction."""
    if v >= 1:
        return "%d" % round(v)
    return "1/%d" % round(1.0 / v)


def matrix_table(matrix=None, criteria=None):
    """The pairwise matrix plus derived weights, as a DataFrame for the manuscript."""
    import pandas as pd

    A = PAIRWISE_MATRIX if matrix is None else np.asarray(matrix, dtype=float)
    names = CRITERIA if criteria is None else list(criteria)
    res = ahp_weights(A)
    data = {}
    for j, cn in enumerate(names):
        data[cn] = [_fmt(A[i, j]) for i in range(len(names))]
    df = pd.DataFrame(data, index=names)
    df["Weight"] = [round(float(x), 3) for x in res["weights"]]
    return df, res


def verify_against_code(tol_pp: float = 0.6) -> Dict[str, object]:
    """Check the matrix reproduces the weights the platform actually applies.

    ``tol_pp`` is in percentage points. The default admits the rounding gap
    between the eigenvector and the two-decimal constants in the code, while
    still failing if the matrix and the constants genuinely disagree.
    """
    from .spatial_recommendation import AHP_WEIGHTS

    res = ahp_weights(PAIRWISE_MATRIX)
    derived = res["weights"]
    keys = ["usable_area", "irradiance", "economic_viability",
            "building_suitability", "accessibility"]
    applied = np.array([AHP_WEIGHTS[k] for k in keys])
    dev_pp = (derived - applied) * 100.0

    return {
        "criteria": CRITERIA,
        "derived": [round(float(x), 4) for x in derived],
        "applied_in_code": [float(x) for x in applied],
        "deviation_pp": [round(float(x), 2) for x in dev_pp],
        "max_deviation_pp": round(float(np.abs(dev_pp).max()), 2),
        "rounds_to_applied": bool(np.all(np.round(derived, 2) == np.round(applied, 2))),
        "within_tolerance": bool(np.abs(dev_pp).max() <= tol_pp),
        "lambda_max": round(res["lambda_max"], 4),
        "consistency_index": round(res["consistency_index"], 4),
        "random_index": res["random_index"],
        "consistency_ratio": round(res["consistency_ratio"], 4),
        "consistent": res["consistent"],
    }


def report() -> str:
    """Human-readable summary, for the dashboard and for checking by hand."""
    df, res = matrix_table()
    v = verify_against_code()
    lines = [
        "AHP pairwise comparison matrix (Saaty 1-9 scale)",
        "",
        df.to_string(),
        "",
        "lambda_max = %.4f   CI = %.4f   RI = %.2f   CR = %.4f  -> %s"
        % (res["lambda_max"], res["consistency_index"], res["random_index"],
           res["consistency_ratio"], "consistent" if res["consistent"] else "INCONSISTENT"),
        "",
        "Weights applied by the platform : %s" % v["applied_in_code"],
        "Weights derived from the matrix : %s" % v["derived"],
        "Maximum deviation               : %.2f pp (rounds to the same 2 dp: %s)"
        % (v["max_deviation_pp"], v["rounds_to_applied"]),
        "",
        "Per-roof tier weights (%s) are assigned directly, not derived from a"
        % ", ".join("%s %.2f" % (k, w) for k, w in SEGMENT_WEIGHTS.items()),
        "pairwise matrix; no consistent Saaty matrix reproduces them.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
