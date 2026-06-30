# solar_calculator/spatial_recommendation.py
"""
Spatial Recommendation Model — Scenario-Based AHP Scoring
==========================================================
Ranks installation scenarios (25/50/75/100% utilization) by composite
AHP-weighted score. No per-building data required.

AHP Weight Sources:
- E3S Conferences (2024) — AHP Solar PV Indonesia (CR=0.04)
  usable_area(0.35), irradiance(0.25), economic_potential(0.20),
  building_type(0.12), accessibility(0.08)
- Bandung Multi-Criteria Study (2022) — 5-tier recommendation classes
- GOMap PMC (2024) — constraint/accessibility scoring
- Res4Africa (2024) — building type scores & normalization

"""

import numpy as np
from typing import Optional


# ─────────────────────────────────────────────
# AHP WEIGHTS
# Source: E3S Conferences (2024) — CR = 0.04 < 0.10 ✓
# ─────────────────────────────────────────────

AHP_WEIGHTS = {
    "usable_area":         0.35,
    "irradiance":          0.25,
    "economic_viability":  0.20,
    "building_suitability":0.12,
    "accessibility":       0.08,
}


RECOMMENDATION_CLASSES = [
    ("Priority 1 — Highly Recommended",   0.75, 1.00),
    ("Priority 2 — Recommended",          0.55, 0.75),
    ("Priority 3 — Conditionally Viable", 0.35, 0.55),
    ("Priority 4 — Low Priority",         0.20, 0.35),
    ("Not Recommended",                   0.00, 0.20),
]

RECOMMENDATION_COLORS = {
    "Priority 1 — Highly Recommended":   "#047857",
    "Priority 2 — Recommended":          "#10b981",
    "Priority 3 — Conditionally Viable": "#f59e0b",
    "Priority 4 — Low Priority":         "#ef4444",
    "Not Recommended":                   "#6b7280",
}

# Building type suitability scores (normalized 0–1)
BUILDING_TYPE_SCORES = {
    "industrial":  1.00,
    "commercial":  0.85,
    "mixed_use":   0.65,
    "residential": 0.50,
    "unknown":     0.40,
}

# Irradiance normalization bounds — Indonesia range
# Source: Global Solar Atlas (2024)
IRRADIANCE_BOUNDS = (3.5, 6.5)   # kWh/m²/day

# NPV normalization upper bound (IDR)
# Scaled to realistic rooftop project size
NPV_UPPER_BOUND = 500_000_000.0  # IDR 


def _normalize(value: float, lo: float, hi: float) -> float:
    """Min-max normalize to [0, 1]."""
    if hi == lo:
        return 0.0
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def _classify_recommendation(score: float) -> str:
    for label, lo, hi in RECOMMENDATION_CLASSES:
        if lo <= score <= hi:
            return label
    return "Not Recommended"


def score_scenario(
    scenario_name: str,
    utilization_pct: float,          #
    usable_area_m2: float,
    max_usable_area_m2: float,       
    irradiance_kwh_m2_day: float,
    npv_rp: float,
    building_type: str,
    near_grid: bool = True,
    road_access: bool = True,
) -> dict:
    """
    Score a single utilization scenario using AHP-weighted MCDA.

    Args:
        scenario_name        : e.g. "50% Utilization"
        utilization_pct      : fraction 0.0–1.0
        usable_area_m2       : usable area for this scenario
        max_usable_area_m2   : usable area at 100% (normalization reference)
        irradiance_kwh_m2_day: local GHI value
        npv_rp               : NPV from economics.py for this scenario
        building_type        : global building type selection
        near_grid            : within 100m of PLN grid
        road_access          : accessible for installation crew

    Returns:
        dict with composite score, criteria breakdown, recommendation class
    """
    # ── Criterion 1: Usable area (normalized to 100% scenario)
    area_score = _normalize(usable_area_m2, 0.0, max_usable_area_m2)

    # ── Criterion 2: Irradiance
    irr_score = _normalize(
        irradiance_kwh_m2_day,
        IRRADIANCE_BOUNDS[0],
        IRRADIANCE_BOUNDS[1]
    )

    # ── Criterion 3: Economic viability (NPV-based)
    eco_score = _normalize(max(npv_rp, 0.0), 0.0, NPV_UPPER_BOUND)

    # ── Criterion 4: Building suitability (categorical)
    bld_score = BUILDING_TYPE_SCORES.get(
        building_type.lower(),
        BUILDING_TYPE_SCORES["unknown"]
    )

    # ── Criterion 5: Accessibility
    # Source: GOMap PMC (2024) — binary grid/road access
    acc_score = (0.60 if near_grid else 0.0) + (0.40 if road_access else 0.0)

    criteria_scores = {
        "usable_area":          round(area_score, 4),
        "irradiance":           round(irr_score, 4),
        "economic_viability":   round(eco_score, 4),
        "building_suitability": round(bld_score, 4),
        "accessibility":        round(acc_score, 4),
    }

    weighted_scores = {
        k: round(criteria_scores[k] * AHP_WEIGHTS[k], 4)
        for k in AHP_WEIGHTS
    }

    composite = float(np.clip(sum(weighted_scores.values()), 0.0, 1.0))
    rec_class = _classify_recommendation(composite)

    return {
        "scenario_name":       scenario_name,
        "utilization_pct":     round(utilization_pct * 100, 1),
        "composite_score":     round(composite, 4),
        "recommendation":      rec_class,
        "recommendation_color":RECOMMENDATION_COLORS.get(rec_class, "#6b7280"),
        "criteria_scores":     criteria_scores,
        "weighted_scores":     weighted_scores,
        "references": [
            "E3S Conferences (2024) — AHP weights Indonesia (CR=0.04)",
            "Bandung Multi-Criteria Study (2022) — 5-tier recommendation",
            "GOMap PMC (2024) — accessibility scoring",
            "Res4Africa (2024) — building type scores",
            "Global Solar Atlas (2024) — irradiance normalization",
        ],
    }


def score_all_scenarios(
    scenarios_economic: list,        
    usable_area_m2: float,           
    irradiance_kwh_m2_day: float,
    building_type: str,
    near_grid: bool = True,
    road_access: bool = True,
) -> list:
    """
    Score all utilization scenarios and return ranked list.

    Args:
        scenarios_economic   : list of dicts from calculate_investment_scenarios()
        usable_area_m2       : technically usable area (100% scenario reference)
        irradiance_kwh_m2_day: local GHI
        building_type        : global building type
        near_grid / road_access: accessibility flags

    Returns:
        list of scored scenario dicts, sorted by composite_score descending
    """
    max_area = usable_area_m2  # 100% scenario = normalization reference

    scored = []
    for s in scenarios_economic:
        result = score_scenario(
            scenario_name         = s["scenario_name"],
            utilization_pct       = s["utilization_pct"] / 100.0,
            usable_area_m2        = s["area_used_m2"],
            max_usable_area_m2    = max_area,
            irradiance_kwh_m2_day = irradiance_kwh_m2_day,
            npv_rp                = s["npv_million_rp"] * 1_000_000,
            building_type         = building_type,
            near_grid             = near_grid,
            road_access           = road_access,
        )
        # Merge economic data into result
        result.update({
            "area_used_m2":              s["area_used_m2"],
            "capacity_kwp":              s["capacity_kwp"],
            "annual_production_kwh":     s["annual_production_kwh"],
            "investment_billion_rp":     s["investment_billion_rp"],
            "annual_savings_million_rp": s["annual_savings_million_rp"],
            "payback_years":             s["payback_years"],
            "roi_lifetime_pct":          s["roi_lifetime_pct"],
            "npv_billion_rp":            s["npv_billion_rp"],
            "annual_co2_reduction_tons": s["annual_co2_reduction_tons"],
            "is_viable":                 s["is_viable"],
        })
        scored.append(result)

    scored.sort(key=lambda x: x["composite_score"], reverse=True)
    for i, s in enumerate(scored):
        s["rank"] = i + 1

    return scored
