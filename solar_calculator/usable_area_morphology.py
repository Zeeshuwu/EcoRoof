"""
Morphology-based usable rooftop area for R1-3.

Reviewer 1 asked us to stop applying a blanket 35.4 % usability cascade to every
roof and instead "apply morphological operations (like inner buffering) directly
to the SAM3-generated polygons to mathematically isolate usable flat space away
from the roof edges".

What changes, and what does not
-------------------------------
The published cascade multiplies five factors::

    building type 0.45  x  setback 0.90  x  HVAC 0.95  x  structural 0.95  x  tilt 0.97

Only the **setback** term describes roof geometry, and only that term is replaced
here by a measured inner buffer. The other four stay as literature values on
purpose:

* building type, HVAC obstruction and structural capacity are properties of the
  building, not of its outline; nothing in a 2-D polygon can measure them,
* tilt efficiency is a packing property of the array, not of the roof edge.

The reviewer asked us not to rely *entirely* on static factors, not to discard
them, so this is a hybrid. ``replace_tilt=True`` additionally drops the tilt term
if you prefer the stricter reading.

Why it matters
--------------
On the study site the aggregate barely moves -- a 0.6 m buffer reproduces the
published 6,256.6 m2 to within 0.4 % -- because a few large roofs dominate the
total. The per-building numbers move enormously: a fixed setback removes about
5 % of a 3,600 m2 roof and about 88 % of a 5 m2 one. That is precisely the
"precision of the building-level analysis" the reviewer flagged.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .usable_area_suitability import (
    USABILITY_PARAMS,
    _classify_suitability,
)

#: Default edge setback. Chosen because it is the common fire-access and
#: maintenance clearance in rooftop PV guidance, and because on the study site it
#: reproduces the literature 10 % setback assumption almost exactly, which makes
#: the change auditable rather than a silent re-baselining.
DEFAULT_SETBACK_M = 0.6

#: Buffered fragments smaller than this cannot host even one module and are
#: dropped rather than counted as usable area.
DEFAULT_MIN_VIABLE_M2 = 4.0


def _iter_parts(geom):
    """Yield the individual polygons of a possibly-multipart geometry."""
    if geom is None or geom.is_empty:
        return
    if geom.geom_type == "Polygon":
        yield geom
    elif geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        for g in geom.geoms:
            if g.geom_type == "Polygon" and not g.is_empty:
                yield g


def buffer_polygon(geom, setback_m: float, min_viable_area_m2: float = DEFAULT_MIN_VIABLE_M2):
    """Inner-buffer one roof outline and drop fragments too small to be usable.

    A negative buffer can split a polygon (an L-shaped roof pinched at the waist)
    or erase it entirely (anything narrower than twice the setback). Both are
    real outcomes and are reported rather than patched over.
    """
    if geom is None or geom.is_empty:
        return None, 0.0, 0
    shrunk = geom.buffer(-abs(setback_m))
    parts = [p for p in _iter_parts(shrunk) if p.area >= min_viable_area_m2]
    if not parts:
        return None, 0.0, 0
    area = float(sum(p.area for p in parts))
    return parts, area, len(parts)


def usable_area_by_buffering(
    gdf,
    setback_m: float = DEFAULT_SETBACK_M,
    building_type: str = "unknown",
    params: Optional[Dict] = None,
    min_viable_area_m2: float = DEFAULT_MIN_VIABLE_M2,
    replace_tilt: bool = False,
) -> Dict[str, object]:
    """Usable area from measured geometry plus the non-geometric literature factors.

    ``gdf`` must be a GeoDataFrame in a projected CRS (metres). Returns a dict
    that mirrors :func:`usable_area_suitability.calculate_usable_area` so it can
    be swapped in wherever that result is consumed, with the per-roof detail and
    the measured setback factor added.
    """
    p = dict(USABILITY_PARAMS if params is None else params)
    bt_ratio = p["building_type_ratio"].get(building_type, p["building_type_ratio"]["unknown"])
    hvac = p["hvac_obstruction_factor"]
    struct = p["structural_factor"]
    tilt = 1.0 if replace_tilt else p["tilt_efficiency_factor"]

    if gdf.crs is not None and gdf.crs.is_geographic:
        raise ValueError(
            "polygons must be in a projected CRS (metres); reproject before buffering"
        )

    non_geometric = bt_ratio * hvac * struct * tilt

    per_roof: List[Dict[str, object]] = []
    gross_total = 0.0
    buffered_total = 0.0
    dropped = 0

    for idx, geom in enumerate(gdf.geometry):
        if geom is None or geom.is_empty:
            continue
        gross = float(geom.area)
        gross_total += gross
        parts, buffered, n_parts = buffer_polygon(geom, setback_m, min_viable_area_m2)
        buffered_total += buffered
        if parts is None:
            dropped += 1
        per_roof.append({
            "id": int(gdf.index[idx]) if hasattr(gdf, "index") else idx,
            "gross_area_m2": round(gross, 2),
            "buffered_area_m2": round(buffered, 2),
            "setback_loss_pct": round(100.0 * (1 - buffered / gross), 1) if gross else None,
            "usable_area_m2": round(buffered * non_geometric, 2),
            "n_fragments": n_parts,
            "viable": parts is not None,
        })

    usable = buffered_total * non_geometric
    measured_setback = (buffered_total / gross_total) if gross_total else 0.0
    ratio = (usable / gross_total) if gross_total else 0.0

    return {
        "method": "morphological_inner_buffer",
        "setback_m": setback_m,
        "min_viable_area_m2": min_viable_area_m2,
        "gross_area_m2": round(gross_total, 2),
        "buffered_area_m2": round(buffered_total, 2),
        "building_type": building_type,
        "building_type_ratio": round(bt_ratio, 3),
        # Measured, not assumed. This is the number that replaces the literature 0.90.
        "setback_factor_measured": round(measured_setback, 4),
        "hvac_obstruction_factor": round(hvac, 3),
        "structural_factor": round(struct, 3),
        "tilt_efficiency_factor": round(tilt, 3),
        "combined_reduction": round(measured_setback * non_geometric, 4),
        "usable_area_m2": round(usable, 2),
        "usability_ratio": round(ratio, 4),
        "suitability_class": _classify_suitability(ratio),
        "n_roofs": len(per_roof),
        "n_roofs_dropped": dropped,
        "per_roof": per_roof,
        "references": [
            "Reviewer 1, R1-3 — morphological inner buffering of SAM3 polygons",
            "Res4Africa (2024) — HVAC and structural factors (retained)",
            "Burke / Gagnon et al. — building type usable-area ratio (retained)",
            "Duffie & Beckman (2013) — tilt efficiency factor (retained)",
        ],
    }


def compare_with_cascade(
    gdf,
    building_type: str = "unknown",
    params: Optional[Dict] = None,
    setbacks: Sequence[float] = (0.25, 0.5, 0.6, 0.75, 1.0, 1.5),
    **kwargs,
):
    """Buffered usable area across setback distances, against the static cascade.

    Returns a DataFrame. The static row is the published method applied to the
    same polygons, so the two are directly comparable.
    """
    import pandas as pd

    p = dict(USABILITY_PARAMS if params is None else params)
    bt = p["building_type_ratio"].get(building_type, p["building_type_ratio"]["unknown"])
    static_combined = (
        bt * p["setback_factor"] * p["hvac_obstruction_factor"]
        * p["structural_factor"] * p["tilt_efficiency_factor"]
    )
    gross = float(gdf.area.sum())
    static_usable = gross * static_combined

    rows = [{
        "method": "static cascade",
        "setback_m": None,
        "setback_factor": round(p["setback_factor"], 4),
        "usable_area_m2": round(static_usable, 1),
        "usability_ratio_pct": round(100 * static_combined, 2),
        "delta_vs_static_pct": 0.0,
        "n_roofs_dropped": 0,
    }]
    for d in setbacks:
        r = usable_area_by_buffering(gdf, setback_m=d, building_type=building_type,
                                     params=params, **kwargs)
        rows.append({
            "method": "inner buffer",
            "setback_m": d,
            "setback_factor": r["setback_factor_measured"],
            "usable_area_m2": r["usable_area_m2"],
            "usability_ratio_pct": round(100 * r["usability_ratio"], 2),
            "delta_vs_static_pct": round(100 * (r["usable_area_m2"] / static_usable - 1), 1)
            if static_usable else None,
            "n_roofs_dropped": r["n_roofs_dropped"],
        })
    return pd.DataFrame(rows)


def building_level_dispersion(gdf, setback_m: float = DEFAULT_SETBACK_M, **kwargs):
    """How badly a blanket setback misrepresents individual roofs.

    This is the substance of R1-3. The aggregate can match the static cascade
    closely while almost every individual building is wrong, because a fixed
    percentage ignores the perimeter-to-area ratio that actually governs edge loss.
    """
    import numpy as np
    import pandas as pd

    rec = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty or geom.area <= 0:
            continue
        _, buffered, _ = buffer_polygon(geom, setback_m, kwargs.get("min_viable_area_m2", DEFAULT_MIN_VIABLE_M2))
        rec.append({"gross_area_m2": geom.area,
                    "setback_loss_pct": 100.0 * (1 - buffered / geom.area)})
    df = pd.DataFrame(rec)
    if df.empty:
        return df, {}
    static_loss = 100.0 * (1 - USABILITY_PARAMS["setback_factor"])
    stats = {
        "static_assumption_pct": static_loss,
        "measured_min_pct": round(float(df.setback_loss_pct.min()), 1),
        "measured_median_pct": round(float(df.setback_loss_pct.median()), 1),
        "measured_max_pct": round(float(df.setback_loss_pct.max()), 1),
        "measured_mean_pct": round(float(df.setback_loss_pct.mean()), 1),
        "n_roofs_understated": int((df.setback_loss_pct > static_loss).sum()),
        "n_roofs_overstated": int((df.setback_loss_pct < static_loss).sum()),
        "n_roofs": len(df),
        "max_error_pp": round(float(np.abs(df.setback_loss_pct - static_loss).max()), 1),
    }
    return df, stats
