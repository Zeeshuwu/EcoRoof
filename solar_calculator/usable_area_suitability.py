"""
Usable Area Suitability Analysis & Roof Spatial Visualization
=============================================================
Two responsibilities in one module:

1. USABLE AREA CALCULATION
   Applies reduction factors to gross roof area to arrive at
   technically usable area for solar PV installation.

2. ROOF SPATIAL VISUALIZATION
   Segments individual roof patches from the SAM3 binary mask,
   scores each via simplified AHP, allocates optimal install
   zones using greedy top-down fill capped at consumption target,
   and renders a two-layer color-coded overlay.

Limitation (journal):
    Since SAM3 produces a single binary mask with no per-pixel
    irradiance, orientation, or tilt data, all roof segments
    receive identical irradiance scores. Ranking is driven
    primarily by segment area (weight 0.55) and shape compactness
    (weight 0.25). Future work should integrate per-pixel DSM or
    irradiance rasters to differentiate shading and tilt per roof.

    The greedy allocation assumes uniform panel productivity across
    all segments. In reality, south-facing or unshaded segments
    would be prioritised first.

References:
    Burke (2021)               — ASF: setback/HVAC/structural factors
    Res4Africa (2024)          — Building type usable area ratios
    Jakubiec & Reinhart (2013) — Min viable area 4 m²
    Duffie & Beckman (2013)    — Tilt efficiency factor
    E3S Conferences (2024)     — AHP weights CR=0.04
    Bandung MCRS (2022)        — 5-tier suitability classification
    Tarigan et al. (2025)      — PSH 4.5 h/day, PR 0.75


"""

import cv2
import numpy as np
from PIL import Image, ImageDraw
from typing import List, Dict, Optional, Tuple


# ============================================================================
# PART 1 — USABLE AREA REDUCTION PARAMETERS
# ============================================================================

USABILITY_PARAMS = {
    "setback_factor":          0.90,  
    "hvac_obstruction_factor": 0.95,  
    "structural_factor":       0.95,   


    "building_type_ratio": {
        "residential":  0.40,
        "commercial":   0.60,
        "industrial":   0.65,
        "mixed_use":    0.50,
        "unknown":      0.45,
    },

    # Minimum viable contiguous area for a single panel cluster
    # Source: Jakubiec & Reinhart (2013), Solar Energy 95:198–208
    "min_viable_area_m2": 4.0,

    # Tilt efficiency for flat roof rack-mounted at 15° (Java latitude ~-7°)
    # Source: Duffie & Beckman (2013), Solar Engineering 4th ed.
    "tilt_efficiency_factor": 0.97,
}

# Suitability classification thresholds
# Source: Res4Africa (2024), Table 5
SUITABILITY_THRESHOLDS = {
    "High":       0.55,
    "Medium":     0.40,
    "Low":        0.20,
    "Unsuitable": 0.00,
}

SUITABILITY_COLORS = {
    "High":       "#10b981",
    "Medium":     "#f59e0b",
    "Low":        "#ef4444",
    "Unsuitable": "#6b7280",
}


def calculate_usable_area(
    gross_area_m2: float,
    building_type: str = "unknown",
    params: dict = None,
) -> dict:
    """
    Compute technically usable rooftop area for solar PV.

    Formula:
        Usable = Gross × building_type_ratio
                       × setback_factor
                       × hvac_obstruction_factor
                       × structural_factor
                       × tilt_efficiency_factor

    Args:
        gross_area_m2 : Total detected roof area in m²
        building_type : One of residential/commercial/industrial/mixed_use
        params        : Override USABILITY_PARAMS if needed

    Returns:
        dict with full reduction breakdown and suitability class
    """
    p     = params or USABILITY_PARAMS
    btype = building_type.lower()
    bt_ratio = p["building_type_ratio"].get(
        btype, p["building_type_ratio"]["unknown"]
    )

    asf  = (
        p["setback_factor"]
        * p["hvac_obstruction_factor"]
        * p["structural_factor"]
    )
    tilt     = p["tilt_efficiency_factor"]
    combined = bt_ratio * asf * tilt
    usable   = gross_area_m2 * combined

    if usable < p["min_viable_area_m2"]:
        usable            = 0.0
        suitability_class = "Unsuitable"
    else:
        ratio             = usable / gross_area_m2 if gross_area_m2 > 0 else 0.0
        suitability_class = _classify_suitability(ratio)

    usability_ratio = usable / gross_area_m2 if gross_area_m2 > 0 else 0.0

    return {
        "gross_area_m2":           round(gross_area_m2, 2),
        "building_type":           btype,
        "building_type_ratio":     round(bt_ratio, 3),
        "setback_factor":          round(p["setback_factor"], 3),
        "hvac_obstruction_factor": round(p["hvac_obstruction_factor"], 3),
        "structural_factor":       round(p["structural_factor"], 3),
        "tilt_efficiency_factor":  round(tilt, 3),
        "combined_reduction":      round(combined, 4),
        "usable_area_m2":          round(usable, 2),
        "usability_ratio":         round(usability_ratio, 4),
        "suitability_class":       suitability_class,
        "references": [
            "Burke (2021) — ASF: setback/HVAC/structural factors",
            "Res4Africa (2024) — Building type usable area ratios",
            "Jakubiec & Reinhart (2013) — Min viable area 4 m²",
            "Duffie & Beckman (2013) — Tilt efficiency factor",
        ],
    }


def _classify_suitability(ratio: float) -> str:
    if ratio >= SUITABILITY_THRESHOLDS["High"]:     return "High"
    elif ratio >= SUITABILITY_THRESHOLDS["Medium"]: return "Medium"
    elif ratio >= SUITABILITY_THRESHOLDS["Low"]:    return "Low"
    else:                                           return "Unsuitable"



# Minimum segment size in pixels
# Corresponds to ~4 m² at 0.5 m/px — Jakubiec & Reinhart (2013)
MIN_SEGMENT_AREA_PX = 50


AHP_WEIGHTS_SEGMENT = {
    "area":        0.55,
    "compactness": 0.25,
    "centrality":  0.20,
}

# Score thresholds → suitability class
# Source: Bandung Multi-Criteria Rooftop Solar Study (2022)
SEGMENT_SCORE_THRESHOLDS = {
    "Priority 1": 0.70,
    "Priority 2": 0.50,
    "Priority 3": 0.35,
    "Priority 4": 0.20,
    "Unsuitable": 0.00,
}


SEGMENT_PALETTE = {
    "Priority 1": (34,  197,  94),   
    "Priority 2": (132, 204,  22),   
    "Priority 3": (234, 179,   8),   
    "Priority 4": (249, 115,  22),   
    "Unsuitable": (156, 163, 175),   
}

#
HATCH_COLOR     = (56, 189, 248)   
HATCH_SPACING   = 8
HATCH_THICKNESS = 1

# Solar production constants
# Source: Tarigan et al. (2025)
_PANEL_EFFICIENCY  = 0.20
_PEAK_SUN_HOURS    = 4.5
_PERFORMANCE_RATIO = 0.75
_DAYS_PER_YEAR     = 365




def extract_roof_segments(
    mask_np: np.ndarray,
    pixel_to_meter: float = 0.5,
    min_area_px: int = MIN_SEGMENT_AREA_PX,
) -> List[Dict]:
    """
    Extract individual roof segments from binary mask via
    connected component labeling (cv2.connectedComponentsWithStats).

    Args:
        mask_np        : Binary mask (H × W), values 0 or >0
        pixel_to_meter : m/pixel conversion ratio
        min_area_px    : Minimum segment area in pixels

    Returns:
        list of segment dicts with geometry, area, compactness, centrality
    """
    binary = (mask_np > 0).astype(np.uint8)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    h, w         = mask_np.shape
    img_center   = np.array([w / 2.0, h / 2.0])
    max_dist     = np.linalg.norm(img_center)

    segments = []
    for i in range(1, n_labels):   # skip background label 0
        area_px = int(stats[i, cv2.CC_STAT_AREA])
        if area_px < min_area_px:
            continue

        x  = int(stats[i, cv2.CC_STAT_LEFT])
        y  = int(stats[i, cv2.CC_STAT_TOP])
        sw = int(stats[i, cv2.CC_STAT_WIDTH])
        sh = int(stats[i, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[i]

        area_m2     = area_px * (pixel_to_meter ** 2)
        bbox_area   = sw * sh
        compactness = area_px / bbox_area if bbox_area > 0 else 0.0
        dist        = np.linalg.norm(np.array([cx, cy]) - img_center)
        centrality  = 1.0 - (dist / max_dist) if max_dist > 0 else 0.0

        segments.append({
            "id":          i,
            "area_px":     area_px,
            "area_m2":     round(area_m2, 2),
            "bbox":        (x, y, sw, sh),
            "centroid":    (round(cx, 1), round(cy, 1)),
            "compactness": round(compactness, 4),
            "centrality":  round(centrality, 4),
            "seg_mask":    (labels == i),
        })

    return segments



def score_segments(segments: List[Dict]) -> List[Dict]:
    """
    Score and classify each segment using simplified AHP model.

    Normalization: min-max across all segments for area and compactness.
    Centrality is already [0, 1].

    Args:
        segments: output of extract_roof_segments()

    Returns:
        segments sorted by composite_score descending, with rank added
    """
    if not segments:
        return segments

    areas       = np.array([s["area_m2"]     for s in segments], dtype=float)
    compacts    = np.array([s["compactness"] for s in segments], dtype=float)
    centrals    = np.array([s["centrality"]  for s in segments], dtype=float)

    def _minmax(arr):
        lo, hi = arr.min(), arr.max()
        return np.ones_like(arr) if hi == lo else (arr - lo) / (hi - lo)

    n_area    = _minmax(areas)
    n_compact = _minmax(compacts)
    n_central = centrals

    for i, seg in enumerate(segments):
        score = float(np.clip(
            AHP_WEIGHTS_SEGMENT["area"]        * n_area[i]    +
            AHP_WEIGHTS_SEGMENT["compactness"] * n_compact[i] +
            AHP_WEIGHTS_SEGMENT["centrality"]  * n_central[i],
            0.0, 1.0
        ))

        if   score >= SEGMENT_SCORE_THRESHOLDS["Priority 1"]: cls = "Priority 1"
        elif score >= SEGMENT_SCORE_THRESHOLDS["Priority 2"]: cls = "Priority 2"
        elif score >= SEGMENT_SCORE_THRESHOLDS["Priority 3"]: cls = "Priority 3"
        elif score >= SEGMENT_SCORE_THRESHOLDS["Priority 4"]: cls = "Priority 4"
        else:                                                  cls = "Unsuitable"

        seg["composite_score"]   = round(score, 4)
        seg["suitability_class"] = cls
        seg["norm_area"]         = round(float(n_area[i]),    4)
        seg["norm_compactness"]  = round(float(n_compact[i]), 4)
        seg["norm_centrality"]   = round(float(n_central[i]), 4)

    segments.sort(key=lambda s: s["composite_score"], reverse=True)
    for rank, seg in enumerate(segments, start=1):
        seg["rank"] = rank

    return segments



def allocate_greedy(
    segments: List[Dict],
    consumption_annual_kwh: float,
    usable_factor: float = 0.40,
) -> List[Dict]:
    """
    Greedy top-down allocation: fill highest-ranked roofs first
    until annual consumption target is met.

    Per segment:
        usable_area = area_m2 × usable_factor
        production  = usable_area × η × PSH × PR × 365

    Allocation:
        remaining > full production  → fill 100% (Full)
        remaining < full production  → fill partial % (Partial)
        consumption already met      → skip (Not needed)
        class == Unsuitable          → skip (Unsuitable)

    Sources:
        Res4Africa (2024)      — usable_factor default 0.40 residential
        Tarigan et al. (2025)  — PSH 4.5, PR 0.75

    Args:
        segments               : scored + sorted segment list
        consumption_annual_kwh : annual kWh target to meet
        usable_factor          : fraction of gross area that is usable

    Returns:
        segments with fill_pct, fill_area_m2, fill_production,
        allocation_status added
    """
    remaining_kwh = consumption_annual_kwh

    for seg in segments:

        if seg["suitability_class"] == "Unsuitable":
            seg.update({
                "usable_area_m2":    0.0,
                "fill_pct":          0.0,
                "fill_area_m2":      0.0,
                "fill_production":   0.0,
                "allocation_status": "Unsuitable",
            })
            continue

        usable_area_m2  = seg["area_m2"] * usable_factor
        full_production = (
            usable_area_m2
            * _PANEL_EFFICIENCY
            * _PEAK_SUN_HOURS
            * _PERFORMANCE_RATIO
            * _DAYS_PER_YEAR
        )
        seg["usable_area_m2"] = round(usable_area_m2, 2)

        if remaining_kwh <= 0:
            seg.update({
                "fill_pct":          0.0,
                "fill_area_m2":      0.0,
                "fill_production":   0.0,
                "allocation_status": "Not needed",
            })

        elif full_production <= remaining_kwh:
            seg.update({
                "fill_pct":          1.0,
                "fill_area_m2":      round(usable_area_m2, 2),
                "fill_production":   round(full_production, 2),
                "allocation_status": "Full",
            })
            remaining_kwh -= full_production

        else:
            fill_pct = float(np.clip(remaining_kwh / full_production, 0.0, 1.0))
            seg.update({
                "fill_pct":          round(fill_pct, 4),
                "fill_area_m2":      round(usable_area_m2 * fill_pct, 2),
                "fill_production":   round(remaining_kwh, 2),
                "allocation_status": "Partial",
            })
            remaining_kwh = 0.0

    return segments



def render_spatial_overlay(
    image_pil: Image.Image,
    segments: List[Dict],
    alpha_suitability: float = 0.45,
    alpha_hatch: float = 0.70,
    show_labels: bool = True,
) -> Image.Image:
    """
    Render two-layer spatial overlay onto original image.

    Layer 1 — Suitability fill:
        Each roof segment filled with its class color.

    Layer 2 — Allocation hatch:
        Diagonal hatch lines over recommended install zone.
        Full segments = full hatch.
        Partial segments = top fill_pct rows hatched.

    Args:
        image_pil         : Original input image (PIL)
        segments          : Scored + allocated segment list
        alpha_suitability : Opacity of suitability fill (0–1)
        alpha_hatch       : Opacity of hatch lines (0–1)
        show_labels       : Draw rank number at each centroid

    Returns:
        PIL Image (RGB) with overlay
    """
    base = image_pil.convert("RGBA")
    h, w = base.size[1], base.size[0]

   
    suit_arr = np.zeros((h, w, 4), dtype=np.uint8)
    for seg in segments:
        if seg["suitability_class"] == "Unsuitable":
            continue
        r, g, b  = SEGMENT_PALETTE[seg["suitability_class"]]
        alpha_v  = int(alpha_suitability * 255)
        suit_arr[seg["seg_mask"]] = [r, g, b, alpha_v]

    composite = Image.alpha_composite(
        base, Image.fromarray(suit_arr, "RGBA")
    )

   
    hatch_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    hatch_draw  = ImageDraw.Draw(hatch_layer)
    alpha_v     = int(alpha_hatch * 255)
    hc          = HATCH_COLOR

    for seg in segments:
        status = seg.get("allocation_status", "")
        if status not in ("Full", "Partial"):
            continue

        ys, xs = np.where(seg["seg_mask"])
        if len(ys) == 0:
            continue

        y_min, y_max = int(ys.min()), int(ys.max())
        x_min, x_max = int(xs.min()), int(xs.max())

        cutoff_y = (
            y_min + int((y_max - y_min) * seg.get("fill_pct", 1.0))
            if status == "Partial" else y_max + 1
        )

        seg_mask = seg["seg_mask"]
        
        if seg_mask.shape != suit_arr.shape[:2]:
            from PIL import Image as _Image
            seg_mask_pil = _Image.fromarray(seg_mask.astype(np.uint8) * 255)
            seg_mask_pil = seg_mask_pil.resize(
                (suit_arr.shape[1], suit_arr.shape[0]),  
                _Image.NEAREST   
            )
            seg_mask = np.array(seg_mask_pil) > 127

        suit_arr[seg_mask] = [r, g, b, alpha_v]

        span     = (y_max - y_min) + (x_max - x_min)

        for offset in range(-span, span, HATCH_SPACING):
            x0c = x_min
            x1c = x_max
            y0c = x0c + offset
            y1c = x1c + offset
            steps = max(abs(x1c - x0c), abs(y1c - y0c))
            if steps == 0:
                continue
            for step in range(steps + 1):
                t  = step / steps
                px = int(x0c + t * (x1c - x0c))
                py = int(y0c + t * (y1c - y0c))
                if 0 <= px < w and 0 <= py < h:
                    if seg_mask[py, px] and py <= cutoff_y:
                        hatch_draw.point(
                            (px, py), fill=(hc[0], hc[1], hc[2], alpha_v)
                        )

    composite = Image.alpha_composite(composite, hatch_layer)


    if show_labels:
        draw = ImageDraw.Draw(composite)
        for seg in segments:
            if seg["suitability_class"] == "Unsuitable":
                continue
            cx, cy = seg["centroid"]
            r      = 10
            draw.ellipse(
                [cx - r, cy - r, cx + r, cy + r],
                fill=(255, 255, 255, 220),
            )
            draw.text(
                (cx, cy),
                f"#{seg['rank']}",
                fill=(30, 50, 30, 255),
                anchor="mm",
            )

    return composite.convert("RGB")



def render_legend(width: int = 280) -> Image.Image:
    """
    Render a standalone legend image for the spatial overlay.

    Returns:
        PIL Image (RGB)
    """
    rows    = list(SEGMENT_PALETTE.keys()) + ["Recommended zone"]
    row_h   = 26
    padding = 14
    height  = padding * 2 + 20 + len(rows) * row_h

    legend = Image.new("RGB", (width, height), (255, 255, 255))
    draw   = ImageDraw.Draw(legend)

    draw.text(
        (padding, padding - 2),
        "Spatial Suitability",
        fill=(30, 50, 30),
    )

    for i, label in enumerate(rows):
        y = padding + 18 + i * row_h

        if label == "Recommended zone":
            swatch = Image.new("RGB", (16, 16), (240, 250, 240))
            sd     = ImageDraw.Draw(swatch)
            for off in range(-16, 32, HATCH_SPACING):
                sd.line([(0, off), (16, off + 16)],
                        fill=HATCH_COLOR, width=HATCH_THICKNESS)
            legend.paste(swatch, (padding, y))
            draw.text((padding + 22, y + 1),
                      "Recommended install zone", fill=(60, 80, 60))
        else:
            draw.rectangle(
                [padding, y, padding + 16, y + 16],
                fill=SEGMENT_PALETTE[label],
            )
            draw.text((padding + 22, y + 1), label, fill=(60, 80, 60))

    return legend




def build_segment_summary(segments: List[Dict]) -> List[Dict]:
    """
    Build a clean summary list for pd.DataFrame display in Streamlit.

    Returns:
        list of row dicts
    """
    return [
        {
            "Rank":                    seg.get("rank", "—"),
            "Segment":                 f"Roof #{seg['id']}",
            "Area (m²)":               seg["area_m2"],
            "Usable (m²)":             seg.get("usable_area_m2", "—"),
            "Score":                   seg.get("composite_score", 0.0),
            "Class":                   seg.get("suitability_class", "—"),
            "Install Area (m²)":       seg.get("fill_area_m2", 0.0),
            "Est. Production (kWh/yr)":round(seg.get("fill_production", 0.0), 1),
            "Status":                  seg.get("allocation_status", "—"),
        }
        for seg in segments
    ]



__version__ = "2.0.0"
__author__  = "EcoPower Roof Team — GEO-AI Twinverse Research Group, UGM"
__references__ = [
    "Burke (2021) — ASF factors",
    "Res4Africa (2024) — Building type ratios & usable factor",
    "Jakubiec & Reinhart (2013) — Min viable area 4 m²",
    "Duffie & Beckman (2013) — Tilt efficiency",
    "E3S Conferences (2024) — AHP weights CR=0.04",
    "Bandung MCRS (2022) — 5-tier suitability classification",
    "Tarigan et al. (2025) — PSH 4.5 h/day, PR 0.75",
]
