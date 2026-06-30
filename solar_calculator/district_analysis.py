# solar_calculator/district_analysis.py


from typing import Optional


# ─────────────────────────────────────────────
# DISTRICT CLASSIFICATION THRESHOLDS
# ─────────────────────────────────────────────
# Source: Bandung Multi-Criteria Rooftop Solar Study (2022)

DISTRICT_THRESHOLDS = {
    "Very High": (0.70, 1.00),
    "High":      (0.50, 0.70),
    "Medium":    (0.35, 0.50),
    "Low":       (0.20, 0.35),
    "Very Low":  (0.00, 0.20),
}

DISTRICT_COLORS = {
    "Very High": "#047857",
    "High":      "#10b981",
    "Medium":    "#f59e0b",
    "Low":       "#ef4444",
    "Very Low":  "#6b7280",
}

# Central Java / Semarang irradiance
# Source: Global Solar Atlas (2024)
JAVA_GHI_KWH_M2_DAY = 4.8

# IEA PVPS Trends Report (2024) — monocrystalline panel efficiency
PANEL_EFFICIENCY     = 0.20

# Duffie & Beckman (2013) — Performance Ratio tropical climate
PERFORMANCE_RATIO    = 0.80

# W/m² standard (IEA PVPS 2024)
PANEL_WATT_PER_M2    = 200.0


def classify_district(usability_ratio: float) -> str:
    """Classify district solar suitability from avg usability ratio."""
    for cls, (lo, hi) in DISTRICT_THRESHOLDS.items():
        if lo <= usability_ratio <= hi:
            return cls
    return "Very Low"


def compute_district_analysis(
    zone_name: str,
    gross_area_m2: float,
    usable_area_m2: float,
    existing_panel_area_m2: float,
    geotiff_handler=None,
    pln_tariff: float = 1_699.53,
) -> dict:
    """
    Compute district-scale solar potential for a single analysis zone.

    Since no admin boundary is available, the GeoTIFF bounding box
    (or image dimensions) defines the zone boundary.

    Args:
        zone_name              : Label for the zone (filename or CRS-derived)
        gross_area_m2          : Total detected roof area
        usable_area_m2         : Technically usable area (after reductions)
        existing_panel_area_m2 : Already-installed panel area
        geotiff_handler        : GeoTIFFHandler for bbox/CRS metadata
        pln_tariff             : IDR/kWh (Permen ESDM No.3/2024)

    Returns:
        dict with zone metadata, potential estimates, and classification
    """
    usability_ratio = (
        usable_area_m2 / gross_area_m2 if gross_area_m2 > 0 else 0.0
    )
    suitability_class = classify_district(usability_ratio)

    # Remaining installable area (exclude existing panels)
    remaining_area_m2 = max(usable_area_m2 - existing_panel_area_m2, 0.0)

    # Capacity estimates
    existing_capacity_kwp  = (existing_panel_area_m2 * PANEL_WATT_PER_M2) / 1000.0
    potential_capacity_kwp = (usable_area_m2 * PANEL_WATT_PER_M2) / 1000.0
    remaining_capacity_kwp = (remaining_area_m2 * PANEL_WATT_PER_M2) / 1000.0

    # Annual energy estimates
    # Formula: E = A × η × PR × H × 365
    def annual_energy(area):
        return area * PANEL_EFFICIENCY * PERFORMANCE_RATIO * JAVA_GHI_KWH_M2_DAY * 365

    existing_annual_kwh  = annual_energy(existing_panel_area_m2)
    potential_annual_kwh = annual_energy(usable_area_m2)
    remaining_annual_kwh = annual_energy(remaining_area_m2)

    # Revenue estimates
    existing_revenue_idr  = existing_annual_kwh  * pln_tariff
    potential_revenue_idr = potential_annual_kwh * pln_tariff
    remaining_revenue_idr = remaining_annual_kwh * pln_tariff

    # GeoTIFF zone metadata
    zone_meta = _extract_zone_meta(geotiff_handler)

    return {
        # Zone identity
        "zone_name":              zone_name,
        "zone_crs":               zone_meta["crs"],
        "zone_bbox":              zone_meta["bbox"],
        "zone_area_m2":           zone_meta["zone_area_m2"],

        # Roof areas
        "gross_roof_area_m2":     round(gross_area_m2, 2),
        "usable_area_m2":         round(usable_area_m2, 2),
        "existing_panel_area_m2": round(existing_panel_area_m2, 2),
        "remaining_area_m2":      round(remaining_area_m2, 2),
        "usability_ratio":        round(usability_ratio, 4),

        # Classification
        "suitability_class":      suitability_class,
        "suitability_color":      DISTRICT_COLORS[suitability_class],

        # Existing system
        "existing_capacity_kwp":  round(existing_capacity_kwp, 2),
        "existing_annual_kwh":    round(existing_annual_kwh, 2),
        "existing_revenue_idr":   round(existing_revenue_idr, 0),

        # Full potential (all usable area)
        "potential_capacity_kwp": round(potential_capacity_kwp, 2),
        "potential_annual_kwh":   round(potential_annual_kwh, 2),
        "potential_revenue_idr":  round(potential_revenue_idr, 0),

        # Remaining (expansion opportunity)
        "remaining_capacity_kwp": round(remaining_capacity_kwp, 2),
        "remaining_annual_kwh":   round(remaining_annual_kwh, 2),
        "remaining_revenue_idr":  round(remaining_revenue_idr, 0),

        "references": [
            "Bandung Multi-Criteria Rooftop Solar Study (2022) — 5-class district tiers",
            "IEA PVPS Trends Report (2024) — panel efficiency 20%",
            "Duffie & Beckman (2013) — Performance Ratio 0.80",
            "Global Solar Atlas (2024) — Central Java GHI 4.8 kWh/m²/day",
            "Permen ESDM No.3/2024 — PLN tariff B2: IDR 1,699.53/kWh",
        ],
    }


def _extract_zone_meta(geotiff_handler) -> dict:
    """Extract zone boundary metadata from GeoTIFF handler if available."""
    if geotiff_handler and getattr(geotiff_handler, "is_geotiff", False):
        meta = geotiff_handler.metadata
        bbox = meta.get("bbox", None)
        if bbox:
            left, bottom, right, top = bbox
            zone_area = abs((right - left) * (top - bottom))
        else:
            zone_area = meta.get("area_m2", 0.0)
        return {
            "crs":          meta.get("crs", "Unknown"),
            "bbox":         bbox,
            "zone_area_m2": round(zone_area, 2),
        }
    return {
        "crs":          "Pixel coordinates (no GeoTIFF)",
        "bbox":         None,
        "zone_area_m2": 0.0,
    }
