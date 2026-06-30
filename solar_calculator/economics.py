"""
Economic analysis for solar PV installations in Indonesia
Based on peer-reviewed research (2020-2025)

References:
[1] Tarigan, E., Kartikasari, F. D., Indrawati, V., & Irawati, F. (2025).
    Sustainability assessment of residential grid-connected monocrystalline
    module solar PV systems in three major cities in Indonesia: A life cycle
    assessment perspective. Future Cities and Environment, 11, 1-13.
    https://doi.org/10.70917/fce-2025-003

[2] Cantiqa, S. P., & Dirkareshza, R. (2025).
    Reformulation of renewable energy incentives: A normative review of the
    implementation of limited-quota feed-in tariffs in Indonesia.
    Walisongo Law Review, 7(2), 389-412.
    https://doi.org/10.21580/walrev.2025.7.2.28425

[3] Suparwoko, & Qamar, F. A. (2022).
    Techno-economic analysis of rooftop solar power plant implementation
    and policy on mosques: An Indonesian case study.
    Scientific Reports, 12(1), Article 4823.
    https://doi.org/10.1038/s41598-022-08968-6

[4] Kunaifi, K., Reinders, A. H. M. E., Lindig, S., Jaeger, M., & Moser, D. (2020).
    Operational performance and degradation of PV systems consisting of six
    technologies in three climates.
    Applied Sciences, 10(16), Article 5412.
    https://doi.org/10.3390/app10165412

[5] Silalahi, D. F., Blakers, A., & Cheng, C. (2024).
    100% renewable electricity in Indonesia.
    Energies, 17(1), Article 3.
    https://doi.org/10.3390/en17010003

Author  : EcoPower Roof Team - GEO-AI Twinverse Research Group, Faculty of Engineering, Universitas Gadjah Mada
Version : 3.0.0
Date    : June 2026
"""

from typing import Dict, List


# ============================================================================
# ECONOMIC CONSTANTS (INDONESIA, 2022-2025)
# ============================================================================

# Installation Cost per kWp
# Source: [3] Suparwoko & Qamar (2022) — IDR 15-24.5M/kWp range in Indonesia
COST_PER_KWP = 17_000_000  # Rp 17,000,000 per kWp installed

# PLN Electricity Tariff
# Source: [2] Cantiqa & Dirkareshza (2025) — actual residential rate IDR 1,699.53/kWh
ELECTRICITY_RATE = 1_500  # Rp 1,500 per kWh (PLN residential, conservative)

# Annual Operation & Maintenance Cost
# Standard international practice: 1-2% of initial investment (IRENA/IEA)
# Applied in Indonesian context per [3] Suparwoko & Qamar (2022)
MAINTENANCE_COST_ANNUAL_PCT = 0.01  # 1% of investment per year

# System Operational Lifetime
# Source: [1] Tarigan et al. (2025) — LCA study baseline 25 years (range 20-30 yr)
SYSTEM_LIFETIME_YEARS = 25  # years

# System Performance Ratio
# Source: [4] Kunaifi et al. (2020) — range 75-92% in tropical Indonesian climate
# Source: [1] Tarigan et al. (2025) — 0.75 used as conservative residential baseline
PERFORMANCE_RATIO = 0.75  # 75% (conservative)

# Peak Sun Hours
# Source: [1] Tarigan et al. (2025) — average for Jakarta, Surabaya, Medan
PEAK_SUN_HOURS = 4.5  # hours per day

# Discount Rate for NPV
# Approximation based on Bank Indonesia BI Rate 2024
DISCOUNT_RATE = 0.05  # 5% per year

# Carbon Emission Factor
# Source: PLN grid emission factor for Indonesian national grid
CO2_EMISSION_FACTOR = 0.85  # kg CO2 per kWh

# Energy Payback Time Range (validation reference)
# Source: [1] Tarigan et al. (2025) — residential PV systems, 2-15 kWp
ENERGY_PAYBACK_TIME_RANGE = (6.88, 8.10)  # years


# ============================================================================
# CORE ECONOMIC FUNCTIONS
# ============================================================================

def calculate_economics(
    capacity_kwp: float,
    annual_production_kwh: float,
    cost_per_kwp: float = COST_PER_KWP,
    electricity_rate: float = ELECTRICITY_RATE,
    maintenance_pct: float = MAINTENANCE_COST_ANNUAL_PCT,
    lifetime_years: int = SYSTEM_LIFETIME_YEARS,
    discount_rate: float = DISCOUNT_RATE,
) -> Dict[str, float]:
    """
    Calculate comprehensive economic metrics for a solar PV installation.

    Methodology:
    - NPV and payback period : Suparwoko & Qamar (2022) [3]
    - LCA cost framework     : Tarigan et al. (2025) [1]
    - LCOE formula           : standard engineering economics

    Args:
        capacity_kwp         : Solar system capacity in kWp
        annual_production_kwh: Annual energy production in kWh
        cost_per_kwp         : Installation cost per kWp (default: IDR 17,000,000)
        electricity_rate     : PLN electricity rate in IDR/kWh (default: 1,500)
        maintenance_pct      : Annual O&M cost as fraction of investment (default: 0.01)
        lifetime_years       : System operational lifetime in years (default: 25)
        discount_rate        : Annual discount rate for NPV (default: 0.05)

    Returns:
        dict with investment, savings, payback, ROI, NPV, LCOE, CO₂ metrics.
    """

    # Investment
    investment_rp         = capacity_kwp * cost_per_kwp

    # Annual cash flows
    annual_savings_rp     = annual_production_kwh * electricity_rate
    annual_maintenance_rp = investment_rp * maintenance_pct
    net_annual_savings_rp = annual_savings_rp - annual_maintenance_rp

    # Simple Payback Period
    payback_years = (
        investment_rp / net_annual_savings_rp
        if net_annual_savings_rp > 0 else float("inf")
    )

    # Lifetime Savings & ROI
    total_savings_lifetime_rp = net_annual_savings_rp * lifetime_years
    roi_lifetime_pct = (
        ((total_savings_lifetime_rp / investment_rp) - 1) * 100
        if investment_rp > 0 else 0.0
    )

    # Net Present Value
    npv_rp = -investment_rp
    for year in range(1, lifetime_years + 1):
        npv_rp += net_annual_savings_rp / ((1 + discount_rate) ** year)

    # Levelized Cost of Energy
    total_lifetime_costs      = investment_rp + (annual_maintenance_rp * lifetime_years)
    total_lifetime_production = annual_production_kwh * lifetime_years
    lcoe_rp_per_kwh = (
        total_lifetime_costs / total_lifetime_production
        if total_lifetime_production > 0 else 0.0
    )

    # Carbon
    annual_co2_reduction_kg     = annual_production_kwh * CO2_EMISSION_FACTOR
    lifetime_co2_reduction_tons = (annual_co2_reduction_kg * lifetime_years) / 1000

    return {
        # System
        "capacity_kwp":                       round(capacity_kwp, 2),
        # Investment
        "investment_rp":                      round(investment_rp, 0),
        "investment_million_rp":              round(investment_rp / 1_000_000, 2),
        "investment_billion_rp":              round(investment_rp / 1_000_000_000, 3),
        "cost_per_kwp":                       round(cost_per_kwp, 0),
        # Annual Economics
        "annual_production_kwh":              round(annual_production_kwh, 2),
        "annual_savings_rp":                  round(annual_savings_rp, 0),
        "annual_savings_million_rp":          round(annual_savings_rp / 1_000_000, 2),
        "annual_maintenance_rp":              round(annual_maintenance_rp, 0),
        "net_annual_savings_rp":              round(net_annual_savings_rp, 0),
        "net_annual_savings_million_rp":      round(net_annual_savings_rp / 1_000_000, 2),
        # Performance Metrics
        "payback_years":                      round(payback_years, 1),
        "total_savings_lifetime_rp":          round(total_savings_lifetime_rp, 0),
        "total_savings_lifetime_million_rp":  round(total_savings_lifetime_rp / 1_000_000, 2),
        "roi_lifetime_pct":                   round(roi_lifetime_pct, 1),
        "npv_rp":                             round(npv_rp, 0),
        "npv_million_rp":                     round(npv_rp / 1_000_000, 2),
        "npv_billion_rp":                     round(npv_rp / 1_000_000_000, 3),
        "lcoe_rp_per_kwh":                    round(lcoe_rp_per_kwh, 2),
        # Environmental
        "annual_co2_reduction_kg":            round(annual_co2_reduction_kg, 2),
        "annual_co2_reduction_tons":          round(annual_co2_reduction_kg / 1000, 2),
        "lifetime_co2_reduction_tons":        round(lifetime_co2_reduction_tons, 2),
        # Viability
        "is_economically_viable":             npv_rp > 0,
        "savings_to_investment_ratio":        round(
            total_savings_lifetime_rp / investment_rp, 2
        ) if investment_rp > 0 else 0.0,
    }


def calculate_investment_scenarios(
    available_area_m2: float,
    panel_efficiency: float = 0.20,
    scenarios: List[float] = [0.25, 0.50, 0.75, 1.0],
    cost_per_kwp: float = COST_PER_KWP,
    electricity_rate: float = ELECTRICITY_RATE,
) -> List[Dict]:
    """
    Calculate economic metrics for multiple roof utilization scenarios.

    Scenario analysis: Suparwoko & Qamar (2022) [3].
    Formula: E = P_kwp × PSH × PR × 365
    PSH = 4.5 h/day [1], PR = 0.75 [1][4].

    Args:
        available_area_m2: Available roof area in m²
        panel_efficiency : kWp/m² (default 0.20 monocrystalline)
        scenarios        : Utilization fractions (default 25/50/75/100%)
        cost_per_kwp     : IDR/kWp
        electricity_rate : PLN tariff IDR/kWh

    Returns:
        list[dict] — one dict per scenario with full economic metrics.
    """
    results = []

    for pct in scenarios:
        area_used_m2          = available_area_m2 * pct
        capacity_kwp          = area_used_m2 * panel_efficiency
        annual_production_kwh = capacity_kwp * PEAK_SUN_HOURS * PERFORMANCE_RATIO * 365

        eco = calculate_economics(
            capacity_kwp,
            annual_production_kwh,
            cost_per_kwp,
            electricity_rate,
        )

        results.append({
            "scenario_name":              f"{int(pct * 100)}% Utilization",
            "utilization_pct":            round(pct * 100, 1),
            "area_used_m2":               round(area_used_m2, 2),
            "capacity_kwp":               eco["capacity_kwp"],
            "annual_production_kwh":      eco["annual_production_kwh"],
            "investment_million_rp":      eco["investment_million_rp"],
            "investment_billion_rp":      eco["investment_billion_rp"],
            "annual_savings_million_rp":  eco["annual_savings_million_rp"],
            "payback_years":              eco["payback_years"],
            "roi_lifetime_pct":           eco["roi_lifetime_pct"],
            "npv_million_rp":             eco["npv_million_rp"],
            "npv_billion_rp":             eco["npv_billion_rp"],
            "lcoe_rp_per_kwh":            eco["lcoe_rp_per_kwh"],
            "annual_co2_reduction_tons":  eco["annual_co2_reduction_tons"],
            "is_viable":                  eco["is_economically_viable"],
        })

    return results


def compare_with_pln(
    annual_consumption_kwh: float,
    annual_production_kwh: float,
    electricity_rate: float = ELECTRICITY_RATE,
) -> Dict[str, float]:
    """
    Compare solar PV cost vs full PLN grid electricity cost.

    Policy context: Net-metering abolished 2024 via PERMEN ESDM No. 2/2024.
    Reference: Cantiqa & Dirkareshza (2025) [2].

    Args:
        annual_consumption_kwh: Annual building energy consumption in kWh
        annual_production_kwh : Annual solar PV production in kWh
        electricity_rate      : PLN tariff IDR/kWh

    Returns:
        dict with cost comparison, savings, energy balance, self-sufficiency.
    """
    annual_pln_cost_rp = annual_consumption_kwh * electricity_rate
    energy_balance_kwh = annual_production_kwh - annual_consumption_kwh

    if energy_balance_kwh < 0:
        deficit_kwh               = abs(energy_balance_kwh)
        annual_cost_with_solar_rp = deficit_kwh * electricity_rate
        surplus_kwh               = 0.0
    else:
        annual_cost_with_solar_rp = 0.0
        deficit_kwh               = 0.0
        surplus_kwh               = energy_balance_kwh

    annual_savings_rp = annual_pln_cost_rp - annual_cost_with_solar_rp

    savings_pct = (
        (annual_savings_rp / annual_pln_cost_rp) * 100
        if annual_pln_cost_rp > 0 else 0.0
    )
    self_sufficiency_pct = (
        min(100.0, (annual_production_kwh / annual_consumption_kwh) * 100)
        if annual_consumption_kwh > 0 else 0.0
    )

    return {
        "annual_pln_cost_rp":                round(annual_pln_cost_rp, 0),
        "annual_pln_cost_million_rp":        round(annual_pln_cost_rp / 1_000_000, 2),
        "annual_cost_with_solar_rp":         round(annual_cost_with_solar_rp, 0),
        "annual_cost_with_solar_million_rp": round(annual_cost_with_solar_rp / 1_000_000, 2),
        "annual_savings_rp":                 round(annual_savings_rp, 0),
        "annual_savings_million_rp":         round(annual_savings_rp / 1_000_000, 2),
        "savings_pct":                       round(savings_pct, 1),
        "deficit_kwh":                       round(deficit_kwh, 2),
        "surplus_kwh":                       round(surplus_kwh, 2),
        "energy_balance_kwh":                round(energy_balance_kwh, 2),
        "self_sufficiency_pct":              round(self_sufficiency_pct, 1),
        "is_self_sufficient":                self_sufficiency_pct >= 100,
        "electricity_rate_used":             electricity_rate,
    }


def calculate_energy_roi(
    capacity_kwp: float,
    annual_production_kwh: float,
) -> Dict[str, float]:
    """
    Calculate Energy Return on Investment (EROI) and energy payback time.

    Source: Tarigan et al. (2025) [1]
    - Energy ROI for Indonesian residential PV: 2.98–4.01
    - Energy payback time: 6.88–8.10 years

    Args:
        capacity_kwp         : System capacity in kWp
        annual_production_kwh: Annual energy production in kWh

    Returns:
        dict with embodied energy, payback time, EROI, literature validation.
    """
    EMBODIED_ENERGY_PER_KWP   = 5_000  # kWh/kWp — Tarigan et al. (2025) [1]
    total_embodied_energy_kwh = capacity_kwp * EMBODIED_ENERGY_PER_KWP

    energy_payback_time = (
        total_embodied_energy_kwh / annual_production_kwh
        if annual_production_kwh > 0 else float("inf")
    )
    lifetime_production_kwh = annual_production_kwh * SYSTEM_LIFETIME_YEARS
    energy_roi = (
        lifetime_production_kwh / total_embodied_energy_kwh
        if total_embodied_energy_kwh > 0 else 0.0
    )

    return {
        "embodied_energy_kwh":       round(total_embodied_energy_kwh, 2),
        "energy_payback_time_years": round(energy_payback_time, 2),
        "energy_roi":                round(energy_roi, 2),
        "lifetime_production_kwh":   round(lifetime_production_kwh, 2),
        "is_within_literature_range": (
            ENERGY_PAYBACK_TIME_RANGE[0]
            <= energy_payback_time
            <= ENERGY_PAYBACK_TIME_RANGE[1]
        ),
        "literature_range_years":    ENERGY_PAYBACK_TIME_RANGE,
    }


# ============================================================================
# CAPPED ECONOMICS — 100% SELF-SUFFICIENCY TARGET
# ============================================================================

def calculate_economics_capped(
    consumption_annual_kwh: float,
    usable_area_m2: float,
    panel_efficiency: float = 0.20,
    cost_per_kwp: float = COST_PER_KWP,
    electricity_rate: float = ELECTRICITY_RATE,
    peak_sun_hours: float = PEAK_SUN_HOURS,
    performance_ratio: float = PERFORMANCE_RATIO,
) -> Dict:
    """
    Calculate economics capped at the area needed to reach 100% self-sufficiency.

    Problem this solves:
        When usable roof area is large (e.g. industrial estate), the full-potential
        economics show unrealistically massive investment figures because they
        multiply the entire available area. This function instead calculates the
        MINIMUM area required to cover consumption, then caps there.

    Returns BOTH:
        A) economics of currently detected panels  → passed in separately via app
        B) 'to_100pct'    — cost to reach exactly 100% self-sufficiency
        C) 'max_potential' — full usable area economics (reference only)

    Formula (rearranged from E = A × η × PR × PSH × 365):
        A_needed = E_consumption / (η × PR × PSH × 365)

    Sources:
        - Tarigan et al. (2025) [1] — PSH 4.5 h/day, PR 0.75
        - Suparwoko & Qamar (2022) [3] — NPV/payback methodology
        - Kunaifi et al. (2020) [4] — Performance Ratio tropical climate

    Args:
        consumption_annual_kwh : Annual energy consumption to cover (kWh)
        usable_area_m2         : Technically usable roof area available (m²)
        panel_efficiency       : kWp per m² (default 0.20 monocrystalline)
        cost_per_kwp           : IDR per kWp installed
        electricity_rate       : PLN tariff IDR/kWh
        peak_sun_hours         : h/day (Tarigan et al. 2025)
        performance_ratio      : system PR (Kunaifi et al. 2020)

    Returns:
        dict with keys:
            consumption_annual_kwh
            area_needed_for_100pct_m2
            area_available_m2
            area_used_for_100pct_m2
            is_feasible
            actual_coverage_pct
            to_100pct      → full economics dict (calculate_economics output)
            max_potential  → full economics dict for reference
    """

    # ── Area needed to produce exactly consumption_annual_kwh ────────────────
    denominator = panel_efficiency * performance_ratio * peak_sun_hours * 365

    area_needed_m2 = (
        consumption_annual_kwh / denominator
        if denominator > 0 and consumption_annual_kwh > 0
        else 0.0
    )

    # Cap: cannot exceed available usable area
    area_for_100pct = min(area_needed_m2, usable_area_m2)
    is_feasible     = area_needed_m2 <= usable_area_m2

    # ── Economics for 100% self-sufficiency target ────────────────────────────
    capacity_100pct_kwp   = area_for_100pct * panel_efficiency
    production_100pct_kwh = (
        capacity_100pct_kwp * peak_sun_hours * performance_ratio * 365
    )
    eco_100pct = calculate_economics(
        capacity_100pct_kwp,
        production_100pct_kwh,
        cost_per_kwp,
        electricity_rate,
    )

    # ── Economics for full potential (all usable area — reference only) ───────
    capacity_max_kwp   = usable_area_m2 * panel_efficiency
    production_max_kwh = (
        capacity_max_kwp * peak_sun_hours * performance_ratio * 365
    )
    eco_max = calculate_economics(
        capacity_max_kwp,
        production_max_kwh,
        cost_per_kwp,
        electricity_rate,
    )

    # ── Actual coverage at 100% target ────────────────────────────────────────
    actual_coverage_pct = (
        min(production_100pct_kwh / consumption_annual_kwh, 1.0) * 100
        if consumption_annual_kwh > 0 else 0.0
    )

    return {
        # Target metadata
        "consumption_annual_kwh":      round(consumption_annual_kwh, 2),
        "area_needed_for_100pct_m2":   round(area_needed_m2, 2),
        "area_available_m2":           round(usable_area_m2, 2),
        "area_used_for_100pct_m2":     round(area_for_100pct, 2),
        "is_feasible":                 is_feasible,
        "actual_coverage_pct":         round(actual_coverage_pct, 1),

        # B) To reach 100% self-sufficiency
        "to_100pct": {
            "capacity_kwp":                  eco_100pct["capacity_kwp"],
            "annual_production_kwh":         eco_100pct["annual_production_kwh"],
            "investment_rp":                 eco_100pct["investment_rp"],
            "investment_million_rp":         eco_100pct["investment_million_rp"],
            "investment_billion_rp":         eco_100pct["investment_billion_rp"],
            "annual_savings_rp":             eco_100pct["annual_savings_rp"],
            "annual_savings_million_rp":     eco_100pct["annual_savings_million_rp"],
            "net_annual_savings_million_rp": eco_100pct["net_annual_savings_million_rp"],
            "payback_years":                 eco_100pct["payback_years"],
            "roi_lifetime_pct":              eco_100pct["roi_lifetime_pct"],
            "npv_rp":                        eco_100pct["npv_rp"],
            "npv_million_rp":                eco_100pct["npv_million_rp"],
            "npv_billion_rp":                eco_100pct["npv_billion_rp"],
            "lcoe_rp_per_kwh":               eco_100pct["lcoe_rp_per_kwh"],
            "annual_co2_reduction_tons":     eco_100pct["annual_co2_reduction_tons"],
            "lifetime_co2_reduction_tons":   eco_100pct["lifetime_co2_reduction_tons"],
            "is_economically_viable":        eco_100pct["is_economically_viable"],
        },

        # C) Full potential — reference only, not recommended target
        "max_potential": {
            "capacity_kwp":              eco_max["capacity_kwp"],
            "annual_production_kwh":     eco_max["annual_production_kwh"],
            "investment_billion_rp":     eco_max["investment_billion_rp"],
            "annual_savings_million_rp": eco_max["annual_savings_million_rp"],
            "net_annual_savings_million_rp": eco_max["net_annual_savings_million_rp"],
            "payback_years":             eco_max["payback_years"],
            "roi_lifetime_pct":          eco_max["roi_lifetime_pct"],
            "npv_billion_rp":            eco_max["npv_billion_rp"],
            "lcoe_rp_per_kwh":           eco_max["lcoe_rp_per_kwh"],
            "annual_co2_reduction_tons": eco_max["annual_co2_reduction_tons"],
            "is_economically_viable":    eco_max["is_economically_viable"],
        },
    }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_currency_idr(amount: float) -> str:
    """Format IDR value with readable denomination."""
    if amount >= 1_000_000_000:
        return f"Rp {amount / 1_000_000_000:.2f} Miliar"
    elif amount >= 1_000_000:
        return f"Rp {amount / 1_000_000:.2f} Juta"
    return f"Rp {amount:,.0f}"


def validate_economics_parameters(
    cost_per_kwp: float,
    electricity_rate: float,
    lifetime_years: int,
) -> Dict:
    """
    Validate input parameters against literature-supported ranges.

    Sources:
    - Cost range  : Suparwoko & Qamar (2022) [3]
    - Rate range  : Cantiqa & Dirkareshza (2025) [2]
    - Lifetime    : Tarigan et al. (2025) [1]
    """
    COST_RANGE     = (15_000_000, 25_000_000)
    RATE_RANGE     = (1_352, 1_700)
    LIFETIME_RANGE = (20, 30)

    return {
        "cost_valid":           COST_RANGE[0]     <= cost_per_kwp     <= COST_RANGE[1],
        "rate_valid":           RATE_RANGE[0]     <= electricity_rate <= RATE_RANGE[1],
        "lifetime_valid":       LIFETIME_RANGE[0] <= lifetime_years   <= LIFETIME_RANGE[1],
        "cost_range_idr":       COST_RANGE,
        "rate_range_idr":       RATE_RANGE,
        "lifetime_range_years": LIFETIME_RANGE,
    }


# ============================================================================
# MODULE METADATA
# ============================================================================

__version__    = "3.0.0"
__author__     = "EcoPower Roof Team — Magister Teknik Geomatika, UGM"
__references__ = [
    "[1] Tarigan et al. (2025) — https://doi.org/10.70917/fce-2025-003",
    "[2] Cantiqa & Dirkareshza (2025) — https://doi.org/10.21580/walrev.2025.7.2.28425",
    "[3] Suparwoko & Qamar (2022) — https://doi.org/10.1038/s41598-022-08968-6",
    "[4] Kunaifi et al. (2020) — https://doi.org/10.3390/app10165412",
    "[5] Silalahi et al. (2024) — https://doi.org/10.3390/en17010003",
]
