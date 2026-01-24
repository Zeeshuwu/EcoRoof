"""
Mathematical models for solar energy calculations
Based on Semarang research studies
"""

import numpy as np
from typing import Dict, Tuple

# Constants from research (Semarang, Indonesia)
PSH_SEMARANG = 5.0  # Peak Sun Hours per day
PERFORMANCE_RATIO = 0.80  # System efficiency
PANEL_POWER_WP = 400  # Watts peak per panel (typical)
PANEL_EFFICIENCY = 0.15  # kWp per m² (15%)
DAILY_CONSUMPTION_PER_BUILDING = 4.5  # kWh/day (from research)
CO2_REDUCTION_FACTOR = 0.5  # kg CO2 per kWh

def calculate_solar_production(
    n_panels: int,
    panel_power_wp: float = PANEL_POWER_WP,
    psh: float = PSH_SEMARANG,
    pr: float = PERFORMANCE_RATIO
) -> Dict[str, float]:
    """
    Calculate solar energy production from detected panels
    
    Based on research formula:
    E = N_panels × P_panel × PSH × PR
    
    Args:
        n_panels: Number of detected solar panels
        panel_power_wp: Power per panel in Watts peak (default: 400 Wp)
        psh: Peak Sun Hours per day (default: 5.0 for Semarang)
        pr: Performance Ratio (default: 0.80)
    
    Returns:
        Dictionary with production metrics
    
    Example:
        >>> calculate_solar_production(80)
        {
            'capacity_kwp': 32.0,
            'daily_kwh': 128.0,
            'annual_kwh': 46720.0,
            'carbon_reduction_tons': 23.36
        }
    """
    # Calculate installed capacity
    capacity_kwp = (n_panels * panel_power_wp) / 1000
    
    # Daily production
    daily_production_kwh = capacity_kwp * psh * pr
    
    # Annual production
    annual_production_kwh = daily_production_kwh * 365
    
    # Carbon reduction (0.5 kg CO2 per kWh)
    carbon_reduction_tons = (annual_production_kwh * CO2_REDUCTION_FACTOR) / 1000
    
    return {
        'n_panels': n_panels,
        'capacity_kwp': round(capacity_kwp, 2),
        'daily_kwh': round(daily_production_kwh, 2),
        'annual_kwh': round(annual_production_kwh, 2),
        'annual_mwh': round(annual_production_kwh / 1000, 2),
        'carbon_reduction_tons': round(carbon_reduction_tons, 2)
    }


def calculate_consumption(
    n_buildings: int,
    daily_per_building: float = DAILY_CONSUMPTION_PER_BUILDING
) -> Dict[str, float]:
    """
    Calculate total energy consumption for detected buildings
    
    Based on Semarang study: 4.5 kWh/day per building
    
    Args:
        n_buildings: Number of detected roofs/buildings
        daily_per_building: Daily consumption per building (default: 4.5 kWh)
    
    Returns:
        Dictionary with consumption metrics
    
    Example:
        >>> calculate_consumption(15)
        {
            'daily_kwh': 67.5,
            'annual_kwh': 24637.5
        }
    """
    daily_consumption_kwh = n_buildings * daily_per_building
    annual_consumption_kwh = daily_consumption_kwh * 365
    
    return {
        'n_buildings': n_buildings,
        'daily_per_building_kwh': daily_per_building,
        'daily_kwh': round(daily_consumption_kwh, 2),
        'annual_kwh': round(annual_consumption_kwh, 2),
        'annual_mwh': round(annual_consumption_kwh / 1000, 2)
    }


def calculate_solar_potential(
    roof_area_m2: float,
    panel_area_m2: float,
    panel_efficiency: float = PANEL_EFFICIENCY,
    psh: float = PSH_SEMARANG,
    pr: float = PERFORMANCE_RATIO
) -> Dict[str, float]:
    """
    Calculate potential solar capacity from unused roof area
    
    Args:
        roof_area_m2: Total detected roof area (m²)
        panel_area_m2: Currently installed panel area (m²)
        panel_efficiency: kWp per m² (default: 0.15)
        psh: Peak Sun Hours (default: 5.0)
        pr: Performance Ratio (default: 0.80)
    
    Returns:
        Dictionary with potential metrics
    
    Example:
        >>> calculate_solar_potential(1250, 85)
        {
            'available_area_m2': 1165.0,
            'potential_kwp': 174.75,
            'potential_daily_kwh': 699.0
        }
    """
    # Available roof area
    available_area_m2 = max(0, roof_area_m2 - panel_area_m2)
    
    # Potential capacity (kWp)
    potential_capacity_kwp = available_area_m2 * panel_efficiency
    
    # Potential daily production
    potential_daily_kwh = potential_capacity_kwp * psh * pr
    
    # Potential annual production
    potential_annual_kwh = potential_daily_kwh * 365
    
    # Number of panels that can be installed (assuming 2 m² per panel)
    potential_n_panels = int(available_area_m2 / 2)
    
    return {
        'total_roof_area_m2': round(roof_area_m2, 2),
        'used_area_m2': round(panel_area_m2, 2),
        'available_area_m2': round(available_area_m2, 2),
        'potential_kwp': round(potential_capacity_kwp, 2),
        'potential_n_panels': potential_n_panels,
        'potential_daily_kwh': round(potential_daily_kwh, 2),
        'potential_annual_kwh': round(potential_annual_kwh, 2),
        'potential_annual_mwh': round(potential_annual_kwh / 1000, 2)
    }


def calculate_coverage_analysis(
    production_daily_kwh: float,
    consumption_daily_kwh: float,
    potential_daily_kwh: float
) -> Dict[str, float]:
    """
    Analyze energy coverage and gaps
    
    Args:
        production_daily_kwh: Current daily production
        consumption_daily_kwh: Total daily consumption
        potential_daily_kwh: Potential daily production
    
    Returns:
        Dictionary with coverage analysis
    """
    # Current coverage
    if consumption_daily_kwh > 0:
        current_coverage_pct = (production_daily_kwh / consumption_daily_kwh) * 100
    else:
        current_coverage_pct = 0
    
    # Energy gap/surplus
    energy_gap_kwh = consumption_daily_kwh - production_daily_kwh
    
    # Potential coverage
    total_potential_production = production_daily_kwh + potential_daily_kwh
    if consumption_daily_kwh > 0:
        potential_coverage_pct = (total_potential_production / consumption_daily_kwh) * 100
    else:
        potential_coverage_pct = 0
    
    # Status
    if current_coverage_pct >= 100:
        status = "✅ Self-sufficient (Surplus)"
    elif current_coverage_pct >= 75:
        status = "⚠️ Partial coverage (Good)"
    elif current_coverage_pct >= 50:
        status = "⚠️ Partial coverage (Moderate)"
    else:
        status = "❌ Insufficient coverage"
    
    return {
        'current_coverage_pct': round(current_coverage_pct, 2),
        'energy_gap_kwh': round(energy_gap_kwh, 2),
        'energy_gap_annual_kwh': round(energy_gap_kwh * 365, 2),
        'potential_coverage_pct': round(potential_coverage_pct, 2),
        'status': status,
        'is_sufficient': current_coverage_pct >= 100,
        'surplus_kwh': round(-energy_gap_kwh if energy_gap_kwh < 0 else 0, 2)
    }


def estimate_panel_area_from_count(n_panels: int, panel_size_m2: float = 2.0) -> float:
    """
    Estimate total panel area from panel count
    
    Args:
        n_panels: Number of panels
        panel_size_m2: Area per panel (default: 2 m²)
    
    Returns:
        Total panel area in m²
    """
    return n_panels * panel_size_m2


def estimate_roof_area_from_pixels(
    mask: np.ndarray,
    pixel_to_meter_ratio: float = 0.5
) -> float:
    """
    Convert detected roof pixels to area in m²
    
    Args:
        mask: Binary mask from SAM3 detection
        pixel_to_meter_ratio: Conversion ratio (default: 0.5 m/pixel)
    
    Returns:
        Area in m²
    """
    pixel_count = np.sum(mask > 0)
    area_m2 = pixel_count * (pixel_to_meter_ratio ** 2)
    return area_m2
