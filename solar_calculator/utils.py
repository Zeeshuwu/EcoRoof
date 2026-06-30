"""
Solar Calculator Utilities
Enhanced with area-based calculations and GeoTIFF support
"""

import numpy as np
from typing import Dict, Optional
import numpy as np
from typing import Any


def pixels_to_area(pixel_count: int, pixel_to_meter: float) -> float:
    """Convert pixel count to area in m²"""
    return pixel_count * (pixel_to_meter ** 2)

def convert_to_serializable(obj: Any) -> Any:
    """
    Recursively convert NumPy / non-JSON-serializable types to native Python.
    Handles: np.integer, np.floating, np.bool_, np.ndarray, dict, list.
    
    Required for json.dumps() on data that passed through NumPy calculations.
    """
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None   # JSON doesn't support NaN/Inf
    else:
        return obj

def count_objects_in_mask(mask: np.ndarray) -> int:
    """Count distinct objects in binary mask"""
    import cv2
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    return len(contours)


def format_currency(amount: float, currency: str = "Rp") -> str:
    """Format currency with thousand separators"""
    return f"{currency} {amount:,.0f}"


def format_energy(kwh: float) -> str:
    """Format energy with appropriate unit"""
    if kwh >= 1000:
        return f"{kwh/1000:.2f} MWh"
    return f"{kwh:.2f} kWh"


def estimate_image_scale(image_width: int, image_height: int, 
                         known_dimension_m: float = None) -> float:
    """
    Estimate pixel-to-meter ratio from image dimensions
    
    Args:
        image_width: Image width in pixels
        image_height: Image height in pixels
        known_dimension_m: Known real-world dimension in meters
    
    Returns:
        Estimated pixel-to-meter ratio
    """
    if known_dimension_m:
        # Use the larger dimension
        max_pixels = max(image_width, image_height)
        return known_dimension_m / max_pixels
    
    # Default estimation (assume ~100m coverage for typical aerial image)
    return 100.0 / max(image_width, image_height)


def estimate_panel_area_from_count(panel_count: int, 
                                   panel_size_m2: float = 2.0) -> float:
    """
    Estimate total panel area from panel count
    
    Args:
        panel_count: Number of panels detected
        panel_size_m2: Size of individual panel in m²
    
    Returns:
        Total panel area in m²
    """
    return panel_count * panel_size_m2


# ============================================
# NEW: AREA-BASED CALCULATIONS (From Research)
# ============================================

# Solar panel specifications based on Magelang research
SOLAR_PANEL_SPECS = {
    'standard': {
        'name': 'Standard Monocrystalline (400-450W)',
        'power_density_wp_m2': 200,
        'panel_size_m2': 2.0,
        'panel_power_wp': 400,
        'efficiency': 0.20
    },
    'high_efficiency': {
        'name': 'High Efficiency (500-550W)',
        'power_density_wp_m2': 250,  # Based on FT 3 Building
        'panel_size_m2': 2.2,
        'panel_power_wp': 540,
        'efficiency': 0.23
    },
    'budget': {
        'name': 'Budget Polycrystalline (300-350W)',
        'power_density_wp_m2': 170,
        'panel_size_m2': 1.8,
        'panel_power_wp': 300,
        'efficiency': 0.17
    }
}

# Magelang solar irradiance data
MAGELANG_SOLAR_DATA = {
    'peak_sun_hours': 4.5,
    'annual_irradiance_kwh_m2': 1642,
    'system_efficiency': 0.75,
    'location': 'Magelang, Central Java'
}


def calculate_area_from_mask(mask: np.ndarray, 
                             pixel_to_meter: float,
                             geotiff_handler=None) -> float:
    """
    Calculate area from mask with GeoTIFF support
    
    Args:
        mask: Binary mask (numpy array)
        pixel_to_meter: Manual pixel-to-meter ratio
        geotiff_handler: Optional GeoTIFF handler for accurate measurements
    
    Returns:
        Area in m²
    """
    pixel_count = np.sum(mask > 0)
    
    # Use GeoTIFF accurate calculation if available
    if geotiff_handler and hasattr(geotiff_handler, 'is_geotiff') and geotiff_handler.is_geotiff:
        return geotiff_handler.calculate_area_m2(pixel_count)
    
    # Fallback to manual ratio
    return pixels_to_area(pixel_count, pixel_to_meter)


def calculate_capacity_from_area(panel_area_m2: float, 
                                 panel_type: str = 'standard') -> Dict:
    """
    Calculate solar capacity from detected panel area
    
    Args:
        panel_area_m2: Total area of detected solar panels (m²)
        panel_type: Type of panel ('standard', 'high_efficiency', 'budget')
    
    Returns:
        dict with capacity calculations
    """
    panel_specs = SOLAR_PANEL_SPECS.get(panel_type, SOLAR_PANEL_SPECS['standard'])
    power_density = panel_specs['power_density_wp_m2']
    
    # Calculate installed capacity
    capacity_wp = panel_area_m2 * power_density
    capacity_kwp = capacity_wp / 1000
    
    # Estimate number of panels (for reference)
    estimated_panel_count = panel_area_m2 / panel_specs['panel_size_m2']
    
    return {
        'capacity_wp': capacity_wp,
        'capacity_kwp': capacity_kwp,
        'panel_area_m2': panel_area_m2,
        'estimated_panel_count': int(estimated_panel_count),
        'power_density': power_density,
        'panel_specs': panel_specs
    }


def calculate_energy_production_from_area(panel_area_m2: float,
                                         panel_type: str = 'standard',
                                         degradation_rate: float = 0.005) -> Dict:
    """
    Calculate energy production from panel area
    
    Based on research data:
    - FT 3: 124.74 kWp → 135.30 MWh/year (1,085 kWh/kWp/year)
    - FT 1: 32 kWp → 26.682 MWh/year (834 kWh/kWp/year)
    
    Args:
        panel_area_m2: Total panel area in m²
        panel_type: Type of panel
        degradation_rate: Annual degradation (0.005 = 0.5%)
    
    Returns:
        dict with energy production estimates
    """
    # Get capacity from area
    capacity_results = calculate_capacity_from_area(panel_area_m2, panel_type)
    capacity_kwp = capacity_results['capacity_kwp']
    
    # Daily production
    daily_production_kwh = (
        capacity_kwp * 
        MAGELANG_SOLAR_DATA['peak_sun_hours'] * 
        MAGELANG_SOLAR_DATA['system_efficiency']
    )
    
    # Annual production (first year)
    annual_production_kwh = daily_production_kwh * 365
    annual_production_mwh = annual_production_kwh / 1000
    
    # Specific yield (kWh/kWp/year)
    specific_yield = annual_production_kwh / capacity_kwp if capacity_kwp > 0 else 0
    
    # 25-year production (accounting for degradation)
    lifetime_production = 0
    for year in range(25):
        degradation_factor = (1 - degradation_rate) ** year
        lifetime_production += annual_production_kwh * degradation_factor
    
    return {
        'capacity_kwp': capacity_kwp,
        'daily_kwh': daily_production_kwh,
        'annual_kwh': annual_production_kwh,
        'annual_mwh': annual_production_mwh,
        'specific_yield': specific_yield,
        'lifetime_25y_mwh': lifetime_production / 1000,
        'peak_sun_hours': MAGELANG_SOLAR_DATA['peak_sun_hours'],
        'system_efficiency': MAGELANG_SOLAR_DATA['system_efficiency']
    }


def calculate_economics_from_area(panel_area_m2: float,
                                  panel_type: str = 'standard',
                                  installation_cost_per_kwp: float = 15000000,
                                  electricity_rate: float = 1500,
                                  lifetime_years: int = 25) -> Dict:
    """
    Calculate economic metrics from panel area
    
    Args:
        panel_area_m2: Total panel area in m²
        panel_type: Type of panel
        installation_cost_per_kwp: Installation cost (Rp/kWp)
        electricity_rate: Electricity price (Rp/kWh)
        lifetime_years: System lifetime
    
    Returns:
        dict with economic analysis
    """
    # Get capacity and energy production
    capacity_results = calculate_capacity_from_area(panel_area_m2, panel_type)
    energy_results = calculate_energy_production_from_area(panel_area_m2, panel_type)
    
    capacity_kwp = capacity_results['capacity_kwp']
    annual_production_kwh = energy_results['annual_kwh']
    
    # Total installation cost
    total_cost = capacity_kwp * installation_cost_per_kwp
    
    # Annual savings
    annual_savings = annual_production_kwh * electricity_rate
    
    # Simple payback period
    payback_years = total_cost / annual_savings if annual_savings > 0 else 999
    
    # Lifetime savings
    lifetime_savings = annual_savings * lifetime_years
    
    # Net profit
    net_profit = lifetime_savings - total_cost
    
    # ROI
    roi_percent = (net_profit / total_cost * 100) if total_cost > 0 else 0
    
    return {
        'total_cost': total_cost,
        'cost_per_kwp': installation_cost_per_kwp,
        'annual_savings': annual_savings,
        'payback_years': payback_years,
        'lifetime_savings': lifetime_savings,
        'net_profit': net_profit,
        'roi_percent': roi_percent
    }


def calculate_carbon_reduction_from_area(panel_area_m2: float,
                                        panel_type: str = 'standard') -> Dict:
    """
    Calculate CO2 emission reduction from panel area
    
    Indonesia grid emission factor: ~0.85 kg CO2/kWh
    
    Args:
        panel_area_m2: Total panel area in m²
        panel_type: Type of panel
    
    Returns:
        dict with carbon reduction metrics
    """
    energy_results = calculate_energy_production_from_area(panel_area_m2, panel_type)
    annual_production_kwh = energy_results['annual_kwh']
    
    co2_factor = 0.85  # kg CO2 per kWh
    
    annual_co2_reduction_kg = annual_production_kwh * co2_factor
    annual_co2_reduction_ton = annual_co2_reduction_kg / 1000
    
    lifetime_25y_co2_ton = annual_co2_reduction_ton * 25
    
    # Equivalent trees planted (1 tree absorbs ~20 kg CO2/year)
    equivalent_trees = annual_co2_reduction_kg / 20
    
    return {
        'annual_co2_kg': annual_co2_reduction_kg,
        'annual_co2_ton': annual_co2_reduction_ton,
        'lifetime_co2_ton': lifetime_25y_co2_ton,
        'equivalent_trees': int(equivalent_trees)
    }


def get_research_comparison_data() -> Dict:
    """
    Get Magelang research benchmark data for comparison
    
    Returns:
        dict with research project data
    """
    return {
        'FT 3 Building': {
            'capacity': 124.74,
            'production': 135.30,
            'yield': 1085,
            'location': 'FT 3 Building, Tuguran Campus'
        },
        'FT 1 Building': {
            'capacity': 32.0,
            'production': 26.682,
            'yield': 834,
            'location': 'FT 1 Building, Tuguran Campus'
        },
        'Magelang City': {
            'capacity': 16.8,
            'production': 27.864,
            'yield': 1659,
            'location': 'Magelang City'
        }
    }


def validate_specific_yield(specific_yield: float) -> Dict:
    """
    Validate specific yield against research data
    
    Args:
        specific_yield: Calculated specific yield (kWh/kWp/year)
    
    Returns:
        dict with validation results
    """
    research_data = get_research_comparison_data()
    yields = [v['yield'] for v in research_data.values()]
    
    avg_yield = sum(yields) / len(yields)
    min_yield = min(yields)
    max_yield = max(yields)
    
    # Check if within reasonable range (±20% of average)
    deviation = abs(specific_yield - avg_yield) / avg_yield
    is_valid = deviation < 0.2
    
    return {
        'specific_yield': specific_yield,
        'avg_research_yield': avg_yield,
        'min_research_yield': min_yield,
        'max_research_yield': max_yield,
        'deviation_percent': deviation * 100,
        'is_valid': is_valid,
        'status': 'Valid' if is_valid else 'Outside typical range'
    }
