"""
Economic analysis for solar installations
"""

from typing import Dict, List

# Economic constants (Indonesia, 2024)
COST_PER_KWP = 15_000_000  # Rp 15 million per kWp installed
ELECTRICITY_RATE = 1_500  # Rp 1,500 per kWh (PLN residential)
MAINTENANCE_COST_ANNUAL_PCT = 0.01  # 1% of investment per year
SYSTEM_LIFETIME_YEARS = 25  # Solar panel lifetime
DISCOUNT_RATE = 0.05  # 5% discount rate for NPV

def calculate_economics(
    capacity_kwp: float,
    annual_production_kwh: float,
    cost_per_kwp: float = COST_PER_KWP,
    electricity_rate: float = ELECTRICITY_RATE,
    maintenance_pct: float = MAINTENANCE_COST_ANNUAL_PCT,
    lifetime_years: int = SYSTEM_LIFETIME_YEARS
) -> Dict[str, float]:
    """
    Calculate economic metrics for solar installation
    
    Args:
        capacity_kwp: Solar capacity in kWp
        annual_production_kwh: Annual energy production
        cost_per_kwp: Installation cost per kWp (default: Rp 15M)
        electricity_rate: PLN electricity rate (default: Rp 1,500/kWh)
        maintenance_pct: Annual maintenance cost as % of investment
        lifetime_years: System lifetime (default: 25 years)
    
    Returns:
        Dictionary with economic metrics
    """
    # Initial investment
    investment_rp = capacity_kwp * cost_per_kwp
    
    # Annual savings (electricity cost avoided)
    annual_savings_rp = annual_production_kwh * electricity_rate
    
    # Annual maintenance cost
    annual_maintenance_rp = investment_rp * maintenance_pct
    
    # Net annual savings
    net_annual_savings_rp = annual_savings_rp - annual_maintenance_rp
    
    # Simple payback period
    if net_annual_savings_rp > 0:
        payback_years = investment_rp / net_annual_savings_rp
    else:
        payback_years = float('inf')
    
    # Total savings over lifetime
    total_savings_lifetime_rp = net_annual_savings_rp * lifetime_years
    
    # ROI over lifetime
    if investment_rp > 0:
        roi_lifetime_pct = ((total_savings_lifetime_rp / investment_rp) - 1) * 100
    else:
        roi_lifetime_pct = 0
    
    # NPV (Net Present Value)
    npv_rp = -investment_rp
    for year in range(1, lifetime_years + 1):
        npv_rp += net_annual_savings_rp / ((1 + DISCOUNT_RATE) ** year)
    
    return {
        'capacity_kwp': round(capacity_kwp, 2),
        'investment_rp': round(investment_rp, 0),
        'investment_billion_rp': round(investment_rp / 1_000_000_000, 2),
        'annual_savings_rp': round(annual_savings_rp, 0),
        'annual_savings_million_rp': round(annual_savings_rp / 1_000_000, 2),
        'annual_maintenance_rp': round(annual_maintenance_rp, 0),
        'net_annual_savings_rp': round(net_annual_savings_rp, 0),
        'payback_years': round(payback_years, 1),
        'total_savings_lifetime_rp': round(total_savings_lifetime_rp, 0),
        'roi_lifetime_pct': round(roi_lifetime_pct, 1),
        'npv_rp': round(npv_rp, 0),
        'npv_billion_rp': round(npv_rp / 1_000_000_000, 2)
    }


def calculate_investment_scenarios(
    available_area_m2: float,
    panel_efficiency: float = 0.15,
    scenarios: List[float] = [0.25, 0.50, 0.75, 1.0]
) -> List[Dict]:
    """
    Calculate multiple investment scenarios
    
    Args:
        available_area_m2: Available roof area for installation
        panel_efficiency: kWp per m²
        scenarios: List of utilization percentages (e.g., 25%, 50%, 75%, 100%)
    
    Returns:
        List of scenario dictionaries
    """
    results = []
    
    for pct in scenarios:
        area_used = available_area_m2 * pct
        capacity_kwp = area_used * panel_efficiency
        annual_production_kwh = capacity_kwp * 5 * 0.80 * 365  # PSH=5, PR=0.80
        
        economics = calculate_economics(capacity_kwp, annual_production_kwh)
        
        results.append({
            'scenario_name': f"{int(pct*100)}% Utilization",
            'utilization_pct': pct * 100,
            'area_used_m2': round(area_used, 2),
            'capacity_kwp': economics['capacity_kwp'],
            'investment_billion_rp': economics['investment_billion_rp'],
            'annual_savings_million_rp': economics['annual_savings_million_rp'],
            'payback_years': economics['payback_years'],
            'roi_lifetime_pct': economics['roi_lifetime_pct'],
            'npv_billion_rp': economics['npv_billion_rp']
        })
    
    return results


def compare_with_pln(
    annual_consumption_kwh: float,
    annual_production_kwh: float,
    electricity_rate: float = ELECTRICITY_RATE
) -> Dict[str, float]:
    """
    Compare solar vs PLN grid costs
    
    Args:
        annual_consumption_kwh: Annual energy consumption
        annual_production_kwh: Annual solar production
        electricity_rate: PLN rate (Rp/kWh)
    
    Returns:
        Comparison metrics
    """
    # Cost if using 100% PLN
    annual_pln_cost_rp = annual_consumption_kwh * electricity_rate
    
    # Cost with solar (only pay for deficit)
    deficit_kwh = max(0, annual_consumption_kwh - annual_production_kwh)
    annual_cost_with_solar_rp = deficit_kwh * electricity_rate
    
    # Annual savings
    annual_savings_rp = annual_pln_cost_rp - annual_cost_with_solar_rp
    
    # Savings percentage
    if annual_pln_cost_rp > 0:
        savings_pct = (annual_savings_rp / annual_pln_cost_rp) * 100
    else:
        savings_pct = 0
    
    return {
        'annual_pln_cost_rp': round(annual_pln_cost_rp, 0),
        'annual_pln_cost_million_rp': round(annual_pln_cost_rp / 1_000_000, 2),
        'annual_cost_with_solar_rp': round(annual_cost_with_solar_rp, 0),
        'annual_savings_rp': round(annual_savings_rp, 0),
        'annual_savings_million_rp': round(annual_savings_rp / 1_000_000, 2),
        'savings_pct': round(savings_pct, 1),
        'deficit_kwh': round(deficit_kwh, 2)
    }
