"""
Solar Calculator Module
Photovoltaic energy analysis based on SAM3 detection
"""

from .models import (
    calculate_solar_production,
    calculate_consumption,
    calculate_solar_potential,
    calculate_coverage_analysis
)

from .economics import (
    calculate_economics,
    calculate_investment_scenarios
)

from .analysis import (
    analyze_with_gemini,
    generate_recommendations
)

from .utils import (
    pixels_to_area,
    format_currency,
    format_energy
)

__version__ = "1.0.0"
