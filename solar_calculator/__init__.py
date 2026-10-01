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

# The Gemini assistant is an optional feature: without `google-generativeai`
# installed, the numeric modules (models, economics, pv_modules, module_yield,
# dem_slope) must still be importable for headless runs and tests.
try:
    from .analysis import (
        analyze_with_gemini,
        generate_recommendations
    )
except ImportError:  # pragma: no cover - optional dependency
    analyze_with_gemini = None
    generate_recommendations = None

from .utils import (
    pixels_to_area,
    format_currency,
    format_energy
)

__version__ = "1.0.0"
