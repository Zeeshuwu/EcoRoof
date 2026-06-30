"""
Gemini AI analysis integration
"""

import google.generativeai as genai
from typing import Dict, Optional
from PIL import Image

def analyze_with_gemini(
    model,
    detection_data: dict,
    production_data: dict,
    consumption_data: dict,
    coverage_data: dict,
    potential_data: dict,
    economics_data: dict,
    image_pil=None,
) -> str:
    """
    Generate a structured English solar analysis report using Gemini AI,
    grounded in the EcoPower Roof journal methodology:
    SAM3 multi-level patch segmentation → usable area reduction →
    PV potential estimation → economic feasibility → spatial recommendation.
    """

    prompt = f"""
You are a solar energy research analyst contributing to the EcoPower Roof project,
developed by the GEO-AI Twinverse Research Group at the Department of Geodetic Engineering,
Faculty of Engineering, Universitas Gadjah Mada, Yogyakarta, Indonesia.

Your task is to interpret the following site analysis data and produce a structured,
journal-quality English report. The report must reflect the exact five-stage methodology
described in the EcoPower Roof paper, as summarized below.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ECOPOWER ROOF — METHODOLOGY CONTEXT (for your reference)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STAGE 1 — SAM3 Multi-Level Patch Segmentation
  The platform uses SAM3 (Segment Anything Model 3), a foundation segmentation model
  with enhanced feature refinement and encoder–decoder coupling. To handle ultra-high-
  resolution UAV orthomosaics, a three-level patch strategy is applied:
    • Level 1 — 256×256 px patches  → fine rooftop boundaries, small structures
    • Level 2 — 512×512 px patches  → intermediate context, urban feature separation
    • Level 3 — full orthomosaic    → global spatial consistency across urban blocks
  Outputs from all three levels are merged via Non-Maximum Suppression (NMS) to remove
  duplicate detections across scales, producing unified georeferenced vector polygons.
  Performance is evaluated using Precision, Recall, F1-Score, and IoU (~91% accuracy).

STAGE 2 — Usable Rooftop Area Estimation
  Raw rooftop area is reduced by a composite suitability factor derived from:
    • Building type ratio        (Res4Africa, 2024)
    • Setback factor             (Burke, 2021 — 1 m edge exclusion)
    • HVAC obstruction factor    (Burke, 2021)
    • Structural constraint factor (Burke, 2021)
    • Tilt efficiency factor     (Duffie & Beckman, 2013 — 15° rack, Java latitude)
  Suitability classes: High (≥55%), Medium (40–55%), Low (20–40%), Unsuitable (<20%)

STAGE 3 — PV Potential Estimation
  Capacity  : P = A_usable × PD        (PD = power density in kWp/m²)
  Generation: E = A_usable × G × η × PR
    where G = annual solar irradiance, η = module efficiency, PR = performance ratio
  Reference values: G = 4.5 h/day (Java avg.), PR = 0.75 (Tarigan et al., 2025)

STAGE 4 — Economic Feasibility Analysis
  Cost      : Cost = P × C_unit        (C_unit = installation cost per kWp)
  Savings   : Saving = E × Tariff      (PLN electricity tariff)
  ROI       : ROI = (Saving − Cost) / Cost × 100
  Payback   : PP = Cost / Saving
  Benchmarks: Tarigan et al. (2025), Suparwoko & Qamar (2022),
              Cantiqa & Dirkareshza (2025), Kunaifi et al. (2020)

STAGE 5 — Spatial Recommendation Model
  Rooftops classified into four tiers using AHP-weighted MCDA
  (weights from E3S Conferences 2024, CR = 0.04):
    • Highly Recommended — top composite score
    • Recommended
    • Moderate
    • Not Recommended
  Greedy allocation fills highest-ranked roofs first until consumption target is met.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SITE ANALYSIS DATA FOR THIS IMAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[STAGE 1 — DETECTION OUTPUT]
  Rooftops detected       : {detection_data['n_roofs']} buildings
  Total roof area         : {detection_data['roof_area_m2']:.2f} m²
  Existing PV panels found: {detection_data['n_panels']} installations
  Existing panel area     : {detection_data['panel_area_m2']:.2f} m²

[STAGE 2 — USABLE AREA]
  (Reduction factors applied per Burke 2021, Res4Africa 2024, Duffie & Beckman 2013)
  Existing panel area used as proxy for currently deployed usable area.

[STAGE 3 — PV POTENTIAL]
  Installed capacity      : {production_data['capacity_kwp']:.2f} kWp
  Daily production        : {production_data['daily_kwh']:.2f} kWh/day
  Annual production       : {production_data['annual_mwh']:.2f} MWh/year
  CO₂ reduction           : {production_data['carbon_reduction_tons']:.2f} t/year
  Estimated consumption   : {consumption_data['daily_kwh']:.2f} kWh/day
                            ({consumption_data.get('annual_mwh', consumption_data['annual_kwh']/1000):.2f} MWh/year)
  Coverage ratio          : {coverage_data['current_coverage_pct']:.1f}%
  Energy gap              : {abs(coverage_data.get('energy_gap_kwh', 0)):.2f} kWh/day
  Status                  : {'Self-sufficient — surplus available' if coverage_data.get('is_sufficient') else 'Deficit — expansion required'}
  Remaining potential     : {potential_data['available_area_m2']:.2f} m² available
                            ({potential_data['potential_kwp']:.2f} kWp additional capacity)

[STAGE 4 — ECONOMIC FEASIBILITY]
  Total investment        : Rp {economics_data['investment_billion_rp']:.3f} billion
  Annual savings          : Rp {economics_data['annual_savings_million_rp']:.2f} million/year
  Payback period          : {economics_data['payback_years']:.1f} years
  ROI (25-year)           : {economics_data['roi_lifetime_pct']:.1f}%

[STAGE 5 — SPATIAL RECOMMENDATION]
  (AHP-weighted MCDA classification applied — see platform Spatial Recommendation tab)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REPORT STRUCTURE — follow exactly, use these headings
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 1. Segmentation & Detection Summary
Describe what the SAM3 multi-level patch segmentation detected at this site.
Reference the three-level strategy (256×256, 512×512, full image) and NMS fusion.
State the number of rooftops and existing PV installations identified.
Comment on what this implies about the site's current solar adoption state.

## 2. Usable Area & Suitability Assessment
Explain how the raw rooftop area is reduced to usable area using the composite
suitability factor. Reference Burke (2021), Res4Africa (2024), and Duffie & Beckman (2013).
Interpret the resulting usability ratio and suitability class for this site.

## 3. PV Potential & Energy Balance
Present the estimated PV capacity, annual generation, and consumption figures.
Apply the formula E = A_usable × G × η × PR and reference Tarigan et al. (2025)
for G = 4.5 h/day and PR = 0.75. State whether the site is in surplus or deficit
and quantify the energy gap in kWh/day and percentage terms.

## 4. Economic Feasibility
Evaluate the investment case using Cost = P × C_unit, Saving = E × Tariff,
ROI = (Saving − Cost)/Cost × 100, and PP = Cost/Saving.
Compare the payback period and ROI against Indonesian benchmarks
(Tarigan et al. 2025; Suparwoko & Qamar 2022). State clearly whether
the project meets the economic viability threshold.

## 5. CO₂ & Environmental Impact
Quantify annual carbon reduction from the detected panel area.
Contextualize against the Indonesian grid emission factor of 0.87 kg CO₂/kWh (PLN, 2022).
Relate this to Indonesia's 3,294 GW untapped solar potential and national
decarbonization targets.

## 6. Spatial Recommendations
Provide 3–4 specific, prioritized recommendations grounded in the AHP-MCDA
spatial recommendation model. Reference the four recommendation tiers
(Highly Recommended → Not Recommended). Address both building-scale and
district-scale deployment opportunities.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WRITING GUIDELINES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Formal academic English suitable for an IEEE-style journal paper
- Use specific numbers from the data — never generalize or round loosely
- Cite references inline: (Tarigan et al., 2025), (Burke, 2021), etc.
- Each section: 3–5 sentences in flowing prose — no bullet points inside sections
- Do not add sections beyond those listed
- Do not mention that you are an AI or that this is generated content
- Write as if this is a Results & Discussion section authored by the research team
"""

    try:
        if image_pil is not None:
            response = model.generate_content([prompt, image_pil])
        else:
            response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ AI analysis error: {str(e)}"



def generate_recommendations(
    coverage_pct: float,
    potential_kwp: float,
    payback_years: float,
    roi_pct: float
) -> Dict[str, str]:
    """
    Generate quick recommendations based on metrics
    
    Args:
        coverage_pct: Current coverage percentage
        potential_kwp: Potential capacity available
        payback_years: Payback period
        roi_pct: ROI percentage
    
    Returns:
        Dictionary with recommendation categories
    """
    recommendations = {}
    
    # Coverage recommendation
    if coverage_pct >= 100:
        recommendations['coverage'] = "✅ Coverage sudah mencukupi. Pertimbangkan ekspansi untuk jual listrik ke PLN."
    elif coverage_pct >= 75:
        recommendations['coverage'] = "⚠️ Coverage cukup baik. Tambahkan 25-50% kapasitas untuk full coverage."
    elif coverage_pct >= 50:
        recommendations['coverage'] = "⚠️ Coverage moderat. Disarankan tambah 50-75% kapasitas."
    else:
        recommendations['coverage'] = "❌ Coverage rendah. Prioritaskan ekspansi segera."
    
    # Expansion recommendation
    if potential_kwp > 50:
        recommendations['expansion'] = f"🚀 Potensi besar! Tersedia {potential_kwp:.1f} kWp. Implementasi bertahap 50% dulu."
    elif potential_kwp > 20:
        recommendations['expansion'] = f"💡 Potensi sedang: {potential_kwp:.1f} kWp. Manfaatkan 75% luas atap."
    else:
        recommendations['expansion'] = f"⚠️ Potensi terbatas: {potential_kwp:.1f} kWp. Maksimalkan semua atap tersedia."
    
    # Economic recommendation
    if payback_years <= 7 and roi_pct >= 200:
        recommendations['economic'] = "✅ Investasi sangat layak! Payback cepat dan ROI tinggi."
    elif payback_years <= 10 and roi_pct >= 150:
        recommendations['economic'] = "✅ Investasi layak. ROI positif dalam jangka panjang."
    elif payback_years <= 15:
        recommendations['economic'] = "⚠️ Investasi cukup layak. Pertimbangkan insentif pemerintah."
    else:
        recommendations['economic'] = "⚠️ Payback period panjang. Evaluasi ulang skenario investasi."
    
    return recommendations
