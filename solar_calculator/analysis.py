"""
Gemini AI analysis integration
"""

import google.generativeai as genai
from typing import Dict, Optional
from PIL import Image

def analyze_with_gemini(
    model,
    detection_data: Dict,
    production_data: Dict,
    consumption_data: Dict,
    coverage_data: Dict,
    potential_data: Dict,
    economics_data: Dict,
    image_pil: Optional[Image.Image] = None
) -> str:
    """
    Comprehensive AI analysis using Gemini 3.0 Flash
    
    Args:
        model: Gemini model instance
        detection_data: Detection results (roofs, panels, areas)
        production_data: Solar production calculations
        consumption_data: Energy consumption calculations
        coverage_data: Coverage analysis
        potential_data: Solar potential calculations
        economics_data: Economic analysis
        image_pil: Optional image for visual analysis
    
    Returns:
        AI-generated analysis text
    """
    
    prompt = f"""
Analisis komprehensif hasil deteksi dan kalkulasi energi surya:

## 📊 DATA DETEKSI
- **Atap terdeteksi**: {detection_data.get('n_roofs', 0)} bangunan
- **Luas total atap**: {detection_data.get('roof_area_m2', 0):.2f} m²
- **Panel surya terdeteksi**: {detection_data.get('n_panels', 0)} panel
- **Luas panel terpasang**: {detection_data.get('panel_area_m2', 0):.2f} m²

## ⚡ PRODUKSI ENERGI SAAT INI
- **Kapasitas terpasang**: {production_data.get('capacity_kwp', 0):.2f} kWp
- **Produksi harian**: {production_data.get('daily_kwh', 0):.2f} kWh/hari
- **Produksi tahunan**: {production_data.get('annual_mwh', 0):.2f} MWh/tahun
- **Reduksi CO₂**: {production_data.get('carbon_reduction_tons', 0):.2f} ton/tahun

## 🏢 KONSUMSI ENERGI
- **Total bangunan**: {consumption_data.get('n_buildings', 0)} unit
- **Konsumsi harian**: {consumption_data.get('daily_kwh', 0):.2f} kWh/hari
- **Konsumsi tahunan**: {consumption_data.get('annual_mwh', 0):.2f} MWh/tahun
- **Basis**: {consumption_data.get('daily_per_building_kwh', 4.5)} kWh/hari per bangunan (studi Semarang)

## 📈 ANALISIS COVERAGE
- **Coverage saat ini**: {coverage_data.get('current_coverage_pct', 0):.1f}%
- **Status**: {coverage_data.get('status', 'Unknown')}
- **Gap/Surplus energi**: {coverage_data.get('energy_gap_kwh', 0):.2f} kWh/hari
- **Potensi coverage**: {coverage_data.get('potential_coverage_pct', 0):.1f}%

## 🎯 POTENSI PENGEMBANGAN
- **Luas atap tersedia**: {potential_data.get('available_area_m2', 0):.2f} m²
- **Potensi kapasitas**: +{potential_data.get('potential_kwp', 0):.2f} kWp
- **Potensi produksi**: +{potential_data.get('potential_daily_kwh', 0):.2f} kWh/hari
- **Potensi panel**: ~{potential_data.get('potential_n_panels', 0)} panel

## 💰 ANALISIS EKONOMI
- **Investasi untuk full coverage**: Rp {economics_data.get('investment_billion_rp', 0):.2f} Miliar
- **Penghematan tahunan**: Rp {economics_data.get('annual_savings_million_rp', 0):.2f} Juta/tahun
- **Payback period**: {economics_data.get('payback_years', 0):.1f} tahun
- **ROI 25 tahun**: {economics_data.get('roi_lifetime_pct', 0):.1f}%
- **NPV**: Rp {economics_data.get('npv_billion_rp', 0):.2f} Miliar

---

**TUGAS ANDA:**

Berikan analisis mendalam dalam bahasa Indonesia yang mencakup:

### 1. 📊 PENILAIAN KONDISI SAAT INI
- Apakah coverage energi saat ini sudah memadai?
- Bagaimana performa sistem solar yang ada?
- Apakah ada surplus atau defisit energi?

### 2. 💡 REKOMENDASI PENGEMBANGAN
- Berapa persen luas atap yang sebaiknya dimanfaatkan?
- Prioritas bangunan mana yang sebaiknya dipasang panel?
- Skenario optimal untuk ekspansi (25%, 50%, 75%, atau 100%)?

### 3. 💰 KELAYAKAN EKONOMI
- Apakah investasi layak secara finansial?
- Berapa lama balik modal (payback period)?
- Perbandingan dengan biaya listrik PLN

### 4. 🌍 DAMPAK LINGKUNGAN
- Berapa reduksi emisi CO₂ yang bisa dicapai?
- Kontribusi terhadap target net-zero emission

### 5. ⚠️ RISIKO & PERTIMBANGAN
- Apa saja risiko teknis dan finansial?
- Pertimbangan maintenance dan operasional
- Rekomendasi sistem penyimpanan energi (battery storage)

### 6. 🎯 KESIMPULAN & ACTION PLAN
- Ringkasan eksekutif
- Langkah-langkah implementasi yang disarankan
- Timeline realistis

**FORMAT:**
- Gunakan emoji untuk memperjelas
- Berikan angka spesifik dan konkret
- Struktur jelas dengan heading dan bullet points
- Bahasa profesional tapi mudah dipahami
"""
    
    try:
        if image_pil:
            response = model.generate_content([prompt, image_pil])
        else:
            response = model.generate_content(prompt)
        
        return response.text
    
    except Exception as e:
        return f"❌ Error dalam analisis AI: {str(e)}"


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
