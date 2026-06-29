# EcoPower Roof

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![SAM3](https://img.shields.io/badge/SAM3-ICLR_2026-10b981.svg)](https://openreview.net/pdf/c01d0dd8d4e7179b0952062eff08eb52c8d8c322.pdf)

**EcoPower Roof** is a scalable geospatial AI framework for urban rooftop solar potential assessment. It integrates hierarchical deep learning segmentation via SAM3, techno-economic modeling, AHP-weighted multi-criteria decision analysis (MCDA), and an LLM-assisted planning dashboard — bridging the gap between complex geospatial outputs and actionable urban energy policy for Indonesian municipalities.

---

## Table of Contents

- [Features](#features)
- [Framework Architecture](#framework-architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Model Performance](#model-performance)
- [Case Study Results](#case-study-results)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Team](#team)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Features

### Hierarchical Rooftop Segmentation
- SAM3-based multi-scale patch inference for precise rooftop boundary detection
- Simultaneous identification of existing PV installations
- GeoTIFF support with automatic geospatial metadata extraction
- Adaptive processing across large-scale, medium-scale, and small-scale features

### Techno-Economic Modeling
- Reduction cascade pipeline: building type, setback, HVAC, structural, and tilt factors
- Automated usable area estimation per rooftop polygon
- Energy balance computation: annual yield, self-sufficiency ratio, and CO2 reduction
- Financial analysis: NPV, LCOE, ROI, and payback period projections

### Spatial Recommendation via MCDA
- AHP-weighted multi-criteria scoring across four utilization scenarios (25%, 50%, 75%, 100%)
- Five-tier prioritization output per site
- GIS-integrated spatial ranking for municipal planning workflows

### LLM-Assisted Planning Dashboard
- Gemini AI integration for natural language report synthesis
- Six-section structured planning narrative per assessed site
- Interactive chatbot interface for planners and energy agencies
- JSON export for downstream GIS or policy workflows

---

## Framework Architecture

```
EcoPower Roof
|
+-- 1. Patch Segmentation (SAM3 Backbone)
|   +-- Multi-level rooftop boundary detection
|   +-- PV installation mask identification
|   +-- GeoTIFF geospatial metadata extraction
|
+-- 2. Techno-Economic Modeling
|   +-- Reduction Cascade
|   |   +-- Building type factor
|   |   +-- Setback factor
|   |   +-- HVAC obstruction factor
|   |   +-- Structural capacity factor
|   |   +-- Panel tilt correction factor
|   +-- Usable area estimation
|   +-- Energy balance (kWh/yr, self-sufficiency %)
|   +-- Financial analysis (NPV / LCOE / ROI)
|   +-- CO2 reduction (t/yr @ 0.87 kg CO2/kWh)
|
+-- 3. Spatial Recommendation (AHP-MCDA)
|   +-- Utilization scenarios: 25% / 50% / 75% / 100%
|   +-- Prioritization tiers:
|       +-- Priority 1 : Highly Recommended
|       +-- Priority 2 : Recommended
|       +-- Priority 3 : Conditionally Viable
|       +-- Priority 4 : Low Priority
|       +-- Tier 5     : Not Recommended
|
+-- 4. LLM Planning Dashboard
    +-- Six-section structured site report
    +-- Interactive natural language Q&A
    +-- Actionable planning narrative output
```

---

## Technology Stack

### Core Framework

| Component | Library / Tool | Purpose |
|---|---|---|
| Web Application | Streamlit 1.28+ | Interactive dashboard interface |
| Deep Learning | PyTorch 2.0+ | SAM3 inference engine |
| Segmentation | SAM3 (ICLR 2026) | Hierarchical patch segmentation |
| LLM Integration | Google Gemini AI | Planning narrative generation |

### Geospatial Processing

| Component | Library | Purpose |
|---|---|---|
| Raster I/O | Rasterio | GeoTIFF reading and metadata extraction |
| Vector Data | GeoPandas | Rooftop polygon management |
| Image Processing | Pillow / OpenCV | Pre- and post-processing |
| Numerical Ops | NumPy | Array computation |

### Data and Visualization

| Component | Library | Purpose |
|---|---|---|
| Data Analysis | Pandas | Tabular metrics and reporting |
| Plotting | Matplotlib | Charts and spatial overlays |
| Interactive UI | Streamlit Charts | Real-time metrics display |

---

## Project Structure

```
EcoPower-Roof/
|
+-- app_solar_calculator.py          # Main Streamlit application entry point
|
+-- sam3/                            # SAM3 model core
|   +-- model_builder.py             # SAM3 model initialization
|   +-- model/
|       +-- sam3_image_processor.py  # Image preprocessing pipeline
|
+-- sam3_advanced/                   # Enhanced multi-scale SAM3 features
|   +-- __init__.py                  # Package initializer
|   +-- config.py                    # Detection thresholds and parameters
|   +-- filtering.py                 # Mask filtering and noise removal
|   +-- geotiff_utils.py             # GeoTIFF spatial utilities
|   +-- inference.py                 # Multi-scale segmentation logic
|   +-- patching.py                  # Image patch extraction and tiling
|   +-- postprocess.py               # Mask refinement and post-processing
|   +-- utils.py                     # GPU memory management
|
+-- solar_calculator/                # Techno-economic computation modules
|   +-- __init__.py                  # Package initializer
|   +-- analysis.py                  # Gemini AI report integration
|   +-- district_analysis.py         # District-level aggregation and scoring
|   +-- economics.py                 # NPV / LCOE / ROI financial analysis
|   +-- export_utils.py              # Report and data export utilities
|   +-- models.py                    # Energy production models
|   +-- spatial_recommendation.py    # AHP-MCDA spatial prioritization
|   +-- usable_area_suitability.py   # Reduction cascade and usable area estimation
|   +-- utils.py                     # Shared helper functions

|
+-- requirements.txt                 # Python dependency list
+-- .streamlit/
|   +-- secrets.toml                 # API keys (excluded from repository)
+-- README.md                        # Project documentation
```

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended; CPU mode supported)
- Google Gemini API Key
- 8 GB+ RAM recommended

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ecopower-roof.git
   cd ecopower-roof
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # Linux / macOS
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download SAM3 model weights**
   ```bash
   # Place sam3.pt in the project root directory
   # Obtain from the SAM3 ICLR 2026 model repository
   ```

5. **Configure API keys**
   ```bash
   mkdir .streamlit
   echo 'GEMINI_API_KEY = "your-api-key-here"' > .streamlit/secrets.toml
   ```

6. **Run the application**
   ```bash
   streamlit run app_solar_calculator.py
   ```

7. **Open in browser**
   ```
   http://localhost:8501
   ```

---

## Usage Guide

### Step 1 — Upload Imagery

| Format | Notes |
|---|---|
| PNG, JPG, JPEG | Standard aerial or satellite images |
| TIF, TIFF | GeoTIFF with automatic geospatial metadata extraction |

- Recommended resolution: 0.3–1.0 m/pixel
- GeoTIFF inputs enable sub-meter area accuracy and coordinate preservation

### Step 2 — Select Detection Mode

| Mode | Description |
|---|---|
| Roof Detection | Detects all rooftop boundaries; estimates energy consumption and solar potential |
| Solar Panel Detection | Identifies existing PV installations; validates against research benchmarks |
| Both Detections | Full pipeline: coverage ratio, expansion opportunity, and complete energy assessment |

### Step 3 — Configure Parameters

**Advanced Settings**
- Pixel-to-meter ratio: manual override or auto-detected from GeoTIFF
- Confidence threshold: detection sensitivity (0.0–1.0)
- Minimum area filter: removes sub-threshold detections (pixels)

**Panel Specifications**

| Type | Power Density | Efficiency |
|---|---|---|
| Monocrystalline | 200 Wp/m2 | 20% |
| Polycrystalline | 170 Wp/m2 | 17% |
| Thin-Film | 100 Wp/m2 | 10% |

**Economic Parameters**
- Installation cost: Rp 10M–25M per kWp
- Electricity rate: PLN tariff (Rp/kWh)
- System lifetime: 25 years
- Annual degradation rate: 0.5%
- Maintenance cost: 1% of installation cost per year

### Step 4 — Review Results

**Detection Summary**
- Number of rooftops and PV installations detected
- Total measured area (m2)
- Installed or potential capacity (kWp)
- Coverage percentage

**Energy Analysis**
- Daily and annual production (kWh)
- Building-level consumption estimate
- Self-sufficiency ratio (%)
- Annual CO2 emission reduction (t/yr)

**Economic Projections**
- Total investment required
- Annual cost savings vs. PLN grid rate
- Payback period (years)
- 25-year ROI (%)

**MCDA Prioritization**

| Tier | Label |
|---|---|
| Priority 1 | Highly Recommended |
| Priority 2 | Recommended |
| Priority 3 | Conditionally Viable |
| Priority 4 | Low Priority |
| — | Not Recommended |

**Investment Scenarios**

| Scenario | Roof Utilization |
|---|---|
| Conservative | 25% |
| Moderate | 50% |
| Aggressive | 75% |
| Full | 100% |

### Step 5 — Export and Share

- Detection overlay images (PNG)
- Segmentation masks (PNG)
- Complete analysis report (JSON)
- LLM-generated six-section planning narrative (PDF / text)

---

## Model Performance

Validated across six sites in Yogyakarta, Jakarta, and Surabaya without site-specific retraining:

| Metric | Roof Segmentation | PV Detection |
|---|---|---|
| IoU | 76.51% | 84.71% |
| Precision | 96.60% | 87.06% |
| Recall | 75.23% | 96.95% |
| F1-Score | 83.89% | 91.69% |
| Accuracy | 90.96% | 99.30% |

Rooftop segmentation achieves high precision on large heterogeneous surfaces, while PV detection prioritizes high recall for smaller, visually distinct panel areas — reflecting a complementary performance trade-off suited to the dual-class inference task.

---

## Case Study Results

Assessed across **34 structures** covering **17,646.76 m2** with **25 existing PV installations** identified:

| Metric | Value |
|---|---|
| Additional PV Potential | 2,580.56 kWp |
| Self-Sufficiency at PV Sites | > 195.5% |
| Return on Investment (ROI) | 183.0% |
| Annual CO2 Reduction | 92.78 t/yr |
| Emission Factor Applied | 0.87 kg CO2/kWh |

---

## Configuration

### Environment Variables

```toml
# .streamlit/secrets.toml
GEMINI_API_KEY = "your-gemini-api-key"
```

### Model Configuration

```python
# sam3_advanced/config.py
class AdvancedConfig:
    conf_threshold_large  = 0.5
    conf_threshold_medium = 0.5
    conf_threshold_small  = 0.5
    min_area_pixels       = 100
    morphology_kernel_size = 5
```

### Solar Parameters

```python
# solar_calculator/utils.py
SOLAR_PANEL_SPECS = {
    'monocrystalline': {
        'power_density_wp_m2': 200,
        'efficiency': 0.20,
        'panel_size_m2': 1.6
    },
    'polycrystalline': {
        'power_density_wp_m2': 170,
        'efficiency': 0.17,
        'panel_size_m2': 1.6
    },
    'thin_film': {
        'power_density_wp_m2': 100,
        'efficiency': 0.10,
        'panel_size_m2': 1.6
    }
}

SOLAR_DATA = {
    'peak_sun_hours'            : 4.5,
    'specific_yield_kwh_kwp_yr' : 1300,
    'performance_ratio'         : 0.75,
    'emission_factor_kg_co2_kwh': 0.87
}
```

---

## Troubleshooting

| Error | Cause | Solution |
|---|---|---|
| `sam3.pt not found` | Missing model weights | Download SAM3 weights and place in project root |
| `CUDA out of memory` | Insufficient GPU VRAM | Reduce image resolution or switch to CPU mode |
| `Rasterio not available` | Missing dependency | Run `pip install rasterio` |
| `Invalid API key` | Misconfigured secrets | Verify `.streamlit/secrets.toml` content |

---

## Deployment

### Streamlit Cloud

1. Push repository to GitHub
2. Connect to Streamlit Cloud
3. Add `GEMINI_API_KEY` in the Streamlit secrets dashboard
4. Deploy

### Docker

```dockerfile
FROM python:3.8-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app_solar_calculator.py"]
```

### Local Server

```bash
streamlit run app_solar_calculator.py --server.port 8501 --server.address 0.0.0.0
```

---

## Team

**GeoAI Twinverse Research Group**  
Faculty of Engineering, Universitas Gadjah Mada  
Yogyakarta, Indonesia

| Name | Role |
|---|---|
| Mohammad Zulfi Rahadi Putra | Researcher |
| Raffi Satya Nugraha | Researcher |
| Najieda Azka | Researcher |
| Salzabila Enzal Putri | Researcher |

---

## Acknowledgments

- **Meta AI** — SAM3: Segment Anything Model 3 (ICLR 2026)
- **Google** — Gemini AI API
- **Streamlit** — Web application framework
- **World Bank Group & Solargis** — Global Solar Atlas 2.0
- **Universitas Gadjah Mada** — Institutional and academic support

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

<div align="center">

Developed by the GeoAI Twinverse Research Group  
Faculty of Engineering, Universitas Gadjah Mada, Indonesia

</div>
