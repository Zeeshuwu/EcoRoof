# EcoPower Roof — Technical Documentation

This document covers the full technical setup, configuration, deployment,
and internal pipeline details for EcoPower Roof.

For the project overview, features, and results, see [README.md](README.md).

---

## Table of Contents

- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [SAM3 Model Setup](#sam3-model-setup)
- [Usage Guide](#usage-guide)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Deployment](#deployment)

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
+-- sam3/                            # SAM3 model core (cloned from Meta AI)
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
+-- sam3.pt                          # SAM3 pre-trained model weights
+-- requirements.txt                 # Python dependency list
+-- .streamlit/
|   +-- secrets.toml                 # API keys (excluded from repository)
+-- README.md                        # Project overview
+-- TECHNICAL.md                     # This file
```

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended; CPU mode supported)
- Google Gemini API Key
- 8 GB+ RAM recommended

### Steps

1. **Clone the EcoPower Roof repository**
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

4. **Configure API keys**
   ```bash
   mkdir .streamlit
   echo 'GEMINI_API_KEY = "your-api-key-here"' > .streamlit/secrets.toml
   ```

5. **Complete SAM3 setup** — see the next section before proceeding.

6. **Run the application**
   ```bash
   streamlit run app_solar_calculator.py
   ```

7. **Open in browser**
   ```
   http://localhost:8501
   ```

---

## SAM3 Model Setup

EcoPower Roof depends on the official SAM3 package from Meta AI Research.
The steps below must be completed **before** running the application.

### Step 1 — Clone the Official SAM3 Repository

```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3
```

### Step 2 — Install SAM3 as a Local Package

```bash
# Install in editable mode so EcoPower Roof can import it directly
pip install -e .
```

This registers the `sam3` package into your Python environment.
After this step, `from sam3 import build_sam3` will resolve correctly
from anywhere in the EcoPower Roof project.

### Step 3 — Download Model Weights

Place the checkpoint in the EcoPower Roof project root as `sam3.pt`:

```bash
# Run from the EcoPower Roof project root
wget -O sam3.pt https://dl.fbaipublicfiles.com/sam3/sam3.pt
```

| File | Size | Purpose |
|---|---|---|
| `sam3.pt` | ~2.4 GB | Default checkpoint used by EcoPower Roof |

### Step 4 — Verify Installation

```python
import torch
from sam3 import build_sam3

model = build_sam3(checkpoint="sam3.pt")
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print(f"SAM3 loaded successfully on: {device}")
```

**Expected output:**
```
SAM3 loaded successfully on: cuda
```

---

### Internal Inference Pipeline

Once loaded, imagery is passed through the full SAM3 Advanced pipeline:

```
Aerial / Satellite Image (PNG or GeoTIFF)
        |
        v
[ sam3_advanced/inference.py ]        — SAM3 automatic mask generation
        |
        v
[ sam3_advanced/filtering.py ]        — Class separation: rooftop vs. PV
        |
        v
[ sam3_advanced/postprocess.py ]      — Morphological refinement + area filtering
        |
        v
[ sam3_advanced/patching.py ]         — Tile-based processing for large images
        |
        v
[ sam3_advanced/geotiff_utils.py ]    — Pixel mask -> real-world area (m2)
        |
        v
[ solar_calculator/usable_area_suitability.py ]   — Reduction cascade
        |
        v
[ solar_calculator/economics.py ]     — NPV / LCOE / ROI computation
        |
        v
[ solar_calculator/spatial_recommendation.py ]    — AHP-MCDA prioritization
        |
        v
[ solar_calculator/analysis.py ]      — Gemini AI six-section report
        |
        v
  Streamlit Dashboard Output
```

### Inference Parameters

| Parameter | Default | Effect |
|---|---|---|
| `points_per_side` | 32 | Grid density for automatic prompt generation |
| `pred_iou_thresh` | 0.88 | Minimum predicted IoU to retain a mask |
| `stability_score_thresh` | 0.95 | Mask stability filter across threshold range |
| `min_mask_region_area` | 100 | Removes sub-threshold pixel regions |

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
    conf_threshold_large   = 0.5
    conf_threshold_medium  = 0.5
    conf_threshold_small   = 0.5
    min_area_pixels        = 100
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
    'peak_sun_hours'             : 4.5,
    'specific_yield_kwh_kwp_yr'  : 1300,
    'performance_ratio'          : 0.75,
    'emission_factor_kg_co2_kwh' : 0.87
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
| `ModuleNotFoundError: sam3` | SAM3 not installed | Run `pip install -e .` inside the cloned sam3 directory |

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
streamlit run app_solar_calculator.py \
  --server.port 8501 \
  --server.address 0.0.0.0
```

---

<div align="center">

Developed by the GeoAI Twinverse Research Group
Faculty of Engineering, Universitas Gadjah Mada, Indonesia

</div>
