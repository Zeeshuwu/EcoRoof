"""
EcoPower Roof - Solar Calculator Dashboard
Photovoltaic energy analysis using SAM3 Advanced detection
Based on Semarang research studies
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import io
import time
import google.generativeai as genai

# SAM3 imports
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# SAM3 Advanced imports
from sam3_advanced.config import AdvancedConfig
from sam3_advanced.inference import segment_multiscale_advanced
from sam3_advanced.postprocess import post_process_mask
from sam3_advanced.utils import clear_memory, get_gpu_memory

# 🆕 NEW: GeoTIFF support
from sam3_advanced.geotiff_utils import load_image_with_geotiff_support, RASTERIO_AVAILABLE

# Solar Calculator imports
from solar_calculator.models import (
    calculate_solar_production,
    calculate_consumption,
    calculate_solar_potential,
    calculate_coverage_analysis,
    estimate_panel_area_from_count
)
from solar_calculator.economics import (
    calculate_economics,
    calculate_investment_scenarios,
    compare_with_pln
)
from solar_calculator.analysis import (
    analyze_with_gemini,
    generate_recommendations
)
from solar_calculator.utils import (
    pixels_to_area,
    count_objects_in_mask,
    format_currency,
    format_energy,
    estimate_image_scale,
    # 🆕 NEW: Area-based functions
    SOLAR_PANEL_SPECS,
    MAGELANG_SOLAR_DATA,
    calculate_area_from_mask,
    calculate_capacity_from_area,
    calculate_energy_production_from_area,
    calculate_economics_from_area,
    calculate_carbon_reduction_from_area,
    get_research_comparison_data,
    validate_specific_yield
)

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="🌱 EcoPower Roof",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    /* Main title */
    .main-title {
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #10b981, #059669, #047857);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 1rem;
        margin-top: 3rem;
        border-top: 2px solid #e5e7eb;
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-radius: 10px;
    }
    
    .footer h3 {
        color: #047857;
        margin-bottom: 1rem;
    }
    
    .footer p {
        color: #374151;
        margin: 0.5rem 0;
    }
    
    .team-member {
        display: inline-block;
        margin: 0.5rem 1rem;
        padding: 0.5rem 1rem;
        background: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Detection buttons */
    .stButton > button {
        width: 100%;
        height: 120px;
        font-size: 1.5rem;
        font-weight: bold;
        border-radius: 15px;
        border: 3px solid #ddd;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        border-color: #10b981;
        box-shadow: 0 5px 15px rgba(16,185,129,0.3);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .metric-card h3 {
        margin: 0;
        font-size: 2rem;
    }
    
    .metric-card p {
        margin: 0;
        font-size: 1rem;
        opacity: 0.9;
    }
    
    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1rem;
    }
    
    .status-success {
        background: #10b981;
        color: white;
    }
    
    .status-warning {
        background: #f59e0b;
        color: white;
    }
    
    .status-danger {
        background: #ef4444;
        color: white;
    }
    
    /* Info box */
    .info-box {
        background: #f0fdf4;
        border-left: 4px solid #10b981;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    /* Results section */
    .results-section {
        background: #f9fafb;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'sam3_model' not in st.session_state:
    st.session_state.sam3_model = None
if 'sam3_processor' not in st.session_state:
    st.session_state.sam3_processor = None
if 'gemini_model' not in st.session_state:
    st.session_state.gemini_model = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
# 🆕 NEW: GeoTIFF handler
if 'geotiff_handler' not in st.session_state:
    st.session_state.geotiff_handler = None
if 'use_area_based' not in st.session_state:
    st.session_state.use_area_based = True

# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================

@st.cache_resource
def load_sam3_model(checkpoint_path: str = "sam3.pt"):
    """Load SAM3 model (cached)"""
    try:
        model = build_sam3_image_model(checkpoint_path=checkpoint_path)
        processor = Sam3Processor(model)
        return model, processor
    except Exception as e:
        st.error(f"❌ Error loading SAM3 model: {str(e)}")
        return None, None

def load_gemini_model(api_key: str):
    """Load Gemini model"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        return model
    except Exception as e:
        st.error(f"❌ Error loading Gemini model: {str(e)}")
        return None

# ============================================================================
# DETECTION FUNCTION
# ============================================================================

def run_solar_detection(
    image_pil: Image.Image,
    detection_mode: str,
    processor: Sam3Processor,
    config: AdvancedConfig,
    pixel_to_meter_ratio: float = 0.5,
    geotiff_handler=None  # 🆕 NEW parameter
) -> dict:
    """
    Run solar detection (roof or panel)
    
    Args:
        image_pil: Input image
        detection_mode: "roof" or "solar_panel"
        processor: SAM3 processor
        config: Advanced config
        pixel_to_meter_ratio: Conversion ratio
        geotiff_handler: GeoTIFF handler for accurate measurements
    
    Returns:
        Dictionary with detection results
    """
    
    # Auto-generate prompts based on mode
    if detection_mode == "roof":
        config.prompts_large = ["roof", "large roof", "big roof", "building roof", "rooftop"]
        config.prompts_medium = ["roof", "medium roof", "rooftop", "house roof"]
        config.prompts_small = ["roof", "small roof", "house roof", "building top"]
    else:  # solar_panel
        config.prompts_large = ["solar panel", "large solar panel", "big solar panel", "photovoltaic array"]
        config.prompts_medium = ["solar panel", "medium solar panel", "photovoltaic panel", "PV panel"]
        config.prompts_small = ["solar panel", "small solar panel", "PV panel", "solar cell"]
    
    # Set image
    processor.set_image(image_pil)
    
    # Run multiscale detection (SAM3 Advanced)
    with st.spinner(f"🔍 Detecting {detection_mode.replace('_', ' ')}..."):
        final_mask, stats = segment_multiscale_advanced(
            image_pil,
            processor,
            config
        )
    
    # Post-process mask
    with st.spinner("🔧 Post-processing..."):
        final_mask_processed = post_process_mask(
            final_mask,
            config,
            verbose=False
        )
    
    # Count objects
    n_objects = count_objects_in_mask(final_mask_processed)
    
    # 🆕 NEW: Calculate area with GeoTIFF support
    area_m2 = calculate_area_from_mask(
        final_mask_processed,
        pixel_to_meter_ratio,
        geotiff_handler
    )
    
    # Calculate coverage percentage
    total_pixels = final_mask_processed.shape[0] * final_mask_processed.shape[1]
    detected_pixels = np.sum(final_mask_processed > 0)
    coverage_pct = (detected_pixels / total_pixels) * 100
    
    # Create overlay
    import cv2
    overlay = image_pil.copy()
    overlay_np = np.array(overlay)
    
    # Apply colored mask
    if detection_mode == "roof":
        color = [255, 100, 100]  # Red for roofs
    else:
        color = [100, 255, 100]  # Green for solar panels
    
    mask_colored = np.zeros_like(overlay_np)
    mask_colored[final_mask_processed > 0] = color
    
    # Blend
    overlay_np = cv2.addWeighted(overlay_np, 0.7, mask_colored, 0.3, 0)
    overlay = Image.fromarray(overlay_np)
    
    return {
        'mode': detection_mode,
        'mask': final_mask_processed,
        'overlay': overlay,
        'n_objects': n_objects,
        'area_m2': area_m2,
        'coverage_pct': coverage_pct,
        'stats': stats
    }

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    
    # Header
    st.markdown('<h1 class="main-title">🌱 EcoPower Roof</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Powered Solar Energy Analysis for Sustainable Future</p>', unsafe_allow_html=True)
    
    # ========================================================================
    # SIDEBAR - Configuration
    # ========================================================================
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # 🆕 NEW: Panel type selection
        st.subheader("🔆 Solar Panel Type")
        panel_type = st.selectbox(
            "Select Panel Type",
            options=list(SOLAR_PANEL_SPECS.keys()),
            format_func=lambda x: SOLAR_PANEL_SPECS[x]['name'],
            help="Based on real installations"
        )
        
        selected_panel = SOLAR_PANEL_SPECS[panel_type]
        
        with st.expander("📋 Panel Specifications"):
            st.write(f"**Power Density:** {selected_panel['power_density_wp_m2']} Wp/m²")
            st.write(f"**Panel Size:** {selected_panel['panel_size_m2']} m²")
            st.write(f"**Efficiency:** {selected_panel['efficiency']*100:.1f}%")
        
        st.divider()
        
        # Gemini API Key
        st.subheader("🤖 Gemini API")
        
        # Try to load from secrets
        try:
            gemini_api_key = st.secrets["GEMINI_API_KEY"]
            st.success("✅ API Key loaded from secrets")
        except:
            gemini_api_key = st.text_input(
                "Gemini API Key",
                type="password",
                help="Enter your Gemini API key for AI analysis"
            )
        
        if gemini_api_key and not st.session_state.gemini_model:
            st.session_state.gemini_model = load_gemini_model(gemini_api_key)
        
        st.divider()
        
        # Advanced Settings
        st.subheader("🔧 Advanced Settings")
        
        pixel_to_meter = st.number_input(
            "Pixel to Meter Ratio",
            min_value=0.1,
            max_value=2.0,
            value=0.5,
            step=0.1,
            help="Conversion ratio from pixels to meters (auto-detected for GeoTIFF)"
        )
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum confidence for detection"
        )
        
        min_area_pixels = st.number_input(
            "Minimum Object Area (pixels)",
            min_value=10,
            max_value=1000,
            value=100,
            step=10,
            help="Minimum area to consider as valid object"
        )
        
        st.divider()
        
        # Economic Parameters
        st.subheader("💰 Economic Parameters")
        
        cost_per_kwp = st.number_input(
            "Installation Cost (Rp/kWp)",
            min_value=10_000_000,
            max_value=25_000_000,
            value=15_000_000,
            step=1_000_000,
            help="Cost per kWp installed (default: Rp 15M)"
        )
        
        electricity_rate = st.number_input(
            "Electricity Rate (Rp/kWh)",
            min_value=1000,
            max_value=3000,
            value=1500,
            step=100,
            help="PLN electricity rate (default: Rp 1,500/kWh)"
        )
        
        st.divider()
        
        # System Info
        st.subheader("💻 System Info")
        if torch.cuda.is_available():
            try:
                gpu_info = get_gpu_memory()
                if isinstance(gpu_info, tuple):
                    st.info(f"🎮 GPU Memory: {gpu_info[0]:.2f} GB / {gpu_info[1]:.2f} GB")
                else:
                    st.info("🎮 GPU Available")
            except:
                st.info("🎮 GPU Available")
        else:
            st.warning("⚠️ Running on CPU")

        if st.button("🗑️ Clear Memory"):
            clear_memory()
            st.success("✅ Memory cleared")

    
    # ========================================================================
    # MAIN CONTENT
    # ========================================================================
    
    # Load SAM3 model
    if st.session_state.sam3_model is None:
        with st.spinner("🔄 Loading SAM3 model..."):
            model, processor = load_sam3_model()
            if model and processor:
                st.session_state.sam3_model = model
                st.session_state.sam3_processor = processor
                st.success("✅ SAM3 model loaded successfully!")
            else:
                st.error("❌ Failed to load SAM3 model. Please check sam3.pt exists.")
                st.stop()
    
    # ========================================================================
    # STEP 1: Image Upload
    # ========================================================================
    
    st.header("📂 Step 1: Upload Aerial/Satellite Image")
    
    # 🆕 NEW: Support GeoTIFF
    uploaded_file = st.file_uploader(
        "Choose an aerial or satellite image (GeoTIFF recommended for accuracy)",
        type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],
        help="Upload an aerial view of buildings with roofs/solar panels"
    )
    
    if uploaded_file:
        # 🆕 NEW: Load with GeoTIFF support
        image_pil, geotiff_handler = load_image_with_geotiff_support(uploaded_file)
        st.session_state.uploaded_image = image_pil
        st.session_state.geotiff_handler = geotiff_handler
        
        # 🆕 NEW: Display GeoTIFF info
        if geotiff_handler and geotiff_handler.is_geotiff:
            st.success("✅ **GeoTIFF Detected** - Using Accurate Geospatial Data!")
            
            with st.expander("🗺️ GeoTIFF Metadata", expanded=True):
                metadata = geotiff_handler.metadata
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("🎯 GSD", f"{metadata['gsd']:.4f} m/pixel")
                with col2:
                    st.metric("📐 Size", f"{metadata['width']} × {metadata['height']} px")
                with col3:
                    st.metric("📍 CRS", metadata['crs'][:20] + "...")
                with col4:
                    st.metric("📏 Area", f"{metadata['area_m2']:,.0f} m²")
            
            # Auto-set pixel-to-meter ratio
            pixel_to_meter = geotiff_handler.get_pixel_to_meter_ratio()
            st.info(f"✅ Auto-detected pixel ratio: **{pixel_to_meter:.4f} m/pixel**")
        else:
            st.info("ℹ️ Regular image - Using manual pixel-to-meter ratio")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image_pil, caption="Uploaded Image", use_container_width=True)
        
        # Image info
        st.info(f"📐 Image size: {image_pil.size[0]} x {image_pil.size[1]} pixels")
    
    # ========================================================================
    # STEP 2: Detection Mode Selection
    # ========================================================================
    
    if st.session_state.uploaded_image:
        st.header("🎯 Step 2: Select Detection Mode")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.markdown("### 🏠 Roof Detection")
            st.markdown("""
            Detect all roofs/buildings in the image to:
            - Calculate total roof area
            - Estimate energy consumption
            - Analyze solar potential
            """)
            roof_button = st.button(
                "🏠 DETECT ROOFS",
                key="roof_btn",
                use_container_width=True
            )
        
        with col2:
            st.markdown("### ☀️ Solar Panel Detection")
            st.markdown("""
            Detect existing solar panels to:
            - Calculate current capacity
            - Estimate energy production
            - Analyze coverage ratio
            """)
            panel_button = st.button(
                "☀️ DETECT SOLAR PANELS",
                key="panel_btn",
                use_container_width=True
            )
        
        with col3:
            st.markdown("### 🔄 Both Detections")
            st.markdown("""
            Run both detections sequentially:
            - Complete analysis
            - Coverage comparison
            - Expansion recommendations
            """)
            both_button = st.button(
                "🔄 DETECT BOTH",
                key="both_btn",
                use_container_width=True
            )
        
        # ====================================================================
        # STEP 3: Run Detection
        # ====================================================================
        
        detection_mode = None
        if roof_button:
            detection_mode = "roof"
        elif panel_button:
            detection_mode = "solar_panel"
        elif both_button:
            detection_mode = "both"
        
        if detection_mode:
            
            # Create config
            config = AdvancedConfig()
            config.conf_threshold_large = confidence_threshold
            config.conf_threshold_medium = confidence_threshold
            config.conf_threshold_small = confidence_threshold
            config.min_area_pixels = min_area_pixels
            
            # Run detection(s)
            results = {}
            
            if detection_mode in ["roof", "both"]:
                st.markdown("---")
                st.subheader("🏠 Running Roof Detection...")
                roof_results = run_solar_detection(
                    st.session_state.uploaded_image,
                    "roof",
                    st.session_state.sam3_processor,
                    config,
                    pixel_to_meter,
                    st.session_state.geotiff_handler  # 🆕 NEW parameter
                )
                results['roof'] = roof_results
                st.success(f"✅ Detected {roof_results['n_objects']} roofs!")
            
            if detection_mode in ["solar_panel", "both"]:
                st.markdown("---")
                st.subheader("☀️ Running Solar Panel Detection...")
                panel_results = run_solar_detection(
                    st.session_state.uploaded_image,
                    "solar_panel",
                    st.session_state.sam3_processor,
                    config,
                    pixel_to_meter,
                    st.session_state.geotiff_handler  # 🆕 NEW parameter
                )
                results['solar_panel'] = panel_results
                st.success(f"✅ Detected {panel_results['n_objects']} solar panels!")
            
            # Store results
            st.session_state.detection_results = results
    
    # ========================================================================
    # STEP 4: Display Results & Analysis
    # ========================================================================
    
    if st.session_state.detection_results:
        
        st.markdown("---")
        st.header("📊 Detection Results & Energy Analysis")
        
        results = st.session_state.detection_results
        
        # ====================================================================
        # Display Detection Overlays
        # ====================================================================
        
        st.subheader("🖼️ Detection Visualization")
        
        if len(results) == 1:
            # Single detection
            mode = list(results.keys())[0]
            st.image(
                results[mode]['overlay'],
                caption=f"{mode.replace('_', ' ').title()} Detection",
                use_container_width=True
            )
        else:
            # Both detections
            col1, col2 = st.columns(2)
            with col1:
                st.image(
                    results['roof']['overlay'],
                    caption="Roof Detection",
                    use_container_width=True
                )
            with col2:
                st.image(
                    results['solar_panel']['overlay'],
                    caption="Solar Panel Detection",
                    use_container_width=True
                )
        
        # ====================================================================
        # Calculate Energy Metrics
        # ====================================================================
        
        st.markdown("---")
        st.subheader("⚡ Energy Analysis")
        
        # Extract data
        n_roofs = results.get('roof', {}).get('n_objects', 0)
        roof_area_m2 = results.get('roof', {}).get('area_m2', 0)
        n_panels = results.get('solar_panel', {}).get('n_objects', 0)
        panel_area_m2 = results.get('solar_panel', {}).get('area_m2', 0)
        
        # 🆕 NEW: Use area-based calculations for panels
        if panel_area_m2 > 0:
            # Area-based calculation
            capacity_results = calculate_capacity_from_area(panel_area_m2, panel_type)
            energy_results = calculate_energy_production_from_area(panel_area_m2, panel_type)
            
            production_data = {
                'capacity_kwp': capacity_results['capacity_kwp'],
                'daily_kwh': energy_results['daily_kwh'],
                'annual_kwh': energy_results['annual_kwh'],
                'annual_mwh': energy_results['annual_mwh'],
                'carbon_reduction_tons': calculate_carbon_reduction_from_area(panel_area_m2, panel_type)['annual_co2_ton']
            }
        elif n_panels > 0:
            # Fallback to panel count
            panel_area_m2 = estimate_panel_area_from_count(n_panels)
            production_data = calculate_solar_production(n_panels)
        else:
            production_data = {
                'capacity_kwp': 0,
                'daily_kwh': 0,
                'annual_kwh': 0,
                'annual_mwh': 0,
                'carbon_reduction_tons': 0
            }
        
        # Calculate consumption (if roofs detected)
        if n_roofs > 0:
            consumption_data = calculate_consumption(n_roofs)
        else:
            consumption_data = {
                'daily_kwh': 0,
                'annual_kwh': 0,
                'annual_mwh': 0
            }
        
        # Calculate potential (if roofs detected)
        if n_roofs > 0:
            potential_data = calculate_solar_potential(roof_area_m2, panel_area_m2)
        else:
            potential_data = {
                'available_area_m2': 0,
                'potential_kwp': 0,
                'potential_daily_kwh': 0
            }
        
        # Calculate coverage
        coverage_data = calculate_coverage_analysis(
            production_data['daily_kwh'],
            consumption_data['daily_kwh'],
            potential_data['potential_daily_kwh']
        )
        
        # 🆕 NEW: Use area-based economics
        if panel_area_m2 > 0:
            economics_data_full = calculate_economics_from_area(
                panel_area_m2,
                panel_type,
                cost_per_kwp,
                electricity_rate
            )
            economics_data = {
                'investment_billion_rp': economics_data_full['total_cost'] / 1_000_000_000,
                'annual_savings_million_rp': economics_data_full['annual_savings'] / 1_000_000,
                'payback_years': economics_data_full['payback_years'],
                'roi_lifetime_pct': economics_data_full['roi_percent']
            }
        elif production_data['capacity_kwp'] > 0:
            economics_data = calculate_economics(
                production_data['capacity_kwp'],
                production_data['annual_kwh'],
                cost_per_kwp,
                electricity_rate
            )
        else:
            economics_data = {
                'investment_billion_rp': 0,
                'annual_savings_million_rp': 0,
                'payback_years': 0,
                'roi_lifetime_pct': 0
            }
        
        # ====================================================================
        # Display Metrics
        # ====================================================================
        
        # Detection Summary
        st.markdown("### 📊 Detection Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🏠 Roofs Detected", f"{n_roofs} buildings")
            st.caption(f"Total area: {roof_area_m2:.2f} m²")
        
        with col2:
            st.metric("☀️ Solar Panels", f"{n_panels} panels")
            st.caption(f"Panel area: {panel_area_m2:.2f} m²")
        
        with col3:
            st.metric("⚡ Current Capacity", f"{production_data['capacity_kwp']:.2f} kWp")
            st.caption(f"Production: {production_data['daily_kwh']:.2f} kWh/day")
        
        with col4:
            st.metric("📈 Coverage", f"{coverage_data['current_coverage_pct']:.1f}%")
            st.caption(coverage_data['status'])
        
        # 🆕 NEW: Validation against research
        if panel_area_m2 > 0:
            energy_results = calculate_energy_production_from_area(panel_area_m2, panel_type)
            validation = validate_specific_yield(energy_results['specific_yield'])
            
            if validation['is_valid']:
                st.success(f"✅ Specific Yield Validated: {validation['specific_yield']:.0f} kWh/kWp/year (within research range)")
            else:
                st.warning(f"⚠️ Specific Yield: {validation['specific_yield']:.0f} kWh/kWp/year (deviation: {validation['deviation_percent']:.1f}%)")
        
        # Energy Analysis
        st.markdown("### ⚡ Energy Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔋 Current Production")
            st.info(f"""
            - **Daily**: {production_data['daily_kwh']:.2f} kWh/day
            - **Annual**: {production_data.get('annual_mwh', production_data['annual_kwh']/1000):.2f} MWh/year
            - **CO₂ Reduction**: {production_data['carbon_reduction_tons']:.2f} tons/year
            """)
        
        with col2:
            st.markdown("#### 🏢 Total Consumption")
            st.info(f"""
            - **Buildings**: {n_roofs} units
            - **Daily**: {consumption_data['daily_kwh']:.2f} kWh/day
            - **Annual**: {consumption_data.get('annual_mwh', consumption_data['annual_kwh']/1000):.2f} MWh/year
            - **Basis**: 4.5 kWh/day per building
            """)
        
        # Coverage Analysis
        st.markdown("### 📈 Coverage Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if coverage_data['is_sufficient']:
                st.success(f"""
                ✅ **Self-Sufficient!**
                
                Current coverage: **{coverage_data['current_coverage_pct']:.1f}%**
                
                Surplus: **{coverage_data['surplus_kwh']:.2f} kWh/day**
                """)
            else:
                st.warning(f"""
                ⚠️ **Insufficient Coverage**
                
                Current coverage: **{coverage_data['current_coverage_pct']:.1f}%**
                
                Deficit: **{abs(coverage_data['energy_gap_kwh']):.2f} kWh/day**
                """)
        
        with col2:
            st.info(f"""
            🎯 **Solar Potential**
            
            Available roof: **{potential_data['available_area_m2']:.2f} m²**
            
            Potential capacity: **+{potential_data['potential_kwp']:.2f} kWp**
            
            Potential production: **+{potential_data['potential_daily_kwh']:.2f} kWh/day**
            """)
        
        with col3:
            st.success(f"""
            🚀 **Full Potential**
            
            Total coverage: **{coverage_data['potential_coverage_pct']:.1f}%**
            
            Can power: **{int(coverage_data['potential_coverage_pct'] / 100 * n_roofs)} buildings**
            """)
        
        # Economic Analysis
        if production_data['capacity_kwp'] > 0:
            st.markdown("### 💰 Economic Analysis")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "💵 Investment",
                    f"Rp {economics_data['investment_billion_rp']:.2f}B"
                )
            
            with col2:
                st.metric(
                    "💰 Annual Savings",
                    f"Rp {economics_data['annual_savings_million_rp']:.2f}M"
                )
            
            with col3:
                st.metric(
                    "⏱️ Payback Period",
                    f"{economics_data['payback_years']:.1f} years"
                )
            
            with col4:
                st.metric(
                    "📊 ROI (25 years)",
                    f"{economics_data['roi_lifetime_pct']:.1f}%"
                )
        
        # ====================================================================
        # Investment Scenarios
        # ====================================================================
        
        if potential_data['available_area_m2'] > 0:
            st.markdown("---")
            st.subheader("💡 Investment Scenarios")
            
            scenarios = calculate_investment_scenarios(
                potential_data['available_area_m2']
            )
            
            # Create table
            import pandas as pd
            df_scenarios = pd.DataFrame(scenarios)
            df_scenarios = df_scenarios[[
                'scenario_name',
                'capacity_kwp',
                'investment_billion_rp',
                'annual_savings_million_rp',
                'payback_years',
                'roi_lifetime_pct'
            ]]
            df_scenarios.columns = [
                'Scenario',
                'Capacity (kWp)',
                'Investment (Rp B)',
                'Annual Savings (Rp M)',
                'Payback (years)',
                'ROI 25y (%)'
            ]
            
            st.dataframe(df_scenarios, use_container_width=True)
        
        # ====================================================================
        # Gemini AI Analysis
        # ====================================================================
        
        if st.session_state.gemini_model:
            st.markdown("---")
            st.subheader("🤖 AI Recommendations")
            
            if st.button("🚀 Generate AI Analysis", use_container_width=True):
                with st.spinner("🤖 Analyzing with Gemini AI..."):
                    
                    detection_data = {
                        'n_roofs': n_roofs,
                        'roof_area_m2': roof_area_m2,
                        'n_panels': n_panels,
                        'panel_area_m2': panel_area_m2
                    }
                    
                    analysis_text = analyze_with_gemini(
                        st.session_state.gemini_model,
                        detection_data,
                        production_data,
                        consumption_data,
                        coverage_data,
                        potential_data,
                        economics_data,
                        st.session_state.uploaded_image
                    )
                    
                    st.session_state.analysis_results = analysis_text
            
            # Display analysis
            if st.session_state.analysis_results:
                st.markdown(st.session_state.analysis_results)
        
        # ====================================================================
        # Download Results
        # ====================================================================
        
        st.markdown("---")
        st.subheader("💾 Download Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Download overlay image
            for mode, data in results.items():
                buf = io.BytesIO()
                data['overlay'].save(buf, format='PNG')
                st.download_button(
                    f"📥 Download {mode.replace('_', ' ').title()} Overlay",
                    data=buf.getvalue(),
                    file_name=f"{mode}_detection.png",
                    mime="image/png"
                )
        
        with col2:
            # Download mask
            for mode, data in results.items():
                mask_img = Image.fromarray((data['mask'] * 255).astype(np.uint8))
                buf = io.BytesIO()
                mask_img.save(buf, format='PNG')
                st.download_button(
                    f"📥 Download {mode.replace('_', ' ').title()} Mask",
                    data=buf.getvalue(),
                    file_name=f"{mode}_mask.png",
                    mime="image/png"
                )
        
        with col3:
            # Download report (JSON)
            import json
            
            # Convert numpy types to native Python types
            def convert_to_serializable(obj):
                """Convert numpy types to native Python types"""
                if isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                elif isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.bool_, bool)):
                    return bool(obj)
                else:
                    return obj
            
            report = {
                'detection': {
                    'n_roofs': int(n_roofs),
                    'roof_area_m2': float(roof_area_m2),
                    'n_panels': int(n_panels),
                    'panel_area_m2': float(panel_area_m2)
                },
                'production': convert_to_serializable(production_data),
                'consumption': convert_to_serializable(consumption_data),
                'coverage': convert_to_serializable(coverage_data),
                'potential': convert_to_serializable(potential_data),
                'economics': convert_to_serializable(economics_data)
            }
            
            st.download_button(
                "📥 Download Full Report (JSON)",
                data=json.dumps(report, indent=2),
                file_name="ecopower_roof_analysis_report.json",
                mime="application/json"
            )
    
    # ========================================================================
    # Interactive Chatbot
    # ========================================================================
    
    if st.session_state.detection_results and st.session_state.gemini_model:
        st.markdown("---")
        st.header("💬 Ask Questions About Your Solar Analysis")
        
        # Quick question buttons
        st.markdown("**Quick Questions:**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("❓ How many panels needed?"):
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': 'How many solar panels do I need for full coverage?'
                })
        
        with col2:
            if st.button("❓ Which roof is best?"):
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': 'Which roofs should I prioritize for solar installation?'
                })
        
        with col3:
            if st.button("❓ Calculate ROI"):
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': 'What is the return on investment for solar installation?'
                })
        
        with col4:
            if st.button("❓ Compare with PLN"):
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': 'How much can I save compared to PLN electricity?'
                })
        
        # Chat input
        user_question = st.chat_input("Ask anything about your solar analysis...")
        
        if user_question:
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_question
            })
        
        # Display chat history
        if st.session_state.chat_history:
            for message in st.session_state.chat_history:
                with st.chat_message(message['role']):
                    st.write(message['content'])
            
            # Generate response for last user message
            if st.session_state.chat_history[-1]['role'] == 'user':
                with st.chat_message("assistant"):
                    with st.spinner("🤖 Thinking..."):
                        
                        results = st.session_state.detection_results
                        n_roofs = results.get('roof', {}).get('n_objects', 0)
                        n_panels = results.get('solar_panel', {}).get('n_objects', 0)
                        panel_area_m2 = results.get('solar_panel', {}).get('area_m2', 0)
                        
                        # Build context
                        context = f"""
You are a solar energy expert assistant. Answer based on this analysis:

Detection: {n_roofs} roofs, {n_panels} panels, {panel_area_m2:.2f} m² panel area
Production: {production_data['daily_kwh']:.2f} kWh/day
Consumption: {consumption_data['daily_kwh']:.2f} kWh/day
Coverage: {coverage_data['current_coverage_pct']:.1f}%
Potential: +{potential_data['potential_kwp']:.2f} kWp available
Investment: Rp {economics_data.get('investment_billion_rp', 0):.2f}B
Payback: {economics_data.get('payback_years', 0):.1f} years

User question: {st.session_state.chat_history[-1]['content']}

Answer in Indonesian, be specific with numbers, use emoji.
"""
                        
                        response = st.session_state.gemini_model.generate_content(context)
                        answer = response.text
                        
                        st.write(answer)
                        
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': answer
                        })
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <h3>🎓 EcoPower Roof</h3>
        <p><strong>A prototype project made by Master's students of Geomatics Engineering</strong></p>
        <p><strong>Universitas Gadjah Mada</strong></p>
        <br>
        <p><strong>👥 Team Members:</strong></p>
        <div>
            <span class="team-member">1️⃣ Mohammad Zulfi Rahadi Putra</span>
            <span class="team-member">2️⃣ Raffi Satya Nugraha</span>
            <span class="team-member">3️⃣ Najieda Azka</span>
            <span class="team-member">4️⃣ Salzabila Enzal Putri</span>
        </div>
        <br>
        <p style="font-size: 0.9rem; color: #6b7280;">
            🌱 Powered by SAM3 Advanced & Gemini AI | 🌍 For a Sustainable Future
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()
