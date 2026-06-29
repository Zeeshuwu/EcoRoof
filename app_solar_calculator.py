"""
EcoPower Roof — Solar PV Analysis Dashboard
============================================
Rooftop solar analysis using SAM3 Advanced segmentation,
multi-criteria spatial modelling, and Gemini AI.

Team   : GEO-AI Twinverse Research Group, Faculty of Engineering, Universitas Gadjah Mada
Version: 3.0.0
Date   : June 2026
"""

import io
import json

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

import google.generativeai as genai

# ── SAM3 core ────────────────────────────────────────────────────────────────
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ── SAM3 Advanced ────────────────────────────────────────────────────────────
from sam3_advanced.config import AdvancedConfig
from sam3_advanced.inference import segment_multiscale_advanced
from sam3_advanced.postprocess import post_process_mask
from sam3_advanced.utils import clear_memory, get_gpu_memory
from sam3_advanced.geotiff_utils import load_image_with_geotiff_support, RASTERIO_AVAILABLE

# ── Solar Calculator ─────────────────────────────────────────────────────────
from solar_calculator.models import (
    calculate_solar_production,
    calculate_consumption,
    calculate_solar_potential,
    calculate_coverage_analysis,
    estimate_panel_area_from_count,
)
from solar_calculator.economics import (
    calculate_economics,
    calculate_investment_scenarios,
    calculate_economics_capped,
    compare_with_pln,
    COST_PER_KWP,
    ELECTRICITY_RATE,
    PEAK_SUN_HOURS,
    PERFORMANCE_RATIO,
)
from solar_calculator.analysis import analyze_with_gemini, generate_recommendations
from solar_calculator.export_utils import (
    export_overlay_as_geotiff,
    export_mask_as_shapefile,
    check_export_dependencies,
)
from solar_calculator.utils import (
    pixels_to_area,
    count_objects_in_mask,
    format_currency,
    format_energy,
    convert_to_serializable,
    SOLAR_PANEL_SPECS,
    MAGELANG_SOLAR_DATA,
    calculate_area_from_mask,
    calculate_capacity_from_area,
    calculate_energy_production_from_area,
    calculate_economics_from_area,
    calculate_carbon_reduction_from_area,
    get_research_comparison_data,
    validate_specific_yield,
)
from solar_calculator.usable_area_suitability import (
    # Part 1 — usable area
    calculate_usable_area,
    SUITABILITY_COLORS,
    USABILITY_PARAMS,
    # Part 2 — spatial visualization
    extract_roof_segments,
    score_segments,
    allocate_greedy,
    render_spatial_overlay,
    render_legend,
    build_segment_summary,
    SEGMENT_PALETTE,
    SEGMENT_SCORE_THRESHOLDS,
)

from solar_calculator.district_analysis import (
    compute_district_analysis,
    DISTRICT_COLORS,
    DISTRICT_THRESHOLDS,
)
from solar_calculator.spatial_recommendation import (
    score_all_scenarios,
    AHP_WEIGHTS,
    RECOMMENDATION_COLORS,
)

import pandas as pd

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="EcoPower Roof",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load external CSS ─────────────────────────────────────────────────────────
def _load_css():
    try:
        with open(".streamlit/style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass  # CSS file optional; app still works without it

_load_css()

# ============================================================================
# SESSION STATE
# ============================================================================

_defaults = {
    "sam3_model":        None,
    "sam3_processor":    None,
    "gemini_model":      None,
    "uploaded_image":    None,
    "detection_results": None,
    "analysis_results":  None,
    "chat_history":      [],
    "geotiff_handler":   None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================================
# MODEL LOADERS
# ============================================================================

@st.cache_resource
def load_sam3_model(checkpoint_path: str = "sam3.pt"):
    try:
        model     = build_sam3_image_model(checkpoint_path=checkpoint_path)
        processor = Sam3Processor(model)
        return model, processor
    except Exception as exc:
        st.error(f"SAM3 load error: {exc}")
        return None, None


def load_gemini_model(api_key: str):
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-3.1-flash-lite")
    except Exception as exc:
        st.error(f"Gemini load error: {exc}")
        return None

# ============================================================================
# DETECTION
# ============================================================================

def run_solar_detection(
    image_pil: Image.Image,
    detection_mode: str,
    processor: Sam3Processor,
    config: AdvancedConfig,
    pixel_to_meter_ratio: float = 0.5,
    geotiff_handler=None,
) -> dict:
    """Run SAM3 multiscale detection for roof or solar panel mode."""

    prompts = {
        "roof": {
            "large":  ["roof", "large roof", "big roof", "building roof", "rooftop"],
            "medium": ["roof", "medium roof", "rooftop", "house roof"],
            "small":  ["roof", "small roof", "house roof", "building top"],
        },
        "solar_panel": {
            "large":  ["solar panel", "large solar panel", "photovoltaic array"],
            "medium": ["solar panel", "medium solar panel", "photovoltaic panel", "PV panel"],
            "small":  ["solar panel", "small solar panel", "PV panel", "solar cell"],
        },
    }

    config.prompts_large  = prompts[detection_mode]["large"]
    config.prompts_medium = prompts[detection_mode]["medium"]
    config.prompts_small  = prompts[detection_mode]["small"]

    processor.set_image(image_pil)

    with st.spinner(f"Running {detection_mode.replace('_', ' ')} detection..."):
        final_mask, stats = segment_multiscale_advanced(image_pil, processor, config)

    with st.spinner("Post-processing mask..."):
        mask = post_process_mask(final_mask, config, verbose=False)

    n_objects    = count_objects_in_mask(mask)
    area_m2      = calculate_area_from_mask(mask, pixel_to_meter_ratio, geotiff_handler)
    total_px     = mask.shape[0] * mask.shape[1]
    coverage_pct = (np.sum(mask > 0) / total_px) * 100

    # ── Overlay — high contrast with hard edge + glow effect ──────────────────
    overlay_np = np.array(image_pil.copy())

    # Stronger, more distinct colors
    color = [255, 60, 60] if detection_mode == "roof" else [0, 230, 120]

    # Step 1: Create colored mask layer
    mask_colored = np.zeros_like(overlay_np)
    mask_colored[mask > 0] = color

    # Step 2: Blend — stronger mask presence (was 0.28, now 0.55)
    blended = cv2.addWeighted(overlay_np, 0.45, mask_colored, 0.55, 0)

    # Step 3: Draw a bright hard border around each detected region
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    border_color = (255, 255, 0) if detection_mode == "roof" else (0, 255, 200)
    cv2.drawContours(blended, contours, -1, border_color, thickness=2)

    # Step 4: Only replace pixels where mask is active — keep background sharp
    overlay_np[mask > 0] = blended[mask > 0]

    return {
        "mode":         detection_mode,
        "mask":         mask,
        "overlay":      Image.fromarray(overlay_np),
        "n_objects":    n_objects,
        "area_m2":      area_m2,
        "coverage_pct": coverage_pct,
        "stats":        stats,
    }

# ============================================================================
# HELPERS
# ============================================================================

def _pill(text: str, style: str = "green") -> str:
    return f'<span class="pill pill-{style}">{text}</span>'


def _section(label: str):
    st.markdown(f'<p class="section-label">{label}</p>', unsafe_allow_html=True)


def _card(title: str, body: str):
    st.markdown(
        f'<div class="card"><p class="card-title">{title}</p>'
        f'<p class="card-body">{body}</p></div>',
        unsafe_allow_html=True,
    )

# ============================================================================
# MAIN
# ============================================================================

def main():

# ── Navbar ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="navbar">
        <div class="navbar-brand">
            <p class="navbar-brand-name">EcoPower Roof</p>
            <p class="navbar-brand-sub">Powered by Geo&#8209;AIT UGM</p>
        </div>
        <div class="navbar-links">
            <a class="navbar-link" href="#">Analysis</a>
            <a class="navbar-link" href="#">Methodology</a>
        </div>
        <div class="navbar-actions">
            <span class="navbar-badge-live">
                <span class="navbar-badge-dot"></span>
                GEO&#8209;AI Twinverse &nbsp;·&nbsp; UGM
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <div class="hero-placeholder-bg"></div>
        <div class="hero-content">
            <h1 class="hero-title">Rooftop solar assessment,<br>from imagery to insight</h1>
            <p class="hero-subtitle">
                Upload UAV orthomosaics or aerial imagery. EcoPower Roof detects
                rooftops and existing panels, estimates generation potential,
                and delivers economic projections — all in one workflow.
            </p>
            <div class="hero-actions">
                <a class="hero-btn-primary" href="#">Start Analysis</a>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


    # ── Stats strip ───────────────────────────────────────────────────────────
    st.markdown("""
    <div class="stats-strip">
        <div class="stat-item">
            <p class="stat-value">4.5<span>h</span></p>
            <p class="stat-label">Peak Sun Hours/day · Java avg.</p>
        </div>
        <div class="stat-item">
            <p class="stat-value">25<span>yr</span></p>
            <p class="stat-label">Panel lifespan projection</p>
        </div>
        <div class="stat-item">
            <p class="stat-value">0.75<span>PR</span></p>
            <p class="stat-label">Performance Ratio · Tarigan 2025</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Features ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="section-heading">
        <h2>Platform capabilities</h2>
        <p>A five-stage workflow from raw imagery to decision-ready outputs</p>
    </div>

    <div class="features-row">
        <div class="feature-card">
            <div class="feature-icon-wrap icon-green">🛰️</div>
            <p class="feature-title">SAM3 Multi-Level Detection</p>
            <p class="feature-body">Three-scale patch segmentation (256, 512 px, full image) with NMS fusion for accurate rooftop and PV extraction.</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon-wrap icon-teal">📐</div>
            <p class="feature-title">Usable Area Estimation</p>
            <p class="feature-body">Composite reduction factors from Burke (2021) and Res4Africa (2024) applied to derive net installable rooftop area.</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon-wrap icon-amber">⚡</div>
            <p class="feature-title">PV Potential & Energy Balance</p>
            <p class="feature-body">Capacity and generation estimated via E = A × G × η × PR using Java irradiance data (Tarigan et al., 2025).</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon-wrap icon-blue">📊</div>
            <p class="feature-title">Economic Feasibility</p>
            <p class="feature-body">ROI, payback period, and NPV calculated per scenario. Benchmarked against Indonesian PV cost studies.</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon-wrap icon-purple">🗺️</div>
            <p class="feature-title">Spatial Recommendation</p>
            <p class="feature-body">AHP-weighted MCDA classifies each roof segment into four deployment priority tiers with greedy allocation.</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon-wrap icon-green">🤖</div>
            <p class="feature-title">LLM-Assisted Interpretation</p>
            <p class="feature-body">Gemini AI generates journal-quality analysis reports and answers natural-language queries about the results.</p>
        </div>
    </div>

    <div class="cta-strip">
        <a class="hero-btn-primary" href="#">Start Analysis</a>
    </div>
    """, unsafe_allow_html=True)

    # =========================================================================
    # SIDEBAR
    # =========================================================================

    with st.sidebar:

        st.markdown("""
        <div class="sidebar-brand">
            <p class="sidebar-brand-name">☀️ EcoPower Roof</p>
            <p class="sidebar-brand-sub">GEO-AI Twinverse · UGM</p>
        </div>
        """, unsafe_allow_html=True)

        # ── Gemini API ────────────────────────────────────────────────────────
        try:
            gemini_api_key = st.secrets["GEMINI_API_KEY"]
            st.markdown(_pill("Gemini AI ready", "green"), unsafe_allow_html=True)
        except Exception:
            gemini_api_key = st.text_input(
                "Gemini API Key", type="password",
                placeholder="Paste your API key...",
            )

        if gemini_api_key and not st.session_state.gemini_model:
            st.session_state.gemini_model = load_gemini_model(gemini_api_key)

        st.divider()

        # ── Step 1: Image Settings ────────────────────────────────────────────
        with st.expander("📐 Image & Detection", expanded=True):
            pixel_to_meter = st.number_input(
                "Pixel / meter ratio",
                min_value=0.1, max_value=2.0, value=0.5, step=0.1,
                help="Set automatically if GeoTIFF is uploaded",
            )
            confidence_threshold = st.slider(
                "Confidence threshold", 0.0, 1.0, 0.5, 0.05,
            )
            min_area_pixels = st.number_input(
                "Min object area (px)",
                min_value=10, max_value=1000, value=100, step=10,
            )

        # ── Step 2: Panel & Economics ─────────────────────────────────────────
        with st.expander("⚡ Panel & Economics", expanded=False):
            panel_type = st.selectbox(
                "Panel type",
                options=list(SOLAR_PANEL_SPECS.keys()),
                format_func=lambda x: SOLAR_PANEL_SPECS[x]["name"],
            )
            selected_panel = SOLAR_PANEL_SPECS[panel_type]
            st.caption(
                f"{selected_panel['power_density_wp_m2']} Wp/m² · "
                f"{selected_panel['panel_size_m2']} m² · "
                f"{selected_panel['efficiency']*100:.0f}% eff."
            )

            st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

            cost_per_kwp = st.number_input(
                "Install cost (Rp/kWp)",
                min_value=10_000_000, max_value=25_000_000,
                value=15_000_000, step=1_000_000,
            )
            electricity_rate = st.number_input(
                "Electricity rate (Rp/kWh)",
                min_value=1000, max_value=3000, value=1500, step=100,
            )

        # ── System status ─────────────────────────────────────────────────────
        with st.expander("🖥️ System", expanded=False):
            if torch.cuda.is_available():
                try:
                    gpu = get_gpu_memory()
                    label = f"GPU {gpu[0]:.1f}/{gpu[1]:.1f} GB" if isinstance(gpu, tuple) else "GPU available"
                    st.markdown(_pill(label, "green"), unsafe_allow_html=True)
                except Exception:
                    st.markdown(_pill("GPU available", "green"), unsafe_allow_html=True)
            else:
                st.markdown(_pill("CPU mode", "yellow"), unsafe_allow_html=True)

            st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
            if st.button("Clear memory", use_container_width=True):
                clear_memory()
                st.success("Memory cleared")


    # =========================================================================
    # MODEL LOAD
    # =========================================================================

    if st.session_state.sam3_model is None:
        with st.spinner("Loading SAM3 model..."):
            model, processor = load_sam3_model()
            if model and processor:
                st.session_state.sam3_model     = model
                st.session_state.sam3_processor = processor
            else:
                st.error("Failed to load SAM3 model. Ensure sam3.pt is present.")
                st.stop()

    # =========================================================================
    # STEP 1 — UPLOAD
    # =========================================================================

    _section("Step 1 — Upload Image")

    uploaded_file = st.file_uploader(
        "Upload an aerial or satellite image. GeoTIFF recommended for accurate measurements.",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
    )

    if uploaded_file:
        image_pil, geotiff_handler = load_image_with_geotiff_support(uploaded_file)
        st.session_state.uploaded_image  = image_pil
        st.session_state.geotiff_handler = geotiff_handler

        if geotiff_handler and geotiff_handler.is_geotiff:
            meta = geotiff_handler.metadata
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("GSD",  f"{meta['gsd']:.4f} m/px")
            c2.metric("Size", f"{meta['width']} × {meta['height']} px")
            c3.metric("CRS",  meta["crs"][:20])
            c4.metric("Area", f"{meta['area_m2']:,.0f} m²")
            pixel_to_meter = geotiff_handler.get_pixel_to_meter_ratio()
            st.markdown(
                _pill(f"GeoTIFF — pixel ratio auto-set to {pixel_to_meter:.4f} m/px", "green"),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                _pill("Standard image — using manual pixel ratio", "gray"),
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image(image_pil, use_container_width=True)
        st.caption(f"{image_pil.size[0]} × {image_pil.size[1]} px")

    # =========================================================================
    # STEP 2 — DETECTION
    # =========================================================================

    if st.session_state.uploaded_image:

        st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)
        _section("Step 2 — Detection Mode")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("""
            <div class="card">
                <p class="card-title">Roof Detection</p>
                <p class="card-body">
                    Detect all rooftops to calculate total area,
                    estimate consumption and solar potential.
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<div class="detect-btn">', unsafe_allow_html=True)
            roof_btn = st.button("Detect Roofs", key="roof_btn", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("""
            <div class="card">
                <p class="card-title">Solar Panel Detection</p>
                <p class="card-body">
                    Detect existing panels to calculate current
                    capacity, production and coverage ratio.
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<div class="detect-btn">', unsafe_allow_html=True)
            panel_btn = st.button("Detect Solar Panels", key="panel_btn", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c3:
            st.markdown("""
            <div class="card">
                <p class="card-title">Full Analysis</p>
                <p class="card-body">
                    Run both detections sequentially for a complete
                    coverage comparison and expansion plan.
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<div class="detect-btn">', unsafe_allow_html=True)
            both_btn = st.button("Detect Both", key="both_btn", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Run detection ─────────────────────────────────────────────────────
        detection_mode = (
            "roof"        if roof_btn  else
            "solar_panel" if panel_btn else
            "both"        if both_btn  else None
        )

        if detection_mode:
            config = AdvancedConfig()
            config.conf_threshold_large  = confidence_threshold
            config.conf_threshold_medium = confidence_threshold
            config.conf_threshold_small  = confidence_threshold
            config.min_area_pixels       = min_area_pixels

            det_results = {}

            if detection_mode in ("roof", "both"):
                st.divider()
                det_results["roof"] = run_solar_detection(
                    st.session_state.uploaded_image, "roof",
                    st.session_state.sam3_processor, config,
                    pixel_to_meter, st.session_state.geotiff_handler,
                )
                st.markdown(
                    _pill(f"Detected {det_results['roof']['n_objects']} roofs", "green"),
                    unsafe_allow_html=True,
                )

            if detection_mode in ("solar_panel", "both"):
                st.divider()
                det_results["solar_panel"] = run_solar_detection(
                    st.session_state.uploaded_image, "solar_panel",
                    st.session_state.sam3_processor, config,
                    pixel_to_meter, st.session_state.geotiff_handler,
                )
                st.markdown(
                    _pill(f"Detected {det_results['solar_panel']['n_objects']} solar panels", "green"),
                    unsafe_allow_html=True,
                )

            st.session_state.detection_results = det_results

    # =========================================================================
    # STEP 3 — RESULTS (tabbed)
    # =========================================================================

    if st.session_state.detection_results:

        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
        _section("Step 3 — Results & Analysis")

        results = st.session_state.detection_results

        # ── Base values ───────────────────────────────────────────────────────
        n_roofs       = results.get("roof",        {}).get("n_objects", 0)
        roof_area_m2  = results.get("roof",        {}).get("area_m2",   0.0)
        n_panels      = results.get("solar_panel", {}).get("n_objects", 0)
        panel_area_m2 = results.get("solar_panel", {}).get("area_m2",   0.0)

        # ── Production ────────────────────────────────────────────────────────
        if panel_area_m2 > 0:
            cap_res  = calculate_capacity_from_area(panel_area_m2, panel_type)
            eng_res  = calculate_energy_production_from_area(panel_area_m2, panel_type)
            co2_res  = calculate_carbon_reduction_from_area(panel_area_m2, panel_type)
            production_data = {
                "capacity_kwp":         cap_res["capacity_kwp"],
                "daily_kwh":            eng_res["daily_kwh"],
                "annual_kwh":           eng_res["annual_kwh"],
                "annual_mwh":           eng_res["annual_mwh"],
                "carbon_reduction_tons":co2_res["annual_co2_ton"],
            }
        elif n_panels > 0:
            panel_area_m2   = estimate_panel_area_from_count(n_panels)
            production_data = calculate_solar_production(n_panels)
        else:
            production_data = {
                "capacity_kwp": 0.0, "daily_kwh": 0.0,
                "annual_kwh": 0.0,   "annual_mwh": 0.0,
                "carbon_reduction_tons": 0.0,
            }

        # ── Consumption ───────────────────────────────────────────────────────
        consumption_data = (
            calculate_consumption(n_roofs) if n_roofs > 0
            else {"daily_kwh": 0.0, "annual_kwh": 0.0, "annual_mwh": 0.0}
        )
        consumption_annual_kwh = consumption_data.get("annual_kwh", 0.0)

        # ── Potential ─────────────────────────────────────────────────────────
        potential_data = (
            calculate_solar_potential(roof_area_m2, panel_area_m2) if n_roofs > 0
            else {"available_area_m2": 0.0, "potential_kwp": 0.0, "potential_daily_kwh": 0.0}
        )

        # ── Coverage ──────────────────────────────────────────────────────────
        coverage_data = calculate_coverage_analysis(
            production_data["daily_kwh"],
            consumption_data["daily_kwh"],
            potential_data["potential_daily_kwh"],
        )

        # ── Usable area suitability ───────────────────────────────────────────
        usability_result = calculate_usable_area(roof_area_m2, building_type=panel_type)
        usable_area_m2   = usability_result["usable_area_m2"]

        # ── Economics — current panels ────────────────────────────────────────
        if production_data["capacity_kwp"] > 0:
            economics_current = calculate_economics(
                production_data["capacity_kwp"],
                production_data["annual_kwh"],
                cost_per_kwp,
                electricity_rate,
            )
        else:
            economics_current = {
                "capacity_kwp": 0.0, "investment_billion_rp": 0.0,
                "annual_savings_million_rp": 0.0, "payback_years": 0.0,
                "roi_lifetime_pct": 0.0, "npv_billion_rp": 0.0,
                "lcoe_rp_per_kwh": 0.0, "annual_co2_reduction_tons": 0.0,
                "is_economically_viable": False,
            }

        # ── Economics — capped at 100% self-sufficiency ───────────────────────
        economics_capped = calculate_economics_capped(
            consumption_annual_kwh=consumption_annual_kwh,
            usable_area_m2=usable_area_m2,
            panel_efficiency=0.20,
            cost_per_kwp=cost_per_kwp,
            electricity_rate=electricity_rate,
        )

        # ── Investment scenarios ──────────────────────────────────────────────
        scenarios = calculate_investment_scenarios(
            usable_area_m2,
            scenarios=[0.25, 0.50, 0.75, 1.0],
            cost_per_kwp=cost_per_kwp,
            electricity_rate=electricity_rate,
        )

        # ── District analysis ─────────────────────────────────────────────────
        zone_name = (
            getattr(st.session_state.geotiff_handler, "filename", "Analysis Zone")
            if st.session_state.geotiff_handler else "Analysis Zone"
        )
        district_result = compute_district_analysis(
            zone_name=zone_name,
            gross_area_m2=roof_area_m2,
            usable_area_m2=usable_area_m2,
            existing_panel_area_m2=panel_area_m2,
            geotiff_handler=st.session_state.geotiff_handler,
            pln_tariff=electricity_rate,
        )

        # ── Spatial recommendation ────────────────────────────────────────────
        irradiance = (
            st.session_state.geotiff_handler.metadata.get("ghi", PEAK_SUN_HOURS)
            if st.session_state.geotiff_handler
               and getattr(st.session_state.geotiff_handler, "is_geotiff", False)
            else PEAK_SUN_HOURS
        )
        spatial_scores = score_all_scenarios(
            scenarios_economic=scenarios,
            usable_area_m2=usable_area_m2,
            irradiance_kwh_m2_day=irradiance,
            building_type=panel_type,
            near_grid=True,
            road_access=True,
        )

        # ── Roof spatial suitability (per-segment scoring + greedy alloc) ────
        roof_segments = []
        spatial_overlay_img = None

        if "roof" in results and results["roof"]["mask"] is not None:
            roof_segments = extract_roof_segments(
                mask_np        = results["roof"]["mask"],
                pixel_to_meter = pixel_to_meter,
                min_area_px    = min_area_pixels,
            )
            if roof_segments:
                roof_segments = score_segments(roof_segments)
                roof_segments = allocate_greedy(
                    segments               = roof_segments,
                    consumption_annual_kwh = consumption_annual_kwh,
                    usable_factor          = usability_result["building_type_ratio"]
                                            * usability_result["setback_factor"]
                                            * usability_result["hvac_obstruction_factor"]
                                            * usability_result["structural_factor"],
                )
                spatial_overlay_img = render_spatial_overlay(
                    image_pil     = st.session_state.uploaded_image,
                    segments      = roof_segments,
                    show_labels   = True,
                )
        # ── PLN comparison ────────────────────────────────────────────────────
        pln_comparison = compare_with_pln(
            consumption_annual_kwh,
            production_data["annual_kwh"],
            electricity_rate,
        )

        # =====================================================================
        # TABS
        # =====================================================================

        (tab_detect, tab_energy, tab_usability,
         tab_district, tab_spatial, tab_econ,
         tab_ai, tab_export) = st.tabs([
            "Detection",
            "Energy",
            "Usable Area",
            "District Analysis",
            "Spatial Recommendation",
            "Economics",
            "AI Analysis",
            "Export",
        ])

        # ─────────────────────────────────────────────────────────────────────
        # TAB 1 — Detection
        # ─────────────────────────────────────────────────────────────────────
        with tab_detect:

            if len(results) == 1:
                mode = list(results.keys())[0]
                st.image(
                    results[mode]["overlay"],
                    caption=mode.replace("_", " ").title(),
                    use_container_width=True,
                )
            else:
                c1, c2 = st.columns(2)
                with c1:
                    st.image(results["roof"]["overlay"],
                             caption="Roof Detection", use_container_width=True)
                with c2:
                    st.image(results["solar_panel"]["overlay"],
                             caption="Solar Panel Detection", use_container_width=True)

            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Roofs",       f"{n_roofs}")
            c2.metric("Roof Area",   f"{roof_area_m2:,.1f} m²")
            c3.metric("Panels",      f"{n_panels}")
            c4.metric("Panel Area",  f"{panel_area_m2:,.1f} m²")

        # ─────────────────────────────────────────────────────────────────────
        # TAB 2 — Energy
        # ─────────────────────────────────────────────────────────────────────
        with tab_energy:

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Capacity",    f"{production_data['capacity_kwp']:.2f} kWp")
            c2.metric("Production",  f"{production_data['daily_kwh']:.1f} kWh/day")
            c3.metric("Consumption", f"{consumption_data['daily_kwh']:.1f} kWh/day")
            c4.metric("Coverage",    f"{coverage_data['current_coverage_pct']:.1f}%")

            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                _section("Current Production")
                st.markdown(f"""
                <table class="info-table">
                    <tr><td>Daily</td><td>{production_data['daily_kwh']:.2f} kWh/day</td></tr>
                    <tr><td>Annual</td><td>{production_data['annual_mwh']:.2f} MWh/year</td></tr>
                    <tr><td>CO₂ saved</td><td>{production_data['carbon_reduction_tons']:.2f} t/year</td></tr>
                </table>
                """, unsafe_allow_html=True)

            with c2:
                _section("Estimated Consumption")
                st.markdown(f"""
                <table class="info-table">
                    <tr><td>Buildings</td><td>{n_roofs} units</td></tr>
                    <tr><td>Daily</td><td>{consumption_data['daily_kwh']:.2f} kWh/day</td></tr>
                    <tr><td>Annual</td><td>{consumption_data.get('annual_mwh', consumption_data['annual_kwh']/1000):.2f} MWh/year</td></tr>
                    <tr><td>Basis</td><td>4.5 kWh/day per building</td></tr>
                </table>
                """, unsafe_allow_html=True)

            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

            if coverage_data["is_sufficient"]:
                st.success(
                    f"Self-sufficient — coverage {coverage_data['current_coverage_pct']:.1f}% "
                    f"| surplus {coverage_data['surplus_kwh']:.2f} kWh/day"
                )
            else:
                st.warning(
                    f"Insufficient — coverage {coverage_data['current_coverage_pct']:.1f}% "
                    f"| deficit {abs(coverage_data['energy_gap_kwh']):.2f} kWh/day"
                )

            # Specific yield validation
            if panel_area_m2 > 0:
                eng_r = calculate_energy_production_from_area(panel_area_m2, panel_type)
                val   = validate_specific_yield(eng_r["specific_yield"])
                if val["is_valid"]:
                    st.success(
                        f"Specific yield validated: {val['specific_yield']:.0f} kWh/kWp/year "
                        f"(within research range)"
                    )
                else:
                    st.warning(
                        f"Specific yield: {val['specific_yield']:.0f} kWh/kWp/year "
                        f"(deviation {val['deviation_percent']:.1f}%)"
                    )

            # PLN comparison
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            _section("PLN Comparison")
            c1, c2, c3 = st.columns(3)
            c1.metric("Full PLN Cost",        f"Rp {pln_comparison['annual_pln_cost_million_rp']:.2f}M/yr")
            c2.metric("Cost with Solar",      f"Rp {pln_comparison['annual_cost_with_solar_million_rp']:.2f}M/yr")
            c3.metric("Self-sufficiency",     f"{pln_comparison['self_sufficiency_pct']:.1f}%")

        # ─────────────────────────────────────────────────────────────────────
        # TAB 3 — Usable Area Suitability
        # ─────────────────────────────────────────────────────────────────────
        with tab_usability:

            st.caption(
                "Reduction factors — Burke (2021) · Res4Africa (2024) · "
                "Jakubiec & Reinhart (2013) · Duffie & Beckman (2013)"
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Gross Roof Area",  f"{roof_area_m2:,.1f} m²")
            c2.metric("Usable Area",      f"{usable_area_m2:,.1f} m²")
            c3.metric("Usability Ratio",  f"{usability_result['usability_ratio']:.1%}")
            c4.metric("Class",            usability_result["suitability_class"])

            cls   = usability_result["suitability_class"]
            color = SUITABILITY_COLORS.get(cls, "#6b7280")
            st.markdown(
                f'<div style="margin:1rem 0;">'
                f'<span class="pill" style="background:{color}20;color:{color};'
                f'border:1px solid {color}60;font-size:0.82rem;padding:0.3rem 0.9rem;">'
                f'Suitability class: {cls}</span></div>',
                unsafe_allow_html=True,
            )

            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            _section("Reduction Factor Breakdown")

            bd = usability_result
            df_bd = pd.DataFrame([
                ("Gross Roof Area",        f"{roof_area_m2:,.2f} m²",
                 "—",       "SAM3 detection output"),
                ("Building Type Ratio",    f"× {bd['building_type_ratio']:.3f}",
                 f"−{(1-bd['building_type_ratio'])*100:.0f}%", "Res4Africa (2024), Table 4"),
                ("Setback Factor",         f"× {bd['setback_factor']:.3f}",
                 "−10%",  "Burke (2021) — 1 m edge exclusion"),
                ("HVAC Obstruction",       f"× {bd['hvac_obstruction_factor']:.3f}",
                 "−5%",   "Burke (2021) — vents & chimneys"),
                ("Structural Factor",      f"× {bd['structural_factor']:.3f}",
                 "−5%",   "Burke (2021) — structural constraints"),
                ("Tilt Efficiency",        f"× {bd['tilt_efficiency_factor']:.3f}",
                 "−3%",   "Duffie & Beckman (2013) — 15° rack, Java lat."),
                ("Combined Reduction",     f"= {bd['combined_reduction']:.4f}",
                 f"−{(1-bd['combined_reduction'])*100:.1f}%", "Composite"),
                ("Usable Area",            f"{usable_area_m2:,.2f} m²",
                 f"{bd['usability_ratio']:.1%} of gross", "Final result"),
            ], columns=["Factor", "Value", "Reduction", "Reference"])
            st.dataframe(df_bd, use_container_width=True, hide_index=True)

            st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
            _section("Suitability Thresholds")
            thresh_df = pd.DataFrame([
                {"Class": "High",       "Usability Ratio": "≥ 55%",  "Color": SUITABILITY_COLORS["High"]},
                {"Class": "Medium",     "Usability Ratio": "40–55%", "Color": SUITABILITY_COLORS["Medium"]},
                {"Class": "Low",        "Usability Ratio": "20–40%", "Color": SUITABILITY_COLORS["Low"]},
                {"Class": "Unsuitable", "Usability Ratio": "< 20%",  "Color": SUITABILITY_COLORS["Unsuitable"]},
            ])
            for _, row in thresh_df.iterrows():
                marker = " ◀ current" if row["Class"] == cls else ""
                st.markdown(
                    f'<span class="pill" style="background:{row["Color"]}20;'
                    f'color:{row["Color"]};border:1px solid {row["Color"]}60;'
                    f'margin-right:0.5rem;">{row["Class"]}</span>'
                    f'<span style="font-size:0.82rem;color:#4a5e4a;">'
                    f'{row["Usability Ratio"]}{marker}</span><br>',
                    unsafe_allow_html=True,
                )

        # ─────────────────────────────────────────────────────────────────────
        # TAB 4 — District Analysis
        # ─────────────────────────────────────────────────────────────────────
        with tab_district:

            st.caption(
                "Zone boundary = GeoTIFF bounding box | "
                "Classification: Bandung Multi-Criteria Study (2022)"
            )

            dr        = district_result
            cls_color = dr["suitability_color"]

            st.markdown(
                f'<div style="margin-bottom:1rem;">'
                f'<span class="pill" style="background:{cls_color}20;color:{cls_color};'
                f'border:1px solid {cls_color}60;font-size:0.82rem;padding:0.3rem 0.9rem;">'
                f'Zone: {dr["zone_name"]} — {dr["suitability_class"]}</span></div>',
                unsafe_allow_html=True,
            )

            if dr["zone_bbox"]:
                st.caption(
                    f"CRS: {dr['zone_crs']} | "
                    f"BBox: {[round(x, 4) for x in dr['zone_bbox']]} | "
                    f"Zone area: {dr['zone_area_m2']:,.0f} m²"
                )
            else:
                st.caption("No GeoTIFF — zone boundary from image dimensions")

            st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                _section("Existing System")
                st.metric("Capacity",      f"{dr['existing_capacity_kwp']:.2f} kWp")
                st.metric("Annual Energy", f"{dr['existing_annual_kwh']:,.0f} kWh/yr")
                st.metric("Annual Revenue",f"Rp {dr['existing_revenue_idr']/1e6:.2f}M")

            with c2:
                _section("Full Potential")
                st.metric("Capacity",      f"{dr['potential_capacity_kwp']:.2f} kWp")
                st.metric("Annual Energy", f"{dr['potential_annual_kwh']:,.0f} kWh/yr")
                st.metric("Annual Revenue",f"Rp {dr['potential_revenue_idr']/1e6:.2f}M")

            with c3:
                _section("Remaining Opportunity")
                st.metric("Capacity",
                          f"{dr['remaining_capacity_kwp']:.2f} kWp",
                          delta=f"+{dr['remaining_capacity_kwp']:.2f} kWp")
                st.metric("Annual Energy", f"{dr['remaining_annual_kwh']:,.0f} kWh/yr")
                st.metric("Annual Revenue",f"Rp {dr['remaining_revenue_idr']/1e6:.2f}M")

            st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
            _section("District Suitability Classification")
            thresh_df2 = pd.DataFrame([
                {
                    "Class":              cls_name,
                    "Avg Usability Ratio":f"{lo:.0%} – {hi:.0%}",
                    "Current Zone":       "Yes" if cls_name == dr["suitability_class"] else "",
                }
                for cls_name, (lo, hi) in DISTRICT_THRESHOLDS.items()
            ])
            st.dataframe(thresh_df2, use_container_width=True, hide_index=True)

            with st.expander("References"):
                for ref in dr["references"]:
                    st.markdown(f"- {ref}")

        # ─────────────────────────────────────────────────────────────────────
        # TAB 5 — Spatial Recommendation
        # ─────────────────────────────────────────────────────────────────────
        with tab_spatial:

            st.caption(
                "AHP-weighted MCDA | Weights: E3S Conferences (2024) CR=0.04 | "
                "Classes: Bandung Multi-Criteria Study (2022) | "
                "Allocation: greedy top-down capped at consumption target"
            )

            # ── Limitation notice ─────────────────────────────────────────────
            with st.expander("Methodological note (journal limitation)"):
                st.markdown("""
                **Uniform irradiance assumption:**
                Since SAM3 produces a single binary roof mask without per-pixel
                irradiance, orientation, or tilt data, all roof segments receive
                identical irradiance scores. Segment ranking is therefore driven
                primarily by **area** (weight 0.55) and **shape compactness**
                (weight 0.25). Future work should integrate per-pixel DSM or
                irradiance rasters to differentiate shading and tilt per roof face.

                **Greedy allocation:**
                The optimal install zone is determined by filling the
                highest-ranked roofs first until the annual consumption target
                is met. Partial fills are shown where a roof only needs to be
                partially covered.
                """)

            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

            # ── Visual overlay ────────────────────────────────────────────────
            if spatial_overlay_img is not None:

                _section("Roof Suitability Map")

                c_img, c_leg = st.columns([3, 1])

                with c_img:
                    st.image(
                        spatial_overlay_img,
                        caption="Roof suitability + recommended installation zones",
                        use_container_width=True,
                    )

                with c_leg:
                    legend_img = render_legend(width=260)
                    st.image(legend_img, use_container_width=True)

                    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

                    # Allocation summary pills
                    _section("Allocation Summary")
                    full_count    = sum(1 for s in roof_segments if s.get("allocation_status") == "Full")
                    partial_count = sum(1 for s in roof_segments if s.get("allocation_status") == "Partial")
                    skip_count    = sum(1 for s in roof_segments if s.get("allocation_status") == "Not needed")
                    unsuit_count  = sum(1 for s in roof_segments if s.get("allocation_status") == "Unsuitable")

                    total_install_area = sum(
                        s.get("fill_area_m2", 0.0) for s in roof_segments
                    )
                    total_production   = sum(
                        s.get("fill_production", 0.0) for s in roof_segments
                    )

                    st.markdown(f"""
                    <table class="info-table">
                        <tr><td>Full install</td><td>{full_count} roof(s)</td></tr>
                        <tr><td>Partial install</td><td>{partial_count} roof(s)</td></tr>
                        <tr><td>Not needed</td><td>{skip_count} roof(s)</td></tr>
                        <tr><td>Unsuitable</td><td>{unsuit_count} roof(s)</td></tr>
                        <tr><td>Total install area</td><td>{total_install_area:,.1f} m²</td></tr>
                        <tr><td>Est. production</td><td>{total_production:,.0f} kWh/yr</td></tr>
                    </table>
                    """, unsafe_allow_html=True)

            else:
                st.info("Run Roof Detection first to generate the spatial suitability map.")

            st.divider()

            # ── Per-segment table ─────────────────────────────────────────────
            if roof_segments:
                _section("Per-Segment Breakdown")
                summary_rows = build_segment_summary(roof_segments)
                df_seg = pd.DataFrame(summary_rows)
                st.dataframe(df_seg, use_container_width=True, hide_index=True)

                # Best segment callout
                best_seg = roof_segments[0]
                st.success(
                    f"Highest priority: Roof #{best_seg['id']} — "
                    f"score {best_seg['composite_score']:.3f} | "
                    f"area {best_seg['area_m2']:.1f} m² | "
                    f"status: {best_seg.get('allocation_status', '—')}"
                )

            st.divider()

            # ── Scenario ranking (existing AHP across utilization %) ──────────
            _section("Utilization Scenario Ranking")
            st.caption("Scores 25/50/75/100% utilization strategies across the full usable area")

            with st.expander("AHP criteria weights"):
                w_df = pd.DataFrame([
                    {"Criterion": k, "Weight": f"{v:.0%}",
                     "Source": "E3S Conferences (2024)"}
                    for k, v in AHP_WEIGHTS.items()
                ])
                st.dataframe(w_df, use_container_width=True, hide_index=True)
                st.caption("Consistency Ratio (CR) = 0.04 < 0.10 — acceptable")

            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

            for s in spatial_scores:
                rc = s["recommendation_color"]
                st.markdown(
                    f'<div class="rec-row" style="border-left:3px solid {rc};">'
                    f'<p class="rec-row-rank">Rank #{s["rank"]}</p>'
                    f'<p class="rec-row-title">{s["scenario_name"]}</p>'
                    f'<span class="pill" style="background:{rc}20;color:{rc};'
                    f'border:1px solid {rc}60;">{s["recommendation"]}</span>'
                    f'&nbsp;&nbsp;'
                    f'<span style="font-size:0.8rem;color:#5a6e5a;">'
                    f'Score: <strong>{s["composite_score"]:.3f}</strong></span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Area",      f"{s['area_used_m2']:,.1f} m²")
                c2.metric("Capacity",  f"{s['capacity_kwp']:.1f} kWp")
                c3.metric("Payback",   f"{s['payback_years']:.1f} yr")
                c4.metric("CO₂ saved", f"{s['annual_co2_reduction_tons']:.1f} t/yr")

                with st.expander(f"Criteria breakdown — {s['scenario_name']}"):
                    crit_df = pd.DataFrame([
                        {
                            "Criterion":      k,
                            "Raw Score":      f"{v:.4f}",
                            "Weight":         f"{AHP_WEIGHTS[k]:.0%}",
                            "Weighted Score": f"{s['weighted_scores'][k]:.4f}",
                        }
                        for k, v in s["criteria_scores"].items()
                    ])
                    st.dataframe(crit_df, use_container_width=True, hide_index=True)

                st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)

            best = spatial_scores[0]
            st.success(
                f"Recommended scenario: {best['scenario_name']} — "
                f"score {best['composite_score']:.3f} | "
                f"payback {best['payback_years']:.1f} yr"
            )

        # ─────────────────────────────────────────────────────────────────────
        # TAB 6 — Economics
        # ─────────────────────────────────────────────────────────────────────
        with tab_econ:

            st.caption(
                "Tarigan et al. (2025) · Suparwoko & Qamar (2022) · "
                "Cantiqa & Dirkareshza (2025) · Kunaifi et al. (2020)"
            )

            ec = economics_capped

            # Feasibility banner
            if ec["is_feasible"]:
                st.success(
                    f"100% self-sufficiency is feasible — "
                    f"needs {ec['area_needed_for_100pct_m2']:,.1f} m² "
                    f"of {ec['area_available_m2']:,.1f} m² available "
                    f"({ec['area_needed_for_100pct_m2']/max(ec['area_available_m2'],1)*100:.1f}% of usable area)"
                )
            else:
                st.warning(
                    f"100% self-sufficiency not fully achievable — "
                    f"max achievable: {ec['actual_coverage_pct']:.1f}%"
                )

            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

            # ── A — Currently detected panels ────────────────────────────────
            _section("A — Currently Detected Panels")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Capacity",       f"{economics_current['capacity_kwp']:.2f} kWp")
            c2.metric("Annual Savings", f"Rp {economics_current['annual_savings_million_rp']:.2f}M/yr")
            c3.metric("Payback",        f"{economics_current['payback_years']:.1f} yr")
            c4.metric("ROI (25 yr)",    f"{economics_current['roi_lifetime_pct']:.1f}%")

            c1, c2, c3 = st.columns(3)
            c1.metric("LCOE",           f"Rp {economics_current['lcoe_rp_per_kwh']:,.0f}/kWh")
            c2.metric("CO₂ saved",      f"{economics_current['annual_co2_reduction_tons']:.2f} t/yr")
            c3.metric("Viable",
                      "Yes" if economics_current["is_economically_viable"] else "No")

            st.divider()

            # ── B — To reach 100% ────────────────────────────────────────────
            _section("B — To Reach 100% Self-Sufficiency")
            t100 = ec["to_100pct"]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Capacity needed", f"{t100['capacity_kwp']:.2f} kWp")
            c2.metric("Annual Savings",  f"Rp {t100['annual_savings_million_rp']:.2f}M/yr")
            c3.metric("Payback",         f"{t100['payback_years']:.1f} yr")
            c4.metric("ROI (25 yr)",     f"{t100['roi_lifetime_pct']:.1f}%")

            c1, c2, c3 = st.columns(3)
            c1.metric("LCOE",            f"Rp {t100['lcoe_rp_per_kwh']:,.0f}/kWh")
            c2.metric("CO₂ saved",       f"{t100['annual_co2_reduction_tons']:.2f} t/yr")
            c3.metric("Viable",
                      "Yes" if t100["is_economically_viable"] else "No")

            st.divider()

            # ── Scenarios table — investment removed ─────────────────────────
            _section("Investment Scenarios")
            st.caption("Scenarios based on usable area utilization")

            df_scen = pd.DataFrame([
                {
                    "Scenario":          s["scenario_name"],
                    "Area (m²)":         f"{s['area_used_m2']:,.1f}",
                    "Capacity (kWp)":    f"{s['capacity_kwp']:.1f}",
                    "Savings (Rp M/yr)": f"{s['annual_savings_million_rp']:.2f}",
                    "Payback (yr)":      f"{s['payback_years']:.1f}",
                    "ROI 25yr (%)":      f"{s['roi_lifetime_pct']:.1f}",
                    "CO₂ (t/yr)":        f"{s['annual_co2_reduction_tons']:.2f}",
                    "Viable":            "Yes" if s["is_viable"] else "No",
                }
                for s in scenarios
            ])
            st.dataframe(df_scen, use_container_width=True, hide_index=True)

        # ─────────────────────────────────────────────────────────────────────
        # TAB 7 — AI Analysis
        # ─────────────────────────────────────────────────────────────────────
        with tab_ai:

            if not st.session_state.gemini_model:
                st.info("Enter a Gemini API key in the sidebar to enable AI analysis.")
            else:
                # ── Generate analysis ─────────────────────────────────────────
                if st.button("Generate AI Analysis", use_container_width=True):
                    with st.spinner("Analysing with Gemini AI..."):
                        detection_data = {
                            "n_roofs":      n_roofs,
                            "roof_area_m2": roof_area_m2,
                            "n_panels":     n_panels,
                            "panel_area_m2":panel_area_m2,
                        }
                        # Pass capped economics so AI sees realistic numbers
                        econ_for_ai = {
                            "investment_billion_rp":    economics_capped["to_100pct"]["investment_billion_rp"],
                            "annual_savings_million_rp":economics_capped["to_100pct"]["annual_savings_million_rp"],
                            "payback_years":            economics_capped["to_100pct"]["payback_years"],
                            "roi_lifetime_pct":         economics_capped["to_100pct"]["roi_lifetime_pct"],
                        }
                        analysis_text = analyze_with_gemini(
                            st.session_state.gemini_model,
                            detection_data,
                            production_data,
                            consumption_data,
                            coverage_data,
                            potential_data,
                            econ_for_ai,
                            st.session_state.uploaded_image,
                        )
                        st.session_state.analysis_results = analysis_text

                if st.session_state.analysis_results:
                    st.markdown(st.session_state.analysis_results)

                st.divider()

                # ── Chatbot ───────────────────────────────────────────────────
                _section("Ask a Question")

                c1, c2, c3, c4 = st.columns(4)
                quick = {
                    c1: "How many panels do I need for full coverage?",
                    c2: "Which scenario has the best ROI?",
                    c3: "What is the payback period for 100% coverage?",
                    c4: "How much CO₂ can I save annually?",
                }
                labels = [
                    "Panels needed",
                    "Best ROI scenario",
                    "Payback period",
                    "CO₂ savings",
                ]
                for (col, question), label in zip(quick.items(), labels):
                    with col:
                        if st.button(label, key=f"quick_{label}"):
                            st.session_state.chat_history.append(
                                {"role": "user", "content": question}
                            )

                user_question = st.chat_input("Ask anything about your solar analysis...")
                if user_question:
                    st.session_state.chat_history.append(
                        {"role": "user", "content": user_question}
                    )

                if st.session_state.chat_history:
                    for msg in st.session_state.chat_history:
                        with st.chat_message(msg["role"]):
                            st.write(msg["content"])

                    if st.session_state.chat_history[-1]["role"] == "user":
                        with st.chat_message("assistant"):
                            with st.spinner("Thinking..."):
                                context = f"""
You are a solar energy expert assistant for EcoPower Roof.
Answer based on this analysis data:

Detection  : {n_roofs} roofs ({roof_area_m2:.1f} m²), {n_panels} panels ({panel_area_m2:.1f} m²)
Production : {production_data['daily_kwh']:.2f} kWh/day | {production_data['annual_mwh']:.2f} MWh/yr
Consumption: {consumption_data['daily_kwh']:.2f} kWh/day
Coverage   : {coverage_data['current_coverage_pct']:.1f}%
Usable area: {usable_area_m2:.1f} m² ({usability_result['suitability_class']} suitability)
District   : {district_result['suitability_class']} zone

Economics (to reach 100% self-sufficiency):
  Investment : Rp {economics_capped['to_100pct']['investment_billion_rp']:.3f}B
  Payback    : {economics_capped['to_100pct']['payback_years']:.1f} years
  NPV        : Rp {economics_capped['to_100pct']['npv_billion_rp']:.3f}B
  ROI 25yr   : {economics_capped['to_100pct']['roi_lifetime_pct']:.1f}%

Best scenario: {spatial_scores[0]['scenario_name']} (score {spatial_scores[0]['composite_score']:.3f})

User question: {st.session_state.chat_history[-1]['content']}

Answer in English. Be specific with numbers. Keep it concise.
"""
                                response = st.session_state.gemini_model.generate_content(context)
                                answer   = response.text
                                st.write(answer)
                                st.session_state.chat_history.append(
                                    {"role": "assistant", "content": answer}
                                )

                    if st.button("Clear chat history", key="clear_chat"):
                        st.session_state.chat_history = []
                        st.rerun()

        # ─────────────────────────────────────────────────────────────────────
        # TAB 8 — Export
        # ─────────────────────────────────────────────────────────────────────
        with tab_export:

            deps = check_export_dependencies()

            if not deps["geotiff_export"]:
                st.warning("rasterio not installed — GeoTIFF export unavailable. Run: pip install rasterio")
            if not deps["shp_export"]:
                st.warning("fiona or shapely not installed — SHP export unavailable. Run: pip install fiona shapely")

            # ── Per-detection exports ─────────────────────────────────────────
            for mode, data in results.items():
                mode_label = mode.replace("_", " ").title()
                with st.expander(f"{mode_label} — Export Options", expanded=True):

                    c1, c2, c3 = st.columns(3)

                    # PNG
                    with c1:
                        _section("Overlay Image")
                        buf_png = io.BytesIO()
                        data["overlay"].save(buf_png, format="PNG")
                        st.download_button(
                            label     = "Download PNG",
                            data      = buf_png.getvalue(),
                            file_name = f"{mode}_overlay.png",
                            mime      = "image/png",
                            key       = f"dl_png_{mode}",
                        )

                    # GeoTIFF
                    with c2:
                        _section("GeoTIFF Overlay")
                        if deps["geotiff_export"]:
                            geotiff_bytes = export_overlay_as_geotiff(
                                overlay_pil    = data["overlay"],
                                geotiff_handler= st.session_state.geotiff_handler,
                                detection_mode = mode,
                            )
                            if geotiff_bytes:
                                georef = (
                                    st.session_state.geotiff_handler
                                    and st.session_state.geotiff_handler.is_geotiff
                                )
                                st.caption(
                                    "Georeferenced — inherits source CRS" if georef
                                    else "Pixel-based transform (no source GeoTIFF)"
                                )
                                st.download_button(
                                    label     = "Download GeoTIFF",
                                    data      = geotiff_bytes,
                                    file_name = f"{mode}_overlay.tif",
                                    mime      = "image/tiff",
                                    key       = f"dl_tif_{mode}",
                                )
                            else:
                                st.error("GeoTIFF export failed")
                        else:
                            st.markdown(_pill("Install rasterio to enable", "gray"), unsafe_allow_html=True)

                    # Shapefile
                    with c3:
                        _section("Shapefile (.shp)")
                        if deps["shp_export"]:
                            shp_bytes = export_mask_as_shapefile(
                                mask_np            = data["mask"],
                                geotiff_handler    = st.session_state.geotiff_handler,
                                detection_mode     = mode,
                                simplify_tolerance = 0.5,
                                min_area_m2        = 1.0,
                            )
                            if shp_bytes:
                                georef = (
                                    st.session_state.geotiff_handler
                                    and st.session_state.geotiff_handler.is_geotiff
                                )
                                st.caption(
                                    f"{'Georeferenced' if georef else 'Pixel coordinates'} "
                                    f"| ~{data['n_objects']} polygon(s)"
                                )
                                st.download_button(
                                    label     = "Download Shapefile (ZIP)",
                                    data      = shp_bytes,
                                    file_name = f"{mode}_mask.zip",
                                    mime      = "application/zip",
                                    key       = f"dl_shp_{mode}",
                                )
                            else:
                                st.warning("No polygons generated — try lowering the confidence threshold")
                        else:
                            st.markdown(_pill("Install fiona + shapely to enable", "gray"), unsafe_allow_html=True)

            # ── JSON report ───────────────────────────────────────────────────
            st.divider()
            _section("Full Analysis Report")

            report = {
                "app":     "EcoPower Roof",
                "version": "3.0.0",
                "detection": {
                    "n_roofs":       int(n_roofs),
                    "roof_area_m2":  float(roof_area_m2),
                    "n_panels":      int(n_panels),
                    "panel_area_m2": float(panel_area_m2),
                },
                "production":          convert_to_serializable(production_data),
                "consumption":         convert_to_serializable(consumption_data),
                "coverage":            convert_to_serializable(coverage_data),
                "potential":           convert_to_serializable(potential_data),
                "usable_area":         convert_to_serializable(usability_result),
                "district_analysis":   convert_to_serializable(district_result),
                "economics_current":   convert_to_serializable(economics_current),
                "economics_capped":    convert_to_serializable(economics_capped),
                "spatial_scores":      convert_to_serializable(spatial_scores),
                "pln_comparison":      convert_to_serializable(pln_comparison),
            }

            st.download_button(
                label     = "Download Full Report (JSON)",
                data      = json.dumps(report, indent=2, ensure_ascii=False),
                file_name = "ecopower_roof_report.json",
                mime      = "application/json",
                key       = "dl_json_report",
            )

    # =========================================================================
    # FOOTER
    # =========================================================================

    st.markdown("""
    <div class="app-footer">
        <p class="app-footer-title">EcoPower Roof</p>
        <p class="app-footer-sub">Spatial Intelligence App for Solar Panel Analysis</p>
        <p class="app-footer-sub">GEO-AI Twinverse Research Group, Faculty of Engineering, Universitas Gadjah Mada</p>
        <p class="app-footer-team">
            Mohammad Zulfi Rahadi Putra &nbsp;·&nbsp;
            Raffi Satya Nugraha &nbsp;·&nbsp;
            Fikri Kurniawan &nbsp;·&nbsp;
            Fairuz Akmal Pradana  &nbsp;·&nbsp;
            Hyatma Adikara Ajrin &nbsp;·&nbsp;
            Ruli Andaru
        </p>
        <p style="font-size:0.72rem;color:#9aaa9a;margin-top:0.75rem;">
            SAM3 Advanced &amp; Gemini AI
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
