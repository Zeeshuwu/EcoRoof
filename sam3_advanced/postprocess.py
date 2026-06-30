"""
Post-processing functions
"""

import numpy as np
import cv2
from .filtering import remove_small_objects, fill_small_holes, filter_by_geometry_relaxed
from .utils import calculate_coverage
from .config import AdvancedConfig


def post_process_mask(mask: np.ndarray, 
                     config: AdvancedConfig,
                     verbose: bool = True) -> np.ndarray:
    """
    Apply post-processing pipeline to mask
    
    Args:
        mask: Input binary mask
        config: AdvancedConfig instance
        verbose: Print progress
    
    Returns:
        Processed mask
    """
    if verbose:
        print("\n" + "="*70)
        print("🧹 POST-PROCESSING")
        print("="*70)

    original_coverage = calculate_coverage(mask)
    if verbose:
        print(f"0️⃣ Input: {original_coverage:.2f}%")

    # Stage 1: Remove small noise
    if verbose:
        print(f"\n1️⃣ Removing noise < {config.min_road_size} pixels...")
    
    cleaned = remove_small_objects(mask, config.min_road_size)
    cleaned = fill_small_holes(cleaned, config.max_hole_size)

    stage1_coverage = calculate_coverage(cleaned)
    if verbose:
        print(f"   ✓ After noise removal: {stage1_coverage:.2f}%")

    # Stage 2: Geometric filtering
    if config.enable_geometric_filter:
        if verbose:
            print(f"\n2️⃣ Filtering LARGE blobs (> {config.min_area_for_geometry} pixels)...")
        
        filtered = filter_by_geometry_relaxed(
            cleaned,
            min_area_for_geometry=config.min_area_for_geometry,
            max_compactness=config.max_compactness,
            min_eccentricity=config.min_eccentricity,
            min_aspect_ratio=config.min_aspect_ratio
        )

        stage2_coverage = calculate_coverage(filtered)
        removed = stage1_coverage - stage2_coverage
        if verbose:
            print(f"   ✓ After geometric filter: {stage2_coverage:.2f}% (removed {removed:.2f}%)")

        final_mask = filtered
    else:
        final_mask = cleaned

    # Stage 3: Light smoothing
    if config.enable_light_smoothing:
        if verbose:
            print(f"\n3️⃣ Light smoothing...")
        
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (config.smooth_kernel_size, config.smooth_kernel_size)
        )
        smoothed = cv2.morphologyEx(
            final_mask, 
            cv2.MORPH_CLOSE, 
            kernel, 
            iterations=config.smooth_iterations
        )
        final_mask = smoothed

    final_coverage = calculate_coverage(final_mask)
    if verbose:
        print(f"\n✅ Final: {final_coverage:.2f}%")

    return final_mask.astype(np.uint8)
