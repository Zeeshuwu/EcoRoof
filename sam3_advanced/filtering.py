"""
Geometric and spatial filtering functions
"""

import numpy as np
from skimage import measure
from typing import Tuple

def validate_with_context(mask_current: np.ndarray, 
                         mask_reference: np.ndarray,
                         min_overlap: float = 0.10) -> np.ndarray:
    """
    Validate mask regions against a reference mask (spatial validation)
    
    Args:
        mask_current: Mask to validate
        mask_reference: Reference mask (e.g., Level 1 results)
        min_overlap: Minimum overlap ratio to keep region
    
    Returns:
        Validated mask
    """
    labeled = measure.label(mask_current, connectivity=2)
    validated = np.zeros_like(mask_current)
    
    rejected_count = 0
    kept_count = 0

    for region in measure.regionprops(labeled):
        coords = region.coords
        region_mask = np.zeros_like(mask_current)
        region_mask[coords[:, 0], coords[:, 1]] = 1

        # Check overlap with reference
        overlap = np.logical_and(region_mask, mask_reference).sum()
        overlap_ratio = overlap / region.area if region.area > 0 else 0

        # Keep if: has overlap OR large area OR elongated shape
        keep = (overlap_ratio >= min_overlap or 
                region.area > 3000 or 
                region.eccentricity > 0.7)

        if keep:
            validated[coords[:, 0], coords[:, 1]] = 1
            kept_count += 1
        else:
            rejected_count += 1

    if rejected_count > 0:
        print(f"     ⚠️ Spatial validation: rejected {rejected_count} isolated regions (kept {kept_count})")

    return validated.astype(np.uint8)


def filter_by_geometry_relaxed(mask: np.ndarray, 
                               min_area_for_geometry: int = 1000,
                               max_compactness: float = 0.45,
                               min_eccentricity: float = 0.60,
                               min_aspect_ratio: float = 0.03) -> np.ndarray:
    """
    Filter out large compact blobs (parking lots, buildings)
    Keep all small/medium objects and elongated shapes
    
    Args:
        mask: Input binary mask
        min_area_for_geometry: Only check objects larger than this
        max_compactness: Maximum compactness (circularity)
        min_eccentricity: Minimum eccentricity (elongation)
        min_aspect_ratio: Minimum aspect ratio
    
    Returns:
        Filtered mask
    """
    labeled = measure.label(mask, connectivity=2)
    filtered = np.zeros_like(mask)
    
    filtered_count = 0
    kept_count = 0

    for region in measure.regionprops(labeled):
        coords = region.coords
        
        # ✅ Keep ALL small/medium objects
        if region.area < min_area_for_geometry:
            filtered[coords[:, 0], coords[:, 1]] = 1
            kept_count += 1
            continue

        # Only check LARGE objects for blob-like characteristics
        minr, minc, maxr, maxc = region.bbox
        width = maxc - minc
        height = maxr - minr
        aspect = min(width, height) / max(width, height) if max(width, height) > 0 else 0

        if region.perimeter > 0:
            compactness = (4 * np.pi * region.area) / (region.perimeter ** 2)
        else:
            compactness = 1.0

        eccentricity = region.eccentricity

        # ✅ Keep if ANY road-like characteristic
        is_road_like = (aspect < min_aspect_ratio or 
                       eccentricity > min_eccentricity or 
                       compactness < max_compactness)

        if is_road_like:
            filtered[coords[:, 0], coords[:, 1]] = 1
            kept_count += 1
        else:
            filtered_count += 1
            print(f"     ❌ Filtered LARGE blob: area={region.area}, aspect={aspect:.3f}, "
                  f"compact={compactness:.3f}, ecc={eccentricity:.3f}")

    if filtered_count > 0:
        print(f"     ✅ Geometric filter: removed {filtered_count} large blobs, kept {kept_count} roads")

    return filtered.astype(np.uint8)


def remove_small_objects(mask: np.ndarray, min_size: int = 50) -> np.ndarray:
    """
    Remove small connected components
    
    Args:
        mask: Input binary mask
        min_size: Minimum object size in pixels
    
    Returns:
        Cleaned mask
    """
    from skimage import morphology
    cleaned = morphology.remove_small_objects(
        mask.astype(bool),
        min_size=min_size
    ).astype(np.uint8)
    return cleaned


def fill_small_holes(mask: np.ndarray, max_hole_size: int = 150) -> np.ndarray:
    """
    Fill small holes in mask
    
    Args:
        mask: Input binary mask
        max_hole_size: Maximum hole size to fill
    
    Returns:
        Filled mask
    """
    from skimage import morphology
    filled = morphology.remove_small_holes(
        mask.astype(bool),
        area_threshold=max_hole_size
    ).astype(np.uint8)
    return filled
