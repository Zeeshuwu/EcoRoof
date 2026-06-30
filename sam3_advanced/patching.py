"""
Patch generation and blending functions
"""

import numpy as np
from typing import List, Dict, Tuple

def generate_patches(image_shape: Tuple[int, int], 
                    patch_size: int,
                    overlap_ratio: float = 0.25) -> List[Dict]:
    """
    Generate overlapping patches for an image
    
    Args:
        image_shape: (height, width) of the image
        patch_size: Size of each patch
        overlap_ratio: Overlap between patches (0.0 to 1.0)
    
    Returns:
        List of patch information dictionaries
    """
    height, width = image_shape
    stride = int(patch_size * (1 - overlap_ratio))
    patches = []
    patch_id = 0

    for y in range(0, height, stride):
        for x in range(0, width, stride):
            x_end = min(x + patch_size, width)
            y_end = min(y + patch_size, height)
            x_start = max(0, x_end - patch_size)
            y_start = max(0, y_end - patch_size)

            patch_info = {
                'id': patch_id,
                'bbox': (x_start, y_start, x_end, y_end),
                'slice': (slice(y_start, y_end), slice(x_start, x_end)),
                'center': ((x_start + x_end) // 2, (y_start + y_end) // 2),
                'size': (y_end - y_start, x_end - x_start)
            }
            patches.append(patch_info)
            patch_id += 1

    return patches


def blend_patches_weighted_average(patches_data: List[Dict],
                                   image_shape: Tuple[int, int],
                                   consensus_threshold_multi: float = 0.25,
                                   consensus_threshold_single: float = 0.40) -> Tuple[np.ndarray, np.ndarray]:
    """
    Blend patches using weighted average with adaptive consensus
    
    Args:
        patches_data: List of patch dictionaries with 'mask', 'bbox', 'confidence'
        image_shape: (height, width) of output image
        consensus_threshold_multi: Threshold when multiple patches overlap
        consensus_threshold_single: Threshold when single patch
    
    Returns:
        (merged_mask, confidence_map)
    """
    height, width = image_shape
    mask_sum = np.zeros((height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.float32)

    # Accumulate masks
    for patch_data in patches_data:
        mask = patch_data['mask'].astype(np.float32)
        bbox = patch_data['bbox']
        x_start, y_start, x_end, y_end = bbox

        mask_sum[y_start:y_end, x_start:x_end] += mask
        count[y_start:y_end, x_start:x_end] += 1

    # Calculate average
    mask_avg = np.zeros_like(mask_sum)
    valid = count > 0
    mask_avg[valid] = mask_sum[valid] / count[valid]

    # Adaptive consensus thresholding
    consensus_threshold = np.where(
        count >= 2,
        consensus_threshold_multi,
        consensus_threshold_single
    )

    merged_mask = (mask_avg > consensus_threshold).astype(np.uint8)

    return merged_mask, mask_avg


def blend_patches_max_confidence(patches_data: List[Dict],
                                 image_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Blend patches by taking maximum confidence
    
    Args:
        patches_data: List of patch dictionaries
        image_shape: (height, width) of output image
    
    Returns:
        (merged_mask, confidence_map)
    """
    height, width = image_shape
    mask_max = np.zeros((height, width), dtype=np.float32)
    conf_max = np.zeros((height, width), dtype=np.float32)

    for patch_data in patches_data:
        mask = patch_data['mask']
        conf = patch_data.get('confidence', 1.0)
        bbox = patch_data['bbox']
        x_start, y_start, x_end, y_end = bbox

        mask_max[y_start:y_end, x_start:x_end] = np.maximum(
            mask_max[y_start:y_end, x_start:x_end],
            mask
        )
        
        if isinstance(conf, np.ndarray):
            conf_value = np.max(conf)
        else:
            conf_value = conf
            
        conf_max[y_start:y_end, x_start:x_end] = np.maximum(
            conf_max[y_start:y_end, x_start:x_end],
            conf_value
        )

    merged_mask_binary = (mask_max > 0.5).astype(np.uint8)
    return merged_mask_binary, conf_max
