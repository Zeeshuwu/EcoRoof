"""
Multiscale inference engine for SAM 3
Uses YOUR existing model interface (build_sam3_image_model + Sam3Processor)
"""

import numpy as np
from typing import List, Tuple, Dict
from PIL import Image

from .patching import generate_patches, blend_patches_weighted_average
from .filtering import validate_with_context
from .utils import clear_memory, calculate_coverage
from .config import AdvancedConfig


def segment_with_prompts(image_pil: Image.Image,
                        processor,
                        prompts: List[str],
                        conf_threshold: float,
                        verbose: bool = True) -> Tuple[List[np.ndarray], List[float]]:
    """
    Segment image with multiple prompts using YOUR processor interface
    
    Args:
        image_pil: PIL Image
        processor: Sam3Processor instance (YOUR interface)
        prompts: List of text prompts
        conf_threshold: Confidence threshold
        verbose: Print progress
    
    Returns:
        (list_of_masks, list_of_confidences)
    """
    all_masks = []
    all_confs = []

    # Set image once
    inference_state = processor.set_image(image_pil)

    for idx, prompt in enumerate(prompts, 1):
        try:
            # YOUR interface
            results = processor.set_text_prompt(
                state=inference_state, 
                prompt=prompt
            )
            
            if results and "masks" in results:
                masks = results["masks"]
                scores = results["scores"]
                
                if masks is not None and len(masks) > 0:
                    # Convert to numpy if needed
                    if hasattr(masks, 'cpu'):
                        masks_np = masks.cpu().numpy()
                    else:
                        masks_np = np.array(masks)
                    
                    if hasattr(scores, 'cpu'):
                        scores_np = scores.cpu().numpy()
                    else:
                        scores_np = np.array(scores)
                    
                    # Filter by confidence
                    for i in range(len(masks_np)):
                        if scores_np[i] >= conf_threshold:
                            all_masks.append(masks_np[i])
                            all_confs.append(float(scores_np[i]))
                    
                    if verbose and len(masks_np) > 0:
                        avg_conf = np.mean(scores_np)
                        print(f"  [{idx:2d}] ✅ '{prompt[:25]:<25}' → {len(masks_np)}m, conf={avg_conf:.2f}")
        
        except Exception as e:
            if verbose:
                print(f"  [{idx:2d}] ❌ '{prompt[:25]:<25}' → Error: {str(e)[:30]}")
            continue

    return all_masks, all_confs


def segment_multiscale_advanced(image_pil: Image.Image,
                                processor,
                                config: AdvancedConfig) -> Tuple[np.ndarray, Dict]:
    """
    Multiscale segmentation with 3 levels
    
    Args:
        image_pil: Input PIL Image
        processor: Sam3Processor instance (YOUR interface)
        config: AdvancedConfig instance
    
    Returns:
        (final_mask, statistics_dict)
    """
    print("="*70)
    print("🔍 MULTISCALE INFERENCE - ADVANCED MODE")
    print("="*70)

    # Convert to numpy for processing
    image_np = np.array(image_pil)
    height, width = image_np.shape[:2]
    
    stats = {
        'level1': {'masks': 0, 'coverage': 0.0},
        'level2': {'masks': 0, 'coverage': 0.0},
        'level3': {'masks': 0, 'coverage': 0.0},
    }

    # ============================================
    # LEVEL 1: FULL IMAGE (Large objects)
    # ============================================
    print("\n" + "="*70)
    print("📊 LEVEL 1: Full Image (Large Objects)")
    print("="*70)
    print(f"  Image size: {height}x{width}")
    print(f"  Confidence threshold: {config.confidence_thresholds['level1']}")

    level1_masks, level1_confs = segment_with_prompts(
        image_pil, processor, config.prompts_large,
        config.confidence_thresholds['level1']
    )

    # Combine Level 1 masks
    level1_combined = np.zeros((height, width), dtype=np.float32)
    if len(level1_masks) > 0:
        for mask in level1_masks:
            # Handle different mask dimensions
            if mask.ndim == 3:
                mask = mask.squeeze()
            if mask.ndim == 2:
                # Resize if needed
                if mask.shape != (height, width):
                    from scipy.ndimage import zoom
                    zoom_factors = (height / mask.shape[0], width / mask.shape[1])
                    mask = zoom(mask, zoom_factors, order=1)
                
                level1_combined = np.maximum(level1_combined, (mask > 0.5).astype(np.float32))

    level1_binary = (level1_combined > 0.5).astype(np.uint8)
    coverage_l1 = calculate_coverage(level1_binary)
    stats['level1']['masks'] = len(level1_masks)
    stats['level1']['coverage'] = coverage_l1
    print(f"  ✅ Level 1: {len(level1_masks)} masks, {coverage_l1:.1f}% coverage")

    clear_memory()

    # ============================================
    # LEVEL 2: 512px PATCHES (Medium objects)
    # ============================================
    print("\n" + "="*70)
    print("📊 LEVEL 2: 512px Patches (Medium Objects)")
    print("="*70)

    patches_l2 = generate_patches(
        (height, width), 
        config.patch_size_level2, 
        config.overlap_ratio
    )
    print(f"  Total patches: {len(patches_l2)}")
    print(f"  Confidence threshold: {config.confidence_thresholds['level2']}")

    level2_patches_data = []
    level2_count = 0

    for idx, patch_info in enumerate(patches_l2, 1):
        y_slice, x_slice = patch_info['slice']
        patch_img_np = image_np[y_slice, x_slice]

        # Skip too small patches
        if patch_img_np.shape[0] < 128 or patch_img_np.shape[1] < 128:
            continue

        # Convert to PIL for processor
        patch_img_pil = Image.fromarray(patch_img_np)

        patch_masks, patch_confs = segment_with_prompts(
            patch_img_pil, processor, config.prompts_medium,
            config.confidence_thresholds['level2'], verbose=False
        )

        if len(patch_masks) > 0:
            patch_combined = np.zeros(patch_img_np.shape[:2], dtype=np.float32)
            for mask in patch_masks:
                if mask.ndim == 3:
                    mask = mask.squeeze()
                if mask.shape != patch_img_np.shape[:2]:
                    from scipy.ndimage import zoom
                    zoom_factors = (patch_img_np.shape[0] / mask.shape[0], 
                                   patch_img_np.shape[1] / mask.shape[1])
                    mask = zoom(mask, zoom_factors, order=1)
                
                patch_combined = np.maximum(patch_combined, (mask > 0.5).astype(np.float32))

            level2_patches_data.append({
                'mask': patch_combined,
                'confidence': np.full(patch_combined.shape, np.mean(patch_confs)),
                'bbox': patch_info['bbox'],
                'level': 2
            })
            level2_count += 1

        if idx % 10 == 0:
            print(f"  Processed {idx}/{len(patches_l2)} patches...")
            clear_memory()

    # Blend Level 2
    print(f"\n  🔗 Blending {level2_count} Level 2 patches...")
    level2_merged, _ = blend_patches_weighted_average(
        level2_patches_data, (height, width),
        config.consensus_threshold_multi,
        config.consensus_threshold_single
    )

    # Optional validation
    if config.enable_spatial_validation and config.validate_level2:
        print(f"  🔍 Validating Level 2 with Level 1...")
        level2_validated = validate_with_context(
            level2_merged, level1_binary,
            min_overlap=config.min_overlap_with_level1
        )
    else:
        level2_validated = level2_merged

    coverage_l2 = calculate_coverage(level2_validated)
    stats['level2']['masks'] = level2_count
    stats['level2']['coverage'] = coverage_l2
    print(f"  ✅ Level 2: {coverage_l2:.1f}% coverage")

    clear_memory()

    # ============================================
    # LEVEL 3: 256px PATCHES (Small objects)
    # ============================================
    temp_merged = np.maximum(level1_binary, level2_validated)
    coverage_l1_l2 = calculate_coverage(temp_merged)
    print(f"\n  Current coverage (L1+L2): {coverage_l1_l2:.1f}%")

    if coverage_l1_l2 >= config.early_stop_coverage['level2'] * 100:
        print(f"  ⚠️ Skipping Level 3 (coverage sufficient)")
        level3_validated = np.zeros((height, width), dtype=np.uint8)
    else:
        print("\n" + "="*70)
        print("📊 LEVEL 3: 256px Patches (Small Objects)")
        print("="*70)

        patches_l3 = generate_patches(
            (height, width), 
            config.patch_size_level3, 
            config.overlap_ratio
        )
        print(f"  Total patches: {len(patches_l3)}")
        print(f"  Confidence threshold: {config.confidence_thresholds['level3']}")

        level3_patches_data = []
        level3_count = 0

        for idx, patch_info in enumerate(patches_l3, 1):
            y_slice, x_slice = patch_info['slice']
            patch_img_np = image_np[y_slice, x_slice]

            if patch_img_np.shape[0] < 64 or patch_img_np.shape[1] < 64:
                continue

            patch_img_pil = Image.fromarray(patch_img_np)

            patch_masks, patch_confs = segment_with_prompts(
                patch_img_pil, processor, config.prompts_small,
                config.confidence_thresholds['level3'], verbose=False
            )

            if len(patch_masks) > 0:
                patch_combined = np.zeros(patch_img_np.shape[:2], dtype=np.float32)
                for mask in patch_masks:
                    if mask.ndim == 3:
                        mask = mask.squeeze()
                    if mask.shape != patch_img_np.shape[:2]:
                        from scipy.ndimage import zoom
                        zoom_factors = (patch_img_np.shape[0] / mask.shape[0], 
                                       patch_img_np.shape[1] / mask.shape[1])
                        mask = zoom(mask, zoom_factors, order=1)
                    
                    patch_combined = np.maximum(patch_combined, (mask > 0.5).astype(np.float32))

                level3_patches_data.append({
                    'mask': patch_combined,
                    'confidence': np.full(patch_combined.shape, np.mean(patch_confs)),
                    'bbox': patch_info['bbox'],
                    'level': 3
                })
                level3_count += 1

            if idx % 20 == 0:
                print(f"  Processed {idx}/{len(patches_l3)} patches...")
                clear_memory()

        # Blend Level 3
        print(f"\n  🔗 Blending {level3_count} Level 3 patches...")
        level3_merged, _ = blend_patches_weighted_average(
            level3_patches_data, (height, width),
            config.consensus_threshold_multi,
            config.consensus_threshold_single
        )

        # Validate Level 3
        if config.enable_spatial_validation and config.validate_level3:
            print(f"  🔍 Validating Level 3 with Level 1...")
            level3_validated = validate_with_context(
                level3_merged, level1_binary,
                min_overlap=config.min_overlap_with_level1
            )
        else:
            level3_validated = level3_merged

        coverage_l3 = calculate_coverage(level3_validated)
        stats['level3']['masks'] = level3_count
        stats['level3']['coverage'] = coverage_l3
        print(f"  ✅ Level 3: {coverage_l3:.1f}% coverage")

        clear_memory()

    # ============================================
    # MERGE ALL LEVELS
    # ============================================
    print("\n" + "="*70)
    print("🔗 MERGING ALL LEVELS")
    print("="*70)

    final_mask = np.maximum(level1_binary, level2_validated)
    final_mask = np.maximum(final_mask, level3_validated)

    final_coverage = calculate_coverage(final_mask)
    print(f"  ✅ Merged coverage: {final_coverage:.2f}%")

    return final_mask, stats
