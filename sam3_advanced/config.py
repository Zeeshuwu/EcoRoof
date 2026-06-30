"""
Configuration for Advanced SAM 3 Processing
"""

from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class AdvancedConfig:
    """Configuration for multiscale road detection"""
    
    # Confidence thresholds per level
    confidence_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'level1': 0.20,  # Full image (large objects)
        'level2': 0.22,  # 512px patches (medium objects)
        'level3': 0.28,  # 256px patches (small objects)
    })
    
    # Early stopping
    early_stop_coverage: Dict[str, float] = field(default_factory=lambda: {
        'level1': 0.90,
        'level2': 0.95,
    })
    
    # Blending configuration
    blend_mode: str = 'weighted_average'  # or 'max_confidence'
    consensus_threshold_multi: float = 0.25   # Multiple patches overlap
    consensus_threshold_single: float = 0.40  # Single patch
    
    # Spatial validation
    enable_spatial_validation: bool = True
    validate_level2: bool = False  # Don't validate medium roads
    validate_level3: bool = True   # Validate small roads only
    min_overlap_with_level1: float = 0.10
    
    # Geometric filtering
    enable_geometric_filter: bool = True
    min_area_for_geometry: int = 1000  # Only check large objects
    max_compactness: float = 0.45
    min_eccentricity: float = 0.60
    min_aspect_ratio: float = 0.03
    
    # Post-processing
    min_road_size: int = 50
    max_hole_size: int = 150
    enable_light_smoothing: bool = True
    smooth_kernel_size: int = 3
    smooth_iterations: int = 1
    
    # Patch generation
    patch_size_level2: int = 512
    patch_size_level3: int = 256
    overlap_ratio: float = 0.25
    
    # Memory management
    batch_clear_frequency: int = 5
    
    # Prompts
    prompts_large: List[str] = field(default_factory=lambda: [
        "highway", "main road", "arterial road", "multi-lane road",
        "expressway", "major road", "primary road", "wide road",
    ])
    
    prompts_medium: List[str] = field(default_factory=lambda: [
        "street", "urban road", "residential street", "paved road",
        "asphalt road", "road with markings", "city street", "secondary road",
    ])
    
    prompts_small: List[str] = field(default_factory=lambda: [
        "narrow road", "alley", "lane", "small road",
        "access road", "service road", "local road", "minor road",
    ])
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {
            'confidence_thresholds': self.confidence_thresholds,
            'early_stop_coverage': self.early_stop_coverage,
            'blend_mode': self.blend_mode,
            'consensus_threshold_multi': self.consensus_threshold_multi,
            'consensus_threshold_single': self.consensus_threshold_single,
            'enable_spatial_validation': self.enable_spatial_validation,
            'validate_level2': self.validate_level2,
            'validate_level3': self.validate_level3,
            'min_overlap_with_level1': self.min_overlap_with_level1,
            'enable_geometric_filter': self.enable_geometric_filter,
            'min_area_for_geometry': self.min_area_for_geometry,
            'max_compactness': self.max_compactness,
            'min_eccentricity': self.min_eccentricity,
            'min_aspect_ratio': self.min_aspect_ratio,
            'min_road_size': self.min_road_size,
            'max_hole_size': self.max_hole_size,
            'enable_light_smoothing': self.enable_light_smoothing,
            'smooth_kernel_size': self.smooth_kernel_size,
            'smooth_iterations': self.smooth_iterations,
            'patch_size_level2': self.patch_size_level2,
            'patch_size_level3': self.patch_size_level3,
            'overlap_ratio': self.overlap_ratio,
            'batch_clear_frequency': self.batch_clear_frequency,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'AdvancedConfig':
        """Create config from dictionary"""
        return cls(**config_dict)
