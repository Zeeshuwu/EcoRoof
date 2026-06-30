"""
SAM 3 Advanced Module
Multiscale patching, geometric filtering, and spatial validation
"""

from .config import AdvancedConfig
from .patching import generate_patches, blend_patches_weighted_average
from .filtering import filter_by_geometry_relaxed, validate_with_context
from .inference import segment_multiscale_advanced
from .utils import clear_memory, get_gpu_memory

__all__ = [
    'AdvancedConfig',
    'generate_patches',
    'blend_patches_weighted_average',
    'filter_by_geometry_relaxed',
    'validate_with_context',
    'segment_multiscale_advanced',
    'clear_memory',
    'get_gpu_memory',
]

__version__ = '1.0.0'
