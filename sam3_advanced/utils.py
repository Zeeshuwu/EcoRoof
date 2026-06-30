"""
Utility functions for memory management and helpers
"""

import gc
import torch
import numpy as np
from typing import Tuple

def clear_memory():
    """Clear GPU and CPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def get_gpu_memory() -> Tuple[float, float]:
    """Get GPU memory usage in GB"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return allocated, reserved
    return 0.0, 0.0

def print_memory_status():
    """Print current GPU memory status"""
    allocated, reserved = get_gpu_memory()
    print(f"  💾 GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

def calculate_coverage(mask: np.ndarray) -> float:
    """Calculate mask coverage percentage"""
    if mask.size == 0:
        return 0.0
    return (mask.sum() / mask.size) * 100.0
