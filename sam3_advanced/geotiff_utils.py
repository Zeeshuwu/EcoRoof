"""
GeoTIFF utilities for accurate geospatial measurements
"""

import numpy as np
from PIL import Image
from typing import Tuple, Dict, Optional
import warnings

try:
    import rasterio
    from rasterio.errors import RasterioIOError
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    warnings.warn("rasterio not installed. GeoTIFF support disabled. Install: pip install rasterio")


class GeoTIFFHandler:
    """Handle GeoTIFF files and extract geospatial metadata"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.metadata = {}
        self.is_geotiff = False
        self.pixel_size_x = None
        self.pixel_size_y = None
        self.crs = None
        self.bounds = None
        
        if RASTERIO_AVAILABLE:
            self._load_metadata()
    
    def _load_metadata(self):
        """Load GeoTIFF metadata using rasterio"""
        try:
            with rasterio.open(self.file_path) as src:
                # Check if it has geospatial info
                if src.crs is not None:
                    self.is_geotiff = True
                    
                    # Get pixel size (resolution)
                    transform = src.transform
                    self.pixel_size_x = abs(transform.a)  # meters per pixel (X)
                    self.pixel_size_y = abs(transform.e)  # meters per pixel (Y)
                    
                    # Get CRS
                    self.crs = src.crs.to_string() if src.crs else "Unknown"
                    
                    # Get bounds
                    self.bounds = src.bounds
                    
                    # Store metadata
                    self.metadata = {
                        'width': src.width,
                        'height': src.height,
                        'crs': self.crs,
                        'pixel_size_x': self.pixel_size_x,
                        'pixel_size_y': self.pixel_size_y,
                        'gsd': (self.pixel_size_x + self.pixel_size_y) / 2,  # Ground Sample Distance
                        'bounds': {
                            'left': self.bounds.left,
                            'bottom': self.bounds.bottom,
                            'right': self.bounds.right,
                            'top': self.bounds.top
                        },
                        'area_m2': (self.bounds.right - self.bounds.left) * (self.bounds.top - self.bounds.bottom)
                    }
                    
                    print(f"✅ GeoTIFF detected!")
                    print(f"   CRS: {self.crs}")
                    print(f"   GSD: {self.metadata['gsd']:.4f} m/pixel")
                    print(f"   Resolution: {self.pixel_size_x:.4f} x {self.pixel_size_y:.4f} m/pixel")
                
        except RasterioIOError:
            print("⚠️ Not a valid GeoTIFF or no geospatial data")
        except Exception as e:
            print(f"⚠️ Error reading GeoTIFF: {e}")
    
    def get_pixel_to_meter_ratio(self) -> Optional[float]:
        """Get automatic pixel-to-meter ratio from GeoTIFF"""
        if self.is_geotiff and self.pixel_size_x:
            # Return average of X and Y resolution
            return (self.pixel_size_x + self.pixel_size_y) / 2
        return None
    
    def calculate_area_m2(self, pixel_count: int) -> float:
        """Calculate real-world area from pixel count"""
        if self.is_geotiff and self.pixel_size_x:
            # Each pixel represents pixel_size_x * pixel_size_y square meters
            pixel_area = self.pixel_size_x * self.pixel_size_y
            return pixel_count * pixel_area
        return 0.0
    
    def get_image_as_pil(self) -> Image.Image:
        """Load GeoTIFF as PIL Image (RGB)"""
        if not RASTERIO_AVAILABLE:
            return Image.open(self.file_path).convert("RGB")
        
        try:
            with rasterio.open(self.file_path) as src:
                # Read RGB bands (assuming bands 1,2,3 are RGB)
                if src.count >= 3:
                    r = src.read(1)
                    g = src.read(2)
                    b = src.read(3)
                    
                    # Stack and normalize
                    rgb = np.dstack([r, g, b])
                    
                    # Normalize to 0-255 if needed
                    if rgb.max() > 255:
                        rgb = ((rgb - rgb.min()) / (rgb.max() - rgb.min()) * 255).astype(np.uint8)
                    else:
                        rgb = rgb.astype(np.uint8)
                    
                    return Image.fromarray(rgb)
                else:
                    # Grayscale - convert to RGB
                    gray = src.read(1)
                    if gray.max() > 255:
                        gray = ((gray - gray.min()) / (gray.max() - gray.min()) * 255).astype(np.uint8)
                    else:
                        gray = gray.astype(np.uint8)
                    return Image.fromarray(gray).convert("RGB")
        
        except Exception as e:
            print(f"⚠️ Error loading GeoTIFF as image: {e}")
            return Image.open(self.file_path).convert("RGB")
    
    def get_info_dict(self) -> Dict:
        """Get all metadata as dictionary"""
        return {
            'is_geotiff': self.is_geotiff,
            'metadata': self.metadata if self.is_geotiff else {},
            'pixel_to_meter': self.get_pixel_to_meter_ratio()
        }


def load_image_with_geotiff_support(uploaded_file) -> Tuple[Image.Image, Optional[GeoTIFFHandler]]:
    """
    Load image with GeoTIFF support
    
    Returns:
        image_pil: PIL Image
        geotiff_handler: GeoTIFFHandler if GeoTIFF, else None
    """
    # Save uploaded file temporarily
    import tempfile
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    # Try to load as GeoTIFF
    if RASTERIO_AVAILABLE and uploaded_file.name.lower().endswith(('.tif', '.tiff')):
        try:
            handler = GeoTIFFHandler(tmp_path)
            if handler.is_geotiff:
                image_pil = handler.get_image_as_pil()
                return image_pil, handler
        except Exception as e:
            print(f"⚠️ Failed to load as GeoTIFF: {e}")
    
    # Fallback to regular PIL
    image_pil = Image.open(uploaded_file).convert("RGB")
    return image_pil, None
