"""
GeoTIFF utilities for accurate geospatial measurements
"""

import os
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import rasterio
    from rasterio.errors import RasterioIOError
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    warnings.warn("rasterio not installed. GeoTIFF support disabled. Install: pip install rasterio")


class GeoTIFFHandler:
    """Handle GeoTIFF files and extract geospatial metadata"""
    
    def __init__(self, file_path: str, display_name: Optional[str] = None):
        self.file_path = file_path
        # Original name of the source raster. Uploads are copied to a tempfile, so
        # without this the site is always reported as "Analysis Zone" and the DEM
        # crop has no site name to write.
        self.filename = display_name or os.path.basename(file_path)
        self.metadata = {}
        self.is_geotiff = False
        self.pixel_size_x = None
        self.pixel_size_y = None
        self.crs = None
        self.bounds = None
        self.transform = None
        self.lat_center = None
        self.lon_center = None

        if RASTERIO_AVAILABLE:
            self._load_metadata()

    @classmethod
    def from_path(cls, path: str) -> "GeoTIFFHandler":
        """Open a raster already on disk, keeping its real filename.

        Used by the site picker so `input/01 DATA/IMAGE/JKT_ROOF.tif` keeps the name
        `JKT_ROOF` all the way through to `input/01 DATA/DEM/JKT_ROOF_DEM.tif`.
        """
        return cls(path)
    
    def _load_metadata(self):
        """Load GeoTIFF metadata using rasterio"""
        try:
            with rasterio.open(self.file_path) as src:
                if src.crs is not None:
                    self.is_geotiff = True

                    # ── Store the FULL affine transform object ────────────────
                    # This is what export_utils.py needs to georeference exports.
                    # Do NOT reconstruct from bounds — read directly from file.
                    self.transform    = src.transform          # ← rasterio Affine
                    self.crs          = src.crs                # ← rasterio CRS object

                    self.pixel_size_x = abs(src.transform.a)  # metres/pixel X
                    self.pixel_size_y = abs(src.transform.e)  # metres/pixel Y
                    self.bounds       = src.bounds

                    # Geographic centre, needed by the slope/orientation model and
                    # by anything that wants to place the site on a map.
                    try:
                        from rasterio.warp import transform_bounds
                        west, south, east, north = transform_bounds(
                            src.crs, "EPSG:4326", *src.bounds
                        )
                        self.lon_center = (west + east) / 2.0
                        self.lat_center = (south + north) / 2.0
                    except Exception as exc:
                        print(f"WARNING: could not derive lat/lon centre: {exc}")

                    self.metadata = {
                        'width':        src.width,
                        'height':       src.height,
                        'crs':          src.crs.to_string(),
                        'filename':     self.filename,
                        'pixel_size_x': self.pixel_size_x,
                        'pixel_size_y': self.pixel_size_y,
                        'gsd':          (self.pixel_size_x + self.pixel_size_y) / 2,
                        'bounds': {
                            'left':   self.bounds.left,
                            'bottom': self.bounds.bottom,
                            'right':  self.bounds.right,
                            'top':    self.bounds.top,
                        },
                        # Flat tuple form. district_analysis.py reads metadata['bbox']
                        # and silently got None before this existed.
                        'bbox': (
                            self.bounds.left, self.bounds.bottom,
                            self.bounds.right, self.bounds.top,
                        ),
                        'lat_center': self.lat_center,
                        'lon_center': self.lon_center,
                        'area_m2': (
                            (self.bounds.right - self.bounds.left) *
                            (self.bounds.top   - self.bounds.bottom)
                        ),
                    }

                    print("GeoTIFF detected:")
                    print(f"   CRS      : {src.crs.to_string()}")
                    print(f"   Transform: {src.transform}")
                    print(f"   GSD      : {self.metadata['gsd']:.4f} m/pixel")

        except RasterioIOError:
            print("WARNING: not a valid GeoTIFF or no geospatial data")
        except Exception as e:
            print(f"WARNING: error reading GeoTIFF: {e}")

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
            print(f"WARNING: error loading GeoTIFF as image: {e}")
            return Image.open(self.file_path).convert("RGB")
    
    def get_info_dict(self) -> Dict:
        """Get all metadata as dictionary"""
        return {
            'is_geotiff': self.is_geotiff,
            'filename': self.filename,
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
            handler = GeoTIFFHandler(tmp_path, display_name=uploaded_file.name)
            if handler.is_geotiff:
                image_pil = handler.get_image_as_pil()
                return image_pil, handler
        except Exception as e:
            print(f"WARNING: failed to load as GeoTIFF: {e}")
    
    # Fallback to regular PIL
    image_pil = Image.open(uploaded_file).convert("RGB")
    return image_pil, None
