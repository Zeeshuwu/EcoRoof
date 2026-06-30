"""
EcoPower Roof — Export Utilities
=================================
Handles GeoTIFF overlay export and Shapefile vectorization.
Coordinate correctness depends on the affine transform being read
directly from the source GeoTIFF via the geotiff_handler object.
"""

import io
import zipfile
import tempfile
import os

import numpy as np
from PIL import Image


try:
    import rasterio
    import rasterio.features
    import rasterio.transform
    from rasterio.crs import CRS
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    import fiona
    import fiona.crs
    from shapely.geometry import shape, mapping
    FIONA_AVAILABLE = True
except ImportError:
    FIONA_AVAILABLE = False




def check_export_dependencies() -> dict:
    return {
        "geotiff_export": RASTERIO_AVAILABLE,
        "shp_export":     RASTERIO_AVAILABLE and FIONA_AVAILABLE,
    }




def export_overlay_as_geotiff(
    overlay_pil,
    geotiff_handler,
    detection_mode: str,
) -> bytes | None:
    """
    Export the coloured overlay PIL image as a GeoTIFF.
    If a valid geotiff_handler is provided, the source CRS and
    affine transform are inherited exactly — no reconstruction.
    """
    if not RASTERIO_AVAILABLE:
        return None

    try:
        overlay_np = np.array(overlay_pil.convert("RGB"))
        h, w       = overlay_np.shape[:2]

        # ── Resolve transform & CRS ───────────────────────────────────────────
        is_georef = (
            geotiff_handler is not None
            and getattr(geotiff_handler, "is_geotiff", False)
        )

        if is_georef:
            src_transform = geotiff_handler.transform          # rasterio Affine
            src_crs       = geotiff_handler.crs                # e.g. CRS object or EPSG string
        else:
            # Pixel-space fallback: 1 px = 1 m, origin (0, 0)
            src_transform = rasterio.transform.from_bounds(0, 0, w, h, w, h)
            src_crs       = CRS.from_epsg(4326)

        buf = io.BytesIO()
        with rasterio.open(
            buf, "w",
            driver    = "GTiff",
            height    = h,
            width     = w,
            count     = 3,
            dtype     = rasterio.uint8,
            crs       = src_crs,
            transform = src_transform,
        ) as dst:
            for band_idx in range(3):
                dst.write(overlay_np[:, :, band_idx], band_idx + 1)

        buf.seek(0)
        return buf.getvalue()

    except Exception as e:
        import streamlit as st
        st.error(f"GeoTIFF export error: {e}")
        return None



def export_mask_as_shapefile(
    mask_np: np.ndarray,
    geotiff_handler,
    detection_mode: str,
    simplify_tolerance: float = 0.5,
    min_area_m2: float        = 1.0,
) -> bytes | None:
    """
    Vectorize a binary mask and export as a georeferenced Shapefile ZIP.

    The critical rule:
        rasterio.features.shapes() must receive the EXACT affine transform
        from the source GeoTIFF so that output polygon coordinates are
        already in the correct CRS — not in pixel space.

    If no GeoTIFF is available, polygons are written in pixel coordinates
    with a WGS84 placeholder CRS (clearly labelled in the UI).
    """
    if not RASTERIO_AVAILABLE or not FIONA_AVAILABLE:
        return None

    try:
        is_georef = (
            geotiff_handler is not None
            and getattr(geotiff_handler, "is_geotiff", False)
        )

        if is_georef:
            src_transform = geotiff_handler.transform
            raw_crs       = geotiff_handler.crs


            if isinstance(raw_crs, CRS):
                src_crs = raw_crs
            elif isinstance(raw_crs, str):
                src_crs = CRS.from_string(raw_crs)
            else:
                src_crs = CRS.from_user_input(raw_crs)
        else:
            h, w          = mask_np.shape[:2]
            src_transform = rasterio.transform.from_bounds(0, 0, w, h, w, h)
            src_crs       = CRS.from_epsg(4326)


        mask_uint8 = (mask_np > 0).astype(np.uint8)


        raw_shapes = list(
            rasterio.features.shapes(
                mask_uint8,
                mask         = mask_uint8,
                transform    = src_transform,   # ← THE FIX
                connectivity = 4,
            )
        )

        if not raw_shapes:
            return None


        valid_geoms = []
        for geom_dict, value in raw_shapes:
            if value == 0:
                continue
            geom = shape(geom_dict)
            if geom.area < min_area_m2:
                continue
            if simplify_tolerance > 0:
                geom = geom.simplify(simplify_tolerance, preserve_topology=True)
            if geom.is_valid and not geom.is_empty:
                valid_geoms.append(geom)

        if not valid_geoms:
            return None


        schema = {
            "geometry":   "Polygon",
            "properties": {
                "id":   "int",
                "mode": "str",
                "area": "float",
            },
        }

        try:
            epsg_code = src_crs.to_epsg()
            fiona_crs = f"EPSG:{epsg_code}" if epsg_code else src_crs.to_wkt()
        except Exception:
            fiona_crs = src_crs.to_wkt()

        zip_buffer = io.BytesIO()

        with tempfile.TemporaryDirectory() as tmpdir:
            shp_path = os.path.join(tmpdir, f"{detection_mode}_mask.shp")

            with fiona.open(
                shp_path, "w",
                driver = "ESRI Shapefile",
                crs    = fiona_crs,
                schema = schema,
            ) as shp:
                for idx, geom in enumerate(valid_geoms):
                    shp.write({
                        "geometry":   mapping(geom),
                        "properties": {
                            "id":   idx + 1,
                            "mode": detection_mode,
                            "area": round(geom.area, 4),
                        },
                    })


            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
                    fpath = os.path.join(tmpdir, f"{detection_mode}_mask{ext}")
                    if os.path.exists(fpath):
                        zf.write(fpath, arcname=f"{detection_mode}_mask{ext}")

        zip_buffer.seek(0)
        return zip_buffer.getvalue()

    except Exception as e:
        import streamlit as st
        st.error(f"Shapefile export error: {e}")
        return None
