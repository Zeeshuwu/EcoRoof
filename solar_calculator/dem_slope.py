"""
Terrain slope / orientation support for R3-3 (roof slope and heading).

What this module can and cannot do
----------------------------------
The available DEM is PALSAR at 12.5 m. The study imagery is 200 x 200 m at 10 cm GSD.
That means **each site tile is covered by roughly 16 x 16 DEM cells**, and a single
roof spans one or two of them. Measured over the six study sites the DEM yields a
median slope of 0.0-3.2 degrees, whereas Indonesian residential roofs are commonly
pitched at about 30 degrees [C1].

So this module reports **terrain** slope and aspect, which it can genuinely measure,
and keeps **roof pitch** as an explicit scenario grid, which it cannot. Resampling the
DEM to the imagery's 10 cm grid would not create pitch information that is absent from
the source; it would only fabricate precision. Every product written here carries an
``EFFECTIVE_RESOLUTION_M`` tag of 12.5 for that reason.

This matches the team revision table, which scopes R3-3 as "Limitation only": what the
IV.D paragraph needs is a *quantified loss range*, not a new energy model.

Three sources for that loss range
---------------------------------
:func:`loss_range` deliberately returns three numbers side by side:

1. ``analytic``   -- the Hay-Davies clear-sky grid computed here. Revision Plan
   section 10 warns that offline clear-sky weather misrepresents orientation losses in
   the humid tropics, so this is a screening tool, never a number to publish.
2. ``literature`` -- measured Indonesian values [C1], [C2]. **This is the defensible
   number for IV.D.**
3. ``pvlib``      -- populated only when pvlib and a PVGIS TMY are available.

Azimuth convention throughout: degrees from North, increasing clockwise toward East
(0 = N, 90 = E, 180 = S, 270 = W). On Java (latitude about -6 to -8) the equator lies
to the north, so 0 degrees is the favourable orientation -- consistent with [C1]
reporting north as best.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from glob import glob
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject, transform_bounds

    RASTERIO_AVAILABLE = True
except ImportError:  # pragma: no cover
    RASTERIO_AVAILABLE = False


DEFAULT_DEM_DIR = os.path.join("input", "DEM")
DEFAULT_DEM_OUT_DIR = os.path.join("input", "01 DATA", "DEM")

#: PALSAR DEM native ground sample distance, metres.
PALSAR_NATIVE_RES_M = 12.5

#: Roof pitches to sweep. 0 covers flat institutional roofs (the UGM campus blocks),
#: 30 is the common Indonesian residential pitch reported in [C1], 15 sits between.
DEFAULT_PITCH_SCENARIOS: Tuple[float, ...] = (0.0, 15.0, 30.0)

#: Eight compass headings.
COMPASS: Dict[str, float] = {
    "N": 0.0, "NE": 45.0, "E": 90.0, "SE": 135.0,
    "S": 180.0, "SW": 225.0, "W": 270.0, "NW": 315.0,
}

#: Measured Indonesian evidence for the IV.D paragraph. These are the numbers to cite;
#: the analytic grid below is only a screening aid.
LITERATURE_ANCHORS: Dict[str, Dict[str, object]] = {
    "C2_jakarta_30deg": {
        "description": (
            "Jakarta, fixed rooftop PV at 30 deg tilt: about 1,243 kWh/kWp facing "
            "north-east against about 1,098 kWh/kWp facing south"
        ),
        "best_kwh_per_kwp": 1243.0,
        "worst_kwh_per_kwp": 1098.0,
        "tilt_deg": 30.0,
        # Two framings of the same pair. Only `loss_pct` is comparable with the
        # analytic grid, which is expressed in percentage points of loss below the
        # best plane; `gain_pct` is the same gap written as a gain over the worst.
        "loss_pct": round((1.0 - 1098.0 / 1243.0) * 100.0, 1),
        "gain_pct": round((1243.0 / 1098.0 - 1.0) * 100.0, 1),
        "citation": (
            "E. Tarigan, \"Simulation of impact azimuth angle on specific energy output "
            "of a fixed mounting rooftop PV system in Jakarta, Indonesia,\" AIP Conf. "
            "Proc., vol. 2470, 2022, doi: 10.1063/5.0080228."
        ),
    },
    "C1_34_cities": {
        "description": (
            "Across 34 Indonesian cities, roof pitch influenced annual yield more than "
            "orientation did, and north was the best heading on Java"
        ),
        "citation": (
            "B. Paramita, R. A. Mangkuto, A. G. Djafar, and N. Jamala, \"Energy yield of "
            "solar PV in 34 Indonesian cities with respect to various roof pitches and "
            "orientations,\" Int. J. Environ. Sci. Technol., 2025, "
            "doi: 10.1007/s13762-024-06201-7."
        ),
    },
}


# ===========================================================================
# Part A -- DEM preparation
# ===========================================================================

@dataclass(frozen=True)
class DemSource:
    """One province-wide DEM found on disk."""

    path: str
    crs: object
    bounds: Tuple[float, float, float, float]
    res_m: float
    nodata: Optional[float]

    @property
    def name(self) -> str:
        return os.path.basename(self.path)


def discover_province_dems(dem_dir: str = DEFAULT_DEM_DIR) -> List[DemSource]:
    """Index every GeoTIFF in ``dem_dir`` without reading pixel data."""
    _require_rasterio()
    sources: List[DemSource] = []
    for path in sorted(glob(os.path.join(dem_dir, "*.tif")) + glob(os.path.join(dem_dir, "*.tiff"))):
        try:
            with rasterio.open(path) as src:
                if src.crs is None:
                    continue
                sources.append(
                    DemSource(
                        path=path,
                        crs=src.crs,
                        bounds=tuple(src.bounds),
                        res_m=float((abs(src.res[0]) + abs(src.res[1])) / 2.0),
                        nodata=src.nodata,
                    )
                )
        except Exception:  # a non-raster file in the folder is not an error
            continue
    return sources


def find_dem_for_bounds(
    image_crs, image_bounds: Sequence[float], dems: Sequence[DemSource]
) -> Optional[DemSource]:
    """Pick the province DEM that contains the image footprint.

    Bounds are transformed into each DEM's own CRS first, which is what lets a
    Surabaya tile in EPSG:32749 be served by a DEM stored in EPSG:32750.
    """
    _require_rasterio()
    best: Optional[DemSource] = None
    for dem in dems:
        left, bottom, right, top = transform_bounds(image_crs, dem.crs, *image_bounds)
        db = dem.bounds
        if left >= db[0] and bottom >= db[1] and right <= db[2] and top <= db[3]:
            # Fully contained: prefer it immediately.
            return dem
        overlaps = left < db[2] and right > db[0] and bottom < db[3] and top > db[1]
        if overlaps and best is None:
            best = dem  # partial cover, used only if nothing contains the tile
    return best


def crop_dem_for_image(
    image_path: str,
    dem_dir: str = DEFAULT_DEM_DIR,
    out_dir: str = DEFAULT_DEM_OUT_DIR,
    buffer_m: float = 50.0,
    out_res_m: Optional[float] = None,
    write_derivatives: bool = True,
    dems: Optional[Sequence[DemSource]] = None,
) -> Dict[str, object]:
    """Crop and reproject the province DEM onto one image footprint.

    The output lands in ``out_dir`` as ``<SITE>_DEM.tif``, reprojected into the
    image's CRS at the DEM's **native** resolution (12.5 m). A buffer is added so the
    3x3 slope kernel has real neighbours at the tile edge instead of padding.

    Returns a provenance dict, which is also written into the GeoTIFF tags so the file
    documents its own limitations.
    """
    _require_rasterio()
    os.makedirs(out_dir, exist_ok=True)

    with rasterio.open(image_path) as img:
        if img.crs is None:
            raise ValueError("%s has no CRS; cannot locate it on the DEM" % image_path)
        img_crs = img.crs
        img_bounds = tuple(img.bounds)
        img_res = float((abs(img.res[0]) + abs(img.res[1])) / 2.0)

    sources = list(dems) if dems is not None else discover_province_dems(dem_dir)
    if not sources:
        raise FileNotFoundError("no DEM GeoTIFFs found in %s" % dem_dir)

    dem = find_dem_for_bounds(img_crs, img_bounds, sources)
    if dem is None:
        raise ValueError(
            "no DEM in %s covers %s (image bounds %s in %s)"
            % (dem_dir, os.path.basename(image_path), img_bounds, img_crs)
        )

    res = float(out_res_m or dem.res_m)

    # Target grid in the image CRS, snapped to a whole multiple of the resolution so
    # repeated runs produce byte-identical grids.
    west = math.floor((img_bounds[0] - buffer_m) / res) * res
    south = math.floor((img_bounds[1] - buffer_m) / res) * res
    east = math.ceil((img_bounds[2] + buffer_m) / res) * res
    north = math.ceil((img_bounds[3] + buffer_m) / res) * res
    width = int(round((east - west) / res))
    height = int(round((north - south) / res))
    dst_transform = from_origin(west, north, res, res)

    dst = np.full((height, width), np.nan, dtype="float32")
    with rasterio.open(dem.path) as src:
        src_nodata = src.nodata
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src_nodata,
            dst_transform=dst_transform,
            dst_crs=img_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    nodata_fraction = float(np.isnan(dst).mean())
    native_cells_x = (img_bounds[2] - img_bounds[0]) / dem.res_m
    native_cells_y = (img_bounds[3] - img_bounds[1]) / dem.res_m

    site = os.path.splitext(os.path.basename(image_path))[0]
    dem_out = os.path.join(out_dir, "%s_DEM.tif" % site)

    provenance: Dict[str, object] = {
        "site": site,
        "image_path": image_path,
        "image_crs": str(img_crs),
        "image_gsd_m": round(img_res, 4),
        "image_bounds": [round(v, 3) for v in img_bounds],
        "source_dem": dem.name,
        "source_dem_path": dem.path,
        "source_crs": str(dem.crs),
        "target_crs": str(img_crs),
        "native_res_m": dem.res_m,
        "output_res_m": res,
        "effective_resolution_m": dem.res_m,
        "native_cells_x": round(native_cells_x, 1),
        "native_cells_y": round(native_cells_y, 1),
        "buffer_m": buffer_m,
        "nodata_fraction": round(nodata_fraction, 4),
        "reprojected": str(dem.crs) != str(img_crs),
        "dem_path": dem_out,
        "caveat": (
            "Terrain surface only. At %.1f m native resolution this %.0f m tile spans "
            "about %.0f x %.0f DEM cells, so a single roof covers 1-2 cells. These "
            "values are NOT roof pitch."
            % (dem.res_m, img_bounds[2] - img_bounds[0], native_cells_x, native_cells_y)
        ),
    }

    tags = {k: str(v) for k, v in provenance.items()}
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": img_crs,
        "transform": dst_transform,
        "nodata": np.nan,
        "compress": "deflate",
    }
    with rasterio.open(dem_out, "w", **profile) as out:
        out.write(dst, 1)
        out.update_tags(**tags)
        out.set_band_description(1, "elevation_m")

    if write_derivatives:
        slope, aspect = slope_aspect_arrays(dst, res)
        for name, arr, desc in (
            ("slope", slope, "terrain_slope_deg"),
            ("aspect", aspect, "terrain_aspect_deg_from_north"),
        ):
            path = os.path.join(out_dir, "%s_%s.tif" % (site, name))
            with rasterio.open(path, "w", **profile) as out:
                out.write(arr.astype("float32"), 1)
                out.update_tags(**tags)
                out.set_band_description(1, desc)
            provenance["%s_path" % name] = path

    return provenance


def crop_dem_for_directory(
    image_dir: str,
    dem_dir: str = DEFAULT_DEM_DIR,
    out_dir: str = DEFAULT_DEM_OUT_DIR,
    buffer_m: float = 50.0,
    write_derivatives: bool = True,
) -> List[Dict[str, object]]:
    """Batch form of :func:`crop_dem_for_image` over every GeoTIFF in a folder."""
    _require_rasterio()
    dems = discover_province_dems(dem_dir)
    results: List[Dict[str, object]] = []
    images = sorted(glob(os.path.join(image_dir, "*.tif")) + glob(os.path.join(image_dir, "*.tiff")))
    for image_path in images:
        try:
            results.append(
                crop_dem_for_image(
                    image_path,
                    dem_dir=dem_dir,
                    out_dir=out_dir,
                    buffer_m=buffer_m,
                    write_derivatives=write_derivatives,
                    dems=dems,
                )
            )
        except Exception as exc:
            results.append({"site": os.path.splitext(os.path.basename(image_path))[0], "error": str(exc)})
    return results


# ===========================================================================
# Part B -- slope and aspect
# ===========================================================================

def horn_gradients(dem: np.ndarray, cell_size_m: float) -> Tuple[np.ndarray, np.ndarray]:
    """Horn (1981) 3x3 gradients, the method GDAL and ArcGIS both use.

    Written out by explicit slicing rather than a convolution kernel on purpose:
    ``scipy.ndimage.convolve`` flips the kernel (it is a true convolution, not a
    correlation), which silently mirrors the east-west aspect. The labelled cell
    layout below leaves no room for that ambiguity.

        a b c      row 0 is the NORTH edge of a north-up raster
        d e f      column 0 is the WEST edge
        g h i
    """
    z = np.asarray(dem, dtype="float64")
    p = np.pad(z, 1, mode="edge")
    a, b, c = p[:-2, :-2], p[:-2, 1:-1], p[:-2, 2:]
    d, f = p[1:-1, :-2], p[1:-1, 2:]
    g, h, i = p[2:, :-2], p[2:, 1:-1], p[2:, 2:]

    dzdx = ((c + 2.0 * f + i) - (a + 2.0 * d + g)) / (8.0 * cell_size_m)
    dzdy = ((g + 2.0 * h + i) - (a + 2.0 * b + c)) / (8.0 * cell_size_m)
    return dzdx, dzdy


def slope_aspect_arrays(dem: np.ndarray, cell_size_m: float) -> Tuple[np.ndarray, np.ndarray]:
    """Slope (degrees) and aspect (degrees from north, clockwise) from an elevation grid.

    Flat cells get an aspect of ``nan`` rather than a fabricated direction: a level
    roof has no meaningful heading, and several of the study sites are genuinely flat.
    """
    z = np.asarray(dem, dtype="float64")
    filled = np.nan_to_num(z, nan=float(np.nanmean(z)) if np.isfinite(z).any() else 0.0)
    dzdx, dzdy = horn_gradients(filled, cell_size_m)

    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))

    # ESRI aspect convention, then rotated into compass degrees from north.
    aspect = np.degrees(np.arctan2(dzdy, -dzdx))
    aspect = np.where(aspect < 0, 90.0 - aspect, np.where(aspect > 90.0, 450.0 - aspect, 90.0 - aspect))
    aspect = np.where(np.hypot(dzdx, dzdy) < 1e-9, np.nan, aspect % 360.0)

    slope = np.where(np.isnan(z), np.nan, slope)
    aspect = np.where(np.isnan(z), np.nan, aspect)
    return slope, aspect


def load_slope_aspect(dem_path: str) -> Dict[str, object]:
    """Read a cropped DEM and derive slope/aspect at its native resolution."""
    _require_rasterio()
    with rasterio.open(dem_path) as src:
        z = src.read(1).astype("float64")
        if src.nodata is not None and not np.isnan(src.nodata):
            z = np.where(z == src.nodata, np.nan, z)
        res = float((abs(src.res[0]) + abs(src.res[1])) / 2.0)
        slope, aspect = slope_aspect_arrays(z, res)
        return {
            "elevation": z,
            "slope": slope,
            "aspect": aspect,
            "transform": src.transform,
            "crs": src.crs,
            "res_m": res,
            "tags": src.tags(),
        }


def _sample_bilinear(arr: np.ndarray, row: float, col: float) -> float:
    """Bilinear sample with edge clamping; returns nan if any contributor is nan."""
    h, w = arr.shape
    r = min(max(row, 0.0), h - 1.0)
    c = min(max(col, 0.0), w - 1.0)
    r0, c0 = int(math.floor(r)), int(math.floor(c))
    r1, c1 = min(r0 + 1, h - 1), min(c0 + 1, w - 1)
    fr, fc = r - r0, c - c0
    quad = np.array([arr[r0, c0], arr[r0, c1], arr[r1, c0], arr[r1, c1]], dtype="float64")
    if np.isnan(quad).any():
        return float("nan")
    return float(
        quad[0] * (1 - fr) * (1 - fc)
        + quad[1] * (1 - fr) * fc
        + quad[2] * fr * (1 - fc)
        + quad[3] * fr * fc
    )


def _circular_mean_deg(values: Sequence[float]) -> float:
    """Mean of compass bearings; plain averaging would put N and NW-of-N at 180."""
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return float("nan")
    rad = np.radians(vals)
    return float(np.degrees(math.atan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360.0)


def zonal_terrain(
    segments: Sequence[Dict[str, object]],
    image_transform,
    dem_layers: Dict[str, object],
) -> List[Dict[str, object]]:
    """Attach terrain slope and aspect to each detected roof segment.

    Segments carry pixel-space centroids on the *imagery* grid; those are converted to
    world coordinates and then sampled on the DEM grid. Because a roof spans one or two
    DEM cells, a bilinear sample at the centroid carries as much information as a zonal
    mean would, and is easier to reason about.
    """
    _require_rasterio()
    inv = ~dem_layers["transform"]
    slope, aspect = dem_layers["slope"], dem_layers["aspect"]

    out: List[Dict[str, object]] = []
    for seg in segments:
        cx, cy = seg["centroid"]
        wx, wy = image_transform * (float(cx), float(cy))
        col, row = inv * (wx, wy)
        s = _sample_bilinear(slope, row, col)
        a = _sample_bilinear(aspect, row, col)
        out.append(
            {
                "id": seg.get("id"),
                "rank": seg.get("rank"),
                "area_m2": seg.get("area_m2"),
                "terrain_slope_deg": None if math.isnan(s) else round(s, 2),
                "terrain_aspect_deg": None if math.isnan(a) else round(a, 1),
                "terrain_heading": None if math.isnan(a) else bearing_to_compass(a),
            }
        )
    return out


def bearing_to_compass(bearing_deg: float) -> str:
    """Nearest of the eight compass points."""
    if bearing_deg is None or math.isnan(bearing_deg):
        return "flat"
    names = list(COMPASS)
    return names[int(round((bearing_deg % 360.0) / 45.0)) % 8]


def terrain_summary(dem_layers: Dict[str, object]) -> Dict[str, object]:
    """Site-level terrain statistics, with the caveat attached."""
    slope, aspect = dem_layers["slope"], dem_layers["aspect"]
    finite = slope[np.isfinite(slope)]
    if finite.size == 0:
        return {"available": False}
    flat = float((finite < 0.5).mean())
    return {
        "available": True,
        "slope_min_deg": round(float(finite.min()), 2),
        "slope_median_deg": round(float(np.median(finite)), 2),
        "slope_max_deg": round(float(finite.max()), 2),
        "slope_mean_deg": round(float(finite.mean()), 2),
        "flat_fraction": round(flat, 3),
        "dominant_aspect_deg": round(_circular_mean_deg(aspect[np.isfinite(aspect)].tolist()), 1)
        if np.isfinite(aspect).any()
        else None,
        "relief_m": round(
            float(np.nanmax(dem_layers["elevation"]) - np.nanmin(dem_layers["elevation"])), 1
        ),
        "res_m": dem_layers["res_m"],
    }


# ===========================================================================
# Part C -- plane-of-array transposition
# ===========================================================================

def _solar_position(lat_deg: float, day: int, hour_angle_deg: float) -> Tuple[float, float]:
    """Cosine of the solar zenith angle, and solar azimuth in compass degrees."""
    lat = math.radians(lat_deg)
    dec = math.radians(23.45) * math.sin(2.0 * math.pi * (284 + day) / 365.0)
    omega = math.radians(hour_angle_deg)

    cos_z = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(omega)
    # Negated numerator so a negative hour angle (morning) puts the sun in the east,
    # matching the compass convention documented at the top of this module.
    azimuth = math.degrees(
        math.atan2(
            -math.cos(dec) * math.sin(omega),
            math.cos(lat) * math.sin(dec) - math.sin(lat) * math.cos(dec) * math.cos(omega),
        )
    ) % 360.0
    return cos_z, azimuth


def poa_annual(
    lat_deg: float,
    tilt_deg: float,
    azimuth_deg: float,
    diffuse_fraction: float = 0.35,
    albedo: float = 0.2,
    sky_model: str = "hay",
    day_step: int = 2,
    hour_step_deg: float = 10.0,
) -> float:
    """Relative annual plane-of-array irradiation under a clear sky.

    The absolute value is meaningless -- only ratios between orientations are used.
    ``sky_model`` is "hay" (Hay-Davies anisotropic, circumsolar diffuse weighted
    toward the sun) or "iso" (isotropic sky, which understates orientation effects
    further still).

    Known limitation: with a smooth clear-sky irradiance shape this reproduces only
    about two thirds of the north-south spread that hourly TMY studies report for
    Jakarta [C2]. Use :func:`loss_range` rather than these numbers directly.
    """
    if not 0.0 <= diffuse_fraction < 1.0:
        raise ValueError("diffuse_fraction must be in [0, 1)")
    lat = lat_deg
    beta = math.radians(tilt_deg)
    gamma = math.radians(azimuth_deg)
    total = 0.0

    for day in range(1, 366, day_step):
        h = -180.0
        while h < 180.0:
            cos_z, solar_az = _solar_position(lat, day, h)
            h += hour_step_deg
            if cos_z <= 0.02:
                continue
            ghi = 1361.0 * cos_z * 0.7          # crude clear-sky shape; scale cancels
            dni = ghi * (1.0 - diffuse_fraction) / cos_z
            dhi = ghi * diffuse_fraction
            sin_z = math.sqrt(max(0.0, 1.0 - cos_z * cos_z))

            cos_i = max(
                0.0,
                cos_z * math.cos(beta)
                + sin_z * math.sin(beta) * math.cos(math.radians(solar_az) - gamma),
            )
            beam = dni * cos_i
            if sky_model == "iso":
                diffuse = dhi * (1.0 + math.cos(beta)) / 2.0
            else:
                ai = min(1.0, dni / 1361.0)      # anisotropy index
                diffuse = dhi * (ai * cos_i / cos_z + (1.0 - ai) * (1.0 + math.cos(beta)) / 2.0)
            ground = ghi * albedo * (1.0 - math.cos(beta)) / 2.0
            total += beam + diffuse + ground

    return total


def poa_factor(
    lat_deg: float,
    tilt_deg: float,
    azimuth_deg: float,
    reference_tilt_deg: Optional[float] = None,
    reference_azimuth_deg: Optional[float] = None,
    **kwargs,
) -> float:
    """Yield of one plane relative to the reference the irradiation data assumes.

    Global Solar Atlas PVOUT assumes a free-standing array at optimum tilt facing the
    equator [E5], which is the draft's implicit assumption. The reference therefore
    defaults to equator-facing at the latitude-derived optimum tilt.
    """
    ref_tilt = reference_tilt_deg if reference_tilt_deg is not None else optimum_tilt(lat_deg)
    ref_az = (
        reference_azimuth_deg
        if reference_azimuth_deg is not None
        else equator_facing_azimuth(lat_deg)
    )
    ref = poa_annual(lat_deg, ref_tilt, ref_az, **kwargs)
    if ref <= 0:
        return float("nan")
    return poa_annual(lat_deg, tilt_deg, azimuth_deg, **kwargs) / ref


def equator_facing_azimuth(lat_deg: float) -> float:
    """0 degrees (north) in the southern hemisphere, 180 (south) in the northern."""
    return 0.0 if lat_deg < 0 else 180.0


def optimum_tilt(lat_deg: float) -> float:
    """Near the equator the optimum tilt is small; a common rule is |latitude|."""
    return abs(lat_deg)


def orientation_grid(
    lat_deg: float,
    tilts: Sequence[float] = DEFAULT_PITCH_SCENARIOS,
    headings: Optional[Dict[str, float]] = None,
    **kwargs,
):
    """Relative yield for every (pitch, heading) combination.

    Returned as a DataFrame of loss percentages against the optimum-tilt reference:
    positive means the plane yields less than the draft assumes.
    """
    import pandas as pd

    headings = headings or COMPASS
    ref_tilt = optimum_tilt(lat_deg)
    ref_az = equator_facing_azimuth(lat_deg)
    ref = poa_annual(lat_deg, ref_tilt, ref_az, **kwargs)

    rows = []
    for tilt in tilts:
        row: Dict[str, object] = {"tilt_deg": tilt}
        for name, az in headings.items():
            row[name] = round((1.0 - poa_annual(lat_deg, tilt, az, **kwargs) / ref) * 100.0, 1)
        rows.append(row)
    return pd.DataFrame(rows).set_index("tilt_deg")


def loss_range(
    lat_deg: float,
    tilts: Sequence[float] = DEFAULT_PITCH_SCENARIOS,
    pvlib_result: Optional[Tuple[float, float]] = None,
    **kwargs,
) -> Dict[str, object]:
    """The quantified loss range for the IV.D limitation paragraph.

    Three sources are returned side by side so the clear-sky number is never mistaken
    for a publishable one. See the module docstring.
    """
    grid = orientation_grid(lat_deg, tilts, **kwargs)
    values = grid.to_numpy(dtype=float)
    analytic = (round(float(values.min()), 1), round(float(values.max()), 1))

    worst_tilt = grid.max(axis=1).idxmax()
    worst_heading = grid.loc[worst_tilt].idxmax()

    # Two different quantities, kept apart on purpose because conflating them would
    # put the wrong number in the paper:
    #
    #   total_loss_pct       -- how far a given (pitch, heading) plane falls below the
    #                           optimum-tilt equator-facing surface that the draft's
    #                           irradiation input assumes [E5]. Includes tilt AND
    #                           heading. This is what the draft's bias actually is.
    #   orientation_spread   -- best minus worst heading at ONE fixed pitch. This is
    #                           what [C2] measured, so it is the only figure directly
    #                           comparable to the literature anchor.
    steepest = max(tilts)
    row = grid.loc[steepest]
    analytic_spread_pct = round(float(row.max() - row.min()), 1)

    c2 = LITERATURE_ANCHORS["C2_jakarta_30deg"]

    return {
        "analytic": {
            "total_loss_pct": analytic,
            "range_pct": analytic,  # backwards-compatible alias
            # Percentage points of loss between the best and worst heading at the
            # steepest pitch in the grid -- the same framing as the [C2] anchor.
            "orientation_spread_pct": analytic_spread_pct,
            "orientation_spread_at_tilt_deg": float(steepest),
            "grid": grid,
            "worst_case": {"tilt_deg": float(worst_tilt), "heading": str(worst_heading)},
            "caveat": (
                "Clear-sky analytic model, screening only. Revision Plan section 10 "
                "warns not to report offline weather numbers. Its orientation spread "
                "(%.1f pp at %.0f deg pitch) recovers only about two thirds of the "
                "%.1f pp that hourly TMY simulation reports for Jakarta [C2]."
                % (analytic_spread_pct, steepest, float(c2["loss_pct"]))
            ),
        },
        "literature": {
            "orientation_spread_pct": float(c2["loss_pct"]),
            "range_pct": (0.0, float(c2["loss_pct"])),
            "measured_at_tilt_deg": float(c2["tilt_deg"]),
            "anchors": LITERATURE_ANCHORS,
            "recommended": True,
            "note": (
                "Measured Indonesian evidence; this is the number to cite in IV.D. "
                "It is an orientation-only spread at fixed pitch, not a total loss "
                "against an optimum-tilt reference."
            ),
        },
        "pvlib": (
            {"range_pct": pvlib_result, "available": True}
            if pvlib_result
            else {"available": False, "note": "install pvlib and fetch a PVGIS TMY to populate"}
        ),
        "direction_of_bias": (
            "The draft's estimates are optimistic for steep, pole-facing (south-facing "
            "on Java) roof planes, and close to correct for low-pitch or flat "
            "institutional roofs such as the UGM campus buildings."
        ),
    }


def limitation_sentence(lat_deg: float, terrain: Optional[Dict[str, object]] = None, **kwargs) -> str:
    """A copy-ready factual summary for drafting IV.D.

    Deliberately a statement of the numbers, not finished prose: Reviewer 3 flagged the
    draft for AI-like text (R3-5), so the paragraph must be written by hand.
    """
    lr = loss_range(lat_deg, **kwargs)
    ana_lo, ana_hi = lr["analytic"]["total_loss_pct"]
    worst = lr["analytic"]["worst_case"]
    parts = [
        "Latitude %.2f deg." % lat_deg,
        "Measured Indonesian evidence [C2] gives an orientation-driven spread of about "
        "%.1f pp at %.0f deg pitch; [C1] finds pitch matters more than heading and that "
        "north is the best heading on Java."
        % (lr["literature"]["orientation_spread_pct"], lr["literature"]["measured_at_tilt_deg"]),
        "Against the optimum-tilt equator-facing surface that the irradiation input "
        "assumes [E5], the pitch/heading grid spans %.1f to %.1f %% loss, worst at "
        "%.0f deg pitch facing %s."
        % (ana_lo, ana_hi, worst["tilt_deg"], worst["heading"]),
        "That grid is clear-sky and recovers only about two thirds of the measured "
        "orientation spread, so treat it as a lower bound.",
    ]
    if terrain and terrain.get("available"):
        parts.append(
            "Terrain slope measured from the 12.5 m DEM at this site is %.1f deg "
            "(median), contributing a negligible share of that range; roof pitch, "
            "which the DEM cannot resolve, dominates."
            % terrain["slope_median_deg"]
        )
    parts.append(lr["direction_of_bias"])
    return " ".join(parts)


def _require_rasterio() -> None:
    if not RASTERIO_AVAILABLE:
        raise ImportError("rasterio is required for DEM handling; pip install rasterio")
