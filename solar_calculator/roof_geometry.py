"""
Roof pitch and aspect measured from a photogrammetric DSM (R3-3 follow-up).

Section IV.D of the manuscript states that roof slope and orientation could not
be derived because the only elevation data available was a 12.5 m terrain model,
too coarse to resolve an individual roof. That remains true for the six study
sites. A separate LOD dataset does, however, provide a 0.11 m DSM over Surabaya
and a 0.31 m DSM over Jakarta, together with digitized roof-plane polygons.

**These DSMs do not overlap the study sites.** They cover different
neighbourhoods of the same two cities, roughly 1.2 km from the Surabaya site and
6 km from the Jakarta one. They therefore cannot correct the energy estimates
reported in Section IV.B, and nothing here should be presented as doing so.

What they can do is replace an assumption with a measurement. The limitation
paragraph currently takes "roofs are commonly pitched at approximately 30 deg"
from the literature. With these data the pitch and aspect of several hundred real
Indonesian roof planes can be measured directly, which turns the magnitude of the
limitation into an observation rather than a citation.

Method
------
For each roof-plane polygon, DSM cells inside an inward-buffered copy of the
polygon are extracted and a plane ``z = ax + by + c`` is fitted by least squares.
The inward buffer keeps eaves, walls and neighbouring vegetation out of the fit.
Slope is ``atan(sqrt(a^2 + b^2))`` and aspect is the compass bearing of steepest
descent, measured clockwise from north to match ``dem_slope``. Planes whose
residual RMSE is large are not planar in reality (vegetation overhang, mixed
structures) and are excluded rather than reported as a pitch.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import geopandas as gpd
    import rasterio
    from rasterio.mask import mask as rio_mask

    GEO_AVAILABLE = True
except ImportError:  # pragma: no cover
    GEO_AVAILABLE = False

#: Inward buffer before sampling, metres. Keeps the fit off eaves and walls.
DEFAULT_EDGE_BUFFER_M = 0.5

#: Minimum valid DSM cells for a fit to be trusted.
DEFAULT_MIN_PIXELS = 40

#: Planes with residual RMSE above this are not flat surfaces; excluded.
DEFAULT_MAX_RMSE_M = 0.35


def fit_plane(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Dict[str, float]:
    """Least-squares plane through (x, y, z); returns slope, aspect and fit quality.

    ``x`` is easting and ``y`` northing in metres, so the fitted gradient is in
    real-world orientation and no raster row/column convention leaks in.

    Aspect is the downslope compass bearing: the gradient ``(a, b)`` points
    uphill, so steepest descent is ``(-a, -b)`` and its bearing is
    ``atan2(east, north)`` measured clockwise from north.
    """
    n = x.size
    if n < 3:
        return {"slope_deg": float("nan"), "aspect_deg": float("nan"),
                "rmse_m": float("nan"), "r2": float("nan"), "n_px": int(n)}

    # Centre the coordinates: a 6-digit UTM easting squared overflows the
    # conditioning of the normal equations and the fit degrades badly.
    xm, ym = x.mean(), y.mean()
    A = np.column_stack([x - xm, y - ym, np.ones(n)])
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    a, b, _ = coef

    resid = z - A @ coef
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    var = float(np.var(z))
    r2 = float(1.0 - np.var(resid) / var) if var > 1e-12 else float("nan")

    grad = math.hypot(a, b)
    slope = math.degrees(math.atan(grad))
    if grad < 1e-9:
        aspect = float("nan")          # a level plane has no heading
    else:
        aspect = math.degrees(math.atan2(-a, -b)) % 360.0

    return {"slope_deg": slope, "aspect_deg": aspect, "rmse_m": rmse,
            "r2": r2, "n_px": int(n)}


def measure_roof_planes(
    dsm_path: str,
    planes_path: str,
    edge_buffer_m: float = DEFAULT_EDGE_BUFFER_M,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    max_rmse_m: float = DEFAULT_MAX_RMSE_M,
):
    """Fit every roof plane in ``planes_path`` against ``dsm_path``.

    Returns a GeoDataFrame with slope, aspect and fit diagnostics per plane, plus
    a boolean ``accepted`` column so rejected planes stay visible rather than
    silently vanishing from the counts.
    """
    if not GEO_AVAILABLE:  # pragma: no cover
        raise ImportError("geopandas and rasterio are required")

    planes = gpd.read_file(planes_path)
    rows = []
    with rasterio.open(dsm_path) as dsm:
        if planes.crs is not None and str(planes.crs) != str(dsm.crs):
            planes = planes.to_crs(dsm.crs)
        nodata = dsm.nodata

        for idx, geom in enumerate(planes.geometry):
            rec = {"plane_id": idx, "area_m2": float(geom.area) if geom else 0.0}
            if geom is None or geom.is_empty:
                rows.append({**rec, "slope_deg": np.nan, "aspect_deg": np.nan,
                             "rmse_m": np.nan, "r2": np.nan, "n_px": 0,
                             "accepted": False, "reason": "empty geometry"})
                continue

            sample = geom.buffer(-abs(edge_buffer_m))
            if sample.is_empty:
                sample = geom  # too small to buffer; use it whole

            try:
                out, tr = rio_mask(dsm, [sample.__geo_interface__], crop=True,
                                   filled=False, all_touched=False)
            except Exception as exc:
                rows.append({**rec, "slope_deg": np.nan, "aspect_deg": np.nan,
                             "rmse_m": np.nan, "r2": np.nan, "n_px": 0,
                             "accepted": False, "reason": str(exc)[:40]})
                continue

            band = out[0]
            valid = ~np.ma.getmaskarray(band)
            data = np.ma.getdata(band)
            if nodata is not None:
                valid &= data != nodata
            valid &= np.isfinite(data)

            if valid.sum() < min_pixels:
                rows.append({**rec, "slope_deg": np.nan, "aspect_deg": np.nan,
                             "rmse_m": np.nan, "r2": np.nan, "n_px": int(valid.sum()),
                             "accepted": False, "reason": "too few pixels"})
                continue

            r_idx, c_idx = np.nonzero(valid)
            xs, ys = rasterio.transform.xy(tr, r_idx, c_idx)
            fit = fit_plane(np.asarray(xs), np.asarray(ys), data[valid].astype(float))

            ok = bool(np.isfinite(fit["rmse_m"]) and fit["rmse_m"] <= max_rmse_m)
            rows.append({**rec, **fit, "accepted": ok,
                         "reason": "" if ok else "rmse %.2f m > %.2f" % (fit["rmse_m"], max_rmse_m)})

    import pandas as pd
    out_gdf = gpd.GeoDataFrame(pd.DataFrame(rows), geometry=planes.geometry.values, crs=planes.crs)
    return out_gdf


def pitch_distribution(gdf, accepted_only: bool = True) -> Dict[str, object]:
    """Summary statistics of measured pitch and aspect, weighted and unweighted.

    Area weighting matters: a solar estimate cares about square metres, not about
    how many polygons happen to have been digitized.
    """
    d = gdf[gdf["accepted"]] if accepted_only else gdf
    d = d[np.isfinite(d["slope_deg"])]
    if len(d) == 0:
        return {"n": 0}

    s = d["slope_deg"].to_numpy()
    w = d["area_m2"].to_numpy()
    flat = s < 5.0

    res: Dict[str, object] = {
        "n_planes": int(len(d)),
        "n_rejected": int((~gdf["accepted"]).sum()),
        "total_area_m2": round(float(w.sum()), 1),
        "slope_median_deg": round(float(np.median(s)), 2),
        "slope_mean_deg": round(float(s.mean()), 2),
        "slope_p10_deg": round(float(np.percentile(s, 10)), 2),
        "slope_p90_deg": round(float(np.percentile(s, 90)), 2),
        "slope_max_deg": round(float(s.max()), 2),
        "slope_area_weighted_deg": round(float(np.average(s, weights=w)), 2),
        "flat_fraction_pct": round(100.0 * flat.mean(), 1),
        "flat_area_fraction_pct": round(100.0 * w[flat].sum() / w.sum(), 1),
    }

    a = d.loc[np.isfinite(d["aspect_deg"]), "aspect_deg"].to_numpy()
    if a.size:
        rad = np.radians(a)
        res["aspect_mean_deg"] = round(float(np.degrees(
            math.atan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360.0), 1)
        # Circular concentration: near 0 means headings are scattered, near 1
        # means the roofs share an orientation.
        res["aspect_concentration"] = round(float(np.hypot(
            np.sin(rad).mean(), np.cos(rad).mean())), 3)
        bins = {"N": 0, "NE": 0, "E": 0, "SE": 0, "S": 0, "SW": 0, "W": 0, "NW": 0}
        names = list(bins)
        for v in a:
            bins[names[int(round((v % 360) / 45.0)) % 8]] += 1
        res["aspect_histogram"] = bins
    return res


def compare_with_assumption(stats: Dict[str, object], assumed_deg: float = 30.0) -> Dict[str, object]:
    """How the measured pitch compares with the literature figure the paper cites."""
    if not stats.get("n_planes"):
        return {}
    med = float(stats["slope_median_deg"])
    return {
        "assumed_deg": assumed_deg,
        "measured_median_deg": med,
        "difference_deg": round(med - assumed_deg, 2),
        "measured_is_lower": bool(med < assumed_deg),
        "flat_area_fraction_pct": stats["flat_area_fraction_pct"],
    }
