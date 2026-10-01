#!/usr/bin/env python
"""
Measure roof pitch and aspect from the LOD photogrammetric DSMs (R3-3 follow-up).

    python run_roof_geometry.py
    python run_roof_geometry.py --lod-dir "C:/path/to/data/lod"

Fits a plane to the DSM inside each digitized roof-plane polygon, then reports
the pitch and aspect distribution and the energy bias those imply against the
optimum-tilt assumption the manuscript's irradiation input carries.

IMPORTANT: these DSMs cover different neighbourhoods from the six study sites
(about 1.2 km away in Surabaya, 6 km in Jakarta). They cannot correct the energy
figures in Section IV.B. What they do is replace the assumed 30 deg roof pitch
with a measurement on real Indonesian roofs in the same two cities.
"""

import argparse
import json
import os
import sys

import numpy as np

try:
    import pandas as pd
    import rasterio
    from rasterio.warp import transform_bounds
except ImportError as exc:
    print("[ERROR] needs pandas + rasterio: %s" % exc)
    sys.exit(1)

from solar_calculator import dem_slope as ds
from solar_calculator.roof_geometry import (
    compare_with_assumption, measure_roof_planes, pitch_distribution,
)

DEFAULT_LOD = os.path.join("C:/", "Users", "GeoAIT_AMD_6", "Desktop",
                           "sam-interactive-github", "data", "lod")

CITIES = [
    ("Surabaya", os.path.join("lod - surabaya", "DSM.tif"),
     os.path.join("lod - surabaya", "Roof Structure.shp")),
    ("Jakarta", os.path.join("lod - dki", "DSM_48S_Clipped.tif"),
     os.path.join("lod - dki", "1_RS_DKI.shp")),
]


def orientation_bias(df, lat):
    """Area-weighted energy loss implied by the measured planes."""
    d = df[df["accepted"] & np.isfinite(df["slope_deg"])]
    ref = ds.poa_annual(lat, ds.optimum_tilt(lat), ds.equator_facing_azimuth(lat))
    # Quantize before evaluating: the transposition integral is the expensive
    # part and hundreds of planes share very similar geometry.
    cache, loss = {}, []
    for t, a in zip(d["slope_deg"], d["aspect_deg"].fillna(0.0)):
        key = (round(t / 2) * 2, round((a if np.isfinite(a) else 0.0) / 15) * 15)
        if key not in cache:
            cache[key] = 100.0 * (1 - ds.poa_annual(lat, key[0], key[1]) / ref)
        loss.append(cache[key])
    d = d.assign(loss_pct=loss)
    worst_uniform = 100.0 * (1 - ds.poa_annual(lat, 30.0, 180.0) / ref)
    aw = float(np.average(d["loss_pct"], weights=d["area_m2"]))
    return d, {
        "latitude": round(lat, 3),
        "area_weighted_loss_pct": round(aw, 2),
        "plane_mean_loss_pct": round(float(d["loss_pct"].mean()), 2),
        "worst_plane_loss_pct": round(float(d["loss_pct"].max()), 2),
        "best_plane_loss_pct": round(float(d["loss_pct"].min()), 2),
        "uniform_30deg_south_loss_pct": round(worst_uniform, 2),
        "aggregate_smaller_than_worst_case_by": round(worst_uniform / aw, 2) if aw > 0 else None,
    }


def main():
    ap = argparse.ArgumentParser(description="Measure roof pitch/aspect from LOD DSMs.")
    ap.add_argument("--lod-dir", default=DEFAULT_LOD)
    ap.add_argument("--out-dir", default="output_inference")
    ap.add_argument("--edge-buffer", type=float, default=0.5)
    ap.add_argument("--max-rmse", type=float, default=0.35)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    results = {}

    for city, dsm_rel, rs_rel in CITIES:
        dsm = os.path.join(args.lod_dir, dsm_rel)
        rs = os.path.join(args.lod_dir, rs_rel)
        if not (os.path.exists(dsm) and os.path.exists(rs)):
            print("[skip] %s: data not found under %s" % (city, args.lod_dir))
            continue

        with rasterio.open(dsm) as d:
            res = (abs(d.res[0]) + abs(d.res[1])) / 2
            w, s, e, n = transform_bounds(d.crs, "EPSG:4326", *d.bounds)
            lat = (s + n) / 2.0

        print("\n%s  (DSM %.3f m, latitude %.2f)" % (city, res, lat))
        print("-" * 68)
        gdf = measure_roof_planes(dsm, rs, edge_buffer_m=args.edge_buffer,
                                  max_rmse_m=args.max_rmse)
        stats = pitch_distribution(gdf)
        if not stats.get("n_planes"):
            print("  no planes could be fitted")
            continue
        vs = compare_with_assumption(stats)
        df, bias = orientation_bias(gdf, lat)

        print("  planes fitted %d, rejected %d as non-planar" % (stats["n_planes"], stats["n_rejected"]))
        print("  roof area %.0f m2" % stats["total_area_m2"])
        print("  pitch  median %.1f  area-weighted %.1f  p10 %.1f  p90 %.1f  max %.1f deg" % (
            stats["slope_median_deg"], stats["slope_area_weighted_deg"],
            stats["slope_p10_deg"], stats["slope_p90_deg"], stats["slope_max_deg"]))
        print("  flat (<5 deg): %.1f %% of planes, %.1f %% of area" % (
            stats["flat_fraction_pct"], stats["flat_area_fraction_pct"]))
        print("  aspect histogram : %s" % stats.get("aspect_histogram"))
        print("  aspect concentration %.3f (0 scattered, 1 aligned)" % stats.get("aspect_concentration", 0))
        print("  vs the assumed 30 deg: measured median %.1f (%+.1f deg)" % (
            vs["measured_median_deg"], vs["difference_deg"]))
        print()
        print("  ENERGY BIAS against the optimum-tilt assumption")
        print("    area-weighted mean loss    %5.1f %%" % bias["area_weighted_loss_pct"])
        print("    individual planes          %5.1f to %.1f %%" % (
            bias["best_plane_loss_pct"], bias["worst_plane_loss_pct"]))
        print("    if every roof were 30 deg facing south: %.1f %%" % bias["uniform_30deg_south_loss_pct"])
        print("    -> aggregate bias is %.1fx smaller than that worst case"
              % bias["aggregate_smaller_than_worst_case_by"])

        gdf.drop(columns="geometry").to_csv(
            os.path.join(args.out_dir, "roof_geometry_%s.csv" % city.lower()), index=False)
        results[city] = {"dsm_resolution_m": round(res, 3), "stats": stats,
                         "vs_assumption": vs, "orientation_bias": bias}

    if results:
        out = os.path.join(args.out_dir, "roof_geometry.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print("\nWritten: %s" % out)
    print()
    print("REMINDER: these DSMs do not overlap the six study sites. Use them to")
    print("characterise Indonesian roof geometry, not to correct Section IV.B.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
