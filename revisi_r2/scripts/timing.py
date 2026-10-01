"""FASE 1 - inference timing per site and per building (supports R1-5 too).

Sources: cache/<KEY>_meta.json (this re-run, per-level wall time, peak VRAM)
and output_inference/manifest.json (the original batch run).
Buildings = GT roof polygons of the site (ROOF class GT), PV objects = GT PV polygons.
Note: the re-run used processor threshold 0.35 (to cache extra instances), which
adds a small amount of mask handling; the original manifest time is the
PAPER_CONFIG reference.
"""
import torch  # noqa: F401
import json
import os

import geopandas as gpd
import pandas as pd

from common import GT_DIR, GT_MAP, PRED_DIR, RESULTS, SITES, CITY
from hierarchy import load_meta

manifest = json.load(open(os.path.join(PRED_DIR, "manifest.json"), encoding="utf-8"))
rows = []
for s in SITES:
    n_bld = len(gpd.read_file(os.path.join(GT_DIR, GT_MAP[s]["roof"] + ".shp")))
    for c in ("roof", "pv"):
        key = "%s_%s" % (s, c)
        meta = load_meta(key)
        t = meta["timing"]
        man = manifest.get(key, {})
        rows.append(dict(site=s, city=CITY[s], cls=c, image_px="%dx%d" % tuple(meta["shape"]),
                         n_gt_buildings=n_bld,
                         t_level1_s=t["level1_s"], t_level2_512_s=t["level2_s"],
                         t_level3_256_s=t["level3_s"], t_segment_total_s=t["segment_total_s"],
                         t_postprocess_s=t["postprocess_s"], peak_vram_gb=meta["peak_vram_gb"],
                         t_original_run_s=man.get("seconds_inference"),
                         t_vectorize_original_s=man.get("seconds_vectorize"),
                         s_per_building=t["segment_total_s"] / n_bld,
                         n_pred_objects_original=man.get("n_objects")))
df = pd.DataFrame(rows)
tot = df.groupby("site").agg(t_both_classes_s=("t_segment_total_s", "sum"),
                             n_gt_buildings=("n_gt_buildings", "first")).reset_index()
tot["s_per_building_both_classes"] = tot.t_both_classes_s / tot.n_gt_buildings
tot["min_per_ha_both_classes"] = tot.t_both_classes_s / 60 / 4.0  # 200 x 200 m = 4 ha
df = df.merge(tot[["site", "t_both_classes_s", "s_per_building_both_classes",
                   "min_per_ha_both_classes"]], on="site")
df.to_csv(os.path.join(RESULTS, "timing.csv"), index=False)
print(df.round(2).to_string(index=False))
