"""FASE 0b - score PAPER_CONFIG (automatic, reproducible) and append to phase0 tables.

PAPER_CONFIG = output_inference/*_mask.tif (batch SAM3, dashboard prompts,
AdvancedConfig defaults, post_process_mask) + dashboard shapefile export
(common.export_roundtrip). No manual editing.
"""
import torch  # noqa: F401  (import before rasterio on Windows)
import os

import pandas as pd

from common import (CITY, METRICS, PRED_DIR, RESULTS, confusion, export_roundtrip,
                    load_gt, load_mask, metrics_from_counts, pairs, ref_profile)
from phase0_reproduce import aggregate

rows = []
for s, c in pairs():
    pred = export_roundtrip(load_mask(os.path.join(PRED_DIR, "%s_%s_mask.tif" % (s, c))),
                            ref_profile(s)["transform"])
    tp, fp, fn, tn = confusion(pred, load_gt(s, c))
    rows.append(dict(source="E_PAPER_CONFIG_auto", site=s, city=CITY[s], cls=c,
                     TP=tp, FP=fp, FN=fn, TN=tn, **metrics_from_counts(tp, fp, fn, tn)))
agg = aggregate(rows, "E_PAPER_CONFIG_auto")
agg += aggregate([r for r in rows if r["site"].endswith("_PV")], "E_PAPER_CONFIG_auto_PVsites_only")

ps = pd.DataFrame(rows)
for k in METRICS:
    ps[k] = 100 * ps[k]
allp = pd.read_csv(os.path.join(RESULTS, "phase0_all_sources_per_site.csv"))
allp = pd.concat([allp[allp.source != "E_PAPER_CONFIG_auto"], ps])
allp.to_csv(os.path.join(RESULTS, "phase0_all_sources_per_site.csv"), index=False)
ps.to_csv(os.path.join(RESULTS, "per_site_metrics.csv"), index=False)

rep = pd.read_csv(os.path.join(RESULTS, "phase0_reproduction.csv"))
rep = rep[~rep.source.str.startswith("E_PAPER_CONFIG")]
rep = pd.concat([rep, pd.DataFrame(agg)]).sort_values("max_abs_delta_pp")
rep.to_csv(os.path.join(RESULTS, "phase0_reproduction.csv"), index=False)
print(pd.DataFrame(agg)[["source", "cls", "aggregation"] + METRICS + ["max_abs_delta_pp"]].round(2).to_string(index=False))
print(ps[["site", "cls"] + METRICS].round(2).to_string(index=False))
