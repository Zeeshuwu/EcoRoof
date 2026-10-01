"""Check that offline re-assembly from cached instances reproduces the live
pipeline masks (cache/<KEY>_merged.tif, <KEY>_mask.tif) and output_inference."""
import torch  # noqa: F401
import json
import os

import numpy as np
import pandas as pd

from common import CACHE, PRED_DIR, RESULTS, SITES, load_mask
from hierarchy import Instances, PAPER_PARAMS, level_masks, load_meta, merge_levels, postprocess

rows = []
for s in SITES:
    for c in ("roof", "pv"):
        key = "%s_%s" % (s, c)
        if not os.path.exists(os.path.join(CACHE, key + "_meta.json")):
            continue
        meta = load_meta(key)
        live = load_mask(os.path.join(CACHE, key + "_merged.tif"))
        live_post = load_mask(os.path.join(CACHE, key + "_mask.tif"))
        inst = Instances(os.path.join(CACHE, key + "_instances.npz"))
        l1, l2, l3 = level_masks(inst, live.shape, PAPER_PARAMS["tau"])
        rec = merge_levels(l1, l2, l3, (1, 2, 3), meta["level3_skipped"]) > 0
        rec_post = postprocess(rec.astype(np.uint8), PAPER_PARAMS["min_size"], PAPER_PARAMS["geo_filter"]) > 0
        oi = load_mask(os.path.join(PRED_DIR, key + "_mask.tif"))
        npx = live.size
        rows.append(dict(key=key, rerun_vs_output_inference_px=int((live_post != oi).sum()),
                         rerun_vs_output_inference_pct=100 * (live_post != oi).sum() / npx,
                         reassembly_merged_diff_px=int((rec != live).sum()),
                         reassembly_post_diff_px=int((rec_post != live_post).sum()),
                         reassembly_post_diff_pct=100 * (rec_post != live_post).sum() / npx,
                         **{"t_" + k: v for k, v in meta["timing"].items()},
                         peak_vram_gb=meta["peak_vram_gb"], n_instances=meta["n_instances_cached"]))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULTS, "determinism.csv"), index=False)
print(df.to_string(index=False))
