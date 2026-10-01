"""FASE 1 - re-run SAM3 once per site x class and cache every hierarchy level.

    .\\venv\\Scripts\\python.exe revisi_r2\\scripts\\cache_levels.py            # all 12 runs
    .\\venv\\Scripts\\python.exe revisi_r2\\scripts\\cache_levels.py --site JKT_PV --classes roof

Same inputs/prompts/config as run_batch_inference.py (the run behind
output_inference/, i.e. PAPER_CONFIG), with two changes that do not alter the
PAPER_CONFIG output:
  * processor.confidence_threshold is lowered from 0.5 to 0.35 so that
    instances in (0.35, 0.5] are also cached for offline calibration, and
  * AdvancedConfig.confidence_thresholds are set to 0.5 for all levels, which
    restores exactly the "score > 0.5" filter the published run used
    (in that run the processor's own 0.5 cut dominated the 0.20/0.22/0.28
    config values).

Per run, into revisi_r2/cache/:
  <KEY>_L1.tif  <KEY>_L2.tif  <KEY>_L3.tif          raw level masks (pre-merge)
  <KEY>_L3val.tif                                     level 3 after validation vs L1
  <KEY>_merged.tif  <KEY>_mask.tif                    merged / post-processed
  <KEY>_instances.npz                                 every instance + score
  <KEY>_meta.json                                     stats, timing, determinism
Resumable: a run is skipped if its _meta.json exists (use --overwrite).
"""
import torch  # noqa: F401  must precede rasterio/GDAL on Windows
import argparse
import json
import os
import sys
import time

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import rasterio

from common import (CACHE, PRED_DIR, REPO, SITES, ensure_dirs, image_path, load_mask,
                    pairs)
from hierarchy import (COLLECT_THRESHOLD, Instances, PAPER_PARAMS, level_masks,
                       merge_levels, postprocess, save_instances)

os.chdir(REPO)  # sam3rooffix / run_batch_inference use repo-relative paths
import run_batch_inference as rbi  # noqa: E402
import sam3rooffix as sr  # noqa: E402
from sam3_advanced.config import AdvancedConfig  # noqa: E402
from sam3_advanced.inference import segment_multiscale_advanced  # noqa: E402
from sam3_advanced.postprocess import post_process_mask  # noqa: E402
from sam3_advanced.utils import clear_memory  # noqa: E402


def write_tif(path, arr, site):
    with rasterio.open(image_path(site)) as ref:
        prof = dict(driver="GTiff", height=ref.height, width=ref.width, count=1,
                    dtype="uint8", crs=ref.crs, transform=ref.transform, nodata=0,
                    compress="deflate")
    with rasterio.open(path, "w", **prof) as dst:
        dst.write((arr > 0).astype("uint8") * 255, 1)


def run(site, cls, processor, overwrite=False):
    key = "%s_%s" % (site, cls)
    meta_path = os.path.join(CACHE, "%s_meta.json" % key)
    if os.path.exists(meta_path) and not overwrite:
        print("[SKIP] %s" % key)
        return
    print("\n[RUN ] %s" % key, flush=True)

    image_pil, handler = sr.load_image(image_path(site))
    config = AdvancedConfig()
    config.confidence_thresholds = {"level1": PAPER_PARAMS["tau"],
                                    "level2": PAPER_PARAMS["tau"],
                                    "level3": PAPER_PARAMS["tau"]}
    p = rbi.PROMPT_SETS["dashboard"][cls]
    config.prompts_large, config.prompts_medium, config.prompts_small = (
        p["large"], p["medium"], p["small"])

    processor.confidence_threshold = COLLECT_THRESHOLD
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    processor.set_image(image_pil)
    final_mask, stats, levels = segment_multiscale_advanced(
        image_pil, processor, config, return_levels=True)
    torch.cuda.synchronize()
    t_seg = time.time() - t0
    t0 = time.time()
    mask = post_process_mask(final_mask, config, verbose=False)
    t_post = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    processor.confidence_threshold = 0.5

    shape = final_mask.shape
    write_tif(os.path.join(CACHE, "%s_L1.tif" % key), levels["level1_binary"], site)
    write_tif(os.path.join(CACHE, "%s_L2.tif" % key), levels["level2_merged"], site)
    l3 = levels["level3_merged"]
    write_tif(os.path.join(CACHE, "%s_L3.tif" % key),
              l3 if l3 is not None else np.zeros(shape, np.uint8), site)
    write_tif(os.path.join(CACHE, "%s_L3val.tif" % key), levels["level3_validated"], site)
    write_tif(os.path.join(CACHE, "%s_merged.tif" % key), final_mask, site)
    write_tif(os.path.join(CACHE, "%s_mask.tif" % key), mask, site)
    inst_path = os.path.join(CACHE, "%s_instances.npz" % key)
    save_instances(inst_path, levels["instances"])

    # --- determinism checks
    ref_mask = load_mask(os.path.join(PRED_DIR, "%s_mask.tif" % key))
    ref_pre = load_mask(os.path.join(PRED_DIR, "%s_mask_prepost.tif" % key))
    diff_mask = int(np.count_nonzero((mask > 0) != ref_mask))
    diff_pre = int(np.count_nonzero((final_mask > 0) != ref_pre))
    # offline re-assembly from cached instances must equal the live pipeline
    inst = Instances(inst_path)
    r1, r2, r3 = level_masks(inst, shape, PAPER_PARAMS["tau"])
    rec = merge_levels(r1, r2, r3, (1, 2, 3), levels["level3_skipped"])
    rec_post = postprocess(rec, PAPER_PARAMS["min_size"], PAPER_PARAMS["geo_filter"])
    diff_reasm = int(np.count_nonzero((rec > 0) != (final_mask > 0)))
    diff_reasm_post = int(np.count_nonzero((rec_post > 0) != (mask > 0)))

    npx = shape[0] * shape[1]
    lv = inst.level
    meta = {
        "key": key, "site": site, "cls": cls, "shape": list(shape),
        "stats": stats, "level3_skipped": levels["level3_skipped"],
        "timing": {**{k: round(v, 2) for k, v in levels["timing"].items()},
                   "segment_total_s": round(t_seg, 2), "postprocess_s": round(t_post, 2)},
        "peak_vram_gb": round(peak_gb, 2),
        "n_instances_cached": int(inst.n),
        "n_instances_by_level": {str(l): int((lv == l).sum()) for l in (1, 2, 3)},
        "n_instances_gt_0.5_by_level": {str(l): int(((lv == l) & (inst.score > 0.5)).sum())
                                         for l in (1, 2, 3)},
        "collect_threshold": COLLECT_THRESHOLD,
        "determinism": {
            "diff_px_vs_output_inference_mask": diff_mask,
            "diff_pct_vs_output_inference_mask": 100.0 * diff_mask / npx,
            "diff_px_vs_output_inference_prepost": diff_pre,
            "diff_pct_vs_output_inference_prepost": 100.0 * diff_pre / npx,
            "diff_px_offline_reassembly_merged": diff_reasm,
            "diff_px_offline_reassembly_postprocessed": diff_reasm_post,
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("      seg %.1f s | vs output_inference: %.4f %% px | reassembly diff %d px"
          % (t_seg, meta["determinism"]["diff_pct_vs_output_inference_mask"], diff_reasm_post),
          flush=True)
    clear_memory()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=None)
    ap.add_argument("--classes", default="roof,pv")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    ensure_dirs()
    classes = args.classes.split(",")
    # all 12 site x class runs (PV is also run on the ROOF sites: needed for
    # timing and to keep the cache complete even though they have no PV GT)
    todo = [(s, c) for s in SITES for c in classes if not args.site or s == args.site]
    processor = sr.load_model(os.path.join(REPO, "sam3.pt"))
    for s, c in todo:
        run(s, c, processor, args.overwrite)


if __name__ == "__main__":
    main()
