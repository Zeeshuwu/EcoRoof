"""FASE 0 - reconcile the manuscript metrics with the outputs in the repo.

Candidates scored (same evaluation logic as evaluate_inference.py):
  A. output_inference/*_mask.tif          (post-processed mask, current batch run)
  B. output_inference/*_mask_prepost.tif  (before post_process_mask)
  C. input/01 DATA/Results/*_REVISED/*.shp  (dashboard exports, 3 PV sites only)
  D. input/01 DATA/Results/{JAKARTA,JOGJA,SURABAYA} original exports
     (stored in local metre coordinates, 0..200 m; re-georeferenced to the
      image grid, both y orientations tried)

Two aggregations are reported for each: pixels pooled over sites, and the
unweighted mean of per-site percentages.

Outputs: results/phase0_reproduction.csv, results/per_site_metrics.csv,
         results/hardware.txt
"""
# torch must be imported before rasterio/GDAL on Windows (c10.dll init error otherwise)
import torch  # noqa: F401
import os
import platform
import subprocess
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.features import rasterize

from common import (CITY, METRICS, PUBLISHED, PRED_DIR, REPO, RESULTS, R2,
                    confusion, ensure_dirs, image_path, load_gt, load_mask,
                    metrics_from_counts, pairs)

RES_DIR = os.path.join(REPO, "input", "01 DATA", "Results")
REVISED = {
    ("JKT_PV", "roof"): "JAKARTA_REVISED/roof_mask_JKT/roof_mask.shp",
    ("JKT_PV", "pv"): "JAKARTA_REVISED/solar_panel_mask_JKT (2)/solar_panel_mask.shp",
    ("JOG_PV", "roof"): "JOGJA_REVISED/roof_mask_JoG/roof_mask.shp",
    ("JOG_PV", "pv"): "JOGJA_REVISED/solar_panel_mask_JOG/solar_panel_mask.shp",
    ("SBY_PV", "roof"): "Surabaya_Revised/roof_mask_SBY/roof_mask.shp",
    ("SBY_PV", "pv"): "Surabaya_Revised/solar_panel_mask_SBY/solar_panel_mask.shp",
}
ORIGINAL = {  # zip, member stem
    ("JKT_PV", "roof"): ("JAKARTA/roof_mask_JKT.zip", None),
    ("JKT_PV", "pv"): (None, "JAKARTA/solar_panel_mask_JKT/solar_panel_mask.shp"),
    ("JOG_PV", "roof"): ("JOGJA/roof_mask_Jogja.zip", None),
    ("JOG_PV", "pv"): ("JOGJA/solar_panel_mask_Jogja.zip", None),
    ("SBY_PV", "roof"): ("SURABAYA/roof_mask_Surabaya.zip", None),
    ("SBY_PV", "pv"): ("SURABAYA/solar_panel_mask_Surabaya.zip", None),
}


def shp_to_mask(gdf, site, local=False, flip=False):
    with rasterio.open(image_path(site)) as ref:
        h, w, tr, crs = ref.height, ref.width, ref.transform, ref.crs
    if local:
        # 0..200 m local frame -> pixel grid at the image GSD
        rx, ry = abs(tr.a), abs(tr.e)
        tr = (Affine(rx, 0, 0, 0, ry, 0) if not flip
              else Affine(rx, 0, 0, 0, -ry, h * ry))
    elif str(gdf.crs) != str(crs):
        gdf = gdf.to_crs(crs)
    geoms = [g for g in gdf.geometry if g is not None and not g.is_empty]
    if not geoms:
        return np.zeros((h, w), bool)
    return rasterize([(g, 1) for g in geoms], out_shape=(h, w), transform=tr,
                     fill=0, all_touched=False, dtype="uint8") > 0


def read_original(site, cls):
    z, member = ORIGINAL[(site, cls)]
    if member:
        return gpd.read_file(os.path.join(RES_DIR, member))
    out = os.path.join(R2, "tmp", "orig", os.path.splitext(os.path.basename(z))[0])
    if not os.path.isdir(out):
        with zipfile.ZipFile(os.path.join(RES_DIR, z)) as zf:
            zf.extractall(out)
    shp = [f for f in os.listdir(out) if f.endswith(".shp")][0]
    return gpd.read_file(os.path.join(out, shp))


def score(source, site, cls, pred):
    gt = load_gt(site, cls)
    tp, fp, fn, tn = confusion(pred, gt)
    m = metrics_from_counts(tp, fp, fn, tn)
    return dict(source=source, site=site, city=CITY[site], cls=cls,
                TP=tp, FP=fp, FN=fn, TN=tn, **m)


def aggregate(rows, label):
    out = []
    df = pd.DataFrame(rows)
    for cls in ("roof", "pv"):
        sub = df[df.cls == cls]
        if sub.empty:
            continue
        pooled = metrics_from_counts(sub.TP.sum(), sub.FP.sum(), sub.FN.sum(), sub.TN.sum())
        mean = {k: sub[k].mean() for k in METRICS}
        for agg, m in (("pooled", pooled), ("site_mean", mean)):
            r = dict(source=label, cls=cls, aggregation=agg, n_sites=len(sub),
                     sites=";".join(sub.site))
            for k in METRICS:
                r[k] = 100 * m[k]
                r["paper_" + k] = PUBLISHED[cls][k.lower() if k != "IoU" else "iou"]
                r["delta_" + k] = r[k] - r["paper_" + k]
            r["max_abs_delta_pp"] = max(abs(r["delta_" + k]) for k in METRICS)
            out.append(r)
    return out


def hardware():
    import psutil
    import torch
    lines = ["platform: %s" % platform.platform(),
             "cpu: %s" % platform.processor(),
             "cpu_cores_logical: %d / physical: %d" % (psutil.cpu_count(), psutil.cpu_count(False)),
             "ram_gb: %.1f" % (psutil.virtual_memory().total / 1e9),
             "python: %s" % platform.python_version(),
             "torch: %s" % torch.__version__,
             "torch_cuda: %s" % torch.version.cuda,
             "cudnn: %s" % torch.backends.cudnn.version(),
             "cuda_available: %s" % torch.cuda.is_available()]
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        lines += ["gpu: %s" % p.name, "gpu_vram_gb: %.1f" % (p.total_memory / 1e9)]
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True).stdout
        lines.append("nvidia-smi: %s" % smi.strip())
    except Exception:
        pass
    try:
        cpu = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_Processor).Name"],
                             capture_output=True, text=True).stdout.strip()
        lines.append("cpu_name: %s" % cpu)
    except Exception:
        pass
    return "\n".join(lines) + "\n"


def main():
    ensure_dirs()
    per_site, agg = [], []

    for suffix, label in (("_mask.tif", "A_output_inference_mask"),
                          ("_mask_prepost.tif", "B_output_inference_prepost")):
        rows = [score(label, s, c, load_mask(os.path.join(PRED_DIR, "%s_%s%s" % (s, c, suffix))))
                for s, c in pairs()]
        per_site += rows
        agg += aggregate(rows, label)
        # same source restricted to the 3 PV sites (the only sites with dashboard exports)
        agg += aggregate([r for r in rows if r["site"].endswith("_PV")], label + "_PVsites_only")

    rows = [score("C_results_REVISED_shp", s, c,
                  shp_to_mask(gpd.read_file(os.path.join(RES_DIR, p)), s))
            for (s, c), p in REVISED.items()]
    per_site += rows
    agg += aggregate(rows, "C_results_REVISED_shp")

    for flip in (False, True):
        label = "D_results_original_local%s" % ("_yflip" if flip else "")
        rows = [score(label, s, c, shp_to_mask(read_original(s, c), s, local=True, flip=flip))
                for (s, c) in ORIGINAL]
        per_site += rows
        agg += aggregate(rows, label)

    ps = pd.DataFrame(per_site)
    for k in METRICS:
        ps[k] = 100 * ps[k]
    ps.to_csv(os.path.join(RESULTS, "phase0_all_sources_per_site.csv"), index=False)
    ps[ps.source == "A_output_inference_mask"].to_csv(
        os.path.join(RESULTS, "per_site_metrics.csv"), index=False)
    ag = pd.DataFrame(agg).sort_values("max_abs_delta_pp")
    ag.to_csv(os.path.join(RESULTS, "phase0_reproduction.csv"), index=False)

    pd.set_option("display.width", 220)
    print(ag[["source", "cls", "aggregation", "n_sites"] + METRICS + ["max_abs_delta_pp"]]
          .round(2).to_string(index=False))
    print()
    print(ps[["source", "site", "cls"] + METRICS].round(2).to_string(index=False))

    with open(os.path.join(RESULTS, "hardware.txt"), "w", encoding="utf-8") as f:
        f.write(hardware())


if __name__ == "__main__":
    main()
