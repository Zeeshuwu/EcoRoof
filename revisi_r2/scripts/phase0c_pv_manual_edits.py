"""FASE 0c - how the *_REVISED PV shapefiles differ from the original dashboard export.

For every polygon of the original export (input/01 DATA/Results/{JAKARTA,JOGJA,SURABAYA},
local 0-200 m frame, y flipped) report whether it survives in the _REVISED file and
how much of it overlaps PV ground truth. Output: results/phase0_pv_revised_edits.csv
"""
import torch  # noqa: F401  (before GDAL on Windows)
import os

import geopandas as gpd
import pandas as pd
import rasterio
from affine import Affine
from rasterio.features import rasterize

from common import RESULTS, image_path, load_gt
from phase0_reproduce import RES_DIR, REVISED, read_original, shp_to_mask

rows = []
for s in ["JKT_PV", "JOG_PV", "SBY_PV"]:
    og = read_original(s, "pv")
    with rasterio.open(image_path(s)) as r:
        h, rx, ry = r.height, abs(r.transform.a), abs(r.transform.e)
    tr = Affine(rx, 0, 0, 0, -ry, h * ry)
    rv = shp_to_mask(gpd.read_file(os.path.join(RES_DIR, REVISED[(s, "pv")])), s)
    gt, gtr = load_gt(s, "pv"), load_gt(s, "roof")
    for i, g in enumerate(og.geometry):
        m = rasterize([(g, 1)], out_shape=rv.shape, transform=tr, fill=0, dtype="uint8") > 0
        n = int(m.sum())
        if n == 0:
            continue
        rows.append(dict(site=s, polygon=i, area_m2=round(g.area, 2),
                         kept_in_revised=bool((m & rv).sum() / n > 0.5),
                         frac_on_gt_pv=round((m & gt).sum() / n, 3),
                         frac_on_gt_roof=round((m & gtr).sum() / n, 3)))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULTS, "phase0_pv_revised_edits.csv"), index=False)
print(df.groupby(["site", "kept_in_revised"]).agg(n=("polygon", "size"),
      area_m2=("area_m2", "sum"), mean_frac_gt_pv=("frac_on_gt_pv", "mean")))
