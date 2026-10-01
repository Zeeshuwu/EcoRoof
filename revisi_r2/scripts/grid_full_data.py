"""Site-mean metrics of every calibration-grid setting on all valid blocks
(descriptive, no split). Output: results/grid_full_data.csv"""
import os

import numpy as np
import pandas as pd

from common import METRICS, RESULTS
from hierarchy import grid
from montecarlo import GRID_IDS, PAPER_ID, load, site_mean

_, _, data = load()
rows = []
for cls, sites in data.items():
    sc = np.stack([s["C"].sum(1) for s in sites])  # (site, cfg, 4)
    m = site_mean(sc)
    for j, (cid, p) in enumerate(zip(GRID_IDS, grid())):
        rows.append(dict(cls=cls, config=cid, **p, is_paper_config=cid == PAPER_ID,
                         **{k: float(m[k][j]) * 100 for k in METRICS}))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULTS, "grid_full_data.csv"), index=False)
for cls in ("roof", "pv"):
    d = df[df.cls == cls].sort_values("F1", ascending=False)
    print(d.head(4)[["cls", "config", "IoU", "Precision", "Recall", "F1"]].round(2).to_string(index=False))
    print(d[d.config.isin([PAPER_ID, "t0.50_m50_g0", "t0.50_m300_g1", "t0.35_m50_g1"])][["config", "IoU", "Precision", "Recall", "F1"]].round(2).to_string(index=False))
