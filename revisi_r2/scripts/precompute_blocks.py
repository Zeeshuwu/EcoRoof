"""Per-block confusion counts for every calibration-grid point and every
hierarchy-level ablation, computed offline from the instance cache.

Output: cache/block_counts.parquet with one row per
(site, cls, kind, config, block_id): TP, FP, FN, TN
  kind = "grid"      config = param_id (42 combinations, levels L1+L2+L3)
  kind = "ablation"  config = level subset (L1, L2, ..., L1+L2+L3) at PAPER_PARAMS
plus cache/blocks.parquet (block geometry and validity).

Every prediction goes through the same chain as PAPER_CONFIG:
levels -> cross-level merge -> post_process_mask -> dashboard export round-trip.
Resumable per (site, cls).
"""
import torch  # noqa: F401
import itertools
import os
import time

import numpy as np
import pandas as pd

from common import (BLOCK_M, CACHE, CITY, block_counts, block_grid, ensure_dirs,
                    export_roundtrip, load_gt, pairs, ref_profile)
from hierarchy import (Instances, PAPER_PARAMS, grid, level_masks, load_meta,
                       merge_levels, param_id, postprocess)

LEVEL_SUBSETS = [(1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]


def subset_name(u):
    return "+".join("L%d" % l for l in u)


def one(site, cls):
    out = os.path.join(CACHE, "blockcounts_%s_%s.parquet" % (site, cls))
    if os.path.exists(out):
        return pd.read_parquet(out)
    t0 = time.time()
    key = "%s_%s" % (site, cls)
    meta = load_meta(key)
    prof = ref_profile(site)
    shape = (prof["height"], prof["width"])
    inst = Instances(os.path.join(CACHE, "%s_instances.npz" % key))
    gt = load_gt(site, cls)
    labels, nr, nc = block_grid(shape, prof["res"])
    nb = nr * nc
    skipped = meta["level3_skipped"]
    rows = []

    def add(kind, cfg, pred):
        c = block_counts(pred, gt, labels, nb)
        for b in range(nb):
            rows.append((site, cls, kind, cfg, b, *c[b]))

    by_tau = {}
    for p in grid():
        if p["tau"] not in by_tau:
            l1, l2, l3 = level_masks(inst, shape, p["tau"])
            by_tau[p["tau"]] = (l1, l2, l3, merge_levels(l1, l2, l3, (1, 2, 3), skipped))
        merged = by_tau[p["tau"]][3]
        post = postprocess(merged, p["min_size"], p["geo_filter"])
        add("grid", param_id(p), export_roundtrip(post, prof["transform"]))

    l1, l2, l3, _ = by_tau[PAPER_PARAMS["tau"]]
    for u in LEVEL_SUBSETS:
        merged = merge_levels(l1, l2, l3, u, skipped)
        post = postprocess(merged, PAPER_PARAMS["min_size"], PAPER_PARAMS["geo_filter"])
        add("ablation", subset_name(u), export_roundtrip(post, prof["transform"]))

    df = pd.DataFrame(rows, columns=["site", "cls", "kind", "config", "block_id",
                                     "TP", "FP", "FN", "TN"])
    df.to_parquet(out, index=False)
    print("  %s %s: %d configs, %.0f s" % (site, cls, df.config.nunique(), time.time() - t0),
          flush=True)
    return df


def blocks_table(counts):
    """Block geometry + validity (valid = GT or ANY prediction touches it)."""
    rows = []
    for (site, cls), d in counts.groupby(["site", "cls"]):
        prof = ref_profile(site)
        labels, nr, nc = block_grid((prof["height"], prof["width"]), prof["res"])
        g = d.groupby("block_id")
        gt_px = g.apply(lambda x: int(x.TP.iloc[0] + x.FN.iloc[0]), include_groups=False)
        any_pred = g.apply(lambda x: int((x.TP + x.FP).max()), include_groups=False)
        npx = np.bincount(labels.ravel(), minlength=nr * nc)
        for b in range(nr * nc):
            rows.append(dict(site=site, city=CITY[site], cls=cls, block_id=b,
                             row=b // nc, col=b % nc, n_px=int(npx[b]),
                             gt_px=int(gt_px[b]), max_pred_px=int(any_pred[b]),
                             valid=bool(gt_px[b] > 0 or any_pred[b] > 0)))
    return pd.DataFrame(rows)


def main():
    ensure_dirs()
    parts = [one(s, c) for s, c in pairs()]
    counts = pd.concat(parts, ignore_index=True)
    counts.to_parquet(os.path.join(CACHE, "block_counts.parquet"), index=False)
    blocks = blocks_table(counts)
    blocks.to_parquet(os.path.join(CACHE, "blocks.parquet"), index=False)
    s = blocks.groupby(["cls", "site"]).agg(n_blocks=("valid", "size"),
                                            n_valid=("valid", "sum"))
    s["n_dropped_empty"] = s.n_blocks - s.n_valid
    print("block size %.0f m" % BLOCK_M)
    print(s)
    s.to_csv(os.path.join(os.path.dirname(CACHE), "results", "block_inventory.csv"))


if __name__ == "__main__":
    main()
