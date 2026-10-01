"""FASE 3B - R2-3: classical bottom-up hierarchical merging baseline (RGB).

SLIC superpixels (Lab, n_segments=2500, compactness=20) -> region adjacency
graph on mean Lab colour -> skimage.graph.merge_hierarchical (agglomerative
merging of the most similar adjacent regions until the colour distance exceeds
a cut threshold). Several cuts of the hierarchy are evaluated
(DELTA_E_CUTS), i.e. several levels of the dendrogram.

Because the method is unsupervised, segments are labelled in two ways:
  oracle      : segment := class if > 50 % of its pixels are GT class.
                Upper bound on what the hierarchy's boundaries allow.
  calibrated  : a depth-3 decision tree on segment colour/size features
                (mean L*, a*, b*, std L*, excess-green, log area; pixel-weighted),
                fitted on segments whose centroid lies in CALIBRATION blocks of
                MC split seed 42, applied to the whole image, scored on TEST
                blocks only. The hierarchy cut is also chosen on calibration.
SAM3 PAPER_CONFIG and SAM3 calibrated (MC iteration 0) are scored on the same
test blocks for a like-for-like comparison.

Outputs: results/baseline_hierarchical.csv, results/baseline_slic_rules.txt,
         figures/fig_baseline_visual.png
"""
import torch  # noqa: F401
import os
import time

import numpy as np
import pandas as pd
import rasterio
from skimage import graph
from skimage.color import rgb2lab
from skimage.segmentation import slic
from sklearn.tree import DecisionTreeClassifier, export_text

from common import (BASE_SEED, CACHE, CITY, FIGURES, METRICS, RESULTS, block_counts,
                    block_grid, ensure_dirs, image_path, load_gt, pairs, ref_profile,
                    vec_metrics)
from montecarlo import GRID_IDS, PAPER_ID, mc_split

N_SEG = 2500
COMPACTNESS = 20
DELTA_E_CUTS = [5, 10, 15, 20, 30]
FEATS = ["L", "a", "b", "stdL", "ExG", "log_area_m2"]


def read_rgb(site):
    with rasterio.open(image_path(site)) as r:
        return r.read([1, 2, 3]).transpose(1, 2, 0)


def _weight(g, src, dst, n):
    return {"weight": np.linalg.norm(g.nodes[dst]["mean color"] - g.nodes[n]["mean color"])}


def _merge(g, src, dst):
    g.nodes[dst]["total color"] += g.nodes[src]["total color"]
    g.nodes[dst]["pixel count"] += g.nodes[src]["pixel count"]
    g.nodes[dst]["mean color"] = g.nodes[dst]["total color"] / g.nodes[dst]["pixel count"]


def segmentations(site):
    """{cut: label image}; cached as npz."""
    path = os.path.join(CACHE, "slic_%s.npz" % site)
    if os.path.exists(path):
        z = np.load(path)
        return {int(k[1:]): z[k] for k in z.files}, 0.0
    t0 = time.time()
    lab = rgb2lab(read_rgb(site))
    sp = slic(lab, n_segments=N_SEG, compactness=COMPACTNESS, convert2lab=False,
              start_label=1, channel_axis=-1)
    out = {}
    for cut in DELTA_E_CUTS:
        rag = graph.rag_mean_color(lab, sp, mode="distance")
        out[cut] = graph.merge_hierarchical(sp, rag, thresh=cut, rag_copy=False,
                                            in_place_merge=True, merge_func=_merge,
                                            weight_func=_weight).astype(np.int32)
    np.savez_compressed(path, **{"t%d" % k: v for k, v in out.items()})
    return out, time.time() - t0


def segment_features(seg, lab, rgb, res):
    ids, inv = np.unique(seg, return_inverse=True)
    inv = inv.reshape(seg.shape)
    n = np.bincount(inv.ravel()).astype(float)
    f = {}
    for j, name in enumerate(("L", "a", "b")):
        s = np.bincount(inv.ravel(), weights=lab[..., j].ravel())
        f[name] = s / n
    s2 = np.bincount(inv.ravel(), weights=(lab[..., 0] ** 2).ravel())
    f["stdL"] = np.sqrt(np.maximum(s2 / n - f["L"] ** 2, 0))
    rgbf = rgb.astype(float) / 255.0
    exg = 2 * rgbf[..., 1] - rgbf[..., 0] - rgbf[..., 2]
    f["ExG"] = np.bincount(inv.ravel(), weights=exg.ravel()) / n
    f["log_area_m2"] = np.log10(n * res * res)
    ys, xs = np.indices(seg.shape)
    cy = np.bincount(inv.ravel(), weights=ys.ravel()) / n
    cx = np.bincount(inv.ravel(), weights=xs.ravel()) / n
    return inv, n, np.column_stack([f[k] for k in FEATS]), cy, cx


def site_mean(sc):
    m = vec_metrics(np.stack(sc))
    return {k: float(v.mean()) * 100 for k, v in m.items()}


def main():
    ensure_dirs()
    blocks = pd.read_parquet(os.path.join(CACHE, "blocks.parquet"))
    counts = pd.read_parquet(os.path.join(CACHE, "block_counts.parquet"))
    mc = pd.read_csv(os.path.join(RESULTS, "mc_iterations.csv"))
    seed = BASE_SEED
    rows, rules = [], []
    timing = {}

    per = {}  # (site, cls) -> dict
    for site, cls in pairs():
        segs, t = segmentations(site)
        timing[site] = timing.get(site) or t
        rgb = read_rgb(site)
        lab = rgb2lab(rgb)
        prof = ref_profile(site)
        labels, nr, nc = block_grid(rgb.shape[:2], prof["res"])
        gt = load_gt(site, cls)
        per[(site, cls)] = dict(segs=segs, lab=lab, rgb=rgb, res=prof["res"], labels=labels,
                                nb=nr * nc, gt=gt, nc=nc)

    for cls in ("roof", "pv"):
        sites = sorted(s for s, c in pairs() if c == cls)
        valid = {s: blocks[(blocks.cls == cls) & (blocks.site == s) & blocks.valid].block_id.values
                 for s in sites}
        split = mc_split([len(valid[s]) for s in sites], seed, cls)
        cal_b = {s: valid[s][ca] for s, (ca, te) in zip(sites, split)}
        test_b = {s: valid[s][te] for s, (ca, te) in zip(sites, split)}

        def score(pred_by_site, which):
            sc = []
            for s in sites:
                P = per[(s, cls)]
                c = block_counts(pred_by_site[s], P["gt"], P["labels"], P["nb"])
                sel = {"test": test_b[s], "cal": cal_b[s], "all_valid": valid[s]}[which]
                sc.append(c[sel].sum(0))
            return site_mean(sc)

        # --- oracle per hierarchy cut + superpixel level (no merging)
        for cut in DELTA_E_CUTS:
            preds, nseg = {}, []
            for s in sites:
                P = per[(s, cls)]
                seg = P["segs"][cut]
                inv, n, _, _, _ = segment_features(seg, P["lab"], P["rgb"], P["res"])
                frac = np.bincount(inv.ravel(), weights=P["gt"].ravel()) / n
                preds[s] = (frac > 0.5)[inv]
                nseg.append(len(n))
            for which in ("test", "all_valid"):
                rows.append(dict(method="SLIC-RAG oracle", cls=cls, cut_deltaE=cut, eval=which,
                                 mean_n_segments=np.mean(nseg), **score(preds, which)))

        # --- calibrated: pick cut and tree on calibration blocks only
        best = None
        for cut in DELTA_E_CUTS:
            X, y, w, cache = [], [], [], {}
            for s in sites:
                P = per[(s, cls)]
                inv, n, F, cy, cx = segment_features(P["segs"][cut], P["lab"], P["rgb"], P["res"])
                frac = np.bincount(inv.ravel(), weights=P["gt"].ravel()) / n
                blk = P["labels"][cy.astype(int), cx.astype(int)]
                m = np.isin(blk, cal_b[s])
                X.append(F[m]); y.append(frac[m] > 0.5); w.append(n[m])
                cache[s] = (inv, F)
            X, y, w = np.vstack(X), np.concatenate(y), np.concatenate(w)
            tree = DecisionTreeClassifier(max_depth=3, class_weight="balanced",
                                          random_state=BASE_SEED).fit(X, y, sample_weight=w)
            preds = {s: tree.predict(cache[s][1]).astype(bool)[cache[s][0]] for s in sites}
            cal = score(preds, "cal")
            if best is None or cal["F1"] > best[0]["F1"]:
                best = (cal, cut, tree, preds)
        cal, cut, tree, preds = best
        rules.append("[%s] selected cut dE=%d (calibration F1 %.2f)\n%s"
                     % (cls, cut, cal["F1"], export_text(tree, feature_names=FEATS)))
        for which in ("test", "all_valid"):
            rows.append(dict(method="SLIC-RAG calibrated", cls=cls, cut_deltaE=cut, eval=which,
                             **score(preds, which)))
        per_cls_pred = preds

        # --- SAM3 on the same test blocks
        for label, cfg in (("SAM3 hierarchical (PAPER_CONFIG)", PAPER_ID),
                           ("SAM3 hierarchical (calibrated, MC seed 42)",
                            mc[(mc.cls == cls) & (mc.seed == seed)].selected.iloc[0])):
            for which in ("test", "all_valid"):
                sc = []
                for s in sites:
                    d = counts[(counts.site == s) & (counts.cls == cls) & (counts.kind == "grid")
                               & (counts.config == cfg)].set_index("block_id")
                    sel = test_b[s] if which == "test" else valid[s]
                    sc.append(d.loc[sel, ["TP", "FP", "FN", "TN"]].values.sum(0))
                rows.append(dict(method=label, cls=cls, cut_deltaE=np.nan, eval=which,
                                 config=cfg, **site_mean(sc)))
        if cls == "roof":
            roof_pred, roof_cut = per_cls_pred, cut

    df = pd.DataFrame(rows)
    for k in METRICS:
        df[k] = df[k].round(3)
    df["slic_rag_seconds_per_site"] = np.mean([v for v in timing.values() if v]) if any(timing.values()) else np.nan
    df.to_csv(os.path.join(RESULTS, "baseline_hierarchical.csv"), index=False)
    with open(os.path.join(RESULTS, "baseline_slic_rules.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(rules))
    pd.set_option("display.width", 220)
    print(df[["method", "cls", "cut_deltaE", "eval"] + METRICS].to_string(index=False))
    print("\n".join(rules))
    figure(per, roof_pred, roof_cut)


def figure(per, roof_pred, cut):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries
    from common import load_mask
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    sites = ["JOG_PV", "SBY_ROOF"]
    fig, axes = plt.subplots(len(sites), 4, figsize=(7.2, 3.8))
    for i, s in enumerate(sites):
        P = per[(s, "roof")]
        sam = load_mask(os.path.join(CACHE, "%s_roof_mask.tif" % s))
        seg = P["segs"][cut]
        slic_pred = roof_pred[s]
        panels = [("Orthomosaic", None), ("Ground truth", P["gt"]),
                  ("SAM3 hierarchical", sam), ("SLIC-RAG (calibrated)", slic_pred)]
        for j, (title, m) in enumerate(panels):
            ax = axes[i, j]
            img = P["rgb"].astype(float) / 255
            if m is not None:
                ov = img.copy()
                ov[m] = 0.45 * ov[m] + 0.55 * np.array([0.165, 0.47, 0.84])
                img = ov
            ax.imshow(img[::2, ::2], interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(title, fontsize=8, color="#0b0b0b")
            if j == 0:
                ax.set_ylabel(s.replace("_", " ") + "\n(%s)" % CITY[s], fontsize=7.5)
    fig.text(0.5, 0.005, "Blue = roof mask. SLIC-RAG: hierarchy cut ΔE = %d selected on calibration blocks, depth-3 colour/size rule." % cut,
             ha="center", fontsize=6.5, color="#52514e")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(FIGURES, "fig_baseline_visual.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
