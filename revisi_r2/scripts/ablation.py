"""FASE 3A - R2-3: hierarchy-level ablation (L1, L2, L3 and their unions).

Every subset goes through the same chain as the full pipeline
(cross-level merge rules of segment_multiscale_advanced -> post_process_mask ->
dashboard export) at PAPER_CONFIG; see precompute_blocks.py / hierarchy.py.

Statistics vs L1+L2+L3, per spatial block (valid blocks only):
  * Wilcoxon signed-rank (two-sided, zero differences dropped = 'wilcox'),
  * Holm correction over the 12 tests of each class (6 subsets x {IoU, F1}),
  * rank-biserial r (matched pairs) = (W+ - W-) / (W+ + W-),
  * median of per-block differences (subset - full) with 95 % bootstrap CI.
Per-block IoU/F1 convention: a block with no GT and no prediction scores 1
(correct rejection). A sensitivity run restricted to GT-positive blocks is
also written.
Outputs: results/ablation.csv, results/ablation_stats.csv, figures/fig_ablation.png
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
from statsmodels.stats.multitest import multipletests

from common import BASE_SEED, CACHE, FIGURES, METRICS, RESULTS, ensure_dirs, vec_metrics
from precompute_blocks import LEVEL_SUBSETS, subset_name

FULL = "L1+L2+L3"
ORDER = [subset_name(u) for u in LEVEL_SUBSETS]
N_BOOT = 1000


def block_scores(c):
    m = vec_metrics(c)
    empty = (c[..., 0] + c[..., 1] + c[..., 2]) == 0
    return {"IoU": np.where(empty, 1.0, m["IoU"]), "F1": np.where(empty, 1.0, m["F1"])}


def site_mean_metrics(d):
    per_site = d.groupby("site")[["TP", "FP", "FN", "TN"]].sum().values
    m = vec_metrics(per_site)
    return {k: float(v.mean()) * 100 for k, v in m.items()}


def main():
    ensure_dirs()
    counts = pd.read_parquet(os.path.join(CACHE, "block_counts.parquet"))
    blocks = pd.read_parquet(os.path.join(CACHE, "blocks.parquet"))
    ab = counts[counts.kind == "ablation"].merge(
        blocks[["site", "cls", "block_id", "valid", "gt_px"]], on=["site", "cls", "block_id"])
    ab = ab[ab.valid]

    # ---------------- aggregate metrics with block-bootstrap CI
    rng = np.random.default_rng(BASE_SEED)
    agg = []
    for cls in ("roof", "pv"):
        d = ab[ab.cls == cls]
        sites = sorted(d.site.unique())
        # resample indices shared by all subsets (paired bootstrap)
        blk = {s: np.sort(d[(d.site == s)].block_id.unique()) for s in sites}
        arr = {cfg: {s: d[(d.config == cfg) & (d.site == s)].set_index("block_id")
                     .loc[blk[s], ["TP", "FP", "FN", "TN"]].values for s in sites} for cfg in ORDER}
        boots = {cfg: [] for cfg in ORDER}
        for _ in range(N_BOOT):
            idx = {s: rng.integers(0, len(blk[s]), len(blk[s])) for s in sites}
            for cfg in ORDER:
                sc = np.stack([arr[cfg][s][idx[s]].sum(0) for s in sites])
                m = vec_metrics(sc)
                boots[cfg].append([float(m[k].mean()) * 100 for k in METRICS])
        for cfg in ORDER:
            full = site_mean_metrics(d[d.config == cfg])
            b = np.array(boots[cfg])
            r = dict(cls=cls, levels=cfg, n_sites=len(sites), n_blocks=len(d[d.config == cfg]))
            for j, k in enumerate(METRICS):
                r[k] = full[k]
                r[k + "_ci_lo"] = np.percentile(b[:, j], 2.5)
                r[k + "_ci_hi"] = np.percentile(b[:, j], 97.5)
            agg.append(r)
    agg = pd.DataFrame(agg)
    agg.to_csv(os.path.join(RESULTS, "ablation.csv"), index=False)

    # ---------------- paired per-block tests
    stats = []
    for subset_label, filt in (("all_valid_blocks", None), ("gt_positive_blocks", "gt")):
        for cls in ("roof", "pv"):
            d = ab[ab.cls == cls]
            if filt == "gt":
                d = d[d.gt_px > 0]
            piv = {cfg: d[d.config == cfg].sort_values(["site", "block_id"]) for cfg in ORDER}
            fs = block_scores(piv[FULL][["TP", "FP", "FN", "TN"]].values)
            rows = []
            for cfg in ORDER:
                if cfg == FULL:
                    continue
                ss = block_scores(piv[cfg][["TP", "FP", "FN", "TN"]].values)
                for k in ("IoU", "F1"):
                    diff = ss[k] - fs[k]
                    nz = diff[np.abs(diff) > 1e-12]
                    if len(nz) == 0:
                        p, W, rrb = 1.0, np.nan, 0.0
                    else:
                        res = wilcoxon(ss[k], fs[k], zero_method="wilcox", alternative="two-sided")
                        p, W = float(res.pvalue), float(res.statistic)
                        rk = rankdata(np.abs(nz))
                        wp, wm = rk[nz > 0].sum(), rk[nz < 0].sum()
                        rrb = (wp - wm) / (wp + wm)
                    bs = [np.median(diff[rng.integers(0, len(diff), len(diff))]) for _ in range(2000)]
                    rows.append(dict(block_set=subset_label, cls=cls, levels=cfg, metric=k,
                                     n_blocks=len(diff), n_nonzero=len(nz),
                                     mean_subset=ss[k].mean() * 100, mean_full=fs[k].mean() * 100,
                                     mean_diff_pp=diff.mean() * 100,
                                     median_diff_pp=np.median(diff) * 100,
                                     median_diff_ci_lo=np.percentile(bs, 2.5) * 100,
                                     median_diff_ci_hi=np.percentile(bs, 97.5) * 100,
                                     W=W, p_raw=p, rank_biserial_r=rrb))
            rej, padj, _, _ = multipletests([r["p_raw"] for r in rows], alpha=0.05, method="holm")
            for r, a, j in zip(rows, padj, rej):
                r["p_holm"] = a
                r["significant_holm_0.05"] = bool(j)
            stats += rows
    stats = pd.DataFrame(stats)
    stats.to_csv(os.path.join(RESULTS, "ablation_stats.csv"), index=False)

    figure(agg)
    pd.set_option("display.width", 250)
    print(agg[["cls", "levels"] + METRICS].round(2).to_string(index=False))
    print(stats[stats.block_set == "all_valid_blocks"][
        ["cls", "levels", "metric", "n_blocks", "n_nonzero", "mean_diff_pp", "median_diff_pp",
         "rank_biserial_r", "p_holm", "significant_holm_0.05"]].round(4).to_string(index=False))


def figure(agg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from montecarlo import INK, INK2, PV_C, ROOF_C, _style
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, k in zip(axes, ("IoU", "F1")):
        x = np.arange(len(ORDER))
        for j, (cls, col, lab) in enumerate((("roof", ROOF_C, "Roof (6 sites)"),
                                             ("pv", PV_C, "PV (3 sites)"))):
            d = agg[agg.cls == cls].set_index("levels").loc[ORDER]
            xx = x + (j - 0.5) * 0.25
            ax.errorbar(xx, d[k], yerr=[d[k] - d[k + "_ci_lo"], d[k + "_ci_hi"] - d[k]],
                        fmt="o", ms=4.5, color=col, ecolor=col, elinewidth=1.2, capsize=2,
                        label=lab, markeredgecolor="white", markeredgewidth=0.6)
        ax.axvspan(len(ORDER) - 1.5, len(ORDER) - 0.5, color="#f1f0ec", zorder=0)
        ax.set_xticks(x, [o.replace("+", "+\n") for o in ORDER])
        ax.set_title(k + " (site mean, 95% block-bootstrap CI)", color=INK, fontsize=8.5, loc="left")
        _style(ax)
    axes[0].set_ylabel("%", color=INK)
    axes[1].legend(frameon=False, fontsize=7, loc="lower right")
    fig.text(0.99, 0.01, "L1 = full image, L2 = 512 px patches, L3 = 256 px patches; shaded = full hierarchy",
             ha="right", fontsize=6.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(FIGURES, "fig_ablation.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
