"""FASE 2 - R2-2: Monte Carlo cross-validation, bootstrap, LOSO / LOCO.

Runs entirely on cache/block_counts.parquet (no SAM3 inference).

Protocol
  unit        : 25 x 25 m spatial block; blocks with neither GT nor any
                prediction (over all configs) are dropped (results/block_inventory.csv)
  calibration : grid of 42 post-processing settings (hierarchy.grid()); the
                setting that maximises calibration F1 is selected
  aggregation : per-site metrics from pooled block counts, then unweighted
                mean over sites (= manuscript convention); pooled also stored
  A  MC-CV    : N=100, seeds 42..141, 70/30 split stratified per site
  B  bootstrap: PAPER_CONFIG, 1000 resamples of blocks with replacement,
                stratified per site, percentile 95 % CI
  C  LOSO/LOCO: calibrate on the other sites / cities, test on the held-out one
"""
import os

import numpy as np
import pandas as pd

from common import (BASE_SEED, CACHE, CITY, FIGURES, METRICS, PUBLISHED, RESULTS,
                    ensure_dirs, vec_metrics)
from hierarchy import PAPER_PARAMS, grid, param_id

N_MC = 100
N_BOOT = 1000
TEST_FRAC = 0.30
PAPER_ID = param_id(PAPER_PARAMS)
GRID = grid()
GRID_IDS = [param_id(p) for p in GRID]
# tie-break: prefer settings closest to PAPER_CONFIG (stable, conservative)
_DIST = np.array([abs(p["tau"] - PAPER_PARAMS["tau"]) * 10
                  + abs(p["min_size"] - PAPER_PARAMS["min_size"]) / 100
                  + (p["geo_filter"] != PAPER_PARAMS["geo_filter"]) for p in GRID])
PUB = {c: {"IoU": PUBLISHED[c]["iou"], "Precision": PUBLISHED[c]["precision"],
           "Recall": PUBLISHED[c]["recall"], "F1": PUBLISHED[c]["f1"],
           "Accuracy": PUBLISHED[c]["accuracy"]} for c in PUBLISHED}


# ------------------------------------------------------------------ data
def load():
    counts = pd.read_parquet(os.path.join(CACHE, "block_counts.parquet"))
    blocks = pd.read_parquet(os.path.join(CACHE, "blocks.parquet"))
    data = {}
    for cls in ("roof", "pv"):
        sites = sorted(blocks[blocks.cls == cls].site.unique())
        g = counts[(counts.cls == cls) & (counts.kind == "grid")]
        per_site = []
        for s in sites:
            valid = blocks[(blocks.cls == cls) & (blocks.site == s) & blocks.valid].block_id.values
            d = g[g.site == s].set_index(["config", "block_id"])[["TP", "FP", "FN", "TN"]]
            arr = np.stack([d.loc[cid].loc[valid].values for cid in GRID_IDS])  # (cfg, blk, 4)
            per_site.append(dict(site=s, blocks=valid, C=arr.astype(np.int64)))
        data[cls] = per_site
    return counts, blocks, data


def site_mean(site_counts):
    """site_counts: (n_sites, ..., 4) -> dict of site-mean metrics (...)."""
    m = vec_metrics(site_counts)
    return {k: v.mean(axis=0) for k, v in m.items()}


def pooled(site_counts):
    return vec_metrics(site_counts.sum(axis=0))


def select(cal_counts):
    """cal_counts (n_sites, n_cfg, 4) -> index of best config by site-mean F1."""
    f1 = site_mean(cal_counts)["F1"]
    best = f1.max()
    cand = np.flatnonzero(np.isclose(f1, best, atol=1e-12))
    return int(cand[np.argmin(_DIST[cand])])


def row(prefix, m, idx=None):
    return {"%s%s" % (prefix, k): float(m[k] if idx is None else m[k][idx]) * 100 for k in METRICS}


# ------------------------------------------------------------------ A. MC-CV
def mc_split(n_blocks_per_site, seed, cls):
    """Positional (cal, test) indices per site; same stream for every consumer
    (MC loop, SLIC baseline), so 'MC split seed 42' is well defined."""
    rng = np.random.default_rng([seed, 0 if cls == "roof" else 1])
    out = []
    for nb in n_blocks_per_site:
        perm = rng.permutation(nb)
        nt = int(round(TEST_FRAC * nb))
        out.append((perm[nt:], perm[:nt]))
    return out


def monte_carlo(data):
    out = []
    pid = GRID_IDS.index(PAPER_ID)
    for it in range(N_MC):
        seed = BASE_SEED + it
        for cls, sites in data.items():
            cal, test = [], []
            n_cal = n_test = 0
            splits = mc_split([len(s["blocks"]) for s in sites], seed, cls)
            for s, (ca, te) in zip(sites, splits):
                cal.append(s["C"][:, ca].sum(1))
                test.append(s["C"][:, te].sum(1))
                n_cal += len(ca)
                n_test += len(te)
            cal, test = np.stack(cal), np.stack(test)  # (site, cfg, 4)
            b = select(cal)
            ms, mp = site_mean(test), pooled(test)
            r = dict(iteration=it, seed=seed, cls=cls, n_cal_blocks=n_cal, n_test_blocks=n_test,
                     selected=GRID_IDS[b], sel_tau=GRID[b]["tau"], sel_min_size=GRID[b]["min_size"],
                     sel_geo_filter=GRID[b]["geo_filter"],
                     cal_F1=float(site_mean(cal)["F1"][b]) * 100)
            r.update(row("test_", ms, b))
            r.update(row("test_pooled_", mp, b))
            r.update(row("paperconfig_test_", ms, pid))
            out.append(r)
    return pd.DataFrame(out)


# ------------------------------------------------------------------ B. bootstrap
def bootstrap(data, roof_subset=None):
    pid = GRID_IDS.index(PAPER_ID)
    rng = np.random.default_rng(BASE_SEED)
    reps, per_site = [], []
    groups = [(cls, cls, sites) for cls, sites in data.items()]
    groups.append(("roof", "roof_PVsites", [s for s in data["roof"] if s["site"].endswith("_PV")]))
    for b in range(N_BOOT):
        for cls, label, sites in groups:
            sc = []
            for s in sites:
                C = s["C"][pid]
                idx = rng.integers(0, len(C), len(C))
                sc.append(C[idx].sum(0))
                if label == cls:
                    m = vec_metrics(sc[-1])
                    per_site.append(dict(rep=b, cls=cls, site=s["site"],
                                         **{k: float(m[k]) * 100 for k in METRICS}))
            sc = np.stack(sc)
            reps.append(dict(rep=b, group=label, **row("", site_mean(sc)),
                             **row("pooled_", pooled(sc))))
    return pd.DataFrame(reps), pd.DataFrame(per_site)


def full_data(data):
    pid = GRID_IDS.index(PAPER_ID)
    res = {}
    for label, cls, filt in (("roof", "roof", None), ("pv", "pv", None),
                             ("roof_PVsites", "roof", "_PV")):
        sites = [s for s in data[cls] if filt is None or s["site"].endswith(filt)]
        sc = np.stack([s["C"][pid].sum(0) for s in sites])
        res[label] = dict(**row("", site_mean(sc)), **row("pooled_", pooled(sc)))
    return res


# ------------------------------------------------------------------ C. LOSO / LOCO
def leave_out(data, by):
    out = []
    pid = GRID_IDS.index(PAPER_ID)
    for cls, sites in data.items():
        key = (lambda s: s["site"]) if by == "site" else (lambda s: CITY[s["site"]])
        for g in sorted({key(s) for s in sites}):
            held = [s for s in sites if key(s) == g]
            train = [s for s in sites if key(s) != g]
            cal = np.stack([s["C"].sum(1) for s in train])
            test = np.stack([s["C"].sum(1) for s in held])
            b = select(cal)
            ms = site_mean(test)
            r = dict(cls=cls, held_out=g, held_sites=";".join(s["site"] for s in held),
                     calib_sites=";".join(s["site"] for s in train),
                     selected=GRID_IDS[b], sel_tau=GRID[b]["tau"],
                     sel_min_size=GRID[b]["min_size"], sel_geo_filter=GRID[b]["geo_filter"])
            r.update(row("test_", ms, b))
            r.update(row("paperconfig_test_", ms, pid))
            out.append(r)
    return pd.DataFrame(out)


# ------------------------------------------------------------------ per-block table
def per_block(counts, blocks):
    d = counts[(counts.kind == "grid") & (counts.config == PAPER_ID)].merge(
        blocks, on=["site", "cls", "block_id"])
    m = vec_metrics(d[["TP", "FP", "FN", "TN"]].values)
    out = d[["site", "city", "block_id", "row", "col", "cls", "valid", "gt_px",
             "TP", "FP", "FN", "TN"]].rename(columns={"cls": "class"}).copy()
    tp, fp, fn = d.TP.values, d.FP.values, d.FN.values
    out["IoU"] = np.where(tp + fp + fn > 0, m["IoU"], np.nan)
    out["P"] = np.where(tp + fp > 0, m["Precision"], np.nan)
    out["R"] = np.where(tp + fn > 0, m["Recall"], np.nan)
    out["F1"] = np.where(tp + fp + fn > 0, m["F1"], np.nan)
    out["config"] = PAPER_ID
    return out


# ------------------------------------------------------------------ summary
def summarize(mc, boot, loso, loco, full):
    rows = []
    for cls in ("roof", "pv"):
        m = mc[mc.cls == cls]
        bt = boot[boot.group == cls]
        lo = loso[loso.cls == cls]
        lc = loco[loco.cls == cls]
        for k in METRICS:
            t = m["test_" + k]
            p = m["paperconfig_test_" + k]
            rows.append({
                "class": cls, "metric": k,
                "paper_value": PUB[cls][k],
                "paperconfig_full": full[cls][k],
                "mc_tuned_mean": t.mean(), "mc_tuned_sd": t.std(ddof=1),
                "mc_tuned_ci_lo": t.quantile(0.025), "mc_tuned_ci_hi": t.quantile(0.975),
                "mc_paperconfig_mean": p.mean(), "mc_paperconfig_sd": p.std(ddof=1),
                "boot_mean": bt[k].mean(), "boot_sd": bt[k].std(ddof=1),
                "boot_ci_lo": bt[k].quantile(0.025), "boot_ci_hi": bt[k].quantile(0.975),
                "loso_mean": lo["test_" + k].mean(), "loso_sd": lo["test_" + k].std(ddof=1),
                "loso_n": len(lo),
                "loco_mean": lc["test_" + k].mean(), "loco_sd": lc["test_" + k].std(ddof=1),
                "loco_n": len(lc),
            })
    return pd.DataFrame(rows)


def summary_md(s, full, boot):
    L = ["# Summary table - R2-2 (all values in %)", "",
         "Site-mean aggregation (manuscript convention). Roof = 6 sites, PV = 3 sites.",
         "PAPER = value printed in the manuscript; PAPER_CONFIG = automatic reproducible "
         "pipeline on all valid blocks; MC = 100 x 70/30 block split, parameters "
         "re-selected on calibration blocks, metrics on test blocks; CI = percentile "
         "95 % interval; LOSO/LOCO = leave-one-site/city-out (mean ± SD over folds).", "",
         "| Class | Metric | PAPER | PAPER_CONFIG | MC mean ± SD | MC 95% CI | Bootstrap 95% CI | LOSO mean ± SD | LOCO mean ± SD |",
         "|---|---|---|---|---|---|---|---|---|"]
    for _, r in s.iterrows():
        L.append("| %s | %s | %.2f | %.2f | %.2f ± %.2f | [%.2f, %.2f] | [%.2f, %.2f] | %.2f ± %.2f | %.2f ± %.2f |" % (
            "Roof" if r["class"] == "roof" else "PV", r.metric, r.paper_value, r.paperconfig_full,
            r.mc_tuned_mean, r.mc_tuned_sd, r.mc_tuned_ci_lo, r.mc_tuned_ci_hi,
            r.boot_ci_lo, r.boot_ci_hi, r.loso_mean, r.loso_sd, r.loco_mean,
            0.0 if np.isnan(r.loco_sd) else r.loco_sd))
    b = boot[boot.group == "roof_PVsites"]
    L += ["", "Roof restricted to the 3 PV sites (the set the manuscript roof numbers trace to):", "",
          "| Metric | PAPER | PAPER_CONFIG | Bootstrap 95% CI |", "|---|---|---|---|"]
    for k in METRICS:
        L.append("| %s | %.2f | %.2f | [%.2f, %.2f] |" % (
            k, PUB["roof"][k], full["roof_PVsites"][k], b[k].quantile(0.025), b[k].quantile(0.975)))
    L += ["", "Notes: (i) PV has one site per city, so PV LOCO folds are identical to PV LOSO folds.",
          "(ii) Accuracy here is computed on valid blocks only (blocks with neither GT nor any "
          "prediction are excluded), so it is lower than whole-tile accuracy (PAPER_CONFIG whole tile: "
          "roof 89.49, PV 99.41; results/phase0_reproduction.csv). IoU/P/R/F1 are unaffected.",
          "(iii) Roof calibration selects the lowest cached threshold (tau = 0.35) in every MC "
          "iteration; lower values would need new SAM3 inference.",
          "(iv) In 2/100 MC iterations the SBY_PV test fold contains no PV ground truth "
          "(results/mc_per_site_paperconfig.csv); such a site scores 0 by convention."]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ figures
ROOF_C, PV_C = "#2a78d6", "#eb6834"
INK, INK2, GRIDC = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.yaxis.grid(True, color=GRIDC, linewidth=0.6)
    ax.set_axisbelow(True)


def figures(mc, boot_site, loso, full_ps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})

    # 1. MC distribution, roof vs PV
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    pos = np.arange(len(METRICS))
    for j, (cls, col) in enumerate((("roof", ROOF_C), ("pv", PV_C))):
        vals = [mc[mc.cls == cls]["test_" + k].values for k in METRICS]
        x = pos + (j - 0.5) * 0.36
        vp = ax.violinplot(vals, positions=x, widths=0.32, showextrema=False)
        for b in vp["bodies"]:
            b.set_facecolor(col)
            b.set_alpha(0.35)
            b.set_edgecolor(col)
        ax.boxplot(vals, positions=x, widths=0.09, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="white", edgecolor=col, linewidth=1),
                   medianprops=dict(color=col, linewidth=1.5),
                   whiskerprops=dict(color=col, linewidth=1), capprops=dict(color=col))
        ax.scatter(x, [PUB[cls][k] for k in METRICS], marker="D", s=22, color=INK,
                   zorder=5, edgecolor="white", linewidth=0.8)
    ax.set_xticks(pos, METRICS)
    ax.set_ylabel("Test-block metric (%)", color=INK)
    _style(ax)
    ax.legend(handles=[Line2D([], [], color=ROOF_C, lw=6, alpha=0.5, label="Roof (6 sites)"),
                       Line2D([], [], color=PV_C, lw=6, alpha=0.5, label="PV (3 sites)"),
                       Line2D([], [], marker="D", color=INK, lw=0, label="Value in manuscript")],
              frameon=False, loc="lower right", fontsize=7.5)
    ax.set_title("Monte Carlo cross-validation (N = 100, 70/30 spatial-block split)",
                 color=INK, fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig_mc_distribution.png"), dpi=300)
    plt.close(fig)

    # 2. per-site metrics with bootstrap CI
    ms = ["IoU", "Precision", "Recall", "F1"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [6, 3]},
                             sharey=True)
    for ax, cls, col in ((axes[0], "roof", ROOF_C), (axes[1], "pv", PV_C)):
        sites = sorted(boot_site[boot_site.cls == cls].site.unique())
        x = np.arange(len(sites))
        shades = [col, col, col, col]
        alphas = [1.0, 0.75, 0.5, 0.3]
        w = 0.19
        for i, k in enumerate(ms):
            pt = [full_ps[(full_ps.site == s) & (full_ps.cls == cls)][k].iloc[0] for s in sites]
            bs = boot_site[boot_site.cls == cls]
            lo = [bs[bs.site == s][k].quantile(0.025) for s in sites]
            hi = [bs[bs.site == s][k].quantile(0.975) for s in sites]
            xx = x + (i - 1.5) * w
            ax.bar(xx, pt, width=w - 0.03, color=shades[i], alpha=alphas[i], label=k)
            ax.errorbar(xx, pt, yerr=[np.array(pt) - lo, np.array(hi) - pt], fmt="none",
                        ecolor=INK, elinewidth=0.7, capsize=1.5)
        ax.set_xticks(x, [s.replace("_", "\n") for s in sites])
        ax.set_title("Roof" if cls == "roof" else "PV", color=INK, fontsize=9, loc="left")
        ax.set_ylim(0, 105)
        _style(ax)
    axes[0].set_ylabel("Metric (%) with 95% block-bootstrap CI", color=INK)
    axes[0].legend(ncol=4, frameon=False, fontsize=7, loc="lower left",
                   bbox_to_anchor=(0, -0.42))
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig_per_site_metrics.png"), dpi=300)
    plt.close(fig)

    # 3. parameter stability
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for ax, col_name, label, levels in (
            (axes[0], "sel_tau", "Confidence threshold τ", sorted({p["tau"] for p in GRID})),
            (axes[1], "sel_min_size", "Min. object size (px)", sorted({p["min_size"] for p in GRID})),
            (axes[2], "sel_geo_filter", "Geometric filter", [True, False])):
        x = np.arange(len(levels))
        for j, (cls, c) in enumerate((("roof", ROOF_C), ("pv", PV_C))):
            v = mc[mc.cls == cls][col_name]
            freq = [100 * (v == lv).mean() for lv in levels]
            ax.bar(x + (j - 0.5) * 0.38, freq, width=0.36, color=c,
                   label="Roof" if cls == "roof" else "PV")
        paper = PAPER_PARAMS[{"sel_tau": "tau", "sel_min_size": "min_size",
                              "sel_geo_filter": "geo_filter"}[col_name]]
        ax.set_xticks(x, [("%.2f" % l if isinstance(l, float) else ("on" if l is True else "off" if l is False else str(l)))
                          + ("*" if l == paper else "") for l in levels],
                      rotation=45 if col_name == "sel_tau" else 0, fontsize=7)
        ax.set_xlabel(label, color=INK)
        ax.set_ylim(0, 100)
        _style(ax)
    axes[0].set_ylabel("% of MC iterations selected", color=INK)
    axes[2].legend(frameon=False, fontsize=7)
    fig.text(0.01, 0.01, "* = PAPER_CONFIG value", fontsize=6.5, color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig_param_stability.png"), dpi=300)
    plt.close(fig)


def main():
    ensure_dirs()
    counts, blocks, data = load()
    full = full_data(data)

    mc_path = os.path.join(RESULTS, "mc_iterations.csv")
    if os.path.exists(mc_path):
        mc = pd.read_csv(mc_path)
    else:
        mc = monte_carlo(data)
        mc.to_csv(mc_path, index=False)
    bpath = os.path.join(RESULTS, "bootstrap.csv")
    if os.path.exists(bpath):
        boot = pd.read_csv(bpath)
        boot_site = pd.read_csv(os.path.join(RESULTS, "bootstrap_per_site.csv"))
    else:
        boot, boot_site = bootstrap(data)
        boot.to_csv(bpath, index=False)
        boot_site.to_csv(os.path.join(RESULTS, "bootstrap_per_site.csv"), index=False)
    loso = leave_out(data, "site")
    loco = leave_out(data, "city")
    loso.to_csv(os.path.join(RESULTS, "loso.csv"), index=False)
    loco.to_csv(os.path.join(RESULTS, "loco.csv"), index=False)

    pb = per_block(counts, blocks)
    pb.to_csv(os.path.join(RESULTS, "per_block_metrics.csv"), index=False)

    # consistency check: PAPER_CONFIG block counts must add up to phase0b per-site counts
    ps = pd.read_csv(os.path.join(RESULTS, "per_site_metrics.csv"))
    tot = pb.groupby(["site", "class"])[["TP", "FP", "FN", "TN"]].sum()
    for _, r in ps.iterrows():
        t = tot.loc[(r.site, r.cls)]
        assert all(int(t[k]) == int(r[k]) for k in ("TP", "FP", "FN", "TN")), (r.site, r.cls)

    s = summarize(mc, boot, loso, loco, full)
    s.to_csv(os.path.join(RESULTS, "summary_table.csv"), index=False)
    with open(os.path.join(RESULTS, "summary_table.md"), "w", encoding="utf-8") as f:
        f.write(summary_md(s, full, boot))

    # parameter selection frequencies
    sel = pd.concat([mc.groupby("cls").selected.value_counts().rename("n").reset_index()
                     .assign(protocol="MC")])
    sel.to_csv(os.path.join(RESULTS, "mc_selected_params.csv"), index=False)

    figures(mc, boot_site, loso, ps)
    pd.set_option("display.width", 250)
    print(s.round(2).to_string(index=False))
    print(loso[["cls", "held_out", "selected", "test_IoU", "test_F1", "paperconfig_test_F1"]].round(2))
    print(loco[["cls", "held_out", "selected", "test_IoU", "test_F1", "paperconfig_test_F1"]].round(2))
    print(sel)


if __name__ == "__main__":
    main()
