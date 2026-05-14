"""
resampling_stability.py
========================
SAPPHIRE resampling stability test.
Randomly subsample 80% of cells x 20 iterations, rebuild the network each time.
Reports: AUC mean +/- std, Spearman rho mean +/- std.

Usage (notebook):
    exec(open("sapphire_core.py").read())
    exec(open("resampling_stability.py").read())
    df_resample = run_resampling_stability()
"""

import gc
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

import os
_here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."
if "load_and_prepare" not in dir():
    print("  -> Auto-loading sapphire_core.py...")
    exec(open(os.path.join(_here, "sapphire_core.py")).read(), globals())

RESAMP_OUTPUT   = DATA_ROOT / "resampling_stability"
RESAMP_OUTPUT.mkdir(exist_ok=True, parents=True)

TARGET_DATASETS = ["Cardiomyocyte", "Endoderm", "Kidney", "Neuro"]
N_ITER          = 20
RESAMP_FRAC     = 0.80
RESAMP_MAX_CELLS = 10000
RESAMP_HVG      = 2000


def _sort_key(tp):
    import re
    m = re.search(r"(\d+(?:\.\d+)?)", str(tp))
    return float(m.group(1)) if m else 0.0


def eval_one_resample(adata_hvg, params, time_col, early_tp, late_tp, seed):
    """Subsample 80% of cells, rebuild network, return AUC and Spearman rho."""
    np.random.seed(seed)
    n         = adata_hvg.n_obs
    k         = int(n * RESAMP_FRAC)
    idx       = np.random.choice(n, k, replace=False)
    adata_sub = adata_hvg[sorted(idx)].copy()

    try:
        modules, _ = build_network(adata_sub, params)
        if len(modules) == 0:
            return float("nan"), float("nan")

        pc_df = compute_per_cell_metrics(adata_sub, modules, time_col)
        pc_df["composite"] = compute_composite(pc_df)

        mask   = pc_df["timepoint"].isin([early_tp, late_tp])
        sub    = pc_df[mask]
        y_true = (sub["timepoint"] == early_tp).astype(int).values
        try:
            auc = roc_auc_score(y_true, sub["composite"].values)
            auc = max(auc, 1 - auc)
        except Exception:
            auc = float("nan")

        tps_sorted = sorted(pc_df["timepoint"].unique(), key=_sort_key)
        time_map   = {tp: i for i, tp in enumerate(tps_sorted)}
        t_num      = pc_df["timepoint"].map(time_map)
        valid      = ~t_num.isna()
        try:
            r, _ = spearmanr(pc_df.loc[valid, "composite"], t_num[valid])
        except Exception:
            r = float("nan")

        del modules, pc_df, adata_sub
        gc.collect()
        return auc, r

    except Exception:
        return float("nan"), float("nan")


def run_resampling():
    all_rows     = []
    summary_rows = []

    for ds_name in TARGET_DATASETS:
        cfg = DATASETS_CONFIG[ds_name]
        print(f"\n{'='*55}\nDataset: {ds_name}\n{'='*55}")

        adata    = load_and_prepare(ds_name, cfg, max_cells=RESAMP_MAX_CELLS)
        time_col = cfg["time_col"]
        early_tp = cfg["early_tp"]
        late_tp  = cfg["late_tp"]

        adata_hvg = hvg_filter(adata, RESAMP_HVG)
        del adata
        gc.collect()

        params = {**SAPPHIRE_PARAMS, **cfg.get("param_overrides", {})}
        print(f"  {N_ITER} iterations x {RESAMP_FRAC:.0%} cells "
              f"({int(adata_hvg.n_obs * RESAMP_FRAC)} cells per resample)...")

        aucs, rhos = [], []

        for i in range(N_ITER):
            auc, r = eval_one_resample(
                adata_hvg, params, time_col, early_tp, late_tp,
                seed=SAPPHIRE_PARAMS["random_state"] + i
            )
            aucs.append(auc)
            rhos.append(r)
            print(f"    iter {i+1:2d}/{N_ITER}  AUC={auc:.3f}  rho={r:.3f}")
            all_rows.append({"dataset": ds_name, "iteration": i + 1, "auc": auc, "spearman_r": r})

        aucs = np.array([x for x in aucs if not np.isnan(x)])
        rhos = np.array([x for x in rhos if not np.isnan(x)])

        summary_rows.append({
            "dataset"      : ds_name,
            "auc_mean"     : aucs.mean(),
            "auc_std"      : aucs.std(),
            "auc_min"      : aucs.min(),
            "spearman_mean": rhos.mean(),
            "spearman_std" : rhos.std(),
            "n_iter"       : len(aucs),
        })

        print(f"\n  -> AUC:        {aucs.mean():.3f} +/- {aucs.std():.3f}  [{aucs.min():.3f} - {aucs.max():.3f}]")
        print(f"  -> Spearman rho: {rhos.mean():.3f} +/- {rhos.std():.3f}")

        del adata_hvg
        gc.collect()

    df_all     = pd.DataFrame(all_rows)
    df_summary = pd.DataFrame(summary_rows)
    df_all.to_csv(RESAMP_OUTPUT / "resampling_iterations.csv", index=False)
    df_summary.to_csv(RESAMP_OUTPUT / "resampling_summary.csv", index=False)
    return df_all, df_summary


def plot_resampling(df_all, df_summary):
    ds_colors = {
        "Cardiomyocyte": "#378ADD",
        "Endoderm"     : "#1D9E75",
        "Kidney"       : "#EF9F27",
        "Neuro"        : "#7F77DD",
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, metric, ylabel, title in zip(
        axes,
        ["auc", "spearman_r"],
        ["AUC (composite, early vs late)", "Spearman rho (composite vs time)"],
        ["A.  AUC stability across 20 resamples",
         "B.  Time correlation stability across 20 resamples"],
    ):
        data_per_ds = [
            df_all[df_all["dataset"] == ds][metric].dropna().values
            for ds in TARGET_DATASETS
        ]
        bp = ax.boxplot(
            data_per_ds, patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker="o", markersize=4, alpha=0.5),
            widths=0.5,
        )
        for patch, ds in zip(bp["boxes"], TARGET_DATASETS):
            patch.set_facecolor(ds_colors[ds])
            patch.set_alpha(0.7)

        for i, ds in enumerate(TARGET_DATASETS):
            m       = df_summary[df_summary["dataset"] == ds]
            col_mean = "spearman_mean" if metric == "spearman_r" else "auc_mean"
            col_std  = "spearman_std"  if metric == "spearman_r" else "auc_std"
            mv = m[col_mean].values[0]
            sv = m[col_std].values[0]
            ax.text(i + 1, 0.52, f"{mv:.3f}\n+/-{sv:.3f}",
                    ha="center", va="bottom", fontsize=9,
                    color=ds_colors[ds], fontweight="bold")

        ax.set_xticks(range(1, len(TARGET_DATASETS) + 1))
        ax.set_xticklabels(TARGET_DATASETS, rotation=15, ha="right")
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(0.5, 1.08)
        ax.axhline(0.95, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.grid(axis="y", alpha=0.3)

    overall_auc_mean = df_summary["auc_mean"].mean()
    overall_auc_std  = df_summary["auc_std"].mean()
    overall_rho_mean = df_summary["spearman_mean"].mean()
    overall_rho_std  = df_summary["spearman_std"].mean()

    fig.suptitle(
        f"SAPPHIRE -- Resampling Stability  ({N_ITER} iterations x {RESAMP_FRAC:.0%} cells)\n"
        f"Mean AUC = {overall_auc_mean:.3f} +/- {overall_auc_std:.3f}  |  "
        f"Mean rho = {overall_rho_mean:.3f} +/- {overall_rho_std:.3f}",
        fontsize=12, fontweight="bold"
    )

    plt.tight_layout()
    fp = RESAMP_OUTPUT / "resampling_stability.pdf"
    plt.savefig(fp, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved: {fp}")


def run_resampling_stability():
    csv_fp = RESAMP_OUTPUT / "resampling_summary.csv"
    if csv_fp.exists():
        print(f"Existing results found -- plotting only (delete {csv_fp.name} to rerun)")
        df_all     = pd.read_csv(RESAMP_OUTPUT / "resampling_iterations.csv")
        df_summary = pd.read_csv(csv_fp)
    else:
        df_all, df_summary = run_resampling()
    plot_resampling(df_all, df_summary)
    return df_summary


print("resampling_stability.py loaded  ->  run_resampling_stability()")
