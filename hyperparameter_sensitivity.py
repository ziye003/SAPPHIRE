"""
hyperparameter_sensitivity.py
==============================
Test SAPPHIRE stability across a grid of hyperparameters.

Parameter grid:
  top_k_edges in {30, 50, 80}
  min_corr    in {0.2, 0.3, 0.4}

Usage (notebook):
    exec(open("sapphire_core.py").read())
    exec(open("hyperparameter_sensitivity.py").read())
    df_sens = run_hyperparameter_sensitivity()
"""

import gc
import itertools
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

import os
_here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."
if "load_and_prepare" not in dir():
    print("  -> Auto-loading sapphire_core.py...")
    exec(open(os.path.join(_here, "sapphire_core.py")).read(), globals())

SENS_OUTPUT     = DATA_ROOT / "hyperparameter_sensitivity"
SENS_OUTPUT.mkdir(exist_ok=True, parents=True)

TARGET_DATASETS = ["Cardiomyocyte", "Endoderm", "Kidney", "Neuro"]
TOP_K_VALUES    = [30, 50, 80]
MIN_CORR_VALUES = [0.2, 0.3, 0.4]
SENS_MAX_CELLS  = 10000
SENS_HVG        = 2000


def _sort_key(tp):
    import re
    m = re.search(r"(\d+(?:\.\d+)?)", str(tp))
    return float(m.group(1)) if m else 0.0


def eval_params(adata, modules, time_col, early_tp, late_tp):
    if len(modules) == 0:
        return dict(auc=float("nan"), spearman_r=float("nan"), n_modules=0)

    pc_df = compute_per_cell_metrics(adata, modules, time_col)
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

    return dict(auc=auc, spearman_r=r, n_modules=len(modules))


def run_sensitivity():
    all_rows = []

    for ds_name in TARGET_DATASETS:
        cfg = DATASETS_CONFIG[ds_name]
        print(f"\n{'='*55}\nDataset: {ds_name}\n{'='*55}")

        adata    = load_and_prepare(ds_name, cfg, max_cells=SENS_MAX_CELLS)
        time_col = cfg["time_col"]
        early_tp = cfg["early_tp"]
        late_tp  = cfg["late_tp"]

        adata_hvg = hvg_filter(adata, SENS_HVG)
        del adata
        gc.collect()

        combos = list(itertools.product(TOP_K_VALUES, MIN_CORR_VALUES))
        print(f"  Running {len(combos)} combos on {adata_hvg.n_obs} cells x {adata_hvg.n_vars} genes...")

        for top_k, min_corr in combos:
            params = {**SAPPHIRE_PARAMS, "top_k_edges": top_k, "min_corr": min_corr}
            try:
                modules, _ = build_network(adata_hvg, params)
                metrics    = eval_params(adata_hvg, modules, time_col, early_tp, late_tp)
                del modules
                gc.collect()
            except Exception as e:
                print(f"    K={top_k}, corr={min_corr}: FAILED -- {e}")
                metrics = dict(auc=float("nan"), spearman_r=float("nan"), n_modules=0)

            all_rows.append({
                "dataset"   : ds_name,
                "top_k"     : top_k,
                "min_corr"  : min_corr,
                "auc"       : metrics["auc"],
                "spearman_r": metrics["spearman_r"],
                "n_modules" : metrics["n_modules"],
            })
            print(f"    K={top_k:2d}, corr={min_corr:.1f} -> "
                  f"AUC={metrics['auc']:.3f}  rho={metrics['spearman_r']:.3f}  "
                  f"modules={metrics['n_modules']}")

        del adata_hvg
        gc.collect()

    df = pd.DataFrame(all_rows)
    df.to_csv(SENS_OUTPUT / "sensitivity_results.csv", index=False)
    return df


def plot_sensitivity(df):
    ds_colors = {
        "Cardiomyocyte": "#378ADD",
        "Endoderm"     : "#1D9E75",
        "Kidney"       : "#EF9F27",
        "Neuro"        : "#7F77DD",
    }

    fig = plt.figure(figsize=(15, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    for ds in TARGET_DATASETS:
        d = df[(df["dataset"] == ds) & (df["min_corr"] == 0.3)]
        ax.plot(d["top_k"], d["auc"], marker="o", color=ds_colors[ds], linewidth=2, label=ds)
    ax.axhline(0.95, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.set_xlabel("top_k_edges", fontsize=11)
    ax.set_ylabel("AUC (composite)", fontsize=11)
    ax.set_title("A.  AUC vs top_k  (min_corr = 0.3)", fontsize=11, fontweight="bold")
    ax.set_ylim(0.5, 1.05); ax.set_xticks(TOP_K_VALUES)
    ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    for ds in TARGET_DATASETS:
        d = df[(df["dataset"] == ds) & (df["top_k"] == 50)]
        ax.plot(d["min_corr"], d["auc"], marker="s", color=ds_colors[ds], linewidth=2, label=ds)
    ax.axhline(0.95, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.set_xlabel("min_corr threshold", fontsize=11)
    ax.set_ylabel("AUC (composite)", fontsize=11)
    ax.set_title("B.  AUC vs min_corr  (top_k = 50)", fontsize=11, fontweight="bold")
    ax.set_ylim(0.5, 1.05); ax.set_xticks(MIN_CORR_VALUES)
    ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 0])
    pivot = df.groupby(["top_k", "min_corr"])["auc"].mean().unstack()
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0.7, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(MIN_CORR_VALUES))); ax.set_xticklabels([f"{v}" for v in MIN_CORR_VALUES])
    ax.set_yticks(range(len(TOP_K_VALUES)));    ax.set_yticklabels([f"{v}" for v in TOP_K_VALUES])
    ax.set_xlabel("min_corr", fontsize=11); ax.set_ylabel("top_k_edges", fontsize=11)
    ax.set_title("C.  Mean AUC heatmap (all datasets)", fontsize=11, fontweight="bold")
    for i in range(len(TOP_K_VALUES)):
        for j in range(len(MIN_CORR_VALUES)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=11,
                    fontweight="bold", color="white" if val > 0.92 else "black")
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = fig.add_subplot(gs[1, 1])
    stds = df.groupby("dataset")["auc"].std().reindex(TARGET_DATASETS)
    bars = ax.bar(range(len(TARGET_DATASETS)), stds.values,
                  color=[ds_colors[d] for d in TARGET_DATASETS], width=0.5, alpha=0.85)
    ax.set_xticks(range(len(TARGET_DATASETS)))
    ax.set_xticklabels(TARGET_DATASETS, rotation=20, ha="right")
    ax.set_ylabel("AUC std across parameter grid", fontsize=11)
    ax.set_title("D.  AUC stability across parameter grid", fontsize=11, fontweight="bold")
    ax.axhline(0.05, color="gray", linestyle="--", alpha=0.5, linewidth=0.8, label="SD=0.05")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, stds.values):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    mean_auc = df["auc"].mean()
    min_auc  = df["auc"].min()
    mean_std = df.groupby("dataset")["auc"].std().mean()
    fig.suptitle(
        f"SAPPHIRE -- Hyperparameter Sensitivity\n"
        f"Mean AUC = {mean_auc:.3f}  |  Min AUC = {min_auc:.3f}  |  Mean within-dataset SD = {mean_std:.3f}",
        fontsize=12, fontweight="bold"
    )
    fp = SENS_OUTPUT / "hyperparameter_sensitivity.pdf"
    plt.savefig(fp, dpi=200, bbox_inches="tight"); plt.close()
    print(f"\nPlot saved: {fp}")


def run_hyperparameter_sensitivity():
    csv_fp = SENS_OUTPUT / "sensitivity_results.csv"
    if csv_fp.exists():
        print(f"Existing results found -- plotting only (delete {csv_fp.name} to rerun)")
        df = pd.read_csv(csv_fp)
    else:
        df = run_sensitivity()
    plot_sensitivity(df)
    return df


print("hyperparameter_sensitivity.py loaded  ->  run_hyperparameter_sensitivity()")
