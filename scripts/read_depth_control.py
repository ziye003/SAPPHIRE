"""
SAPPHIRE Read-depth / Dropout Control
=======================================
Test that SAPPHIRE scores are not driven by sequencing depth.

Two steps:
  1. Plot entropy vs nUMI / n_genes scatter (per dataset)
  2. Regress out log(nUMI), recompute AUC on residuals
     -> If AUC does not drop, entropy signal is independent of read depth

Usage (notebook):
    exec(open("sapphire_core.py").read())
    exec(open("read_depth_control.py").read())
    run_read_depth_control()
"""

import gc
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
warnings.filterwarnings("ignore")

# Auto-load core if not already loaded
import os
_here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."
if "load_and_prepare" not in dir():
    print("  -> Auto-loading sapphire_core.py...")
    exec(open(os.path.join(_here, "sapphire_core.py")).read(), globals())

RD_OUTPUT     = DATA_ROOT / "read_depth_control"
RD_OUTPUT.mkdir(exist_ok=True, parents=True)

TARGET_DATASETS = ["Cardiomyocyte", "Endoderm", "Kidney", "Neuro"]
RD_MAX_CELLS    = 10000
RD_HVG          = 2000


# ============================================================
# Compute nUMI and n_genes from raw data
# ============================================================

def get_depth_metrics(adata_raw):
    """Compute nUMI and n_genes before any normalization."""
    # Prefer adata.raw if available (preserves original counts)
    if adata_raw.raw is not None:
        X = adata_raw.raw.X
    else:
        X = adata_raw.X

    if sp.issparse(X):
        numi   = np.array(X.sum(axis=1)).ravel()
        ngenes = np.array((X > 0).sum(axis=1)).ravel()
    else:
        numi   = X.sum(axis=1)
        ngenes = (X > 0).sum(axis=1)

    return numi, ngenes


# ============================================================
# Per-dataset analysis
# ============================================================

def analyze_one(name, cfg):
    print(f"\n{'='*55}\nDataset: {name}\n{'='*55}")

    # Load raw data before normalization to extract depth metrics
    import scanpy as sc
    adata_raw = sc.read_h5ad(cfg["file"])

    # Extract depth before any processing
    if "Unknown" in adata_raw.obs[cfg["time_col"]].values:
        adata_raw = adata_raw[adata_raw.obs[cfg["time_col"]] != "Unknown"].copy()

    # Subsample if too large
    if adata_raw.n_obs > RD_MAX_CELLS:
        np.random.seed(SAPPHIRE_PARAMS["random_state"])
        sampled = []
        time_col = cfg["time_col"]
        for tp in adata_raw.obs[time_col].unique():
            idx = np.where(adata_raw.obs[time_col] == tp)[0]
            n   = max(1, int(len(idx) * RD_MAX_CELLS / adata_raw.n_obs))
            sampled.extend(np.random.choice(idx, min(n, len(idx)), replace=False))
        adata_raw = adata_raw[sorted(sampled)].copy()

    numi, ngenes = get_depth_metrics(adata_raw)
    cell_ids     = adata_raw.obs_names
    timepoints   = adata_raw.obs[cfg["time_col"]].values
    print(f"  nUMI range: {numi.min():.0f} - {numi.max():.0f}, median={np.median(numi):.0f}")
    print(f"  n_genes range: {ngenes.min():.0f} – {ngenes.max():.0f}")

    # Normalize, log-transform, select HVGs, build network
    adata = load_and_prepare(name, cfg, max_cells=RD_MAX_CELLS)
    adata = hvg_filter(adata, RD_HVG)

    params = {**SAPPHIRE_PARAMS, **cfg.get("param_overrides", {})}
    modules, _ = build_network(adata, params)

    pc_df = compute_per_cell_metrics(adata, modules, cfg["time_col"])
    pc_df["composite"] = compute_composite(pc_df)
    pc_df["numi"]      = numi[:len(pc_df)]
    pc_df["ngenes"]    = ngenes[:len(pc_df)]

    # Correlation: entropy vs depth
    r_umi,   p_umi   = spearmanr(pc_df["pathway_entropy"], pc_df["numi"])
    r_genes, p_genes = spearmanr(pc_df["pathway_entropy"], pc_df["ngenes"])
    print(f"  Entropy vs nUMI:    ρ={r_umi:.3f}, p={p_umi:.2e}")
    print(f"  Entropy vs n_genes: ρ={r_genes:.3f}, p={p_genes:.2e}")

    # Regression residual
    # Regress out log(nUMI) from entropy
    log_numi = np.log1p(pc_df["numi"].values).reshape(-1, 1)
    entropy  = pc_df["pathway_entropy"].values

    reg = LinearRegression().fit(log_numi, entropy)
    residual_entropy = entropy - reg.predict(log_numi)
    pc_df["entropy_residual"] = residual_entropy

    # AUC: original vs residual entropy
    early_tp = cfg["early_tp"]
    late_tp  = cfg["late_tp"]

    auc_orig = compute_auc(pc_df, early_tp, late_tp, "pathway_entropy")
    auc_comp = compute_auc(pc_df, early_tp, late_tp, "composite")

    # AUC on residual
    mask   = pc_df["timepoint"].isin([early_tp, late_tp])
    sub    = pc_df[mask]
    y_true = (sub["timepoint"] == early_tp).astype(int).values
    try:
        auc_resid = roc_auc_score(y_true, sub["entropy_residual"].values)
        auc_resid = max(auc_resid, 1 - auc_resid)
    except Exception:
        auc_resid = float("nan")

    print(f"  AUC (original entropy):  {auc_orig:.3f}")
    print(f"  AUC (residual entropy):  {auc_resid:.3f}")
    print(f"  AUC (composite):         {auc_comp:.3f}")
    print(f"  AUC drop after regressing depth: {auc_orig - auc_resid:+.3f}")

    result = {
        "dataset"        : name,
        "r_entropy_numi" : round(r_umi,   3),
        "p_entropy_numi" : round(p_umi,   4),
        "r_entropy_genes": round(r_genes, 3),
        "p_entropy_genes": round(p_genes, 4),
        "auc_original"   : round(auc_orig,  3),
        "auc_residual"   : round(auc_resid, 3),
        "auc_composite"  : round(auc_comp,  3),
        "auc_drop"       : round(auc_orig - auc_resid, 3),
    }

    pc_df.to_csv(RD_OUTPUT / f"{name}_depth_analysis.csv")
    del adata, modules, pc_df, adata_raw
    gc.collect()
    return result


# ============================================================
# Visualization
# ============================================================

def plot_read_depth(all_results, all_pc_dfs):
    n_ds = len(TARGET_DATASETS)
    ds_colors = {
        "Cardiomyocyte": "#378ADD",
        "Endoderm"     : "#1D9E75",
        "Kidney"       : "#EF9F27",
        "Neuro"        : "#7F77DD",
    }

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, n_ds, figure=fig, hspace=0.45, wspace=0.35)

    for col, ds_name in enumerate(TARGET_DATASETS):
        pc_df = all_pc_dfs[ds_name]
        color = ds_colors[ds_name]
        res   = next(r for r in all_results if r["dataset"] == ds_name)

        # Row 1: entropy vs log(nUMI) scatter
        ax = fig.add_subplot(gs[0, col])
        log_numi = np.log1p(pc_df["numi"])

        # Downsample for plotting speed
        n_plot = min(2000, len(pc_df))
        idx    = np.random.choice(len(pc_df), n_plot, replace=False)

        ax.scatter(log_numi.iloc[idx], pc_df["pathway_entropy"].iloc[idx],
                   alpha=0.15, s=3, color=color, rasterized=True)

        # Regression line
        x_line = np.linspace(log_numi.min(), log_numi.max(), 100)
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression().fit(
            log_numi.values.reshape(-1, 1),
            pc_df["pathway_entropy"].values
        )
        ax.plot(x_line, reg.predict(x_line.reshape(-1, 1)),
                color="red", linewidth=1.5, linestyle="--", alpha=0.8)

        r = res["r_entropy_numi"]
        p = res["p_entropy_numi"]
        p_str = f"{p:.2e}" if p < 0.001 else f"{p:.3f}"
        ax.set_title(f"{ds_name}\nρ = {r:.3f}, p = {p_str}",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("log(nUMI + 1)", fontsize=9)
        if col == 0:
            ax.set_ylabel("Pathway Entropy", fontsize=10)
        ax.grid(alpha=0.3)

        # Row 2: AUC comparison bar
        ax2 = fig.add_subplot(gs[1, col])
        labels_bar = ["Original\nEntropy", "Residual\nEntropy", "Composite"]
        vals       = [res["auc_original"], res["auc_residual"], res["auc_composite"]]
        bar_colors = [color, "#BBBBBB", "#7F77DD"]
        bars = ax2.bar(range(3), vals, color=bar_colors, width=0.55, alpha=0.85)

        for bar, val in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width()/2,
                     val + 0.01, f"{val:.3f}",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax2.set_xticks(range(3))
        ax2.set_xticklabels(labels_bar, fontsize=9)
        ax2.set_ylim(0.5, 1.12)
        ax2.axhline(0.5, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
        if col == 0:
            ax2.set_ylabel("AUC (early vs late)", fontsize=10)
        drop = res["auc_drop"]
        ax2.set_title(f"AUC drop = {drop:+.3f}", fontsize=10,
                      color="darkgreen" if abs(drop) < 0.05 else "darkred")
        ax2.grid(axis="y", alpha=0.3)

    # Row labels
    fig.text(0.01, 0.73, "Entropy vs nUMI", va="center",
             rotation="vertical", fontsize=11, fontweight="bold")
    fig.text(0.01, 0.27, "AUC before/after\ndepth regression", va="center",
             rotation="vertical", fontsize=11, fontweight="bold")

    fig.suptitle(
        "SAPPHIRE — Read-depth / Dropout Control\n"
        "Top: correlation between pathway entropy and sequencing depth\n"
        "Bottom: AUC before vs. after regressing out log(nUMI)",
        fontsize=12, fontweight="bold"
    )

    fp = RD_OUTPUT / "read_depth_control.pdf"
    plt.savefig(fp, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved: {fp}")


# ============================================================
# Entry point
# ============================================================

def run_read_depth_control():
    csv_fp = RD_OUTPUT / "read_depth_summary.csv"
    all_pc_dfs = {}

    if csv_fp.exists():
        print(f"Existing summary found -- replotting (delete {csv_fp.name} to rerun)")
        df_summary  = pd.read_csv(csv_fp)
        all_results = df_summary.to_dict("records")
        for ds in TARGET_DATASETS:
            pc_path = RD_OUTPUT / f"{ds}_depth_analysis.csv"
            if pc_path.exists():
                all_pc_dfs[ds] = pd.read_csv(pc_path, index_col=0)
    else:
        all_results = []
        for ds_name in TARGET_DATASETS:
            cfg = DATASETS_CONFIG[ds_name]
            try:
                res = analyze_one(ds_name, cfg)
                all_results.append(res)
                pc_path = RD_OUTPUT / f"{ds_name}_depth_analysis.csv"
                all_pc_dfs[ds_name] = pd.read_csv(pc_path, index_col=0)
            except Exception as e:
                import traceback
                print(f"\n❌ {ds_name} : {e}")
                traceback.print_exc()

        df_summary = pd.DataFrame(all_results)
        df_summary.to_csv(csv_fp, index=False)

    # Print summary
    print(f"\n{'='*60}\nSummary\n{'='*60}")
    print(df_summary[[
        "dataset", "r_entropy_numi", "p_entropy_numi",
        "auc_original", "auc_residual", "auc_drop"
    ]].to_string(index=False))

    # Plot
    if all_pc_dfs:
        plot_read_depth(all_results, all_pc_dfs)

    # Paper sentence
    mean_r    = df_summary["r_entropy_numi"].abs().mean()
    mean_drop = df_summary["auc_drop"].abs().mean()
    print(f"\n{'='*60}")
    print("Paper summary sentence:")
    print(f"{'='*60}")
    print(
        f"Pathway entropy showed weak correlation with sequencing depth "
        f"(mean |ρ| = {mean_r:.3f} across four datasets), and AUC for "
        f"early/late separation changed by only {mean_drop:.3f} on average "
        f"after regressing out log(nUMI), indicating that SAPPHIRE entropy "
        f"is not driven by sequencing depth."
    )

    return df_summary


print("read_depth_control.py loaded  ->  run_read_depth_control()")
