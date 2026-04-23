"""
SAPPHIRE Method Comparison — 
======================================
Compare SAPPHIRE composite score against:
  1. CytoTRACE    (gene count-based potency)
  2. Expr Entropy (SCENT-like, per-gene Shannon entropy)
  3. PAGA         (diffusion pseudotime)
  4. Gene Count   (baseline)

Evaluation metrics:
  - AUC (early vs late separation)
  - Spearman ρ with experimental time
  - Monotonicity (|ρ|)

Note: exec sapphire_validation_all.py first in the notebook, then exec this file.
      Requires DATASETS_CONFIG and run_one from the core pipeline.
"""

import time
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# Variables inherited from sapphire_core.py / sapphire_validation_all.py:
# DATA_ROOT, OUTPUT_DIR, DATASETS_CONFIG, SAPPHIRE_PARAMS
# load_and_prepare, compute_per_cell_metrics, compute_composite

MC_OUTPUT = DATA_ROOT / "method_comparison"
MC_OUTPUT.mkdir(exist_ok=True, parents=True)

# Run these four datasets
TARGET_DATASETS = ["Cardiomyocyte", "Endoderm", "Kidney", "Neuro"]

# ============================================================
# Baseline method implementations
# ============================================================

def cytotrace_score(adata):
    """Gene count per cell, normalized 0-1. Higher = more stem-like."""
    X = adata.X
    if sp.issparse(X):
        counts = np.array((X > 0).sum(axis=1)).ravel()
    else:
        counts = (X > 0).sum(axis=1)
    s = (counts - counts.min()) / (counts.max() - counts.min() + 1e-10)
    return pd.Series(s, index=adata.obs_names, name="CytoTRACE")


def expression_entropy_score(adata):
    """Shannon entropy of per-cell expression distribution. Higher = less committed."""
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = np.maximum(X, 0)
    eps = 1e-10
    entropies = []
    for i in range(X.shape[0]):
        expr = X[i] + eps
        p    = expr / expr.sum()
        entropies.append(-np.sum(p * np.log2(p)))
    s = np.array(entropies)
    s = (s - s.min()) / (s.max() - s.min() + 1e-10)
    return pd.Series(s, index=adata.obs_names, name="Expr_Entropy")


def gene_count_score(adata):
    """Simple detected gene count, normalized 0-1."""
    X = adata.X
    if sp.issparse(X):
        counts = np.array((X > 0).sum(axis=1)).ravel()
    else:
        counts = (X > 0).sum(axis=1)
    s = (counts - counts.min()) / (counts.max() - counts.min() + 1e-10)
    return pd.Series(s, index=adata.obs_names, name="Gene_Count")


def paga_pseudotime_score(adata, early_tp, time_col):
    """PAGA-based diffusion pseudotime from early state. Higher = earlier."""
    adata_c = adata.copy()
    try:
        if "X_pca" not in adata_c.obsm:
            sc.tl.pca(adata_c, n_comps=min(50, adata_c.n_vars - 1))
        sc.pp.neighbors(adata_c, n_neighbors=15, n_pcs=40)
        sc.tl.leiden(adata_c, resolution=1.0, key_added="_leiden_tmp")
        sc.tl.paga(adata_c, groups="_leiden_tmp")

        # Root = cluster with most early cells
        early_mask  = adata_c.obs[time_col] == early_tp
        cluster_cts = adata_c.obs.loc[early_mask, "_leiden_tmp"].value_counts()
        root_clust  = cluster_cts.idxmax()
        root_cell   = np.where(
            (adata_c.obs["_leiden_tmp"] == root_clust).values
        )[0][0]
        adata_c.uns["iroot"] = root_cell

        sc.tl.dpt(adata_c, n_dcs=10)
        pt = adata_c.obs["dpt_pseudotime"].copy()

        # Flip so early = high score
        early_mean = pt[adata_c.obs[time_col] == early_tp].mean()
        late_mean  = pt[adata_c.obs[time_col] != early_tp].mean()
        if late_mean > early_mean:
            pt = 1 - pt

        pt = (pt - pt.min()) / (pt.max() - pt.min() + 1e-10)
        return pd.Series(pt.values, index=adata.obs_names, name="PAGA")
    except Exception as e:
        print(f"    PAGA failed: {e}")
        return None


# ============================================================
# Evaluation functions
# ============================================================

def sort_key(tp):
    import re
    m = re.search(r"(\d+(?:\.\d+)?)", str(tp))
    return float(m.group(1)) if m else 0.0


def compute_auc_score(scores, labels, early_tp, late_tp):
    mask = labels.isin([early_tp, late_tp])
    if mask.sum() < 10:
        return np.nan
    y_true  = (labels[mask] == early_tp).astype(int).values
    y_score = scores[mask].values
    try:
        auc = roc_auc_score(y_true, y_score)
        return max(auc, 1 - auc)
    except Exception:
        return np.nan


def compute_time_corr(scores, labels, tps_sorted):
    time_map = {tp: i for i, tp in enumerate(tps_sorted)}
    t_num    = labels.map(time_map)
    valid    = ~t_num.isna()
    if valid.sum() < 10:
        return np.nan, np.nan
    r, p = spearmanr(scores[valid], t_num[valid])
    return r, p


def evaluate_method(scores, labels, early_tp, late_tp, tps_sorted):
    if scores is None:
        return dict(auc=np.nan, spearman_r=np.nan, monotonicity=np.nan)
    auc      = compute_auc_score(scores, labels, early_tp, late_tp)
    r, _     = compute_time_corr(scores, labels, tps_sorted)
    return dict(auc=auc, spearman_r=r, monotonicity=abs(r) if not np.isnan(r) else np.nan)


# ============================================================
# Per-dataset comparison
# ============================================================

def compare_one(name, cfg):
    print(f"\n{'='*60}")
    print(f"Method comparison: {name}")
    print("="*60)

    out_dir = MC_OUTPUT / name
    out_dir.mkdir(exist_ok=True, parents=True)

    # 1. Load data
    adata    = load_and_prepare(name, cfg)
    time_col = cfg["time_col"]
    early_tp = cfg["early_tp"]
    late_tp  = cfg["late_tp"]
    labels   = adata.obs[time_col].astype(str)
    tps_sorted = sorted(labels.unique(), key=sort_key)

    # 2. SAPPHIRE composite score
    print("  SAPPHIRE...")
    t0 = time.time()
    params  = {**SAPPHIRE_PARAMS, **cfg.get("param_overrides", {})}
    modules, _ = build_network(adata, params)
    pc_df   = compute_per_cell_metrics(adata, modules, time_col)
    pc_df["composite"] = compute_composite(pc_df)
    sapphire_scores = pc_df["composite"]
    t_sapphire = time.time() - t0
    print(f"    done ({t_sapphire:.1f}s)")

    results = []

    # 3. Evaluate baseline methods
    methods = {
        "SAPPHIRE"    : (sapphire_scores, t_sapphire),
    }

    for mname, func, kwargs in [
        ("CytoTRACE",   cytotrace_score,          {}),
        ("Expr_Entropy", expression_entropy_score, {}),
        ("Gene_Count",  gene_count_score,          {}),
        ("PAGA",        paga_pseudotime_score,     {"early_tp": early_tp, "time_col": time_col}),
    ]:
        print(f"  {mname}...")
        t0 = time.time()
        try:
            s   = func(adata, **kwargs)
            elapsed = time.time() - t0
            methods[mname] = (s, elapsed)
            print(f"    done ({elapsed:.1f}s)")
        except Exception as e:
            print(f"    failed: {e}")
            methods[mname] = (None, np.nan)

    # 4. Compute metrics
    for mname, (scores, elapsed) in methods.items():
        m = evaluate_method(scores, labels, early_tp, late_tp, tps_sorted)
        results.append({
            "dataset"    : name,
            "method"     : mname,
            "auc"        : round(m["auc"],         3) if not np.isnan(m["auc"])         else np.nan,
            "spearman_r" : round(m["spearman_r"],  3) if not np.isnan(m["spearman_r"])  else np.nan,
            "monotonicity": round(m["monotonicity"],3) if not np.isnan(m["monotonicity"]) else np.nan,
            "time_sec"   : round(elapsed,           1) if not np.isnan(elapsed)          else np.nan,
        })
        print(f"    {mname:15s} AUC={m['auc']:.3f}  ρ={m['spearman_r']:.3f}  mono={m['monotonicity']:.3f}")

    df = pd.DataFrame(results)
    df.to_csv(out_dir / f"{name}_method_comparison.csv", index=False)

    # 5. Plot
    _plot_comparison(df, name, out_dir)

    return df


def _plot_comparison(df, name, out_dir):
    methods_order = ["SAPPHIRE", "CytoTRACE", "Expr_Entropy", "PAGA", "Gene_Count"]
    colors = {
        "SAPPHIRE"    : "#7F77DD",
        "CytoTRACE"   : "#378ADD",
        "Expr_Entropy": "#1D9E75",
        "PAGA"        : "#EF9F27",
        "Gene_Count"  : "#888780",
    }

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    metrics   = ["auc", "spearman_r", "monotonicity"]
    titles    = ["AUC (early vs late)", "Spearman ρ (vs time)", "Monotonicity (|ρ|)"]

    for ax, metric, title in zip(axes, metrics, titles):
        sub = df.set_index("method").reindex(methods_order)[metric].dropna()
        bar_colors = [colors.get(m, "#888780") for m in sub.index]
        bars = ax.bar(range(len(sub)), sub.values, color=bar_colors, width=0.6)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub.index, rotation=35, ha="right", fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.grid(axis="y", alpha=0.3)

        # Highlight SAPPHIRE bar
        if "SAPPHIRE" in sub.index:
            idx = list(sub.index).index("SAPPHIRE")
            bars[idx].set_edgecolor("black")
            bars[idx].set_linewidth(1.5)

        # Value labels
        for i, v in enumerate(sub.values):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(f"Method Comparison — {name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fp = out_dir / f"{name}_method_comparison.pdf"
    plt.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot: {fp.name}")


# ============================================================
# Cross-dataset summary plot
# ============================================================

def plot_summary(all_df, out_dir):
    methods_order = ["SAPPHIRE", "CytoTRACE", "Expr_Entropy", "PAGA", "Gene_Count"]
    colors = {
        "SAPPHIRE"    : "#7F77DD",
        "CytoTRACE"   : "#378ADD",
        "Expr_Entropy": "#1D9E75",
        "PAGA"        : "#EF9F27",
        "Gene_Count"  : "#888780",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ["auc", "spearman_r", "monotonicity"]
    titles  = ["AUC (early vs late)", "Spearman ρ (vs time)", "Monotonicity (|ρ|)"]

    for ax, metric, title in zip(axes, metrics, titles):
        pivot = all_df.pivot_table(index="dataset", columns="method", values=metric)
        pivot = pivot.reindex(columns=[m for m in methods_order if m in pivot.columns])

        x      = np.arange(len(pivot))
        n_meth = len(pivot.columns)
        width  = 0.8 / n_meth

        for i, method in enumerate(pivot.columns):
            vals = pivot[method].values
            offset = (i - n_meth / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width=width * 0.9,
                          color=colors.get(method, "#888780"),
                          label=method if ax == axes[0] else "")
            if method == "SAPPHIRE":
                for b in bars:
                    b.set_edgecolor("black")
                    b.set_linewidth(1.2)

        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=20, ha="right")
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.grid(axis="y", alpha=0.3)

    axes[0].legend(title="Method", bbox_to_anchor=(0, -0.25),
                   loc="upper left", ncol=3, fontsize=9)
    fig.suptitle("SAPPHIRE vs Baseline Methods — All Datasets",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fp = out_dir / "ALL_method_comparison_summary.pdf"
    plt.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSummary plot: {fp}")


# ============================================================
# Main entry point
# ============================================================

def run_method_comparison(dataset=None):
    targets = [dataset] if dataset else TARGET_DATASETS
    all_dfs = []

    for name in targets:
        if name not in DATASETS_CONFIG:
            print(f"Unknown dataset: {name}")
            continue
        try:
            df = compare_one(name, DATASETS_CONFIG[name])
            all_dfs.append(df)
        except Exception as e:
            import traceback
            print(f"\nFailed: {name} -- {e}")
            traceback.print_exc()

    if all_dfs:
        all_df = pd.concat(all_dfs, ignore_index=True)
        all_df.to_csv(MC_OUTPUT / "ALL_method_comparison.csv", index=False)

        print(f"\n\n{'='*65}")
        print("Summary")
        print("="*65)
        pivot = all_df.pivot_table(
            index="method", columns="dataset", values="auc"
        ).round(3)
        # Add mean AUC column
        pivot["Mean AUC"] = pivot.mean(axis=1).round(3)
        pivot = pivot.sort_values("Mean AUC", ascending=False)
        print(pivot.to_string())

        if len(all_dfs) > 1:
            plot_summary(all_df, MC_OUTPUT)

        print(f"\nOutput directory: {MC_OUTPUT}")
        return all_df


# ──  ─────────────────────────────────────────────────
# Test on one dataset first (fastest):
run_method_comparison(dataset="Endoderm")

# Then run all:
# run_method_comparison()
# ─────────────────────────────────────────────────────────────
