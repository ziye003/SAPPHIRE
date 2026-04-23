"""
sapphire_core.py
================
SAPPHIRE core module — shared functions and configuration for all pipelines.

Usage:
    exec(open("sapphire_core.py").read())
    # or from another script:
    # All functions become available in the current namespace after exec()

Key functions:
    load_and_prepare    -- load, subsample, normalize an h5ad dataset
    hvg_filter          -- select top-N highly variable genes by variance
    build_network       -- label-free Spearman correlation network + community detection
    compute_per_cell_metrics -- pathway entropy and network dispersion per cell
    compute_composite   -- rank-based composite score
    compute_auc         -- AUC for early vs late separation
    compute_marker_corr -- Spearman correlation with marker gene expression
    shuffle_time_null   -- permutation null test for trajectory monotonicity
    plot_dataset        -- trajectory plot: entropy / dispersion / composite
"""

import gc
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

# ============================================================
# Global configuration
# ============================================================

DATA_ROOT  = Path("/Users/ziye/Documents/paper/data")
OUTPUT_DIR = DATA_ROOT / "sapphire_validation_v2"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

MAX_CELLS = 30000

SAPPHIRE_PARAMS = {
    "n_top_genes"      : 2000,
    "top_k_edges"      : 10,
    "min_corr"         : 0.25,
    "leiden_resolution": 1.5,
    "min_module_size"  : 10,
    "random_state"     : 0,
}

DATASETS_CONFIG = {
    "Cardiomyocyte": {
        "file"         : DATA_ROOT / "E-MTAB-6268_allTP_merged.h5ad",
        "time_col"     : "timepoint",
        "early_tp"     : "D0",
        "late_tp"      : "D30",
        "already_log1p": False,
        "stem_markers" : ["POU5F1", "SOX2", "NANOG", "KLF4"],
        "diff_markers" : ["TNNT2", "MYH6", "MYL2", "NKX2-5"],
    },
    "Endoderm": {
        "file"         : DATA_ROOT / "GSE75748_Endoderm_merged_qc_umap.h5ad",
        "time_col"     : "timepoint",
        "early_tp"     : "00h",
        "late_tp"      : "96h",
        "already_log1p": False,
        "stem_markers" : ["POU5F1", "SOX2", "NANOG"],
        "diff_markers" : ["SOX17", "FOXA2", "GATA6", "HNF1B"],
    },
    "Kidney": {
        "file"         : DATA_ROOT / "GSE118184_Takasato_iPS_timecourse.h5ad",
        "time_col"     : "timepoint",
        "early_tp"     : "Day7",
        "late_tp"      : "Day26",
        "already_log1p": False,
        "stem_markers" : ["POU5F1", "SOX2"],
        "diff_markers" : ["PAX2", "LHX1", "WT1"],
    },
    "EB": {
        "file"         : DATA_ROOT / "EB_merged_qc.h5ad",
        "time_col"     : "timepoint",
        "early_tp"     : "D0",
        "late_tp"      : "D12",
        "already_log1p": False,
        "stem_markers" : ["SOX2"],
        "diff_markers" : ["PAX6", "T"],
        "param_overrides": {"min_corr": 0.15, "n_top_genes": 2000},
    },
    "Neuro": {
        "file"         : DATA_ROOT / "Neuro_dopa_allTP_small_merged.h5ad",
        "time_col"     : "timepoint",
        "early_tp"     : "D11",
        "late_tp"      : "D52",
        "already_log1p": True,
        "stem_markers" : ["SOX2", "NES"],
        "diff_markers" : ["TH", "MAP2"],
    },
}

# ============================================================
# Data loading
# ============================================================

def load_and_prepare(name, cfg, max_cells=None):
    """
    Load, subsample, and normalize a dataset.
    max_cells defaults to global MAX_CELLS if not specified.
    """
    if max_cells is None:
        max_cells = MAX_CELLS

    print(f"  Loading {cfg['file'].name}...")
    adata    = sc.read_h5ad(cfg["file"])
    time_col = cfg["time_col"]

    # Drop cells with unknown timepoint
    if "Unknown" in adata.obs[time_col].values:
        n_before = adata.n_obs
        adata = adata[adata.obs[time_col] != "Unknown"].copy()
        print(f"  Removed Unknown timepoint: {n_before:,} -> {adata.n_obs:,} cells")

    # Stratified subsampling
    if adata.n_obs > max_cells:
        print(f"  Subsampling {adata.n_obs:,} -> {max_cells:,}...")
        np.random.seed(SAPPHIRE_PARAMS["random_state"])
        sampled = []
        for tp in adata.obs[time_col].unique():
            idx = np.where(adata.obs[time_col] == tp)[0]
            n   = max(1, int(len(idx) * max_cells / adata.n_obs))
            sampled.extend(np.random.choice(idx, min(n, len(idx)), replace=False))
        adata = adata[sorted(sampled)].copy()

    # Normalize and log-transform if not already done
    if not cfg.get("already_log1p", False):
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    print(f"  Final shape: {adata.shape}")
    for tp, n in adata.obs[time_col].value_counts().sort_index().items():
        print(f"    {tp}: {n:,}")

    return adata


def hvg_filter(adata, n_top=2000):
    """Select top-N highly variable genes by variance. Returns filtered AnnData."""
    if adata.n_vars <= n_top:
        return adata
    X = adata.X
    if sp.issparse(X):
        mean    = np.array(X.mean(axis=0)).ravel()
        mean_sq = np.array(X.power(2).mean(axis=0)).ravel()
        var     = mean_sq - mean**2
    else:
        var = np.array(X.var(axis=0)).ravel()
    top_idx = np.argsort(var)[::-1][:n_top]
    result  = adata[:, top_idx].copy()
    print(f"  HVG filter: {adata.n_vars} -> {result.n_vars} genes")
    return result


# ============================================================
# Network construction
# ============================================================

def build_network(adata, params):
    """
    Label-free gene co-expression network construction.

    Steps:
      1. Spearman rank correlation (batched matrix computation)
      2. Top-k edges per gene filtered by min_corr threshold
      3. Greedy modularity community detection (networkx)
      4. Filter communities by min_module_size

    Returns:
      modules  -- dict {module_id: [gene_col_indices]}
      gene_list -- list of gene names (adata.var_names)
    """
    print("  Building label-free network...")

    X = adata.X
    if sp.issparse(X):
        X = X.toarray()

    n_cells, n_genes = X.shape
    print(f"  Rank transform ({n_genes} genes)...")

    from scipy.stats import rankdata
    X_rank = np.zeros_like(X, dtype=np.float32)
    for j in range(n_genes):
        X_rank[:, j] = rankdata(X[:, j])

    print("  Computing correlation matrix...")
    batch = 200
    corr  = np.zeros((n_genes, n_genes), dtype=np.float32)
    mu    = X_rank.mean(axis=0)
    std   = X_rank.std(axis=0) + 1e-10
    Xz    = (X_rank - mu) / std / np.sqrt(n_cells)

    for i in range(0, n_genes, batch):
        end = min(i + batch, n_genes)
        corr[i:end, :] = Xz[:, i:end].T @ Xz

    np.fill_diagonal(corr, 0)
    del X_rank, Xz
    gc.collect()

    top_k = params["top_k_edges"]
    min_c = params["min_corr"]
    edges = []
    for i in range(n_genes):
        row = np.abs(corr[i])
        row[i] = 0
        row[row < min_c] = 0
        top = np.argsort(row)[::-1][:top_k]
        for j in top:
            if row[j] > 0 and i < j:
                edges.append((i, j))

    print(f"  Network: {n_genes} genes, {len(edges)} edges")

    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(n_genes))
    G.add_edges_from(edges)
    communities = nx.community.greedy_modularity_communities(
        G, resolution=params["leiden_resolution"]
    )

    min_size = params["min_module_size"]
    modules  = {}
    for i, comm in enumerate(communities):
        genes = list(comm)
        if len(genes) >= min_size:
            modules[f"M{i}"] = genes

    print(f"  Modules detected: {len(modules)}")
    return modules, list(adata.var_names)


# ============================================================
# Per-cell metrics
# ============================================================

def compute_per_cell_metrics(adata, modules, time_col):
    """
    Compute Pathway Entropy and Network Dispersion per cell.

    Pathway Entropy:
      Per-cell Shannon entropy of module activation distribution.
      High entropy = diffuse activation across modules (stem-like).

    Network Dispersion:
      Per-timepoint median cosine distance to population centroid.
      High dispersion = heterogeneous cell states (early/plastic).

    Returns:
      pd.DataFrame with columns: timepoint, pathway_entropy, network_dispersion
    """
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()

    n_cells  = X.shape[0]
    mod_keys = list(modules.keys())
    n_mods   = len(mod_keys)

    # Module activation matrix (cells x modules)
    A = np.zeros((n_cells, n_mods), dtype=np.float32)
    for k, mod_id in enumerate(mod_keys):
        A[:, k] = X[:, modules[mod_id]].mean(axis=1)

    # Pathway entropy: Shannon entropy of |A| / row_sum
    A_abs   = np.abs(A)
    row_sum = A_abs.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    P       = np.clip(A_abs / row_sum, 1e-10, 1)
    entropy = -(P * np.log2(P)).sum(axis=1)

    # Network dispersion: median cosine distance to timepoint centroid
    from sklearn.metrics.pairwise import cosine_distances
    dispersion = np.zeros(n_cells)
    for tp in adata.obs[time_col].unique():
        mask = (adata.obs[time_col] == tp).values
        idx  = np.where(mask)[0]
        if len(idx) < 2:
            continue
        A_tp     = A[idx]
        centroid = A_tp.mean(axis=0, keepdims=True)
        dists    = cosine_distances(A_tp, centroid).ravel()
        dispersion[idx] = np.median(dists)

    return pd.DataFrame({
        "timepoint"         : adata.obs[time_col].values,
        "pathway_entropy"   : entropy,
        "network_dispersion": dispersion,
    }, index=adata.obs_names)


def compute_composite(pc_df):
    """Rank-based composite score = (rank_entropy + rank_dispersion) / 2."""
    n      = len(pc_df)
    r_ent  = pc_df["pathway_entropy"].rank()    / n
    r_disp = pc_df["network_dispersion"].rank() / n
    return (r_ent + r_disp) / 2


# ============================================================
# Evaluation metrics
# ============================================================

def compute_auc(pc_df, early_tp, late_tp, metric):
    """AUC for binary early vs late separation. Returns max(auc, 1-auc)."""
    mask = pc_df["timepoint"].isin([early_tp, late_tp])
    sub  = pc_df[mask]
    if len(sub) < 10:
        return np.nan
    y_true  = (sub["timepoint"] == early_tp).astype(int).values
    y_score = sub[metric].values
    try:
        auc = roc_auc_score(y_true, y_score)
        return max(auc, 1 - auc)
    except Exception:
        return np.nan


def compute_marker_corr(adata, pc_df, stem_markers, diff_markers):
    """
    Spearman correlation between pathway_entropy and stem/differentiation
    marker gene expression scores.
    """
    results = {}
    if adata.raw is not None:
        gene_names = list(adata.raw.var_names)
        get_expr   = lambda g: np.array(
            adata.raw[:, g].X.toarray() if sp.issparse(adata.raw[:, g].X)
            else adata.raw[:, g].X
        ).ravel()
    else:
        gene_names = list(adata.var_names)
        get_expr   = lambda g: np.array(
            adata[:, g].X.toarray() if sp.issparse(adata[:, g].X)
            else adata[:, g].X
        ).ravel()

    for label, markers in [("stem", stem_markers), ("diff", diff_markers)]:
        available = [g for g in markers if g in gene_names]
        if not available:
            results[f"corr_{label}"] = np.nan
            results[f"p_{label}"]    = np.nan
            continue
        score = np.mean([get_expr(g) for g in available], axis=0)
        r, p  = spearmanr(pc_df["pathway_entropy"], score)
        results[f"corr_{label}"] = r
        results[f"p_{label}"]    = p
        print(f"    entropy vs {label}: rho={r:.3f}, p={p:.2e}  (genes: {available})")

    return results


def shuffle_time_null(adata, modules, time_col, n_iter=50, random_state=0):
    """
    Permutation null test for trajectory monotonicity.
    Shuffles timepoint labels and measures sign changes in mean entropy trajectory.

    Returns:
      real_sign_changes -- int
      null_list         -- list of sign change counts under null
      p_value           -- fraction of null >= real
    """
    rng = np.random.default_rng(random_state)

    def sign_changes(arr):
        diffs = np.diff(arr)
        return int(np.sum(diffs[:-1] * diffs[1:] < 0))

    pc_real    = compute_per_cell_metrics(adata, modules, time_col)
    tps_sorted = sorted(
        adata.obs[time_col].unique(),
        key=lambda x: float("".join(c for c in str(x) if c.isdigit() or c == ".") or "0")
    )
    real_means = [pc_real[pc_real["timepoint"] == tp]["pathway_entropy"].mean()
                  for tp in tps_sorted]
    real_sc    = sign_changes(real_means)

    null_scs  = []
    adata_sh  = adata.copy()
    for _ in range(n_iter):
        adata_sh.obs[time_col] = rng.permutation(adata.obs[time_col].values)
        pc_null    = compute_per_cell_metrics(adata_sh, modules, time_col)
        null_means = [pc_null[pc_null["timepoint"] == tp]["pathway_entropy"].mean()
                      for tp in tps_sorted]
        null_scs.append(sign_changes(null_means))

    p_val = np.mean(np.array(null_scs) >= real_sc)
    print(f"    real sign changes={real_sc}, null mean={np.mean(null_scs):.1f}"
          f"+/-{np.std(null_scs):.1f}, p={p_val:.3f}")
    return real_sc, null_scs, p_val


# ============================================================
# Visualization
# ============================================================

def plot_dataset(name, pc_df, cfg, out_dir):
    """Trajectory plot: entropy / dispersion / composite across timepoints."""
    time_col = cfg["time_col"]
    tps = sorted(
        pc_df["timepoint"].unique(),
        key=lambda x: float("".join(c for c in str(x) if c.isdigit() or c == ".") or "0")
    )
    means = pc_df.groupby("timepoint")[
        ["pathway_entropy", "network_dispersion", "composite"]
    ].mean().reindex(tps)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, col, color, title in zip(
        axes,
        ["pathway_entropy", "network_dispersion", "composite"],
        ["#378ADD", "#1D9E75", "#7F77DD"],
        ["Pathway Entropy", "Network Dispersion", "Composite (rank-based)"],
    ):
        ax.plot(range(len(tps)), means[col], marker="o", color=color, linewidth=2)
        ax.set_xticks(range(len(tps)))
        ax.set_xticklabels(tps, rotation=45, ha="right")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Timepoint")
        ax.grid(alpha=0.3)

    early_x = tps.index(cfg["early_tp"]) if cfg["early_tp"] in tps else None
    late_x  = tps.index(cfg["late_tp"])  if cfg["late_tp"]  in tps else None
    for ax in axes:
        if early_x is not None:
            ax.axvline(early_x, color="blue",  linestyle="--", alpha=0.4, label="early")
        if late_x is not None:
            ax.axvline(late_x,  color="red",   linestyle="--", alpha=0.4, label="late")
        ax.legend(fontsize=8)

    fig.suptitle(f"SAPPHIRE -- {name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fp = Path(out_dir) / f"{name}_sapphire_metrics.pdf"
    plt.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {fp.name}")


print("sapphire_core.py loaded successfully")
