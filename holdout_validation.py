"""
holdout_validation.py
=====================
Strict holdout-cell validation for SAPPHIRE.
 sapphire_core.py 


----
 dataset20 split timepoint 80/20

  Construction set (80%):
    1. hvg_filter_subset()  —  top N genes core.hvg_filter
    2. build_network_from_array()  — Spearman rank corr + greedy modularity
        core.build_network
    3. compute_centroids()  —  timepoint  centroid

  Holdout set (20%):
    4.  modules +  centroids 
       - Pathway Entropyper cell core.compute_per_cell_metrics
       - Network Dispersionper cell centroidmedian within-tp
       - Compositerank  core.compute_composite

normalize 
--------------
   sapphire_core.DATASETS_CONFIG  already_log1p 
  - False → normalize_total + log1p load_and_prepare 
  - True  →  normalizeNeuro
  


----
  holdout_{dataset}_raw.csv  —  split  dataset 
  holdout_all_splits.csv     —  dataset
  holdout_summary.csv        — mean ± std / min / max dataset
"""

import gc
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import networkx as nx
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import rankdata, spearmanr

warnings.filterwarnings("ignore")

# ──  exec()  resampling_stability.py────────────
import os
_here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."
if "load_and_prepare" not in dir():
    print("  →  sapphire_core.py...")
    exec(open(os.path.join(_here, "sapphire_core.py")).read(), globals())
#  exec(sapphire_core) DATASETS_CONFIG / SAPPHIRE_PARAMS  globals() 

N_SPLITS    = 20
TEST_SIZE   = 0.20
RANDOM_SEED = 42

OUTPUT_DIR = Path("/Users/ziye/Documents/paper/data/sapphire_validation_v2")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


# ──────────────────────────────────────────────────────────────────────────────
#  log1p 
# ──────────────────────────────────────────────────────────────────────────────

def check_log1p_status(adata, n_sample: int = 1000):
    """
     log1p
    
      - max_val < 20  frac_integer < 0.5  →  log1p
      - frac_integer > 0.8                  →  raw counts
      -                                 → 
    """
    X   = adata.X
    idx = np.random.choice(adata.n_obs, min(n_sample, adata.n_obs), replace=False)
    sample = X[idx]
    if sp.issparse(sample):
        sample = sample.toarray()

    max_val      = float(sample.max())
    frac_integer = float(np.mean(np.abs(sample - np.round(sample)) < 1e-6))

    if max_val < 20 and frac_integer < 0.5:
        verdict = "✅ likely log1p-normalized"
    elif frac_integer > 0.8:
        verdict = "⚠️  likely raw counts"
    else:
        verdict = "❓ uncertain — verify manually"

    print(f"    [log1p check] max={max_val:.2f}, "
          f"frac_integer={frac_integer:.2f}  →  {verdict}")


# ──────────────────────────────────────────────────────────────────────────────
# HVG  sapphire_core.hvg_filter
# ──────────────────────────────────────────────────────────────────────────────

def hvg_filter_subset(X_dense: np.ndarray, gene_names: list, n_top: int):
    """
     top n_top 
     sapphire_core.hvg_filter scanpy HVG pipeline

    :
        X_filt     — (n_cells, n_top) dense array
        genes_filt — list of gene names
        sel_idx    — np.ndarray of selected column indices holdout 
    """
    if X_dense.shape[1] <= n_top:
        return X_dense, gene_names, np.arange(X_dense.shape[1])

    var     = X_dense.var(axis=0)
    top_idx = np.argsort(var)[::-1][:n_top]
    return (
        X_dense[:, top_idx],
        [gene_names[i] for i in top_idx],
        top_idx,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  sapphire_core.build_network
# ──────────────────────────────────────────────────────────────────────────────

def build_network_from_array(X: np.ndarray, params: dict):
    """
    Spearman rank correlation → edge list → greedy modularity communities
     sapphire_core.build_network

    :
        X      — (n_cells, n_genes) dense float array HVG
        params — SAPPHIRE_PARAMS

    :
        modules — dict {module_id: [gene_col_indices]}
        n_edges — int
    """
    n_cells, n_genes = X.shape
    top_k  = params["top_k_edges"]
    min_c  = params["min_corr"]
    res    = params["leiden_resolution"]   # greedy modularity resolution
    min_ms = params["min_module_size"]

    # Spearman: rank transform per gene
    X_rank = np.zeros_like(X, dtype=np.float32)
    for j in range(n_genes):
        X_rank[:, j] = rankdata(X[:, j])

    #  Pearson on ranks = Spearman
    mu  = X_rank.mean(axis=0)
    std = X_rank.std(axis=0) + 1e-10
    Xz  = (X_rank - mu) / std / np.sqrt(n_cells)

    batch = 200
    corr  = np.zeros((n_genes, n_genes), dtype=np.float32)
    for i in range(0, n_genes, batch):
        end = min(i + batch, n_genes)
        corr[i:end, :] = Xz[:, i:end].T @ Xz

    np.fill_diagonal(corr, 0)
    del X_rank, Xz
    gc.collect()

    #  edge listtop_k  + min_corr 
    edges = []
    for i in range(n_genes):
        row       = np.abs(corr[i]).copy()
        row[i]    = 0
        row[row < min_c] = 0
        top_j     = np.argsort(row)[::-1][:top_k]
        for j in top_j:
            if row[j] > 0 and i < j:
                edges.append((i, j))

    G = nx.Graph()
    G.add_nodes_from(range(n_genes))
    G.add_edges_from(edges)

    communities = nx.community.greedy_modularity_communities(G, resolution=res)

    modules = {}
    for idx, comm in enumerate(communities):
        genes = list(comm)
        if len(genes) >= min_ms:
            modules[f"M{idx}"] = genes   # gene indices relative to filtered X

    return modules, len(edges)


# ──────────────────────────────────────────────────────────────────────────────
# Metric helpers sapphire_core.compute_per_cell_metrics
# ──────────────────────────────────────────────────────────────────────────────

def module_activation_matrix(X: np.ndarray, modules: dict) -> np.ndarray:
    """
    cells × modules 
     core: A[:, k] = X[:, module_genes].mean(axis=1)
    """
    mod_keys = list(modules.keys())
    A = np.zeros((X.shape[0], len(mod_keys)), dtype=np.float32)
    for k, mk in enumerate(mod_keys):
        idx = modules[mk]
        if len(idx) > 0:
            A[:, k] = X[:, idx].mean(axis=1)
    return A


def pathway_entropy_from_A(A: np.ndarray) -> np.ndarray:
    """
     coreP = |A| / row_sumShannon entropy (log2)
    """
    A_abs   = np.abs(A)
    row_sum = A_abs.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    P = np.clip(A_abs / row_sum, 1e-10, 1)
    return -(P * np.log2(P)).sum(axis=1)


def compute_centroids(A: np.ndarray, tp_labels: np.ndarray) -> dict:
    """
     construction set  timepoint  A  centroid
    : {timepoint: np.ndarray (n_modules,)}
    """
    return {
        tp: A[tp_labels == tp].mean(axis=0)
        for tp in np.unique(tp_labels)
    }


def dispersion_fixed_centroids(
    A: np.ndarray,
    tp_labels: np.ndarray,
    fixed_centroids: dict,
) -> np.ndarray:
    """
    Holdout dispersion cell  timepoint FIXED centroid  cosine distance
     core cell  timepoint  median 

     fixed centroid construction set
    holdout cells  centroid 
    """
    dispersion = np.zeros(A.shape[0])
    for tp in np.unique(tp_labels):
        if tp not in fixed_centroids:
            continue
        mask     = tp_labels == tp
        idx      = np.where(mask)[0]
        A_tp     = A[idx]
        centroid = fixed_centroids[tp].reshape(1, -1)
        dists    = cosine_distances(A_tp, centroid).ravel()
        dispersion[idx] = np.median(dists)    #  core
    return dispersion


# ──────────────────────────────────────────────────────────────────────────────
# AUC & Spearman sapphire_core.compute_auc
# ──────────────────────────────────────────────────────────────────────────────

def compute_auc(scores: np.ndarray, tp_labels: np.ndarray,
                early_tp: str, late_tp: str) -> float:
    """ coremax(auc, 1-auc) 10"""
    mask = np.isin(tp_labels, [early_tp, late_tp])
    if mask.sum() < 10:
        return np.nan
    y_true  = (tp_labels[mask] == early_tp).astype(int)
    y_score = scores[mask]
    try:
        auc = roc_auc_score(y_true, y_score)
        return max(auc, 1 - auc)
    except Exception:
        return np.nan


def compute_spearman_rho(scores: np.ndarray, tp_labels: np.ndarray,
                         tp_order: list) -> float:
    """Mean score per timepoint vs ordered index → Spearman ρ"""
    means   = {tp: scores[tp_labels == tp].mean() for tp in np.unique(tp_labels)}
    ordered = [tp for tp in tp_order if tp in means]
    if len(ordered) < 3:
        return np.nan
    rho, _ = spearmanr(np.arange(len(ordered)),
                       [means[tp] for tp in ordered])
    return rho


# ──────────────────────────────────────────────────────────────────────────────
#  split
# ──────────────────────────────────────────────────────────────────────────────

def run_one_split(X_full: np.ndarray, gene_names: list,
                  tp_labels: np.ndarray, cfg: dict,
                  params: dict, split_seed: int) -> dict:
    """
     80/20 holdout split

    X_full, gene_names, tp_labels —  adata normalize
    """
    early_tp = cfg["early_tp"]
    late_tp  = cfg["late_tp"]
    tp_order = sorted(
        np.unique(tp_labels),
        key=lambda x: float("".join(c for c in str(x) if c.isdigit() or c == ".") or "0")
    )

    # ── Stratified split ───────────────────────────────────────────────────────
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=split_seed
    )
    train_idx, holdout_idx = next(
        splitter.split(np.zeros(len(tp_labels)), tp_labels)
    )

    X_train, X_holdout = X_full[train_idx], X_full[holdout_idx]
    tp_train, tp_hold  = tp_labels[train_idx], tp_labels[holdout_idx]

    # ── Construction set ───────────────────────────────────────────────────────
    n_hvgs = params.get("n_top_genes", 2000)
    X_tr_filt, _, sel_idx = hvg_filter_subset(X_train, gene_names, n_hvgs)

    modules, n_edges = build_network_from_array(X_tr_filt, params)

    if not modules:
        return {"error": "no modules detected"}

    A_train         = module_activation_matrix(X_tr_filt, modules)
    fixed_centroids = compute_centroids(A_train, tp_train)

    # ── Holdout:  HVG  ────────────────────────────────────────────
    X_ho_filt = X_holdout[:, sel_idx]
    A_holdout = module_activation_matrix(X_ho_filt, modules)

    entropy    = pathway_entropy_from_A(A_holdout)
    dispersion = dispersion_fixed_centroids(A_holdout, tp_hold, fixed_centroids)

    # Compositerank  core.compute_composite
    n      = len(entropy)
    r_ent  = rankdata(entropy)   / n
    r_disp = rankdata(dispersion) / n
    composite = (r_ent + r_disp) / 2

    # ── Metrics ────────────────────────────────────────────────────────────────
    return {
        "entropy_auc"    : compute_auc(entropy,    tp_hold, early_tp, late_tp),
        "dispersion_auc" : compute_auc(dispersion, tp_hold, early_tp, late_tp),
        "composite_auc"  : compute_auc(composite,  tp_hold, early_tp, late_tp),
        "rho_entropy"    : compute_spearman_rho(entropy,    tp_hold, tp_order),
        "rho_dispersion" : compute_spearman_rho(dispersion, tp_hold, tp_order),
        "n_modules"      : len(modules),
        "n_edges"        : n_edges,
        "n_hvgs"         : len(sel_idx),
        "n_train"        : len(train_idx),
        "n_holdout"      : len(holdout_idx),
    }


# ──────────────────────────────────────────────────────────────────────────────
#  dataset  N_SPLITS 
# ──────────────────────────────────────────────────────────────────────────────

def run_dataset(name: str, cfg: dict, params: dict) -> pd.DataFrame:
    print(f"\n{'='*62}")
    print(f"  Dataset: {name}")
    print(f"{'='*62}")

    # load_and_prepare  already_log1p  normalize
    adata = load_and_prepare(name, cfg, max_cells=MAX_CELLS)
    check_log1p_status(adata)   # 

    # apply dataset-level param overrides EB  min_corr
    p = {**params, **cfg.get("param_overrides", {})}

    #  dense  split 
    X_full = adata.X
    if sp.issparse(X_full):
        X_full = X_full.toarray()
    gene_names = list(adata.var_names)
    tp_labels  = adata.obs[cfg["time_col"]].values

    rng   = np.random.default_rng(RANDOM_SEED)
    seeds = rng.integers(0, 100_000, size=N_SPLITS)

    results = []
    for i, seed in enumerate(seeds):
        print(f"  Split {i+1:2d}/{N_SPLITS}...", end="", flush=True)
        try:
            res = run_one_split(X_full, gene_names, tp_labels, cfg, p, int(seed))
            res.update({"split": i + 1, "dataset": name})
            results.append(res)

            if "error" in res:
                print(f"  ERROR: {res['error']}")
            else:
                print(
                    f"  ent={res['entropy_auc']:.3f}  "
                    f"disp={res['dispersion_auc']:.3f}  "
                    f"comp={res['composite_auc']:.3f}  "
                    f"mods={res['n_modules']}  edges={res['n_edges']}"
                )
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            results.append({"split": i + 1, "dataset": name, "error": str(e)})

    return pd.DataFrame(results)


# ──────────────────────────────────────────────────────────────────────────────
# Summary table
# ──────────────────────────────────────────────────────────────────────────────

def summarise(df_all: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "entropy_auc", "dispersion_auc", "composite_auc",
        "rho_entropy", "rho_dispersion",
        "n_modules", "n_edges",
    ]
    rows = []
    for ds, grp in df_all.groupby("dataset"):
        row = {"dataset": ds}
        for m in metrics:
            if m not in grp.columns:
                continue
            col = pd.to_numeric(grp[m], errors="coerce").dropna()
            row[f"{m}_mean"] = col.mean() if len(col) else np.nan
            row[f"{m}_std"]  = col.std()  if len(col) else np.nan
            row[f"{m}_min"]  = col.min()  if len(col) else np.nan
            row[f"{m}_max"]  = col.max()  if len(col) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def print_summary(summary: pd.DataFrame):
    print("\n" + "="*70)
    print("HOLDOUT VALIDATION SUMMARY  (mean ± std  [min, max])")
    print("="*70)
    for _, row in summary.iterrows():
        print(f"\nDataset: {row['dataset']}")
        for m in ["entropy_auc", "dispersion_auc", "composite_auc"]:
            v = row.get(f"{m}_mean", np.nan)
            if pd.isna(v):
                print(f"  {m:22s}  N/A")
                continue
            print(
                f"  {m:22s}  {v:.3f} ± {row[f'{m}_std']:.3f}"
                f"  [{row[f'{m}_min']:.3f}, {row[f'{m}_max']:.3f}]"
            )
        for m in ["rho_entropy", "rho_dispersion"]:
            v = row.get(f"{m}_mean", np.nan)
            if not pd.isna(v):
                print(f"  {m:22s}  {v:.3f} ± {row[f'{m}_std']:.3f}")
        v_mod = row.get("n_modules_mean", np.nan)
        if not pd.isna(v_mod):
            print(f"  {'n_modules (mean)':22s}  {v_mod:.1f}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # EB NaN corruption 4 
    datasets_to_run = [k for k in DATASETS_CONFIG if k != "EB"]

    all_results = []
    for name in datasets_to_run:
        cfg = DATASETS_CONFIG[name]
        if not cfg["file"].exists():
            print(f"[SKIP] {name}: file not found → {cfg['file']}")
            continue

        df_ds = run_dataset(name, cfg, SAPPHIRE_PARAMS)
        all_results.append(df_ds)

        #  dataset 
        out_path = OUTPUT_DIR / f"holdout_{name}_raw.csv"
        df_ds.to_csv(out_path, index=False)
        print(f"  → saved: {out_path.name}")

    if not all_results:
        print("No datasets processed.")
        return

    df_all = pd.concat(all_results, ignore_index=True)
    df_all.to_csv(OUTPUT_DIR / "holdout_all_splits.csv", index=False)

    summary = summarise(df_all)
    summary.to_csv(OUTPUT_DIR / "holdout_summary.csv", index=False)

    print_summary(summary)
    print(f"All outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
