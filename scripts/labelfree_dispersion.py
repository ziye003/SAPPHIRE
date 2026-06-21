"""
labelfree_dispersion.py
=======================
Three fully label-free per-cell dispersion measures, compared against
the original timepoint-based Network Dispersion.

Methods
-------
1. kNN_local   : cosine distance from cell to centroid of its k nearest neighbors
2. kNN_density : mean cosine distance to k nearest neighbors (local sparsity)
3. NMF_soft    : cosine distance from cell to its NMF soft-weighted centroid

All three are per-cell and require no timepoint labels at any stage.

Usage
-----
    python labelfree_dispersion.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_distances
from sklearn.decomposition import NMF
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
import os
# Override with: export SAPPHIRE_DATA_ROOT=/path/to/your/data
_DATA_ROOT = Path(os.environ.get(
    "SAPPHIRE_DATA_ROOT",
    Path(__file__).resolve().parent.parent / "data"
))
VAL_DIR = _DATA_ROOT / "sapphire_validation_v2"
OUT_DIR = _DATA_ROOT / "labelfree_dispersion"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["Cardiomyocyte", "Endoderm", "Kidney", "Neuro"]

EARLY_LATE = {
    "Cardiomyocyte": ("D0",    "D30"),
    "Endoderm":      ("00h",   "96h"),
    "Kidney":        ("Day7",  "Day26"),
    "Neuro":         ("D11",   "D52"),
}

KNN_K       = 30    # neighbors for kNN methods; try 15/30/50 if needed
NMF_K       = 2     # components for NMF soft method (best k from previous run)
NMF_INIT    = "nndsvda"
RANDOM_STATE = 0


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def rank_normalize(s):
    r = pd.Series(np.array(s).ravel()).rank()
    return (r - r.min()) / (r.max() - r.min())


def compute_auc(scores, labels, early, late):
    mask = labels.isin([early, late])
    s = np.array(scores).ravel()
    y = (labels[mask] == late).astype(int).values
    s_masked = s[mask.values]
    if s_masked.std() == 0:
        return np.nan
    auc = roc_auc_score(y, s_masked)
    return max(auc, 1 - auc)


# ─────────────────────────────────────────────
# METHOD 1: kNN local centroid
# ─────────────────────────────────────────────

def knn_local_dispersion(A, k=KNN_K):
    """
    For each cell, find its k nearest neighbors in cosine space,
    compute their centroid, return cosine distance from cell to centroid.
    High value = cell sits far from its local neighborhood center = heterogeneous.
    """
    A_norm = normalize(A, norm="l2")
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
    nn.fit(A_norm)
    _, indices = nn.kneighbors(A_norm)   # shape (n_cells, k+1); col 0 = self

    disp = np.zeros(len(A))
    for i in range(len(A)):
        neighbor_idx = indices[i, 1:]    # exclude self
        centroid = A[neighbor_idx].mean(axis=0)
        d = cosine_distances(A[i:i+1], centroid.reshape(1, -1))[0, 0]
        disp[i] = d
    return disp


# ─────────────────────────────────────────────
# METHOD 2: kNN density (local sparsity)
# ─────────────────────────────────────────────

def knn_density_dispersion(A, k=KNN_K):
    """
    Mean cosine distance to k nearest neighbors.
    High value = cell is in a sparse region = locally heterogeneous/unusual.
    """
    A_norm = normalize(A, norm="l2")
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
    nn.fit(A_norm)
    distances, _ = nn.kneighbors(A_norm)  # shape (n_cells, k+1)
    return distances[:, 1:].mean(axis=1)  # exclude self (col 0 = 0)


# ─────────────────────────────────────────────
# METHOD 3: NMF soft weighted centroid
# ─────────────────────────────────────────────

def nmf_soft_dispersion(A, k=NMF_K):
    """
    Fit NMF with k components. Each cell has a soft weight vector W[i].
    Weighted centroid = sum_j(W[i,j] * basis_j) / sum(W[i])  — this is just
    the NMF reconstruction of cell i: W[i] @ H.
    Dispersion = cosine distance between original A[i] and its NMF reconstruction.
    High value = cell is poorly explained by smooth NMF basis = transcriptionally unusual.
    """
    # Cap k at n_features
    k = min(k, A.shape[1])
    model = NMF(n_components=k, init=NMF_INIT, random_state=RANDOM_STATE,
                max_iter=1000)
    W = model.fit_transform(A)    # (n_cells, k)
    H = model.components_          # (k, n_modules)
    A_recon = W @ H                # NMF reconstruction

    disp = np.array([
        cosine_distances(A[i:i+1], A_recon[i:i+1])[0, 0]
        for i in range(len(A))
    ])
    return disp


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

results = []

for dataset in DATASETS:
    print(f"\n{'='*55}")
    print(f"  {dataset}")
    print(f"{'='*55}")

    early, late = EARLY_LATE[dataset]

    # Load per-cell metrics (has original dispersion + entropy)
    csv_path = VAL_DIR / dataset / f"{dataset}_per_cell_metrics.csv"
    act_path = VAL_DIR / dataset / f"{dataset}_module_activation.csv"

    if not csv_path.exists() or not act_path.exists():
        print(f"  [SKIP] Missing files")
        continue

    df  = pd.read_csv(csv_path, index_col=0)
    act = pd.read_csv(act_path, index_col=0)
    A   = act.values.astype(float)
    tp  = df["timepoint"]

    print(f"  Activation matrix: {A.shape}")

    # ── Compute three label-free dispersion measures ──────────
    print(f"  Method 1: kNN local centroid  (k={KNN_K})...")
    d1 = knn_local_dispersion(A)

    print(f"  Method 2: kNN density         (k={KNN_K})...")
    d2 = knn_density_dispersion(A)

    print(f"  Method 3: NMF soft reconstruction (k={NMF_K})...")
    d3 = nmf_soft_dispersion(A)

    # ── Composite scores ──────────────────────────────────────
    ent = df["pathway_entropy"]

    comp_knn_local   = (rank_normalize(ent) + rank_normalize(d1)) / 2
    comp_knn_density = (rank_normalize(ent) + rank_normalize(d2)) / 2
    comp_nmf_soft    = (rank_normalize(ent) + rank_normalize(d3)) / 2

    # ── AUC ───────────────────────────────────────────────────
    row = {
        "dataset": dataset,

        # Baselines
        "entropy_auc":        round(compute_auc(ent,                        tp, early, late), 4),
        "orig_pop_disp_auc":  round(compute_auc(df["network_dispersion"],   tp, early, late), 4),
        "orig_composite_auc": round(compute_auc(df["composite"],            tp, early, late), 4),

        # Method 1: kNN local centroid
        "knn_local_auc":      round(compute_auc(pd.Series(d1), tp, early, late), 4),
        "knn_local_comp_auc": round(compute_auc(comp_knn_local, tp, early, late), 4),

        # Method 2: kNN density
        "knn_density_auc":      round(compute_auc(pd.Series(d2), tp, early, late), 4),
        "knn_density_comp_auc": round(compute_auc(comp_knn_density, tp, early, late), 4),

        # Method 3: NMF soft reconstruction
        "nmf_soft_auc":      round(compute_auc(pd.Series(d3), tp, early, late), 4),
        "nmf_soft_comp_auc": round(compute_auc(comp_nmf_soft, tp, early, late), 4),
    }
    results.append(row)

    # Save per-cell
    df["knn_local_disp"]   = d1
    df["knn_density_disp"] = d2
    df["nmf_soft_disp"]    = d3
    df.to_csv(OUT_DIR / f"{dataset}_labelfree_dispersion.csv")

    # Print summary
    print(f"\n  {'Metric':<28} {'AUC':>6}")
    print(f"  {'-'*35}")
    for k, v in row.items():
        if k == "dataset":
            continue
        print(f"  {k:<28} {v:>6.4f}")

# ─────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────
if results:
    df_out = pd.DataFrame(results)
    df_out.to_csv(OUT_DIR / "labelfree_dispersion_summary.csv", index=False)

    print("\n\n" + "="*75)
    print("FULL COMPARISON TABLE")
    print("="*75)

    # Dispersion AUC only
    disp_cols = ["dataset", "orig_pop_disp_auc",
                 "knn_local_auc", "knn_density_auc", "nmf_soft_auc"]
    print("\nDispersion AUC (per-cell, label-free vs original):")
    print(df_out[disp_cols].to_string(index=False))

    # Composite AUC
    comp_cols = ["dataset", "entropy_auc", "orig_composite_auc",
                 "knn_local_comp_auc", "knn_density_comp_auc", "nmf_soft_comp_auc"]
    print("\nComposite AUC (entropy + each dispersion):")
    print(df_out[comp_cols].to_string(index=False))

    # Mean AUC summary
    print("\nMean AUC across 4 datasets:")
    for col in disp_cols[1:] + comp_cols[2:]:
        print(f"  {col:<28} {df_out[col].mean():.4f}")
