"""
Unsupervised Dispersion via NMF + Silhouette
=============================================
Replaces timepoint-based centroid with NMF-derived clusters.
Computes both:
  - per-cell dispersion  : each cell's cosine distance to its cluster centroid
  - population dispersion: median per-cell dispersion within each NMF cluster
                           (assigned back to cells as a scalar, label-free equivalent
                            of the original timepoint-based dispersion)

Compares AUC against:
  - original timepoint-based population dispersion
  - original per-cell dispersion (simple distance to timepoint centroid)
  - pathway entropy (unchanged)
  - composite scores (rank average of entropy + each dispersion variant)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import NMF
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cosine


# ─────────────────────────────────────────────
# CONFIG  (edit paths to match your environment)
# ─────────────────────────────────────────────
VAL_DIR = Path("/Users/ziye/Documents/paper/data/sapphire_validation_v2")
OUT_DIR = Path("/Users/ziye/Documents/paper/data/unsupervised_dispersion")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["Cardiomyocyte", "Endoderm", "Kidney", "Neuro"]

EARLY_LATE = {
    "Cardiomyocyte": ("D0",    "D30"),
    "Endoderm":      ("00h",   "96h"),
    "Kidney":        ("Day7",  "Day26"),
    "Neuro":         ("D11",   "D52"),
}

K_RANGE = range(3, 9)   # silhouette search: k = 3 … 8; capped at n_modules at runtime
NMF_INIT = "nndsvda"
RANDOM_STATE = 0


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def cosine_dist(a, b):
    """Cosine distance between two 1-D vectors."""
    return cosine(a, b)


def rank_normalize(series):
    """Min-max rank normalisation to [0, 1]."""
    r = series.rank()
    return (r - r.min()) / (r.max() - r.min())


def compute_auc(scores, labels, early_label, late_label):
    """
    Binary AUC: early = 0, late = 1.
    Higher score → later timepoint (more differentiated).
    If AUC < 0.5, flip (handles both directions).
    """
    mask = labels.isin([early_label, late_label])
    y = (labels[mask] == late_label).astype(int)
    s = scores[mask]
    if s.std() == 0:
        return np.nan
    auc = roc_auc_score(y, s)
    return max(auc, 1 - auc)


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

results = []

for dataset in DATASETS:
    print(f"\n{'='*55}")
    print(f"  {dataset}")
    print(f"{'='*55}")

    early, late = EARLY_LATE[dataset]

    # ── Load per-cell metrics from existing pipeline output ──
    csv_path = VAL_DIR / dataset / f"{dataset}_per_cell_metrics.csv"
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path} not found")
        continue

    df = pd.read_csv(csv_path)
    # columns: timepoint, pathway_entropy, network_dispersion, composite

    # ── Build module activation matrix ──────────────────────
    # The activation matrix lives in the pipeline; here we approximate
    # it by loading if available, otherwise reconstruct from metrics.
    # Preferred: load the actual activation matrix if you saved it.
    act_path = VAL_DIR / dataset / f"{dataset}_module_activation.csv"
    if act_path.exists():
        act_df = pd.read_csv(act_path, index_col=0)
        A = act_df.values.astype(float)          # shape (n_cells, n_modules)
        print(f"  Loaded activation matrix: {A.shape}")
    else:
        print(f"  [WARN] Activation matrix not found at {act_path}")
        print(f"         Falling back to entropy + dispersion columns only.")
        print(f"         For full NMF clustering, save the activation matrix")
        print(f"         from sapphire_core.py run_pipeline().")
        # Demonstrate pipeline structure without real data
        A = None

    if A is None:
        print(f"  Skipping NMF step for {dataset} (no activation matrix).")
        continue

    n_cells, n_modules = A.shape

    # ── Silhouette search for best k ─────────────────────────
    # Normalise rows so cosine distance = euclidean on unit sphere
    A_norm = normalize(A, norm="l2")

    best_k, best_sil = 2, -1
    sil_scores = {}
    for k in K_RANGE:
        if k >= n_cells:
            break
        if k > n_modules:   # NMF requires k <= n_features (n_modules)
            break
        model = NMF(n_components=k, init=NMF_INIT,
                    random_state=RANDOM_STATE, max_iter=500)
        W = model.fit_transform(A)          # (n_cells, k)  – soft assignments
        labels_nmf = W.argmax(axis=1)       # hard cluster assignment

        # Need ≥ 2 clusters actually populated
        if len(np.unique(labels_nmf)) < 2:
            continue

        sil = silhouette_score(A_norm, labels_nmf, metric="cosine")
        sil_scores[k] = round(sil, 4)
        print(f"  k={k}  silhouette={sil:.4f}")
        if sil > best_sil:
            best_sil, best_k = sil, k

    print(f"  → Best k = {best_k}  (silhouette = {best_sil:.4f})")

    # ── Fit final NMF with best k ────────────────────────────
    model_final = NMF(n_components=best_k, init=NMF_INIT,
                      random_state=RANDOM_STATE, max_iter=500)
    W_final = model_final.fit_transform(A)           # (n_cells, best_k)
    cluster_labels = W_final.argmax(axis=1)          # hard assignment

    # Cluster centroids in original (un-normalised) space
    centroids = np.array([
        A[cluster_labels == c].mean(axis=0)
        for c in range(best_k)
    ])                                               # (best_k, n_modules)

    # ── Per-cell dispersion (label-free) ─────────────────────
    # Each cell's cosine distance to its own cluster centroid
    percell_disp_nmf = np.array([
        cosine_dist(A[i], centroids[cluster_labels[i]])
        for i in range(n_cells)
    ])

    # ── Population dispersion (label-free) ───────────────────
    # Median per-cell dispersion within each NMF cluster,
    # then assign that median back to every cell in the cluster.
    pop_disp_nmf = np.zeros(n_cells)
    for c in range(best_k):
        mask = cluster_labels == c
        median_d = np.median(percell_disp_nmf[mask])
        pop_disp_nmf[mask] = median_d

    # ── Attach to dataframe ──────────────────────────────────
    df["nmf_cluster"]          = cluster_labels
    df["percell_disp_nmf"]     = percell_disp_nmf
    df["pop_disp_nmf"]         = pop_disp_nmf

    # Original timepoint-based dispersion is already in df["network_dispersion"]

    # ── Composite scores ─────────────────────────────────────
    # rank average of entropy + each dispersion variant
    df["composite_nmf_percell"] = (
        rank_normalize(df["pathway_entropy"]) +
        rank_normalize(df["percell_disp_nmf"])
    ) / 2

    df["composite_nmf_pop"] = (
        rank_normalize(df["pathway_entropy"]) +
        rank_normalize(df["pop_disp_nmf"])
    ) / 2

    # Original composite (timepoint-based) already in df["composite"]

    # ── AUC comparison ───────────────────────────────────────
    tp = df["timepoint"]

    auc_results = {
        "dataset":                dataset,
        "best_k":                 best_k,
        "best_silhouette":        round(best_sil, 4),
        "n_timepoints":           df["timepoint"].nunique(),

        # Entropy (unchanged, truly label-free)
        "entropy_auc":            round(compute_auc(df["pathway_entropy"],        tp, early, late), 4),

        # Original timepoint-based dispersion (uses label → semi-supervised)
        "orig_pop_disp_auc":      round(compute_auc(df["network_dispersion"],     tp, early, late), 4),
        "orig_composite_auc":     round(compute_auc(df["composite"],              tp, early, late), 4),

        # NMF per-cell dispersion (fully label-free)
        "nmf_percell_disp_auc":   round(compute_auc(df["percell_disp_nmf"],       tp, early, late), 4),
        "nmf_percell_composite":  round(compute_auc(df["composite_nmf_percell"],  tp, early, late), 4),

        # NMF population dispersion (fully label-free)
        "nmf_pop_disp_auc":       round(compute_auc(df["pop_disp_nmf"],           tp, early, late), 4),
        "nmf_pop_composite":      round(compute_auc(df["composite_nmf_pop"],      tp, early, late), 4),
    }

    results.append(auc_results)

    # Save per-cell results
    df.to_csv(OUT_DIR / f"{dataset}_unsupervised_dispersion.csv", index=False)

    # Print summary
    print(f"\n  AUC Summary:")
    print(f"  {'Metric':<35} {'AUC':>6}")
    print(f"  {'-'*42}")
    for k, v in auc_results.items():
        if "_auc" in k or "composite" in k:
            label = k.replace("_auc", "").replace("_", " ")
            print(f"  {label:<35} {v:>6.4f}")

# ─────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────
if results:
    summary = pd.DataFrame(results)
    summary.to_csv(OUT_DIR / "unsupervised_dispersion_summary.csv", index=False)

    print("\n\n" + "="*70)
    print("FULL AUC COMPARISON TABLE")
    print("="*70)
    cols = [
        "dataset", "best_k",
        "entropy_auc",
        "orig_pop_disp_auc",    "orig_composite_auc",
        "nmf_percell_disp_auc", "nmf_percell_composite",
        "nmf_pop_disp_auc",     "nmf_pop_composite",
    ]
    print(summary[cols].to_string(index=False))
    print(f"\nResults saved to: {OUT_DIR}")
else:
    print("\n[!] No results — check that activation matrices are saved.")
    print("    Add this to run_pipeline() in sapphire_core.py:")
    print()
    print("    act_df = pd.DataFrame(")
    print("        module_activation_matrix,   # shape (n_cells, n_modules)")
    print("        index=cell_barcodes,")
    print("        columns=[f'M{i}' for i in range(n_modules)]")
    print("    )")
    print("    act_df.to_csv(")
    print("        VAL_DIR / dataset / f'{dataset}_module_activation.csv'")
    print("    )")
