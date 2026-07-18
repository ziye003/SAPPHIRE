"""
export_for_r.py
================
Exports, for each SAPPHIRE benchmark dataset, the inputs that the official
SCENT and SLICE R packages need:

  - SCENT (DoIntegPPI / CompSRana) expects a LINEAR-SCALE, non-negative
    genes x cells expression matrix (it internally does the log/normalisation
    assumptions documented in the SCENT vignette; do not feed it already
    z-scored data).
  - SLICE (construct / getEntropy) expects a genes x cells expression matrix,
    typically log2(FPKM/CPM+1)-like values (see xu-lab/SLICE demo/FB.R), plus
    a cell identity vector (timepoint label is fine here since SLICE only
    uses it for downstream plotting/lineage steps, NOT for entropy itself).

This script does NOT compute anything — it only exports CSVs, so that
run_scent.R and run_slice.R can each be run with a real R installation of
the official packages on your machine (this sandbox has no internet access
and cannot install R/Bioconductor packages or fetch the bundled PPI network
/ kappa gene-set similarity matrix).

Usage (run from the directory containing sapphire_core.py / DATA_ROOT setup,
i.e. exec this after sapphire_validation_all.py as with the other scripts):

    exec(open("scripts/comparison/export_for_r.py").read())
    export_all()
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path

# Variables inherited from sapphire_core.py / sapphire_validation_all.py:
# DATA_ROOT, DATASETS_CONFIG, load_and_prepare

EXPORT_DIR = DATA_ROOT / "scent_slice_export"
EXPORT_DIR.mkdir(exist_ok=True, parents=True)

TARGET_DATASETS = ["Cardiomyocyte", "Endoderm", "Kidney", "Neuro"]


def _to_linear(X):
    """SAPPHIRE stores log1p-normalised data (see Methods 3.1). SCENT and
    SLICE both expect (or assume) a roughly linear-scale expression matrix
    as their starting point, so we invert the log1p transform here.
    If your adata.X is already linear-scale for some dataset, set
    already_log1p=False in export_one() for that dataset."""
    return np.expm1(X)


def export_one(name, cfg, already_log1p=True, max_cells=None):
    print(f"Exporting {name} ...")
    adata = load_and_prepare(name, cfg)

    if max_cells is not None and adata.n_obs > max_cells:
        idx = np.random.RandomState(0).choice(adata.n_obs, max_cells, replace=False)
        adata = adata[idx].copy()

    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)

    if already_log1p:
        X_linear = _to_linear(X)
    else:
        X_linear = X

    genes = adata.var_names.astype(str)
    cells = adata.obs_names.astype(str)
    time_col = cfg["time_col"]
    labels = adata.obs[time_col].astype(str).values

    out_dir = EXPORT_DIR / name
    out_dir.mkdir(exist_ok=True, parents=True)

    # genes x cells orientation, as both SCENT and SLICE expect
    expr_df = pd.DataFrame(X_linear.T, index=genes, columns=cells)
    expr_df.to_csv(out_dir / f"{name}_expr_linear_genes_x_cells.csv")

    meta_df = pd.DataFrame({"cell": cells, "timepoint": labels})
    meta_df.to_csv(out_dir / f"{name}_metadata.csv", index=False)

    print(f"  {expr_df.shape[0]} genes x {expr_df.shape[1]} cells -> {out_dir}")
    return out_dir


def export_all(max_cells=8000):
    """max_cells caps per-dataset cell count for export: SCENT's CompSRana
    (exact random-walk entropy-rate) and SLICE's bootstrap entropy are both
    far slower than SAPPHIRE's closed-form metrics. 8000 cells/dataset is a
    reasonable starting point; raise it if runtime allows. Cardiomyocyte and
    Neuro (n=29,998 full) are downsampled; Endoderm (n=758) and Kidney
    (n=5,244) are used in full when below max_cells."""
    for name in TARGET_DATASETS:
        export_one(name, DATASETS_CONFIG[name], max_cells=max_cells)
    print(f"\nAll exports written under {EXPORT_DIR}")
    print("Next: run run_scent.R and run_slice.R (see SETUP_SCENT_SLICE.md) "
          "on each exported dataset, then run merge_scent_slice_results.py")
