"""
gene_umap_B.py
==============
Gene UMAP (Method B):
  - Transpose expression matrix (genes x cells), then PCA -> UMAP on genes
  - Each point = one gene, color = module assignment
  - One plot per dataset + a 2x2 overview figure

Usage:
    python gene_umap_B.py

Output (data/umap/):
    gene_umap_B_ALL.png
    {Dataset}_gene_umap_B.png
"""

import os, sys, gc, warnings
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as ssp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
warnings.filterwarnings("ignore")

_here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."
exec(open(os.path.join(_here, "sapphire_core.py")).read(), globals())

try:
    import umap as umap_lib
except ImportError:
    print("[ERROR] Please install: conda install -c conda-forge umap-learn")
    raise

OUT_DIR = os.path.join(str(DATA_ROOT), "umap")
os.makedirs(OUT_DIR, exist_ok=True)
sc.settings.verbosity = 0

MODULE_COLORS = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261",
    "#6A0572", "#1D7874", "#C77DFF", "#4CC9F0", "#F77F00",
    "#90BE6D", "#577590", "#F94144", "#43AA8B", "#277DA1",
]

def get_module_color(mod_id, mod_list):
    idx = mod_list.index(mod_id) if mod_id in mod_list else 0
    return MODULE_COLORS[idx % len(MODULE_COLORS)]


def compute_gene_umap_B(adata):
    """
    Transpose expression matrix -> genes x cells -> PCA (50 PCs) -> UMAP.
    """
    X = adata.X
    if ssp.issparse(X):
        X = X.toarray()

    n_cells, n_genes = X.shape
    print(f"  Transposing ({n_genes} genes x {n_cells} cells)...")

    # Standardize genes (zero mean, unit variance across cells)
    mu  = X.mean(axis=0)
    std = X.std(axis=0) + 1e-10
    Xz  = ((X - mu) / std).T   # shape: (n_genes, n_cells)

    # PCA via scanpy (stable, efficient)
    print(f"  PCA on gene x cell matrix...")
    adata_gene = sc.AnnData(X=Xz)
    n_comps = min(50, n_genes - 1, n_cells - 1)
    sc.tl.pca(adata_gene, n_comps=n_comps, random_state=42)

    print(f"  UMAP...")
    sc.pp.neighbors(adata_gene, n_neighbors=min(15, n_genes - 1),
                    n_pcs=n_comps, random_state=42)
    sc.tl.umap(adata_gene, random_state=42, min_dist=0.3)

    embedding = adata_gene.obsm["X_umap"]   # (n_genes, 2)
    del adata_gene, Xz; gc.collect()
    return embedding


def plot_gene_umap(embedding, gene_names, modules, dataset_name,
                   ax=None, standalone=False, suffix=""):
    mod_list = sorted(modules.keys())

    gene_to_mod = {}
    for mod_id, gene_indices in modules.items():
        for gi in gene_indices:
            if gi < len(gene_names):
                gene_to_mod[gene_names[gi]] = mod_id

    colors = [
        get_module_color(gene_to_mod[g], mod_list) if g in gene_to_mod else "#CCCCCC"
        for g in gene_names
    ]

    if standalone:
        fig, ax = plt.subplots(figsize=(9, 8))
        fig.suptitle(f"{dataset_name} — Gene UMAP (method B, colored by module)",
                     fontsize=13, fontweight="bold")

    gray_mask = np.array([c == "#CCCCCC" for c in colors])
    ax.scatter(embedding[gray_mask, 0], embedding[gray_mask, 1],
               c="#CCCCCC", s=6, alpha=0.35, linewidths=0,
               rasterized=True, zorder=1)

    for mod_id in mod_list:
        color = get_module_color(mod_id, mod_list)
        mask  = np.array([gene_to_mod.get(g) == mod_id for g in gene_names])
        if mask.sum() == 0:
            continue
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=color, s=12, alpha=0.85, linewidths=0,
                   rasterized=True, zorder=2, label=mod_id)

    ax.set_title(dataset_name, fontsize=11, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(False)

    handles = [
        mpatches.Patch(color=get_module_color(m, mod_list), label=m)
        for m in mod_list
    ] + [mpatches.Patch(color="#CCCCCC", label="unassigned")]
    ax.legend(handles=handles, title="Module", fontsize=6.5,
              title_fontsize=7, loc="best", framealpha=0.7,
              ncol=2 if len(mod_list) > 8 else 1)

    if standalone:
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f"{dataset_name}_gene_umap_B.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved -> {out_path}")


# ════════════════════════════════════════════════════════════════
TARGET = ["Cardiomyocyte", "Endoderm", "Kidney", "Neuro"]

fig_all, axes_all = plt.subplots(2, 2, figsize=(16, 14))
fig_all.suptitle("SAPPHIRE — Gene UMAP (Method B: transposed expression)",
                 fontsize=14, fontweight="bold", y=1.01)
axes_all = axes_all.ravel()

for ax_all, ds_name in zip(axes_all, TARGET):
    cfg    = DATASETS_CONFIG[ds_name]
    params = {**SAPPHIRE_PARAMS, **cfg.get("param_overrides", {})}
    print(f"\n{'='*55}\n  {ds_name}\n{'='*55}")

    adata = load_and_prepare(ds_name, cfg)
    if adata.n_vars > params["n_top_genes"]:
        adata = hvg_filter(adata, params["n_top_genes"])

    modules, gene_names = build_network(adata, params)
    print(f"  Modules: {len(modules)}  |  Genes: {len(gene_names)}")

    embedding = compute_gene_umap_B(adata)

    plot_gene_umap(embedding, gene_names, modules, ds_name,
                   standalone=True, suffix="_B")
    plot_gene_umap(embedding, gene_names, modules, ds_name,
                   ax=ax_all, standalone=False)

    del adata, modules, embedding; gc.collect()

plt.tight_layout()
out_all = os.path.join(OUT_DIR, "gene_umap_B_ALL.png")
fig_all.savefig(out_all, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved overview -> {out_all}")
print("\nDone!\n")
