"""
umap_with_module_overlay.py
============================
Generate three types of UMAP plots. Run as-is; no configuration needed.

Output (data/umap_overlay/):
  1. ALL_datasets_umap_timepoint.png       -- 4 datasets, timepoint, 2x2
  2. {Dataset}_umap_sapphire_scores.png    -- timepoint/entropy/dispersion/composite, 1 per dataset
  3. Cardiomyocyte_umap_module_overlay.png -- key module activation overlays (M4, etc.)
  4. ALL_umap_module_overlay.png           -- key module per dataset, 2x2 overview

Usage:
    cd SAPPHIRE/scripts
    python umap_with_module_overlay.py
"""

import os, sys, gc, re, warnings
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
warnings.filterwarnings("ignore")

# Load core
_here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."
exec(open(os.path.join(_here, "sapphire_core.py")).read(), globals())

VAL_DIR = os.path.join(str(DATA_ROOT), "sapphire_validation_v2")
OUT_DIR = os.path.join(str(DATA_ROOT), "umap_overlay")
os.makedirs(OUT_DIR, exist_ok=True)
sc.settings.verbosity = 0

TARGET = ["Cardiomyocyte", "Endoderm", "Kidney", "Neuro"]

# Key modules per dataset (based on GO enrichment results in the paper)
KEY_MODULES = {
    "Cardiomyocyte": {
        "M4": ("Cardiac Commitment\n(heart contraction, D2 peak)", "#E63946"),
        "M6": ("Cell Cycle\n(mitotic segregation)", "#457B9D"),
        "M2": ("Terminal Maturation\n(RNA splicing, D15-D30)", "#2A9D8F"),
    },
    "Endoderm": {
        "M4": ("Cell Cycle\n(chromosome segregation)", "#457B9D"),
        "M3": ("Stem Cell Diff.\n(neural crest)", "#E63946"),
        "M8": ("Wnt Signalling\n(gut morphogenesis)", "#2A9D8F"),
    },
    "Kidney": {
        "M0": ("Metal Detox\n(proximal tubule)", "#E63946"),
        "M2": ("ECM Organisation\n(collagen fibril)", "#2A9D8F"),
        "M5": ("Oxidative Phosph.\n(metabolic maturation)", "#F4A261"),
    },
    "Neuro": {
        "M2": ("Axonogenesis\n(neuron projection)", "#E63946"),
        "M4": ("Dopaminergic Spec.\n(catecholamine transport)", "#457B9D"),
        "M5": ("Cell Cycle\n(nuclear segregation)", "#2A9D8F"),
    },
}

# Utility functions

def sort_tp(tp):
    m = re.search(r"(\d+(?:\.\d+)?)", str(tp))
    return float(m.group(1)) if m else 0.0

def tp_palette(tps):
    cmap = plt.cm.Blues
    return {tp: cmap(0.3 + 0.6 * i / max(len(tps)-1, 1)) for i, tp in enumerate(tps)}

def get_umap(adata):
    if "X_umap" in adata.obsm:
        return adata
    print("    Computing UMAP...")
    if "X_pca" not in adata.obsm:
        sc.tl.pca(adata, n_comps=min(50, adata.n_vars-1), random_state=0)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=40, random_state=0)
    sc.tl.umap(adata, random_state=0)
    return adata

def scatter_ax(ax, x, y, c, cmap, vmin, vmax, s=3, alpha=0.6, title="", xlabel=True, cbar=True):
    sc_plot = ax.scatter(x, y, c=c, cmap=cmap, vmin=vmin, vmax=vmax,
                         s=s, alpha=alpha, linewidths=0, rasterized=True)
    if cbar:
        plt.colorbar(sc_plot, ax=ax, shrink=0.75, pad=0.02, aspect=20)
    ax.set_title(title, fontsize=10, fontweight="bold")
    if xlabel:
        ax.set_xlabel("UMAP 1", fontsize=8)
    ax.set_ylabel("UMAP 2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(False)
    ax.set_aspect("equal", adjustable="datalim")


# ════════════════════════════════════════════════════════════════
# Figure 1: Timepoint UMAP — 4 datasets, 2x2
#
# NOTE: Disabled by default. This panel duplicates the "Timepoint"
# column already produced as panel 0 of each {Dataset}_umap_sapphire_scores.png
# below (advisor feedback: do not repeat the same image across figures,
# e.g. old Fig 1A = old Fig 2A). Set RUN_FIGURE_1 = True to regenerate the
# standalone 2x2 timepoint-only overview if needed elsewhere.
# ════════════════════════════════════════════════════════════════
RUN_FIGURE_1 = False

if RUN_FIGURE_1:
    print("\n" + "="*60)
    print("  Figure 1: Timepoint UMAP (all 4 datasets)")
    print("="*60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("SAPPHIRE — Cell UMAP Coloured by Timepoint",
                 fontsize=14, fontweight="bold")
    axes = axes.ravel()

    for ax, ds_name in zip(axes, TARGET):
        cfg      = DATASETS_CONFIG[ds_name]
        time_col = cfg["time_col"]
        print(f"\n  {ds_name}")

        adata  = load_and_prepare(ds_name, cfg)
        if adata.n_vars > SAPPHIRE_PARAMS["n_top_genes"]:
            adata = hvg_filter(adata, SAPPHIRE_PARAMS["n_top_genes"])
        adata  = get_umap(adata)
        u1, u2 = adata.obsm["X_umap"][:, 0], adata.obsm["X_umap"][:, 1]
        tps    = sorted(adata.obs[time_col].unique(), key=sort_tp)
        pal    = tp_palette(tps)

        for tp in tps:
            mask = (adata.obs[time_col] == tp).values
            ax.scatter(u1[mask], u2[mask], c=[pal[tp]], s=3, alpha=0.5,
                       linewidths=0, rasterized=True)

        handles = [mpatches.Patch(color=pal[tp], label=tp) for tp in tps]
        ax.legend(handles=handles, title="Timepoint", fontsize=7,
                  title_fontsize=8, loc="best", framealpha=0.8)
        ax.set_title(ds_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("UMAP 1", fontsize=9)
        ax.set_ylabel("UMAP 2", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(False)
        ax.set_aspect("equal", adjustable="datalim")

        del adata; gc.collect()

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "ALL_datasets_umap_timepoint.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\n  -> {out}")


# ════════════════════════════════════════════════════════════════
# Figure 2: SAPPHIRE score overlay — per dataset, 1x4 panels
#   [timepoint | entropy | dispersion | composite]
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  Figure 2: SAPPHIRE score overlays (per dataset)")
print("="*60)

for ds_name in TARGET:
    cfg      = DATASETS_CONFIG[ds_name]
    time_col = cfg["time_col"]
    print(f"\n  {ds_name}")

    adata  = load_and_prepare(ds_name, cfg)
    if adata.n_vars > SAPPHIRE_PARAMS["n_top_genes"]:
        adata = hvg_filter(adata, SAPPHIRE_PARAMS["n_top_genes"])
    adata  = get_umap(adata)
    u1, u2 = adata.obsm["X_umap"][:, 0], adata.obsm["X_umap"][:, 1]

    # Load per-cell scores
    pc_df = pd.read_csv(
        os.path.join(VAL_DIR, ds_name, f"{ds_name}_per_cell_metrics.csv"),
        index_col=0)
    shared = adata.obs_names.intersection(pc_df.index)
    if len(shared) < len(adata):
        adata = adata[shared].copy()
        u1, u2 = adata.obsm["X_umap"][:, 0], adata.obsm["X_umap"][:, 1]
    pc_df = pc_df.loc[adata.obs_names]

    tps = sorted(adata.obs[time_col].unique(), key=sort_tp)
    pal = tp_palette(tps)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"{ds_name} — UMAP with SAPPHIRE Score Overlays",
                 fontsize=13, fontweight="bold")

    # Panel 0: timepoint
    ax = axes[0]
    for tp in tps:
        mask = (adata.obs[time_col] == tp).values
        ax.scatter(u1[mask], u2[mask], c=[pal[tp]], s=3, alpha=0.55,
                   linewidths=0, rasterized=True)
    handles = [mpatches.Patch(color=pal[tp], label=tp) for tp in tps]
    ax.legend(handles=handles, title="Timepoint", fontsize=7,
              title_fontsize=7, loc="best", framealpha=0.8)
    ax.set_title("Timepoint", fontsize=11, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=9); ax.set_ylabel("UMAP 2", fontsize=9)
    ax.tick_params(labelsize=7); ax.grid(False)
    ax.set_aspect("equal", adjustable="datalim")

    # Panels 1-3: SAPPHIRE scores
    for ax, (col, label, cmap) in zip(axes[1:], [
        ("pathway_entropy",    "Pathway Entropy",    "Blues"),
        ("network_dispersion", "Network Dispersion", "Greens"),
        ("composite",          "Composite Score",    "Purples"),
    ]):
        vals = pc_df[col].values.astype(float)
        vmin, vmax = np.percentile(vals, 2), np.percentile(vals, 98)
        sc_plot = ax.scatter(u1, u2, c=vals, cmap=cmap, vmin=vmin, vmax=vmax,
                             s=3, alpha=0.6, linewidths=0, rasterized=True)
        plt.colorbar(sc_plot, ax=ax, shrink=0.75, pad=0.02)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("UMAP 1", fontsize=9)
        ax.tick_params(labelsize=7); ax.grid(False)
        ax.set_aspect("equal", adjustable="datalim")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"{ds_name}_umap_sapphire_scores.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    del adata, pc_df; gc.collect()
    print(f"  -> {out}")


# ════════════════════════════════════════════════════════════════
# Figure 3: Module activation overlay — Cardiomyocyte (key figure)
#   [timepoint | M4 cardiac | M6 cell cycle | M2 maturation | composite]
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  Figure 3: Module activation overlay — Cardiomyocyte")
print("="*60)

ds_name  = "Cardiomyocyte"
cfg      = DATASETS_CONFIG[ds_name]
time_col = cfg["time_col"]

adata  = load_and_prepare(ds_name, cfg)
if adata.n_vars > SAPPHIRE_PARAMS["n_top_genes"]:
    adata = hvg_filter(adata, SAPPHIRE_PARAMS["n_top_genes"])
params = {**SAPPHIRE_PARAMS, **cfg.get("param_overrides", {})}
modules, _ = build_network(adata, params)
adata = get_umap(adata)
u1, u2 = adata.obsm["X_umap"][:, 0], adata.obsm["X_umap"][:, 1]

# Compute module activation matrix
import scipy.sparse as ssp
X = adata.X.toarray() if ssp.issparse(adata.X) else adata.X
mod_activation = {}
for mod_id, gene_idx in modules.items():
    mod_activation[mod_id] = X[:, gene_idx].mean(axis=1)

pc_df = pd.read_csv(
    os.path.join(VAL_DIR, ds_name, f"{ds_name}_per_cell_metrics.csv"),
    index_col=0)
shared = adata.obs_names.intersection(pc_df.index)
if len(shared) < len(adata):
    keep = [i for i, b in enumerate((adata.obs_names.isin(shared)).values) if b]
    u1, u2 = u1[keep], u2[keep]
    X = X[keep]
    for mod_id in mod_activation:
        mod_activation[mod_id] = mod_activation[mod_id][keep]
    adata = adata[shared].copy()
pc_df = pc_df.loc[adata.obs_names]

tps = sorted(adata.obs[time_col].unique(), key=sort_tp)
pal = tp_palette(tps)

# 5-panel figure: timepoint + M4 + M6 + M2 + composite
fig, axes = plt.subplots(1, 5, figsize=(24, 5))
fig.suptitle(
    "Cardiomyocyte — UMAP with Module Activation Overlays\n"
    "M4 shows transient cardiac commitment peak at D2 (1.67x baseline, p < 2.2x10^-308)",
    fontsize=12, fontweight="bold")

# Panel 0: timepoint
ax = axes[0]
for tp in tps:
    mask = (adata.obs[time_col] == tp).values
    ax.scatter(u1[mask], u2[mask], c=[pal[tp]], s=4, alpha=0.6,
               linewidths=0, rasterized=True)
handles = [mpatches.Patch(color=pal[tp], label=tp) for tp in tps]
ax.legend(handles=handles, title="Timepoint", fontsize=7.5,
          title_fontsize=8, loc="best", framealpha=0.8)
ax.set_title("Timepoint", fontsize=11, fontweight="bold")
ax.set_xlabel("UMAP 1", fontsize=9); ax.set_ylabel("UMAP 2", fontsize=9)
ax.tick_params(labelsize=7); ax.grid(False)
ax.set_aspect("equal", adjustable="datalim")

# Panels 1-3: key modules
for ax, (mod_id, cmap_name, title) in zip(axes[1:4], [
    ("M4", "YlOrRd",  "M4: Cardiac Commitment\n(heart contraction, D2 peak)"),
    ("M6", "Blues",   "M6: Cell Cycle\n(mitotic segregation, early)"),
    ("M2", "Greens",  "M2: Terminal Maturation\n(RNA splicing, D15-D30)"),
]):
    vals = mod_activation.get(mod_id, np.zeros(len(u1)))
    vmin, vmax = np.percentile(vals, 2), np.percentile(vals, 98)
    sc_plot = ax.scatter(u1, u2, c=vals, cmap=cmap_name, vmin=vmin, vmax=vmax,
                         s=4, alpha=0.65, linewidths=0, rasterized=True)
    plt.colorbar(sc_plot, ax=ax, shrink=0.75, pad=0.02, label="Activation")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.tick_params(labelsize=7); ax.grid(False)
    ax.set_aspect("equal", adjustable="datalim")

# Panel 4: composite score
ax = axes[4]
vals = pc_df["composite"].values.astype(float)
vmin, vmax = np.percentile(vals, 2), np.percentile(vals, 98)
sc_plot = ax.scatter(u1, u2, c=vals, cmap="Purples", vmin=vmin, vmax=vmax,
                     s=4, alpha=0.65, linewidths=0, rasterized=True)
plt.colorbar(sc_plot, ax=ax, shrink=0.75, pad=0.02, label="Score")
ax.set_title("Composite Score\n(SAPPHIRE)", fontsize=10, fontweight="bold")
ax.set_xlabel("UMAP 1", fontsize=9)
ax.tick_params(labelsize=7); ax.grid(False)
ax.set_aspect("equal", adjustable="datalim")

plt.tight_layout()
out = os.path.join(OUT_DIR, "Cardiomyocyte_umap_module_overlay.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
del adata, mod_activation, pc_df; gc.collect()
print(f"  -> {out}")


# ════════════════════════════════════════════════════════════════
# Figure 4: 2x2 module overlay overview — all 4 datasets
#   Each dataset: timepoint (left) + key transitional module (right)
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  Figure 4: Module overlay overview (all datasets, 2x4 panels)")
print("="*60)

# Key module per dataset (most biologically informative transitional module)
HIGHLIGHT = {
    "Cardiomyocyte": ("M4", "YlOrRd",  "M4: Cardiac Commitment (D2 peak)"),
    "Endoderm":      ("M8", "Oranges", "M8: Wnt Signalling (gut morphogenesis)"),
    "Kidney":        ("M0", "Reds",    "M0: Metal Detox (proximal tubule)"),
    "Neuro":         ("M2", "PuRd",    "M2: Axonogenesis (neuron projection)"),
}

fig, axes = plt.subplots(2, 4, figsize=(24, 11))
fig.suptitle(
    "SAPPHIRE — Module Activation Overlays Across Differentiation Systems\n"
    "Left: timepoint identity  |  Right: key module activation score",
    fontsize=13, fontweight="bold")

# Explicit (row, timepoint_col, module_col) assignment for each dataset.
# FIX: the previous version used conditional indexing (axes[0 if ... else 1, ...])
# separately in two loops. Endoderm's module panel landed on axes[1,3], which the
# second loop (Kidney/Neuro) never touched, but Kidney's panels at axes[0,0]/[0,1]
# and Neuro's at axes[1,0]/[1,1] silently overwrote Cardiomyocyte's panels at the
# same coordinates, and axes[0,2]/[0,3] (meant for Endoderm) were partially never
# drawn into because Endoderm's timepoint panel used axes[0,2] but nothing wrote
# axes[1,2]. Net effect: two of the eight panels rendered blank. Explicit per-dataset
# layout below guarantees every one of the 8 axes is written to exactly once.
PANEL_LAYOUT = {
    "Cardiomyocyte": (0, 0, 1),
    "Endoderm":      (0, 2, 3),
    "Kidney":        (1, 0, 1),
    "Neuro":         (1, 2, 3),
}

for ds_name in ["Cardiomyocyte", "Endoderm", "Kidney", "Neuro"]:
    cfg      = DATASETS_CONFIG[ds_name]
    time_col = cfg["time_col"]
    mod_id, cmap_name, mod_title = HIGHLIGHT[ds_name]
    row, tp_col, mod_col = PANEL_LAYOUT[ds_name]
    print(f"\n  {ds_name}")

    adata  = load_and_prepare(ds_name, cfg)
    if adata.n_vars > SAPPHIRE_PARAMS["n_top_genes"]:
        adata = hvg_filter(adata, SAPPHIRE_PARAMS["n_top_genes"])
    params = {**SAPPHIRE_PARAMS, **cfg.get("param_overrides", {})}
    modules, _ = build_network(adata, params)
    adata = get_umap(adata)
    u1, u2 = adata.obsm["X_umap"][:, 0], adata.obsm["X_umap"][:, 1]

    X = adata.X.toarray() if ssp.issparse(adata.X) else adata.X
    gene_idx = modules.get(mod_id, [])
    mod_vals = X[:, gene_idx].mean(axis=1) if len(gene_idx) > 0 else np.zeros(len(u1))

    tps = sorted(adata.obs[time_col].unique(), key=sort_tp)
    pal = tp_palette(tps)

    # Timepoint panel
    ax = axes[row, tp_col]
    for tp in tps:
        mask = (adata.obs[time_col] == tp).values
        ax.scatter(u1[mask], u2[mask], c=[pal[tp]], s=3, alpha=0.55,
                   linewidths=0, rasterized=True)
    handles = [mpatches.Patch(color=pal[tp], label=tp) for tp in tps]
    ax.legend(handles=handles, title="Timepoint", fontsize=6.5,
              title_fontsize=7, loc="best", framealpha=0.8, ncol=2 if len(tps)>4 else 1)
    ax.set_title(f"{ds_name}\nTimepoint", fontsize=10, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=8); ax.set_ylabel("UMAP 2", fontsize=8)
    ax.tick_params(labelsize=6); ax.grid(False)
    ax.set_aspect("equal", adjustable="datalim")

    # Module activation panel
    ax2 = axes[row, mod_col]
    vmin, vmax = np.percentile(mod_vals, 2), np.percentile(mod_vals, 98)
    sc_plot = ax2.scatter(u1, u2, c=mod_vals, cmap=cmap_name,
                          vmin=vmin, vmax=vmax, s=3, alpha=0.65,
                          linewidths=0, rasterized=True)
    plt.colorbar(sc_plot, ax=ax2, shrink=0.75, pad=0.02, label="Activation")
    ax2.set_title(f"{ds_name}\n{mod_title}", fontsize=10, fontweight="bold")
    ax2.set_xlabel("UMAP 1", fontsize=8)
    ax2.tick_params(labelsize=6); ax2.grid(False)
    ax2.set_aspect("equal", adjustable="datalim")

    del adata, modules; gc.collect()

plt.tight_layout()
out = os.path.join(OUT_DIR, "ALL_umap_module_overlay.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  -> {out}")

print("\n" + "="*60)
print(f"  Done! Output dir: {OUT_DIR}")
print("="*60)
print("\nFiles generated:")
for f in sorted(os.listdir(OUT_DIR)):
    print(f"  {f}")
