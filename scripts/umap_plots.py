"""
umap_plots.py
=============
1. 细胞 UMAP，颜色 = timepoint（多面板，每个数据集一张子图）
2. 细胞 UMAP，颜色 = pathway_entropy / network_dispersion / composite
   （叠加 SAPPHIRE 分数，直观展示与轨迹的对应关系）

用法：
    conda activate liver_adar1_py
    python umap_plots.py

输出（data/umap/）：
    ALL_datasets_umap_timepoint.png   — 4个数据集 timepoint UMAP 并排
    {Dataset}_umap_scores.png         — 每个数据集 3个分数的 UMAP
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
warnings.filterwarnings("ignore")

# ── 路径 ──────────────────────────────────────────────────────
sys.path.insert(0, "/Users/ziye/Documents/sapphire_package/")
exec(open("/Users/ziye/Documents/sapphire_package/sapphire_core.py").read(), globals())

VAL_DIR = os.path.join(str(DATA_ROOT), "sapphire_validation_v2")
OUT_DIR = os.path.join(str(DATA_ROOT), "umap")
os.makedirs(OUT_DIR, exist_ok=True)

sc.settings.verbosity = 0

# ── Timepoint 颜色方案（每个数据集独立渐变）────────────────────
def tp_palette(tps):
    """浅→深蓝绿渐变，early=浅，late=深"""
    cmap = plt.cm.viridis
    return {tp: cmap(i / max(len(tps) - 1, 1)) for i, tp in enumerate(tps)}

def sort_tp(tp):
    m = re.search(r"(\d+(?:\.\d+)?)", str(tp))
    return float(m.group(1)) if m else 0.0

# ── 计算或复用 UMAP ────────────────────────────────────────────
def get_umap(adata, dataset_name):
    """
    如果 adata.obsm 已有 X_umap，直接用。
    否则：PCA → neighbors → UMAP。
    """
    if "X_umap" in adata.obsm:
        print(f"  [INFO] 已有 UMAP，直接使用")
        return adata

    print(f"  [INFO] 计算 UMAP...")
    if "X_pca" not in adata.obsm:
        sc.tl.pca(adata, n_comps=min(50, adata.n_vars - 1), random_state=0)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=40, random_state=0)
    sc.tl.umap(adata, random_state=0)
    return adata


# ════════════════════════════════════════════════════════════════
# Figure 1：4个数据集 timepoint UMAP，2×2 布局
# ════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  Figure 1: Timepoint UMAP (all datasets)")
print("="*60)

TARGET = ["Cardiomyocyte", "Endoderm", "Kidney", "Neuro"]
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for ax, ds_name in zip(axes, TARGET):
    cfg = DATASETS_CONFIG[ds_name]
    print(f"\n  {ds_name}")

    adata = load_and_prepare(ds_name, cfg)
    if adata.n_vars > SAPPHIRE_PARAMS["n_top_genes"]:
        adata = hvg_filter(adata, SAPPHIRE_PARAMS["n_top_genes"])
    adata = get_umap(adata, ds_name)

    time_col = cfg["time_col"]
    tps      = sorted(adata.obs[time_col].unique(), key=sort_tp)
    palette  = tp_palette(tps)

    umap1 = adata.obsm["X_umap"][:, 0]
    umap2 = adata.obsm["X_umap"][:, 1]

    # 按 timepoint 分层画（early 在下，late 在上）
    for tp in tps:
        mask = (adata.obs[time_col] == tp).values
        ax.scatter(umap1[mask], umap2[mask],
                   c=[palette[tp]], s=3, alpha=0.5,
                   linewidths=0, rasterized=True, label=tp)

    ax.set_title(ds_name, fontsize=13, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=10)
    ax.set_ylabel("UMAP 2", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(False)

    # 图例（小点）
    handles = [mpatches.Patch(color=palette[tp], label=tp) for tp in tps]
    ax.legend(handles=handles, title="Timepoint", fontsize=8,
              title_fontsize=8, markerscale=2,
              loc="best", framealpha=0.7)

    del adata; gc.collect()

fig.suptitle("SAPPHIRE — Cell UMAP colored by Timepoint",
             fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
out1 = os.path.join(OUT_DIR, "ALL_datasets_umap_timepoint.png")
fig.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved → {out1}")


# ════════════════════════════════════════════════════════════════
# Figure 2：每个数据集，4 panel UMAP
#   左1：timepoint  |  右3：entropy / dispersion / composite
# ════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  Figure 2: UMAP colored by SAPPHIRE scores")
print("="*60)

SCORE_CMAPS = {
    "pathway_entropy":    ("Pathway Entropy",    "Blues"),
    "network_dispersion": ("Network Dispersion", "Greens"),
    "composite":          ("Composite Score",    "Purples"),
}

for ds_name in TARGET:
    cfg      = DATASETS_CONFIG[ds_name]
    time_col = cfg["time_col"]
    print(f"\n  {ds_name}")

    # 加载数据 + UMAP
    adata = load_and_prepare(ds_name, cfg)
    if adata.n_vars > SAPPHIRE_PARAMS["n_top_genes"]:
        adata = hvg_filter(adata, SAPPHIRE_PARAMS["n_top_genes"])
    adata = get_umap(adata, ds_name)

    umap1 = adata.obsm["X_umap"][:, 0]
    umap2 = adata.obsm["X_umap"][:, 1]

    # 读 per-cell 分数
    csv_path = os.path.join(VAL_DIR, ds_name, f"{ds_name}_per_cell_metrics.csv")
    pc_df    = pd.read_csv(csv_path, index_col=0)
    pc_df["timepoint"] = pc_df["timepoint"].astype(str)

    # 对齐 index（subsampling 可能导致顺序不同）
    shared_idx = adata.obs_names.intersection(pc_df.index)
    if len(shared_idx) < len(adata):
        print(f"  [WARN] {len(adata) - len(shared_idx)} cells not in CSV, subsetting")
        adata  = adata[shared_idx].copy()
        pc_df  = pc_df.loc[shared_idx]
        umap1  = adata.obsm["X_umap"][:, 0]
        umap2  = adata.obsm["X_umap"][:, 1]
    else:
        pc_df = pc_df.loc[adata.obs_names]

    tps     = sorted(adata.obs[time_col].unique(), key=sort_tp)
    palette = tp_palette(tps)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"{ds_name} — UMAP", fontsize=13, fontweight="bold")

    # Panel 0：timepoint
    ax = axes[0]
    for tp in tps:
        mask = (adata.obs[time_col] == tp).values
        ax.scatter(umap1[mask], umap2[mask],
                   c=[palette[tp]], s=3, alpha=0.5,
                   linewidths=0, rasterized=True, label=tp)
    handles = [mpatches.Patch(color=palette[tp], label=tp) for tp in tps]
    ax.legend(handles=handles, title="Timepoint", fontsize=7,
              title_fontsize=7, loc="best", framealpha=0.7)
    ax.set_title("Timepoint", fontsize=11)
    ax.set_xlabel("UMAP 1", fontsize=9); ax.set_ylabel("UMAP 2", fontsize=9)
    ax.tick_params(labelsize=7); ax.grid(False)

    # Panel 1–3：SAPPHIRE scores
    for ax, (col, (label, cmap_name)) in zip(axes[1:], SCORE_CMAPS.items()):
        vals = pc_df[col].values.astype(float)
        vmin, vmax = np.percentile(vals, 2), np.percentile(vals, 98)

        sc_plot = ax.scatter(umap1, umap2,
                             c=vals, cmap=cmap_name,
                             vmin=vmin, vmax=vmax,
                             s=3, alpha=0.6,
                             linewidths=0, rasterized=True)
        plt.colorbar(sc_plot, ax=ax, shrink=0.7, pad=0.02)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("UMAP 1", fontsize=9)
        ax.set_ylabel("")
        ax.tick_params(labelsize=7); ax.grid(False)

    plt.tight_layout()
    out2 = os.path.join(OUT_DIR, f"{ds_name}_umap_scores.png")
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out2}")

    del adata, pc_df; gc.collect()

print("\n" + "="*60)
print(f"  ✅ Done! 输出目录：{OUT_DIR}")
print("="*60 + "\n")
