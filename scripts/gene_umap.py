"""
gene_umap.py
============
基因 UMAP（方案 A）：
  - 用基因间 Spearman 相关矩阵做降维（与 SAPPHIRE 网络构建逻辑一致）
  - 每个点 = 一个基因，颜色 = 所属模块
  - 4个数据集各出一张图，再出一张 2×2 总览图

用法：
    conda activate liver_adar1_py
    python gene_umap.py

输出（data/umap/）：
    gene_umap_ALL.png              — 2×2 总览
    {Dataset}_gene_umap.png        — 每个数据集单独一张（更大更清晰）
"""

import os, sys, gc, warnings
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as ssp
from scipy.stats import rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
warnings.filterwarnings("ignore")

sys.path.insert(0, "/Users/ziye/Documents/sapphire_package/")
exec(open("/Users/ziye/Documents/sapphire_package/sapphire_core.py").read(), globals())

OUT_DIR = os.path.join(str(DATA_ROOT), "umap")
os.makedirs(OUT_DIR, exist_ok=True)

sc.settings.verbosity = 0

# ── 固定颜色方案（最多支持 15 个模块）────────────────────────
MODULE_COLORS = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261",
    "#6A0572", "#1D7874", "#C77DFF", "#4CC9F0", "#F77F00",
    "#90BE6D", "#577590", "#F94144", "#43AA8B", "#277DA1",
]

def get_module_color(mod_id, mod_list):
    idx = mod_list.index(mod_id) if mod_id in mod_list else 0
    return MODULE_COLORS[idx % len(MODULE_COLORS)]


# ── 计算基因相关矩阵 UMAP ─────────────────────────────────────
def compute_gene_umap(adata, params):
    """
    1. Rank transform（和 build_network 完全一致）
    2. 计算基因×基因 Spearman 相关矩阵
    3. 转成距离矩阵（1 - |corr|）
    4. UMAP 降维
    返回：gene_names list, corr_matrix, umap coords (n_genes x 2)
    """
    X = adata.X
    if ssp.issparse(X):
        X = X.toarray()

    n_cells, n_genes = X.shape
    print(f"  Rank transform ({n_genes} genes × {n_cells} cells)...")

    X_rank = np.zeros_like(X, dtype=np.float32)
    for j in range(n_genes):
        X_rank[:, j] = rankdata(X[:, j])

    print("  Computing gene-gene correlation matrix...")
    mu  = X_rank.mean(axis=0)
    std = X_rank.std(axis=0) + 1e-10
    Xz  = (X_rank - mu) / std / np.sqrt(n_cells)

    batch = 200
    corr  = np.zeros((n_genes, n_genes), dtype=np.float32)
    for i in range(0, n_genes, batch):
        end = min(i + batch, n_genes)
        corr[i:end, :] = Xz[:, i:end].T @ Xz
    np.fill_diagonal(corr, 1)
    del X_rank, Xz; gc.collect()

    # 距离矩阵：1 - |corr|（相关性越高 = 距离越近）
    dist = (1 - np.abs(corr)).astype(np.float32)
    dist = np.clip(dist, 0, 1)

    print("  Running UMAP on gene correlation space...")
    import umap as umap_lib
    reducer = umap_lib.UMAP(
        metric="precomputed",
        n_neighbors=min(15, n_genes - 1),
        min_dist=0.3,
        random_state=42,
        verbose=False,
    )
    embedding = reducer.fit_transform(dist)
    return list(adata.var_names), corr, embedding


def plot_gene_umap(embedding, gene_names, modules, dataset_name, ax=None, standalone=False):
    """
    在 ax 上画基因 UMAP，颜色按模块。
    standalone=True 时创建独立 figure 并保存。
    """
    mod_list = sorted(modules.keys())

    # 给每个基因分配模块颜色（不在任何模块的基因 = 灰色）
    gene_to_mod = {}
    for mod_id, gene_indices in modules.items():
        for gi in gene_indices:
            if gi < len(gene_names):
                gene_to_mod[gene_names[gi]] = mod_id

    colors = []
    for g in gene_names:
        if g in gene_to_mod:
            colors.append(get_module_color(gene_to_mod[g], mod_list))
        else:
            colors.append("#CCCCCC")  # 不在模块内的基因 = 浅灰

    if standalone:
        fig, ax = plt.subplots(figsize=(9, 8))
        fig.suptitle(f"{dataset_name} — Gene UMAP (colored by module)",
                     fontsize=13, fontweight="bold")

    # 先画灰色（背景）
    gray_mask = [c == "#CCCCCC" for c in colors]
    ax.scatter(
        embedding[gray_mask, 0], embedding[gray_mask, 1],
        c="#CCCCCC", s=6, alpha=0.4, linewidths=0, rasterized=True, zorder=1
    )

    # 按模块画（前景）
    for mod_id in mod_list:
        color = get_module_color(mod_id, mod_list)
        mask  = [gene_to_mod.get(g) == mod_id for g in gene_names]
        if sum(mask) == 0:
            continue
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=color, s=12, alpha=0.8, linewidths=0,
            rasterized=True, zorder=2, label=mod_id
        )

    ax.set_title(dataset_name, fontsize=11, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(False)

    # 图例
    handles = [
        mpatches.Patch(color=get_module_color(m, mod_list), label=m)
        for m in mod_list
    ] + [mpatches.Patch(color="#CCCCCC", label="unassigned")]
    ax.legend(handles=handles, title="Module", fontsize=6.5,
              title_fontsize=7, loc="best", framealpha=0.7,
              ncol=2 if len(mod_list) > 8 else 1)

    if standalone:
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f"{dataset_name}_gene_umap.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {out_path}")


# ════════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════════

TARGET = ["Cardiomyocyte", "Endoderm", "Kidney", "Neuro"]

# 检查 umap 包是否安装
try:
    import umap
except ImportError:
    print("[ERROR] 请先安装 umap-learn：")
    print("  conda install -c conda-forge umap-learn")
    raise

fig_all, axes_all = plt.subplots(2, 2, figsize=(16, 14))
fig_all.suptitle("SAPPHIRE — Gene UMAP colored by Module",
                 fontsize=15, fontweight="bold", y=1.01)
axes_all = axes_all.ravel()

for ax_all, ds_name in zip(axes_all, TARGET):
    cfg = DATASETS_CONFIG[ds_name]
    print(f"\n{'='*55}\n  {ds_name}\n{'='*55}")

    # 加载数据 + HVG 筛选（与主 pipeline 一致）
    adata  = load_and_prepare(ds_name, cfg)
    params = {**SAPPHIRE_PARAMS, **cfg.get("param_overrides", {})}
    if adata.n_vars > params["n_top_genes"]:
        adata = hvg_filter(adata, params["n_top_genes"])

    # 构建网络（拿模块）
    modules, gene_names = build_network(adata, params)
    print(f"  Modules: {len(modules)}  |  Genes: {len(gene_names)}")

    # 计算基因 UMAP
    gene_names_out, corr, embedding = compute_gene_umap(adata, params)

    # 独立大图
    plot_gene_umap(embedding, gene_names_out, modules,
                   ds_name, standalone=True)

    # 总览图
    plot_gene_umap(embedding, gene_names_out, modules,
                   ds_name, ax=ax_all, standalone=False)

    del adata, corr, embedding; gc.collect()

# 保存总览图
plt.tight_layout()
out_all = os.path.join(OUT_DIR, "gene_umap_ALL.png")
fig_all.savefig(out_all, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved overview → {out_all}")

print("\n" + "="*55)
print("  ✅ Done!")
print(f"  输出目录：{OUT_DIR}")
print("="*55 + "\n")
