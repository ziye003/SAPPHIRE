"""
test_percell_dispersion.py
==========================
快速验证：把 Network Dispersion 改成 per-cell（每个细胞到 timepoint centroid 的
个人距离，而不是 timepoint 中位数）之后，AUC 是否还能保持。

不重跑 pipeline，直接读已有的 per_cell_metrics.csv + 重新从 module activation 算。

用法：
    conda activate liver_adar1_py
    python test_percell_dispersion.py
"""

import os, re, sys, gc, warnings
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as ssp
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_distances
warnings.filterwarnings("ignore")

# ── 路径 ──────────────────────────────────────────────────────
DATA_ROOT = "/Users/ziye/Documents/paper/data"
VAL_DIR   = os.path.join(DATA_ROOT, "sapphire_validation_v2")

# ── 加载 sapphire_core（需要 build_network / hvg_filter 等）──
sys.path.insert(0, "/Users/ziye/Documents/sapphire_package/")
exec(open("/Users/ziye/Documents/sapphire_package/sapphire_core.py").read(), globals())

# ── 工具函数 ──────────────────────────────────────────────────

def compute_auc(scores, labels):
    try:
        auc = roc_auc_score(labels, scores)
        return max(auc, 1 - auc)
    except Exception:
        return np.nan

def sort_tp(tp):
    m = re.search(r"(\d+(?:\.\d+)?)", str(tp))
    return float(m.group(1)) if m else 0.0


def compute_percell_dispersion(adata, modules, time_col):
    """
    Per-cell dispersion：每个细胞到其所在 timepoint centroid 的个人余弦距离。
    （原版是取 median，所有细胞共享同一个值）
    """
    X = adata.X
    if ssp.issparse(X):
        X = X.toarray()

    n_cells  = X.shape[0]
    mod_keys = list(modules.keys())
    n_mods   = len(mod_keys)

    # Module activation matrix
    A = np.zeros((n_cells, n_mods), dtype=np.float32)
    for k, mod_id in enumerate(mod_keys):
        A[:, k] = X[:, modules[mod_id]].mean(axis=1)

    # Per-cell cosine distance to timepoint centroid
    dispersion = np.zeros(n_cells)
    for tp in adata.obs[time_col].unique():
        mask = (adata.obs[time_col] == tp).values
        idx  = np.where(mask)[0]
        if len(idx) < 2:
            continue
        A_tp     = A[idx]
        centroid = A_tp.mean(axis=0, keepdims=True)
        dists    = cosine_distances(A_tp, centroid).ravel()
        dispersion[idx] = dists          # ← 每个细胞自己的距离，不取 median

    return dispersion, A


# ════════════════════════════════════════════════════════════════
# 主流程：对4个数据集跑一遍
# ════════════════════════════════════════════════════════════════

results = []

for ds_name, cfg in DATASETS_CONFIG.items():
    if ds_name == "EB":
        continue

    print(f"\n{'='*55}\n  {ds_name}\n{'='*55}")

    # 1. 读已有 per_cell CSV（拿 entropy 和 timepoint，不用重算）
    csv_path = os.path.join(VAL_DIR, ds_name, f"{ds_name}_per_cell_metrics.csv")
    if not os.path.exists(csv_path):
        print(f"  [SKIP] 找不到 {csv_path}")
        continue
    pc_df = pd.read_csv(csv_path, index_col=0)
    pc_df["timepoint"] = pc_df["timepoint"].astype(str)

    # 2. 重新加载数据 + 构建网络（需要 module activation）
    print("  Loading data & building network...")
    adata  = load_and_prepare(ds_name, cfg)
    params = {**SAPPHIRE_PARAMS, **cfg.get("param_overrides", {})}
    if adata.n_vars > params["n_top_genes"]:
        adata = hvg_filter(adata, params["n_top_genes"])
    modules, _ = build_network(adata, params)

    # 3. 计算 per-cell dispersion（新版）
    print("  Computing per-cell dispersion...")
    disp_percell, _ = compute_percell_dispersion(adata, modules, cfg["time_col"])

    # 4. 原版 dispersion（从 CSV 读）
    disp_original = pc_df["network_dispersion"].values

    # 5. 把新算的列加回 pc_df（对齐 index）
    pc_df = pc_df.reset_index(drop=True)
    pc_df["disp_percell"]  = disp_percell
    n = len(pc_df)
    r_ent              = pc_df["pathway_entropy"].rank() / n
    r_disp_new         = pc_df["disp_percell"].rank()    / n
    pc_df["composite_new"] = (r_ent + r_disp_new) / 2

    # 6. AUC 对比
    early, late = cfg["early_tp"], cfg["late_tp"]
    mask   = pc_df["timepoint"].isin([early, late])
    sub    = pc_df[mask]
    labels = (sub["timepoint"] == early).astype(int).values

    auc_disp_orig  = compute_auc(sub["network_dispersion"].values, labels)
    auc_disp_new   = compute_auc(sub["disp_percell"].values,       labels)
    auc_comp_orig  = compute_auc(sub["composite"].values,          labels)
    auc_comp_new   = compute_auc(sub["composite_new"].values,      labels)
    auc_entropy    = compute_auc(pc_df["pathway_entropy"][mask], labels)

    print(f"\n  Metric                  | Original | Per-cell")
    print(f"  ----------------------- | -------- | --------")
    print(f"  Entropy AUC             | {auc_entropy:.3f}    | {auc_entropy:.3f}  (unchanged)")
    print(f"  Dispersion AUC          | {auc_disp_orig:.3f}    | {auc_disp_new:.3f}")
    print(f"  Composite AUC           | {auc_comp_orig:.3f}    | {auc_comp_new:.3f}")

    results.append({
        "Dataset":              ds_name,
        "Entropy_AUC":          round(auc_entropy,    3),
        "Dispersion_orig_AUC":  round(auc_disp_orig,  3),
        "Dispersion_new_AUC":   round(auc_disp_new,   3),
        "Composite_orig_AUC":   round(auc_comp_orig,  3),
        "Composite_new_AUC":    round(auc_comp_new,   3),
    })

    del adata, modules
    gc.collect()

# ── 汇总 ──────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("  SUMMARY")
print(f"{'='*65}")
df = pd.DataFrame(results)
print(df.to_string(index=False))

print(f"\n  Mean Dispersion AUC:  original={df['Dispersion_orig_AUC'].mean():.3f}  "
      f"per-cell={df['Dispersion_new_AUC'].mean():.3f}")
print(f"  Mean Composite AUC:   original={df['Composite_orig_AUC'].mean():.3f}  "
      f"per-cell={df['Composite_new_AUC'].mean():.3f}")

out_path = os.path.join(DATA_ROOT, "ablation", "percell_dispersion_test.csv")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
df.to_csv(out_path, index=False)
print(f"\n  结果保存到：{out_path}")
print(f"\n{'='*65}")
print("  结论参考：")
print("  - 如果 per-cell Dispersion AUC ≈ original → 可以改，violin 更好看")
print("  - 如果 per-cell Dispersion AUC 明显下降   → 原版更好，保持 bar chart")
print(f"{'='*65}\n")
