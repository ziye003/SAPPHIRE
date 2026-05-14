"""
export_module_genes.py
======================
导出 SAPPHIRE 每个模块包含哪些基因。

用法（bash 终端）：
    python export_module_genes.py                         # 默认跑 Cardiomyocyte
    python export_module_genes.py --dataset Endoderm
    python export_module_genes.py --dataset all           # 跑全部4个数据集

输出（在 data/module_genes/<dataset>/ 下）：
    <dataset>_module_genes_long.csv    -- 长格式：module, gene
    <dataset>_module_genes_wide.csv    -- 宽格式：每列一个模块
    <dataset>_module_summary.csv       -- 每个模块的基因数量汇总
"""

import os
import sys
import gc
import warnings
warnings.filterwarnings("ignore")

# ── 兼容 Jupyter kernel 的 argparse ──────────────────────────
import argparse
_jupyter = any("jupyter" in a or "ipykernel" in a for a in sys.argv)
parser = argparse.ArgumentParser(description="SAPPHIRE: Export module gene lists")
parser.add_argument("--dataset", default="Cardiomyocyte",
                    help="数据集名称，或 'all' 跑全部（Cardiomyocyte/Endoderm/Kidney/Neuro）")
parser.add_argument("--data_dir", default="/Users/ziye/Documents/paper/data",
                    help="数据根目录")
args = parser.parse_args([] if _jupyter else None)

# ── 加载 sapphire_core.py ────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."
_core = os.path.join(_here, "sapphire_core.py")
if not os.path.exists(_core):
    print(f"[ERROR] 找不到 sapphire_core.py，请确认它和本脚本在同一文件夹：{_here}")
    sys.exit(1)

print("正在加载 sapphire_core.py...")
exec(open(_core).read(), globals())   # 把所有核心函数加载进当前命名空间

import numpy as np
import pandas as pd
from pathlib import Path

# ── 输出目录 ─────────────────────────────────────────────────
OUT_ROOT = Path(args.data_dir) / "module_genes"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# ── 要跑的数据集 ──────────────────────────────────────────────
ALL_DATASETS = ["Cardiomyocyte", "Endoderm", "Kidney", "Neuro"]

if args.dataset.lower() == "all":
    targets = ALL_DATASETS
elif args.dataset in DATASETS_CONFIG:
    targets = [args.dataset]
else:
    # 大小写容错
    matched = [d for d in DATASETS_CONFIG if d.lower() == args.dataset.lower()]
    if matched:
        targets = matched
    else:
        print(f"[ERROR] 未知数据集：{args.dataset}")
        print(f"  可用选项：{list(DATASETS_CONFIG.keys())} 或 all")
        sys.exit(1)


# ── 核心函数 ─────────────────────────────────────────────────

def export_one(dataset_name: str):
    cfg = DATASETS_CONFIG[dataset_name]
    print(f"\n{'='*55}")
    print(f"  数据集：{dataset_name}")
    print(f"{'='*55}")

    # 1. 加载数据（只需要基因信息，用最少的细胞）
    adata = load_and_prepare(dataset_name, cfg, max_cells=10000)

    # 2. HVG 筛选（和主流程保持一致）
    n_top = SAPPHIRE_PARAMS["n_top_genes"]
    if adata.n_vars > n_top:
        adata = hvg_filter(adata, n_top)

    # 3. 构建网络（不使用时间点标签）
    params = {**SAPPHIRE_PARAMS, **cfg.get("param_overrides", {})}
    modules, gene_list = build_network(adata, params)
    # modules: {module_id: [gene_col_indices]}
    # gene_list: list of gene names (adata.var_names after HVG filter)

    # 4. 转换：列索引 → 基因名
    module_genes = {}
    for mod_id, col_indices in modules.items():
        gene_names = [gene_list[i] for i in col_indices]
        module_genes[mod_id] = sorted(gene_names)  # 字母排序方便查阅

    print(f"\n  共 {len(module_genes)} 个模块，基因数分布：")
    for mod_id, genes in sorted(module_genes.items()):
        print(f"    {mod_id}: {len(genes)} 个基因")

    # 5. 导出
    out_dir = OUT_ROOT / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 5a. 长格式 CSV（enrichment_analysis.py 需要这个格式）
    long_rows = []
    for mod_id, genes in module_genes.items():
        for g in genes:
            long_rows.append({"module": mod_id, "gene": g})
    long_df = pd.DataFrame(long_rows)
    long_path = OUT_ROOT / f"{dataset_name.lower()}_module_genes_long.csv"
    long_df.to_csv(long_path, index=False)
    print(f"\n  长格式 CSV → {long_path}")

    # 5b. 宽格式 CSV（每列一个模块）
    max_len  = max(len(g) for g in module_genes.values())
    wide_dict = {mod_id: pd.Series(genes + [""] * (max_len - len(genes)))
                 for mod_id, genes in module_genes.items()}
    wide_df  = pd.DataFrame(wide_dict)
    wide_path = OUT_ROOT / f"{dataset_name.lower()}_module_genes_wide.csv"
    wide_df.to_csv(wide_path, index=False)
    print(f"  宽格式 CSV → {wide_path}")

    # 5c. 汇总：每个模块的基因数
    summary_df = pd.DataFrame([
        {"module": mod_id, "n_genes": len(genes),
         "top5_genes": ", ".join(genes[:5])}
        for mod_id, genes in sorted(module_genes.items())
    ])
    summary_path = OUT_ROOT / f"{dataset_name.lower()}_module_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  模块汇总   → {summary_path}")

    # 5d. Excel（可选，需要 openpyxl）
    try:
        import openpyxl
        xlsx_path = OUT_ROOT / f"{dataset_name.lower()}_module_genes.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            long_df.to_excel(writer, sheet_name="Long", index=False)
            wide_df.to_excel(writer, sheet_name="Wide", index=False)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
        print(f"  Excel      → {xlsx_path}")
    except ImportError:
        print("  [跳过 Excel] openpyxl 未安装（pip install openpyxl 可启用）")

    gc.collect()
    return module_genes


# ── 主流程 ───────────────────────────────────────────────────

print(f"\n{'='*55}")
print("  SAPPHIRE — Export Module Gene Lists")
print(f"{'='*55}")
print(f"  目标数据集：{targets}")
print(f"  输出目录：{OUT_ROOT}")

for ds in targets:
    try:
        export_one(ds)
    except Exception as e:
        import traceback
        print(f"\n[ERROR] {ds} 失败：{e}")
        traceback.print_exc()

print(f"\n{'='*55}")
print("  ✓ 完成！")
print(f"  文件在：{OUT_ROOT}")
print(f"{'='*55}\n")
