"""
export_module_genes.py
======================
Export the gene list for each SAPPHIRE module.

Usage (terminal):
    python export_module_genes.py                         # default: Cardiomyocyte
    python export_module_genes.py --dataset Endoderm
    python export_module_genes.py --dataset all           # all 4 datasets

Output (under data/module_genes/<dataset>/):
    <dataset>_module_genes_long.csv    -- long format: module, gene
    <dataset>_module_genes_wide.csv    -- wide format: one column per module
    <dataset>_module_summary.csv       -- gene count summary per module
"""

import os
import sys
import gc
import warnings
warnings.filterwarnings("ignore")

# Argument parsing (Jupyter-compatible)
import argparse
_jupyter = any("jupyter" in a or "ipykernel" in a for a in sys.argv)
parser = argparse.ArgumentParser(description="SAPPHIRE: Export module gene lists")
parser.add_argument("--dataset", default="Cardiomyocyte",
                    help="Dataset name, or 'all' for all (Cardiomyocyte/Endoderm/Kidney/Neuro)")
parser.add_argument("--data_dir",
                    default=os.environ.get(
                        "SAPPHIRE_DATA_ROOT",
                        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
                    ),
                    help="Root data directory")
args = parser.parse_args([] if _jupyter else None)

# Load sapphire core
_here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."
_core = os.path.join(_here, "sapphire_core.py")
if not os.path.exists(_core):
    print(f"[ERROR] sapphire_core.py not found. Expected location: {_here}")
    sys.exit(1)

print("Loading sapphire_core.py...")
exec(open(_core).read(), globals())   # load all core functions into current namespace

import numpy as np
import pandas as pd
from pathlib import Path

# Output directory
OUT_ROOT = Path(args.data_dir) / "module_genes"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Determine target datasets
ALL_DATASETS = ["Cardiomyocyte", "Endoderm", "Kidney", "Neuro"]

if args.dataset.lower() == "all":
    targets = ALL_DATASETS
elif args.dataset in DATASETS_CONFIG:
    targets = [args.dataset]
else:
    # Case-insensitive fallback
    matched = [d for d in DATASETS_CONFIG if d.lower() == args.dataset.lower()]
    if matched:
        targets = matched
    else:
        print(f"[ERROR] Unknown dataset: {args.dataset}")
        print(f"  Available options: {list(DATASETS_CONFIG.keys())} or all")
        sys.exit(1)


# Core export function

def export_one(dataset_name: str):
    cfg = DATASETS_CONFIG[dataset_name]
    print(f"\n{'='*55}")
    print(f"  Dataset: {dataset_name}")
    print(f"{'='*55}")

    # 1. Load data (only gene info needed; use reduced cell count)
    adata = load_and_prepare(dataset_name, cfg, max_cells=10000)

    # 2. HVG filtering (consistent with main pipeline)
    n_top = SAPPHIRE_PARAMS["n_top_genes"]
    if adata.n_vars > n_top:
        adata = hvg_filter(adata, n_top)

    # 3. Build network (no timepoint labels used)
    params = {**SAPPHIRE_PARAMS, **cfg.get("param_overrides", {})}
    modules, gene_list = build_network(adata, params)
    # modules:   {module_id: [gene_col_indices]}
    # gene_list: list of gene names (adata.var_names after HVG filter)

    # 4. Convert column indices to gene names
    module_genes = {}
    for mod_id, col_indices in modules.items():
        gene_names = [gene_list[i] for i in col_indices]
        module_genes[mod_id] = sorted(gene_names)  # alphabetical order

    print(f"\n  {len(module_genes)} modules found:")
    for mod_id, genes in sorted(module_genes.items()):
        print(f"    {mod_id}: {len(genes)} genes")

    # 5. Export
    out_dir = OUT_ROOT / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 5a. Long-format CSV (required by enrichment_analysis.py)
    long_rows = []
    for mod_id, genes in module_genes.items():
        for g in genes:
            long_rows.append({"module": mod_id, "gene": g})
    long_df = pd.DataFrame(long_rows)
    long_path = OUT_ROOT / f"{dataset_name.lower()}_module_genes_long.csv"
    long_df.to_csv(long_path, index=False)
    print(f"\n  Long-format CSV -> {long_path}")

    # 5b. Wide-format CSV (one column per module)
    max_len  = max(len(g) for g in module_genes.values())
    wide_dict = {mod_id: pd.Series(genes + [""] * (max_len - len(genes)))
                 for mod_id, genes in module_genes.items()}
    wide_df  = pd.DataFrame(wide_dict)
    wide_path = OUT_ROOT / f"{dataset_name.lower()}_module_genes_wide.csv"
    wide_df.to_csv(wide_path, index=False)
    print(f"  Wide-format CSV  -> {wide_path}")

    # 5c. Summary: gene count per module
    summary_df = pd.DataFrame([
        {"module": mod_id, "n_genes": len(genes),
         "top5_genes": ", ".join(genes[:5])}
        for mod_id, genes in sorted(module_genes.items())
    ])
    summary_path = OUT_ROOT / f"{dataset_name.lower()}_module_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  Module summary   -> {summary_path}")

    # 5d. Excel (optional, requires openpyxl)
    try:
        import openpyxl
        xlsx_path = OUT_ROOT / f"{dataset_name.lower()}_module_genes.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            long_df.to_excel(writer, sheet_name="Long", index=False)
            wide_df.to_excel(writer, sheet_name="Wide", index=False)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
        print(f"  Excel            -> {xlsx_path}")
    except ImportError:
        print("  [Skipping Excel] openpyxl not installed (pip install openpyxl to enable)")

    gc.collect()
    return module_genes


# Main

print(f"\n{'='*55}")
print("  SAPPHIRE — Export Module Gene Lists")
print(f"{'='*55}")
print(f"  Target datasets: {targets}")
print(f"  Output dir: {OUT_ROOT}")

for ds in targets:
    try:
        export_one(ds)
    except Exception as e:
        import traceback
        print(f"\n[ERROR] {ds} failed: {e}")
        traceback.print_exc()

print(f"\n{'='*55}")
print("  Done!")
print(f"  Files saved to: {OUT_ROOT}")
print(f"{'='*55}\n")
