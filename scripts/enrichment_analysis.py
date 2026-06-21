"""
SAPPHIRE Module Enrichment Analysis
=====================================
Run GO_BP / KEGG / MSigDB Hallmark enrichment on module genes.

Usage:
    python enrichment_analysis.py                        # default: cardiomyocyte
    python enrichment_analysis.py --dataset endoderm
    python enrichment_analysis.py --dataset all          # all 4 datasets

Dependencies:
    pip install gseapy matplotlib seaborn pandas numpy
"""

import argparse
import sys
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import gseapy as gp
except ImportError:
    print("[ERROR] gseapy not installed. Run: pip install gseapy")
    sys.exit(1)


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

GENE_SETS = {
    "GO_BP":    "GO_Biological_Process_2023",
    "KEGG":     "KEGG_2021_Human",
    "Hallmark": "MSigDB_Hallmark_2020",
}

ORGANISM = "human"

ALL_DATASETS = ["cardiomyocyte", "endoderm", "kidney", "neuro"]


# ─────────────────────────────────────────────
# Argument parsing (Jupyter-compatible)
# ─────────────────────────────────────────────

_jupyter = any("jupyter" in a or "ipykernel" in a for a in sys.argv)
parser = argparse.ArgumentParser(description="SAPPHIRE Module Enrichment Analysis")
parser.add_argument("--dataset",  default="cardiomyocyte",
                    help="Dataset name, or 'all' to run all datasets")
parser.add_argument("--data_dir",
                    default=os.environ.get(
                        "SAPPHIRE_DATA_ROOT",
                        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
                    ),
                    help="Root data directory")
parser.add_argument("--top_n",   type=int, default=5,
                    help="Top N terms to show per module in bubble plot")
parser.add_argument("--entropy_csv", default=None,
                    help="(Optional) CSV with module entropy values; columns: module, entropy")
args = parser.parse_args([] if _jupyter else None)


# ─────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────

def load_module_genes(long_csv_path):
    df = pd.read_csv(long_csv_path)
    module_col = next((c for c in df.columns if "module" in c.lower()), None)
    gene_col   = next((c for c in df.columns if "gene"   in c.lower()), None)
    if module_col is None or gene_col is None:
        raise ValueError(f"Cannot identify column names. Columns found: {df.columns.tolist()}")
    module_genes = {}
    for mod, grp in df.groupby(module_col):
        genes = grp[gene_col].dropna().unique().tolist()
        if len(genes) >= 5:
            module_genes[str(mod)] = genes
    print(f"  [INFO] {len(module_genes)} modules, {sum(len(v) for v in module_genes.values())} total gene entries")
    return module_genes


def run_enrichr_for_module(genes, gene_set_name, module_id):
    try:
        enr = gp.enrichr(
            gene_list=genes,
            gene_sets=gene_set_name,
            organism=ORGANISM,
            outdir=None,
            verbose=False,
            cutoff=0.05,
        )
        res = enr.results.copy()
        if res.empty:
            return pd.DataFrame()
        res.insert(0, "Module", module_id)
        return res
    except Exception as e:
        print(f"    [WARN] Module {module_id} / {gene_set_name} failed: {e}")
        return pd.DataFrame()


def run_all_enrichments(module_genes, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    for gs_key, gs_name in GENE_SETS.items():
        print(f"\n  [INFO] Gene set: {gs_key} ({gs_name})")
        all_dfs = []
        for mod_id, genes in module_genes.items():
            print(f"    Module {mod_id} ({len(genes)} genes)...", end=" ", flush=True)
            df = run_enrichr_for_module(genes, gs_name, mod_id)
            if not df.empty:
                print(f"{len(df)} significant results")
                all_dfs.append(df)
            else:
                print("no significant results")
        if all_dfs:
            merged = pd.concat(all_dfs, ignore_index=True)
            merged.columns = [c.strip() for c in merged.columns]
            merged.to_csv(os.path.join(out_dir, f"enrichment_{gs_key}.csv"), index=False)
            results[gs_key] = merged
        else:
            results[gs_key] = pd.DataFrame()
    return results


def _get_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ─────────────────────────────────────────────
# Summary table (one per dataset, includes Dataset column)
# ─────────────────────────────────────────────

def make_summary_table(results, dataset_name, out_dir, top_n_per_module=3):
    """
    Build summary table with columns: Dataset, Module, Gene_Set, Term, Adj_Pval, neg_log10_p.
    Output filename: <dataset>_enrichment_summary.csv
    """
    rows = []
    for gs_key, df in results.items():
        if df.empty:
            continue
        pval_col = _get_col(df, ["Adjusted P-value", "P-value", "Adjusted_P-value"])
        term_col = _get_col(df, ["Term", "term"])
        if not pval_col or not term_col:
            continue
        for mod, grp in df.groupby("Module"):
            top = grp.nsmallest(top_n_per_module, pval_col)
            for _, row in top.iterrows():
                rows.append({
                    "Dataset":     dataset_name,
                    "Module":      mod,
                    "Gene_Set":    gs_key,
                    "Term":        row[term_col],
                    "Adj_Pval":    round(float(row[pval_col]), 6),
                    "neg_log10_p": round(-np.log10(float(row[pval_col]) + 1e-300), 3),
                })

    if not rows:
        return pd.DataFrame()

    summary  = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, f"{dataset_name.lower()}_enrichment_summary.csv")
    summary.to_csv(out_path, index=False)
    print(f"\n  [INFO] {dataset_name} summary -> {out_path}")
    return summary


# ─────────────────────────────────────────────
# Bubble plot
# ─────────────────────────────────────────────

def plot_bubble(df, gs_key, dataset_name, top_n, out_dir):
    if df.empty:
        return
    pval_col    = _get_col(df, ["Adjusted P-value", "P-value", "Adjusted_P-value"])
    term_col    = _get_col(df, ["Term", "term"])
    overlap_col = _get_col(df, ["Overlap", "overlap"])
    if not pval_col or not term_col:
        return

    df = df.copy()
    if overlap_col and df[overlap_col].dtype == object:
        def parse_overlap(s):
            try:
                a, b = str(s).split("/")
                return int(a) / int(b)
            except Exception:
                return 0.05
        df["overlap_ratio"] = df[overlap_col].apply(parse_overlap)
    else:
        df["overlap_ratio"] = 0.05

    df["neg_log10_p"] = -np.log10(df[pval_col].clip(lower=1e-300))

    plot_rows = []
    for mod, grp in df.groupby("Module"):
        plot_rows.append(grp.nsmallest(top_n, pval_col))
    plot_df = pd.concat(plot_rows, ignore_index=True)
    if plot_df.empty:
        return

    plot_df[term_col] = plot_df[term_col].apply(
        lambda x: x[:55] + "..." if len(str(x)) > 55 else x)

    modules = sorted(plot_df["Module"].unique(),
                     key=lambda x: int(x[1:]) if x[1:].isdigit() else x)
    n_mods  = len(modules)
    fig_h   = max(6, len(plot_df) * 0.28)
    fig_w   = max(10, n_mods * 1.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    vmax = plot_df["neg_log10_p"].quantile(0.95) or 1

    for _, row in plot_df.iterrows():
        x = modules.index(row["Module"])
        ax.scatter(x, row[term_col],
                   s=row["overlap_ratio"] * 1500,
                   c=[row["neg_log10_p"]],
                   cmap="YlOrRd", vmin=0, vmax=vmax,
                   alpha=0.85, edgecolors="grey", linewidths=0.4)

    sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=plt.Normalize(0, vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.4, pad=0.02)
    cbar.set_label("-log10(adj. p-value)", fontsize=9)

    ax.set_xticks(range(n_mods))
    ax.set_xticklabels(modules, fontsize=9)
    ax.set_xlabel("Module", fontsize=11)
    ax.set_title(f"{dataset_name} — {gs_key} (top {top_n} per module)",
                 fontsize=12, fontweight="bold")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"bubble_{gs_key}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [INFO] Bubble plot -> {out_path}")


# ─────────────────────────────────────────────
# Single-dataset pipeline
# ─────────────────────────────────────────────

def run_one_dataset(dataset, data_dir, top_n, entropy_csv=None):
    dataset  = dataset.lower()
    long_csv = os.path.join(data_dir, "module_genes",
                            f"{dataset}_module_genes_long.csv")
    out_dir  = os.path.join(data_dir, "enrichment", dataset)

    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset.upper()}")
    print(f"{'='*60}")

    if not os.path.exists(long_csv):
        print(f"  [SKIP] File not found: {long_csv}")
        print(f"  Run first: python export_module_genes.py --dataset {dataset}")
        return pd.DataFrame()

    module_genes = load_module_genes(long_csv)
    results      = run_all_enrichments(module_genes, out_dir)

    for gs_key, df in results.items():
        plot_bubble(df, gs_key, dataset.capitalize(), top_n, out_dir)

    summary = make_summary_table(results, dataset.capitalize(), out_dir,
                                 top_n_per_module=3)
    return summary


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    data_dir    = args.data_dir
    top_n       = args.top_n
    entropy_csv = args.entropy_csv

    targets = ALL_DATASETS if args.dataset.lower() == "all" else [args.dataset.lower()]

    print(f"\n{'='*60}")
    print(f"  SAPPHIRE Module Enrichment Analysis")
    print(f"  Target datasets: {targets}")
    print(f"{'='*60}")

    all_summaries = []
    for ds in targets:
        summary = run_one_dataset(ds, data_dir, top_n, entropy_csv)
        if not summary.empty:
            all_summaries.append(summary)

    # Combine summaries across datasets when running all
    if len(all_summaries) > 1:
        combined      = pd.concat(all_summaries, ignore_index=True)
        combined_path = os.path.join(data_dir, "enrichment",
                                     "ALL_datasets_enrichment_summary.csv")
        os.makedirs(os.path.join(data_dir, "enrichment"), exist_ok=True)
        combined.to_csv(combined_path, index=False)
        print(f"\n  [INFO] Combined summary across datasets -> {combined_path}")

    print(f"\n{'='*60}")
    print(f"  All done!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
