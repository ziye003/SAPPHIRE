"""
SAPPHIRE Validation Pipeline
=============================
All core functions come from sapphire_core.py.

Usage (notebook):
    exec(open("sapphire_core.py").read())
    exec(open("sapphire_validation_all.py").read())
    run_pipeline(dataset="Endoderm")   # single dataset
    run_pipeline()                      # all four (automatically skips EB)
"""

import gc
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore")
from pathlib import Path

# ── Auto-load core (if not already loaded) ──────────────────────────
import os
_here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."
_core_path = os.path.join(_here, "sapphire_core.py")
if "load_and_prepare" not in dir():
    print("  -> Auto-loading sapphire_core.py...")
    exec(open(_core_path).read(), globals())

_VAL_OUTPUT = DATA_ROOT / "sapphire_validation_v2"
_VAL_OUTPUT.mkdir(exist_ok=True, parents=True)


def run_one(name, cfg):
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    out_dir = _VAL_OUTPUT / name
    out_dir.mkdir(exist_ok=True, parents=True)

    adata_full = load_and_prepare(name, cfg)
    time_col   = cfg["time_col"]

    params = {**SAPPHIRE_PARAMS, **cfg.get("param_overrides", {})}
    if adata_full.n_vars > params.get("n_top_genes", 2000):
        adata = hvg_filter(adata_full, params["n_top_genes"])
    else:
        adata = adata_full

    modules, _ = build_network(adata, params)

    print("  Computing per-cell metrics...")
    pc_df = compute_per_cell_metrics(adata, modules, time_col)
    pc_df["composite"] = compute_composite(pc_df)

    print("  AUC (early vs late):")
    early, late = cfg["early_tp"], cfg["late_tp"]
    auc_ent  = compute_auc(pc_df, early, late, "pathway_entropy")
    auc_disp = compute_auc(pc_df, early, late, "network_dispersion")
    auc_comp = compute_auc(pc_df, early, late, "composite")
    print(f"    entropy    AUC = {auc_ent:.3f}")
    print(f"    dispersion AUC = {auc_disp:.3f}")
    print(f"    composite  AUC = {auc_comp:.3f}")

    print("  Marker correlation:")
    # FIX: use adata_full (pre-HVG-filter), not the HVG-2000-filtered adata.
    # Curated marker genes (e.g. POU5F1, SOX2, PAX2, LHX1, WT1) are identity
    # markers, not necessarily high-variance genes -- in Kidney specifically,
    # none of the 5 configured markers survived the HVG-2000 variance cutoff,
    # which silently produced NaN correlations even though every marker gene
    # is present in the dataset. Scoring against the full normalized gene
    # panel avoids this HVG-selection artifact.
    marker_r = compute_marker_corr(
        adata_full, pc_df, cfg["stem_markers"], cfg["diff_markers"]
    )

    print("  Shuffle-time null test (n=50):")
    real_sc, _, null_p = shuffle_time_null(
        adata, modules, time_col, n_iter=50,
        random_state=params["random_state"]
    )

    plot_dataset(name, pc_df, cfg, out_dir)
    pc_df.to_csv(out_dir / f"{name}_per_cell_metrics.csv")

    summary = {
        "dataset"          : name,
        "n_cells"          : adata.n_obs,
        "n_modules"        : len(modules),
        "auc_entropy"      : round(auc_ent,  3),
        "auc_dispersion"   : round(auc_disp, 3),
        "auc_composite"    : round(auc_comp, 3),
        "marker_corr_stem" : round(marker_r.get("corr_stem", float("nan")), 3),
        "marker_corr_diff" : round(marker_r.get("corr_diff", float("nan")), 3),
        "shuffle_p"        : round(null_p, 3),
        "real_sign_changes": real_sc,
    }
    pd.DataFrame([summary]).to_csv(out_dir / f"{name}_summary.csv", index=False)
    gc.collect()
    return summary


def run_pipeline(dataset=None):
    """dataset=None -> run all four datasets (automatically skips EB)."""
    targets = (
        [dataset] if dataset
        else [d for d in DATASETS_CONFIG if d != "EB"]
    )
    summaries = []
    for name in targets:
        if name not in DATASETS_CONFIG:
            print(f"Unknown dataset: {name}")
            continue
        try:
            summaries.append(run_one(name, DATASETS_CONFIG[name]))
        except Exception as e:
            import traceback
            print(f"\nFailed: {name} -- {e}")
            traceback.print_exc()

    if summaries:
        df = pd.DataFrame(summaries)
        df.to_csv(_VAL_OUTPUT / "ALL_datasets_summary.csv", index=False)
        print(f"\n{'='*70}\nSummary\n{'='*70}")
        print(df.to_string(index=False))
        print(f"\nOutput directory: {_VAL_OUTPUT}")
        return df


print("sapphire_validation_all.py loaded  ->  run_pipeline()")

if __name__ == "__main__":
    run_pipeline()
