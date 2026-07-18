"""
merge_scent_slice_results.py
=============================
Reads the CSV outputs of run_scent.R and run_slice.R for each dataset,
evaluates them with the SAME AUC / Spearman / monotonicity framework used
in scripts/method_comparison.py, and merges them with the existing
SAPPHIRE / CytoTRACE / Expr_Entropy results into one comparison table ---
ready to replace Table 3 in the paper and to ground the Discussion
paragraph on SLICE/SCENT benchmarking.

Run this AFTER:
  1. export_for_r.py -> export_all()
  2. run_scent.R for each of the four datasets
  3. run_slice.R for each of the four datasets  (needs the human kappa
     gene-set matrix resolved first -- see run_slice.R header)
  4. method_comparison.py (for the existing SAPPHIRE/CytoTRACE/Expr_Entropy
     numbers) -- or reuse sapphire_package/docs results if already run.

Usage (exec after sapphire_validation_all.py, same convention as the other
scripts in this repo):

    exec(open("scripts/comparison/merge_scent_slice_results.py").read())
    final_df = build_full_comparison_table()
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

# Variables inherited from sapphire_core.py / sapphire_validation_all.py /
# method_comparison.py:
# DATA_ROOT, DATASETS_CONFIG, MC_OUTPUT (from method_comparison.py),
# sort_key, compute_auc_score, compute_time_corr

EXPORT_DIR = DATA_ROOT / "scent_slice_export"
SCENT_SLICE_OUTPUT_DIR = DATA_ROOT / "scent_slice_results"

TARGET_DATASETS = ["Cardiomyocyte", "Endoderm", "Kidney", "Neuro"]


def _sort_key(tp):
    import re
    m = re.search(r"(\d+(?:\.\d+)?)", str(tp))
    return float(m.group(1)) if m else 0.0


def _compute_auc(scores, labels, early_tp, late_tp):
    mask = labels.isin([early_tp, late_tp])
    if mask.sum() < 10:
        return np.nan
    y_true = (labels[mask] == early_tp).astype(int).values
    y_score = scores[mask].values
    try:
        auc = roc_auc_score(y_true, y_score)
        return max(auc, 1 - auc)
    except Exception:
        return np.nan


def _compute_time_corr(scores, labels, tps_sorted):
    time_map = {tp: i for i, tp in enumerate(tps_sorted)}
    t_num = labels.map(time_map)
    valid = ~t_num.isna()
    if valid.sum() < 10:
        return np.nan
    r, _ = spearmanr(scores[valid], t_num[valid])
    return r


def evaluate_one_method(name, method_label, score_series, cfg):
    meta_path = EXPORT_DIR / name / f"{name}_metadata.csv"
    meta_df = pd.read_csv(meta_path)
    labels = meta_df.set_index("cell")["timepoint"].astype(str)
    labels = labels.reindex(score_series.index)

    early_tp, late_tp = cfg["early_tp"], cfg["late_tp"]
    tps_sorted = sorted(labels.unique(), key=_sort_key)

    auc = _compute_auc(score_series, labels, early_tp, late_tp)
    r = _compute_time_corr(score_series, labels, tps_sorted)

    return {
        "dataset": name,
        "method": method_label,
        "auc": round(auc, 3) if not np.isnan(auc) else np.nan,
        "spearman_r": round(r, 3) if not np.isnan(r) else np.nan,
        "monotonicity": round(abs(r), 3) if not np.isnan(r) else np.nan,
    }


def load_scent(name):
    fp = SCENT_SLICE_OUTPUT_DIR / f"{name}_scent_SR.csv"
    if not fp.exists():
        print(f"  [missing] {fp} -- run run_scent.R for {name} first")
        return None
    df = pd.read_csv(fp).set_index("cell")
    return df["SCENT_SR"]


def load_slice(name):
    fp = SCENT_SLICE_OUTPUT_DIR / f"{name}_slice_entropy.csv"
    if not fp.exists():
        print(f"  [missing] {fp} -- run run_slice.R for {name} first")
        return None
    df = pd.read_csv(fp).set_index("cell")
    return df["SLICE_entropy"]


def build_full_comparison_table(existing_comparison_csv=None):
    """existing_comparison_csv: path to method_comparison.py's
    ALL_method_comparison.csv (SAPPHIRE / CytoTRACE / Expr_Entropy / PAGA /
    Gene_Count rows). If None, defaults to MC_OUTPUT/ALL_method_comparison.csv
    if that variable/file exists from a prior method_comparison.py run."""
    rows = []

    for name in TARGET_DATASETS:
        cfg = DATASETS_CONFIG[name]

        scent_scores = load_scent(name)
        if scent_scores is not None:
            rows.append(evaluate_one_method(name, "SCENT", scent_scores, cfg))

        slice_scores = load_slice(name)
        if slice_scores is not None:
            rows.append(evaluate_one_method(name, "SLICE", slice_scores, cfg))

    new_df = pd.DataFrame(rows)

    if existing_comparison_csv is None:
        try:
            existing_comparison_csv = MC_OUTPUT / "ALL_method_comparison.csv"
        except NameError:
            existing_comparison_csv = None

    if existing_comparison_csv is not None and Path(existing_comparison_csv).exists():
        existing_df = pd.read_csv(existing_comparison_csv)
        # Flag the pre-existing "Expr_Entropy" baseline honestly: it is a
        # naive per-cell Shannon entropy of the expression vector, NOT the
        # SCENT signalling-entropy algorithm, despite both being cited to
        # Teschendorff & Enver in earlier paper drafts.
        existing_df["method"] = existing_df["method"].replace(
            {"Expr_Entropy": "Naive_Expression_Entropy"}
        )
        full_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        print("No existing method_comparison.py output found -- "
              "returning SCENT/SLICE rows only. Merge manually with "
              "Table 3's SAPPHIRE/CytoTRACE numbers.")
        full_df = new_df

    out_path = SCENT_SLICE_OUTPUT_DIR / "ALL_method_comparison_with_SCENT_SLICE.csv"
    SCENT_SLICE_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    full_df.to_csv(out_path, index=False)

    pivot = full_df.pivot_table(index="method", columns="dataset", values="auc").round(3)
    if not pivot.empty:
        pivot["Mean AUC"] = pivot.mean(axis=1).round(3)
        pivot = pivot.sort_values("Mean AUC", ascending=False)
    print("\n" + "=" * 65)
    print("Full method comparison (AUC, early vs late)")
    print("=" * 65)
    print(pivot.to_string())
    print(f"\nWrote {out_path}")

    return full_df
