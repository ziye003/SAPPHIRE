"""
ablation.py
===========
Ablation analysis: AUC comparison and per-timepoint score distributions.

Fixes applied:
  1. p-value stars overflowing plot box -> fixed-position annotation inside plot
  2. Network Dispersion violin collapsed to line -> replaced with bar + bootstrap 95% CI

Usage:
    python ablation.py
"""

import os, re, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_auc_score
from scipy import stats

try:
    import scikit_posthocs as sp
except ImportError:
    print("[ERROR] Please install: conda install -c conda-forge scikit-posthocs")
    raise

warnings.filterwarnings("ignore")

# ── Path configuration ────────────────────────────────────────────────────────
# Override with: export SAPPHIRE_DATA_ROOT=/path/to/your/data
DATA_ROOT = os.environ.get(
    "SAPPHIRE_DATA_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
)
VAL_DIR   = os.path.join(DATA_ROOT, "sapphire_validation_v2")
OUT_DIR   = os.path.join(DATA_ROOT, "ablation")
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = {
    "Cardiomyocyte": {"early": "D0",   "late": "D30"},
    "Endoderm":      {"early": "00h",  "late": "96h"},
    "Kidney":        {"early": "Day7", "late": "Day26"},
    "Neuro":         {"early": "D11",  "late": "D52"},
}

METRIC_LABELS = {
    "pathway_entropy":    "Pathway Entropy",
    "network_dispersion": "Network Dispersion",
    "composite":          "Composite Score",
}

# ── Utility functions ─────────────────────────────────────────────────────────

def sort_tp(tp):
    m = re.search(r"(\d+(?:\.\d+)?)", str(tp))
    return float(m.group(1)) if m else 0.0

def compute_auc(scores, labels):
    try:
        auc = roc_auc_score(labels, scores)
        return max(auc, 1 - auc)
    except Exception:
        return np.nan

def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

def bootstrap_ci(values, n_boot=500, ci=95, seed=42):
    """Return mean, lower_err, upper_err (suitable for ax.errorbar yerr format)."""
    rng = np.random.default_rng(seed)
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    boots = [rng.choice(values, size=len(values), replace=True).mean()
             for _ in range(n_boot)]
    lo = np.percentile(boots, (100 - ci) / 2)
    hi = np.percentile(boots, 100 - (100 - ci) / 2)
    m  = np.mean(values)
    return m, m - lo, hi - m   # mean, lower_err, upper_err


# ════════════════════════════════════════════════════════════════════════════════
# Part 1: Ablation AUC
# ════════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  Part 1: Ablation AUC")
print("="*60)

records = []
for ds, cfg in DATASETS.items():
    csv_path = os.path.join(VAL_DIR, ds, f"{ds}_per_cell_metrics.csv")
    if not os.path.exists(csv_path):
        print(f"  [SKIP] {csv_path}")
        continue
    df = pd.read_csv(csv_path, index_col=0)
    df["timepoint"] = df["timepoint"].astype(str)
    mask   = df["timepoint"].isin([cfg["early"], cfg["late"]])
    sub    = df[mask].copy()
    labels = (sub["timepoint"] == cfg["early"]).astype(int).values
    records.append({
        "Dataset":        ds,
        "Entropy_AUC":    round(compute_auc(sub["pathway_entropy"].values,    labels), 3),
        "Dispersion_AUC": round(compute_auc(sub["network_dispersion"].values, labels), 3),
        "Composite_AUC":  round(compute_auc(sub["composite"].values,          labels), 3),
    })

auc_df   = pd.DataFrame(records)
mean_row = pd.DataFrame([{
    "Dataset":        "Mean",
    "Entropy_AUC":    round(auc_df["Entropy_AUC"].mean(),    3),
    "Dispersion_AUC": round(auc_df["Dispersion_AUC"].mean(), 3),
    "Composite_AUC":  round(auc_df["Composite_AUC"].mean(),  3),
}])
auc_df_full = pd.concat([auc_df, mean_row], ignore_index=True)
auc_df_full.to_csv(os.path.join(OUT_DIR, "ablation_auc_table.csv"), index=False)
print(auc_df_full.to_string(index=False))

# Bar chart
fig, ax = plt.subplots(figsize=(9, 5))
n_ds = len(auc_df); x = np.arange(n_ds); w = 0.24
for offset, col, color, lbl in [
    (-w, "Entropy_AUC",    "#378ADD", "Entropy only"),
    ( 0, "Dispersion_AUC", "#1D9E75", "Dispersion only"),
    ( w, "Composite_AUC",  "#7F77DD", "Composite"),
]:
    bars = ax.bar(x + offset, auc_df[col], w*0.9, color=color, label=lbl, alpha=0.85)
    for bar in bars:
        v = bar.get_height()
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(auc_df["Dataset"], fontsize=11)
ax.set_ylim(0.5, 1.13); ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4)
ax.set_ylabel("AUC (early vs late)", fontsize=11)
ax.set_title("Ablation Analysis: Entropy vs Dispersion vs Composite",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10, loc="lower right"); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "ablation_auc_barplot.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Ablation barplot saved.")


# ════════════════════════════════════════════════════════════════════════════════
# Part 2: Score distribution plots
#   - Pathway Entropy & Composite: violin + jitter, p-value annotated inside plot
#   - Network Dispersion: bar + bootstrap 95% CI
# ════════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  Part 2: Violin plots (fixed)")
print("="*60)

for ds, cfg in DATASETS.items():
    csv_path = os.path.join(VAL_DIR, ds, f"{ds}_per_cell_metrics.csv")
    if not os.path.exists(csv_path):
        continue

    df = pd.read_csv(csv_path, index_col=0)
    df["timepoint"] = df["timepoint"].astype(str)
    tps  = sorted(df["timepoint"].unique(), key=sort_tp)
    n_tp = len(tps)

    cmap      = plt.cm.Blues
    tp_colors = [cmap(0.35 + 0.55 * i / max(n_tp - 1, 1)) for i in range(n_tp)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle(f"{ds} — Per-cell Score Distributions",
                 fontsize=13, fontweight="bold")

    rng = np.random.default_rng(42)

    for ax_i, (ax, metric) in enumerate(
        zip(axes, ["pathway_entropy", "network_dispersion", "composite"])
    ):
        label  = METRIC_LABELS[metric]
        groups = [df.loc[df["timepoint"] == tp, metric].dropna().values for tp in tps]

        # Kruskal-Wallis test
        valid = [g for g in groups if len(g) >= 3]
        if len(valid) >= 2:
            _, p_kw = stats.kruskal(*valid)
        else:
            p_kw = 1.0

        # Dunn post-hoc (compute only; annotate inside plot corner)
        dunn_text = ""
        if p_kw < 0.05 and len(valid) == n_tp:
            try:
                dunn  = sp.posthoc_dunn(groups, p_adjust="bonferroni")
                i0, j0 = 0, n_tp - 1          # early vs late
                p_el   = dunn.iloc[i0, j0]
                s_el   = stars(p_el)
                dunn_text = f"early vs late: {s_el} (p={p_el:.2e})"
            except Exception:
                pass

        # Network Dispersion -> bar + 95% CI (per-timepoint constant value)
        if metric == "network_dispersion":
            means, lo_errs, hi_errs = [], [], []
            for grp in groups:
                if len(grp) == 0:
                    means.append(np.nan); lo_errs.append(0); hi_errs.append(0)
                else:
                    m, lo, hi = bootstrap_ci(grp)
                    means.append(m); lo_errs.append(lo); hi_errs.append(hi)

            x_pos = np.arange(n_tp)
            bars  = ax.bar(x_pos, means, color=tp_colors, width=0.6, alpha=0.85,
                           zorder=2)
            ax.errorbar(x_pos, means,
                        yerr=[lo_errs, hi_errs],
                        fmt="none", color="black", capsize=4, linewidth=1.2, zorder=3)

            # Label value on top of each bar
            for xi, m in zip(x_pos, means):
                if not np.isnan(m):
                    ax.text(xi, m + max(hi_errs) * 0.15,
                            f"{m:.4f}", ha="center", va="bottom", fontsize=7.5)

            ax.set_xticks(x_pos)
            ax.set_xticklabels(tps, rotation=40, ha="right", fontsize=8)
            ax.set_xlabel("Timepoint", fontsize=9)
            ax.set_ylabel(label, fontsize=9)
            ax.grid(axis="y", alpha=0.3, linewidth=0.5)

            # KW p in title; Dunn in upper-left corner inside plot
            p_str = f"{p_kw:.2e}" if p_kw < 0.001 else f"{p_kw:.4f}"
            ax.set_title(f"{label}\nKruskal-Wallis p = {p_str}", fontsize=10)
            if dunn_text:
                ax.text(0.02, 0.97, dunn_text,
                        transform=ax.transAxes, fontsize=8,
                        va="top", ha="left",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                  ec="gray", alpha=0.8))
            continue   # skip violin code below

        # Pathway Entropy & Composite -> violin + jitter

        # Violin
        parts = ax.violinplot(
            groups, positions=range(n_tp),
            showmedians=True, showextrema=False, widths=0.7
        )
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(tp_colors[i])
            pc.set_alpha(0.65)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(2)

        # Jitter
        for i, grp in enumerate(groups):
            if len(grp) == 0:
                continue
            n_plot = min(len(grp), 600)
            idx    = rng.choice(len(grp), n_plot, replace=False)
            jitter = rng.uniform(-0.18, 0.18, size=n_plot)
            ax.scatter(i + jitter, grp[idx],
                       color=tp_colors[i], alpha=0.22, s=5,
                       zorder=2, linewidths=0)

        ax.set_xticks(range(n_tp))
        ax.set_xticklabels(tps, rotation=40, ha="right", fontsize=8)
        ax.set_xlabel("Timepoint", fontsize=9)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)

        # KW p in title; Dunn in upper-left corner inside plot (no overflow)
        p_str = f"{p_kw:.2e}" if p_kw < 0.001 else f"{p_kw:.4f}"
        ax.set_title(f"{label}\nKruskal-Wallis p = {p_str}", fontsize=10)

        if dunn_text:
            ax.text(0.02, 0.97, dunn_text,
                    transform=ax.transAxes, fontsize=8,
                    va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="gray", alpha=0.8))

    # Legend
    patches = [mpatches.Patch(color=tp_colors[i], label=tps[i]) for i in range(n_tp)]
    fig.legend(handles=patches, title="Timepoint", loc="lower center",
               ncol=min(n_tp, 7), fontsize=8.5,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    out_path = os.path.join(OUT_DIR, f"{ds}_violin_v2.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

print("\n" + "="*60)
print("  Done! Output dir: " + OUT_DIR)
print("="*60 + "\n")
