"""
Generate validation results summary report (PDF + CSV).
=========================================================
Reads per-dataset summary CSVs and produces a combined figure and CSV.
Run after exec(sapphire_validation_all.py) in the notebook.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

DATA_ROOT  = Path("/Users/ziye/Documents/paper/data")
VAL_DIR    = DATA_ROOT / "sapphire_validation_v2"
REPORT_DIR = DATA_ROOT / "sapphire_report"
REPORT_DIR.mkdir(exist_ok=True, parents=True)

# ============================================================================================================================
# 
# ============================================================================================================================

datasets = ["Cardiomyocyte", "Endoderm", "Kidney", "Neuro"]

summaries = []
for ds in datasets:
    fp = VAL_DIR / ds / f"{ds}_summary.csv"
    if fp.exists():
        summaries.append(pd.read_csv(fp))
    else:
        print(f"⚠️   {fp} run_pipeline()")

if not summaries:
    raise FileNotFoundError("No result files found. Run sapphire_validation_all.py first.")

df = pd.concat(summaries, ignore_index=True)
print("Results loaded:")
print(df[["dataset","n_cells","n_modules","auc_entropy",
          "auc_dispersion","auc_composite",
          "marker_corr_stem","marker_corr_diff","shuffle_p"]].to_string(index=False))

# ============================================================================================================================
# 1 + AUC 
# ============================================================================================================================

fig = plt.figure(figsize=(16, 20))
gs  = gridspec.GridSpec(4, 1, figure=fig, hspace=0.45)

# Panel A: AUC summary bar chart
ax_auc = fig.add_subplot(gs[0])

x      = np.arange(len(df))
width  = 0.25
colors = {"auc_entropy": "#378ADD", "auc_dispersion": "#1D9E75", "auc_composite": "#7F77DD"}
labels = {"auc_entropy": "Pathway Entropy", "auc_dispersion": "Network Dispersion", "auc_composite": "Composite"}

for i, (col, color) in enumerate(colors.items()):
    bars = ax_auc.bar(x + (i-1)*width, df[col], width=width*0.9,
                      color=color, label=labels[col], alpha=0.85)
    for bar, val in zip(bars, df[col]):
        if not np.isnan(val):
            ax_auc.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                       f"{val:.3f}", ha="center", va="bottom", fontsize=8)

ax_auc.set_xticks(x)
ax_auc.set_xticklabels(df["dataset"], fontsize=11)
ax_auc.set_ylim(0, 1.12)
ax_auc.axhline(0.5, color="gray", linestyle="--", alpha=0.4)
ax_auc.set_ylabel("AUC (early vs late)", fontsize=11)
ax_auc.set_title("A.  Early/Late Separation Performance", fontsize=12, fontweight="bold", loc="left")
ax_auc.legend(fontsize=10, loc="lower right")
ax_auc.grid(axis="y", alpha=0.3)

# Panel B: Marker correlation
ax_mk = fig.add_subplot(gs[1])

x2 = np.arange(len(df))
w2 = 0.35
stem_vals = df["marker_corr_stem"].values
diff_vals = df["marker_corr_diff"].values

b1 = ax_mk.bar(x2 - w2/2, stem_vals, w2*0.9, color="#E85D24", alpha=0.8, label="vs Stem markers")
b2 = ax_mk.bar(x2 + w2/2, diff_vals, w2*0.9, color="#185FA5", alpha=0.8, label="vs Diff markers")

for bars in [b1, b2]:
    for bar in bars:
        val = bar.get_height()
        if not np.isnan(val):
            y_pos = val + 0.02 if val >= 0 else val - 0.04
            ax_mk.text(bar.get_x() + bar.get_width()/2, y_pos,
                      f"{val:.2f}", ha="center", va="bottom", fontsize=8)

ax_mk.axhline(0, color="black", linewidth=0.8)
ax_mk.set_xticks(x2)
ax_mk.set_xticklabels(df["dataset"], fontsize=11)
ax_mk.set_ylabel("Spearman ρ", fontsize=11)
ax_mk.set_title("B.  Marker Gene Correlation (Pathway Entropy)", fontsize=12, fontweight="bold", loc="left")
ax_mk.legend(fontsize=10)
ax_mk.grid(axis="y", alpha=0.3)
ax_mk.set_ylim(-0.85, 0.75)

# Panel C: Shuffle-time null p-values
ax_sh = fig.add_subplot(gs[2])

colors_sh = ["#3B6D11" if p > 0.05 else "#A32D2D" for p in df["shuffle_p"]]
bars_sh = ax_sh.bar(x, df["shuffle_p"], color=colors_sh, alpha=0.8, width=0.5)
ax_sh.axhline(0.05, color="red", linestyle="--", alpha=0.6, label="p=0.05")
ax_sh.set_xticks(x)
ax_sh.set_xticklabels(df["dataset"], fontsize=11)
ax_sh.set_ylabel("Shuffle-time null p-value", fontsize=11)
ax_sh.set_title("C.  Shuffle-time Null Test\n     (high p = real trajectory more monotone than random)", fontsize=12, fontweight="bold", loc="left")
ax_sh.legend(fontsize=10)
ax_sh.grid(axis="y", alpha=0.3)
for bar, val in zip(bars_sh, df["shuffle_p"]):
    ax_sh.text(bar.get_x() + bar.get_width()/2, val + 0.01,
              f"{val:.2f}", ha="center", va="bottom", fontsize=9)

# Panel D: Numerical summary table
ax_tb = fig.add_subplot(gs[3])
ax_tb.axis("off")

table_data = []
for _, row in df.iterrows():
    table_data.append([
        row["dataset"],
        f"{int(row['n_cells']):,}",
        f"{int(row['n_modules'])}",
        f"{row['auc_entropy']:.3f}",
        f"{row['auc_dispersion']:.3f}",
        f"{row['auc_composite']:.3f}",
        f"{row['marker_corr_stem']:.3f}" if not np.isnan(row['marker_corr_stem']) else "—",
        f"{row['marker_corr_diff']:.3f}"  if not np.isnan(row['marker_corr_diff'])  else "—",
        f"{row['shuffle_p']:.2f}",
    ])

col_labels = ["Dataset", "Cells", "Modules",
              "AUC\nEntropy", "AUC\nDispersion", "AUC\nComposite",
              "ρ Stem", "ρ Diff", "Shuffle p"]

tbl = ax_tb.table(
    cellText=table_data,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 2)

# Header color
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor("#7F77DD")
    tbl[0, j].set_text_props(color="white", fontweight="bold")

# Highlight composite AUC column
for i in range(1, len(table_data) + 1):
    tbl[i, 5].set_facecolor("#EEF0FF")

ax_tb.set_title("D.  Numerical Summary", fontsize=12, fontweight="bold", loc="left", pad=20)

# Overall title
fig.suptitle(
    "SAPPHIRE -- Label-free Transcriptional Plasticity Framework\n"
    "Validation across 4 differentiation datasets",
    fontsize=14, fontweight="bold", y=0.98
)

out_fp = REPORT_DIR / "SAPPHIRE_results_summary.pdf"
plt.savefig(out_fp, dpi=200, bbox_inches="tight")
plt.close()
print(f"\nReport PDF saved: {out_fp}")

# ============================================================================================================================
#  CSV
# ============================================================================================================================

clean = df[[
    "dataset", "n_cells", "n_modules",
    "auc_entropy", "auc_dispersion", "auc_composite",
    "marker_corr_stem", "marker_corr_diff",
    "shuffle_p", "real_sign_changes"
]].round(3)
clean.columns = [
    "Dataset", "N_cells", "N_modules",
    "AUC_Entropy", "AUC_Dispersion", "AUC_Composite",
    "Marker_rho_Stem", "Marker_rho_Diff",
    "Shuffle_p", "Real_sign_changes"
]
csv_fp = REPORT_DIR / "SAPPHIRE_results_summary.csv"
clean.to_csv(csv_fp, index=False)
print(f"Results CSV saved: {csv_fp}")
print(f"\nOutput directory: {REPORT_DIR}")
print("\nFiles ready for sharing:")
print(f"  {out_fp.name}")
print(f"  {csv_fp.name}")
