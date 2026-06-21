"""
heatmap_pseudopathway.py
========================
Transposed pseudo-pathway heatmap:
  - Rows    = modules (labeled on left)
  - Columns = cells   (timepoint color bar on top)
  - Two versions per dataset:
      1. Ordered by timepoint (early -> late)
      2. Ordered by composite score (high -> low)

Usage:
    python heatmap_pseudopathway.py

Output (data/heatmap/):
    {Dataset}_heatmap_by_timepoint.png
    {Dataset}_heatmap_by_score.png
"""

import os, sys, gc, re, warnings
import numpy as np
import pandas as pd
import scipy.sparse as ssp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
warnings.filterwarnings("ignore")

_here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."
exec(open(os.path.join(_here, "sapphire_core.py")).read(), globals())

VAL_DIR = os.path.join(str(DATA_ROOT), "sapphire_validation_v2")
OUT_DIR = os.path.join(str(DATA_ROOT), "heatmap")
os.makedirs(OUT_DIR, exist_ok=True)

MAX_CELLS_PER_TP = 500   # max cells sampled per timepoint
CMAP_HEATMAP     = "RdBu_r"

def sort_tp(tp):
    m = re.search(r"(\d+(?:\.\d+)?)", str(tp))
    return float(m.group(1)) if m else 0.0

def tp_colors_map(tps):
    cmap = plt.cm.viridis
    return {tp: cmap(i / max(len(tps) - 1, 1)) for i, tp in enumerate(tps)}


def compute_activation_matrix(adata, modules):
    X = adata.X
    if ssp.issparse(X):
        X = X.toarray()
    mod_keys = sorted(modules.keys())
    A = np.zeros((adata.n_obs, len(mod_keys)), dtype=np.float32)
    for k, mod_id in enumerate(mod_keys):
        A[:, k] = X[:, modules[mod_id]].mean(axis=1)
    return pd.DataFrame(A, index=adata.obs_names, columns=mod_keys)


def draw_heatmap(matrix_z, mod_cols, cell_tp_labels, tp_cmap, tps,
                 title, out_path, order_mode="timepoint",
                 composite_scores=None):
    """
    matrix_z  : (n_cells, n_mods) z-score matrix, columns already sorted
    order_mode : "timepoint" or "score"
    """
    n_cells, n_mods = matrix_z.shape

    # Transpose: rows = modules, columns = cells
    mat_T = matrix_z.T   # (n_mods, n_cells)

    # Figure size: width scales with cell count, height with module count
    fig_w = max(12, n_cells * 0.012 + 3)
    fig_h = max(4,  n_mods  * 0.55  + 2.5)

    fig = plt.figure(figsize=(fig_w, fig_h))

    # GridSpec: top timepoint bar + main heatmap + right colorbar
    gs = gridspec.GridSpec(
        2, 2,
        height_ratios=[0.05, 1],
        width_ratios=[1, 0.025],
        hspace=0.01, wspace=0.02,
        left=0.10, right=0.90, top=0.88, bottom=0.18
    )

    ax_tp   = fig.add_subplot(gs[0, 0])   # top timepoint bar
    ax_heat = fig.add_subplot(gs[1, 0])   # main heatmap
    ax_cb   = fig.add_subplot(gs[1, 1])   # colorbar

    # Top timepoint color bar
    tp_color_array = np.array([tp_cmap[tp] for tp in cell_tp_labels])  # (n_cells, 4)
    ax_tp.imshow(
        tp_color_array.reshape(1, n_cells, 4),
        aspect="auto", interpolation="none"
    )
    ax_tp.set_xticks([])
    ax_tp.set_yticks([0])
    ax_tp.set_yticklabels(["Timepoint"], fontsize=8)
    ax_tp.tick_params(left=False)

    # Timepoint dividers (only in timepoint mode)
    if order_mode == "timepoint":
        prev_tp = cell_tp_labels[0]
        for i, tp in enumerate(cell_tp_labels):
            if tp != prev_tp:
                ax_tp.axvline(i - 0.5, color="white", linewidth=1.2)
                ax_heat.axvline(i - 0.5, color="white", linewidth=0.6, alpha=0.5)
                prev_tp = tp

    # Main heatmap
    im = ax_heat.imshow(
        mat_T,
        aspect="auto",
        cmap=CMAP_HEATMAP,
        vmin=-3, vmax=3,
        interpolation="none",
    )
    ax_heat.set_yticks(range(n_mods))
    ax_heat.set_yticklabels(mod_cols, fontsize=9)
    ax_heat.set_xticks([])
    ax_heat.set_xlabel(
        "Cells (ordered by timepoint, early->late)" if order_mode == "timepoint"
        else "Cells (ordered by Composite Score, high->low)",
        fontsize=9
    )
    ax_heat.set_ylabel("Module (Pseudo-pathway)", fontsize=10)

    # When ordered by score, replace top bar with a composite score gradient
    if order_mode == "score" and composite_scores is not None:
        score_arr = composite_scores.values.reshape(1, -1)
        ax_tp.imshow(score_arr, aspect="auto",
                     cmap="Purples", vmin=0, vmax=1,
                     interpolation="none")
        ax_tp.set_yticklabels(["Composite"], fontsize=8)

    # Colorbar
    cbar = plt.colorbar(im, cax=ax_cb)
    cbar.set_label("Z-score", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Legend + title
    if order_mode == "timepoint":
        handles = [mpatches.Patch(color=tp_cmap[tp], label=tp) for tp in tps]
        fig.legend(handles=handles, title="Timepoint",
                   loc="lower center", ncol=min(len(tps), 8),
                   fontsize=8, title_fontsize=8,
                   bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out_path}")


# ════════════════════════════════════════════════════════════════
TARGET = ["Cardiomyocyte", "Endoderm", "Kidney", "Neuro"]

for ds_name in TARGET:
    cfg = DATASETS_CONFIG[ds_name]
    print(f"\n{'='*55}\n  {ds_name}\n{'='*55}")

    # 1. Load data + build network
    adata  = load_and_prepare(ds_name, cfg)
    params = {**SAPPHIRE_PARAMS, **cfg.get("param_overrides", {})}
    if adata.n_vars > params["n_top_genes"]:
        adata = hvg_filter(adata, params["n_top_genes"])
    modules, _ = build_network(adata, params)

    time_col = cfg["time_col"]
    tps      = sorted(adata.obs[time_col].unique(), key=sort_tp)
    tp_cmap  = tp_colors_map(tps)

    # 2. Module activation matrix
    act_df = compute_activation_matrix(adata, modules)
    act_df["timepoint"] = adata.obs[time_col].values

    # 3. Load composite scores
    csv_path = os.path.join(VAL_DIR, ds_name, f"{ds_name}_per_cell_metrics.csv")
    pc_df    = pd.read_csv(csv_path, index_col=0)
    shared   = act_df.index.intersection(pc_df.index)
    act_df   = act_df.loc[shared]
    pc_df    = pc_df.loc[shared]
    act_df["composite"] = pc_df["composite"].values

    # 4. Subsample cells per timepoint
    rng = np.random.default_rng(42)
    sampled = []
    for tp in tps:
        idx = act_df.index[act_df["timepoint"] == tp].tolist()
        if len(idx) > MAX_CELLS_PER_TP:
            idx = list(rng.choice(idx, MAX_CELLS_PER_TP, replace=False))
        sampled.extend(idx)
    act_df = act_df.loc[sampled]

    mod_cols = sorted(modules.keys())
    matrix   = act_df[mod_cols].values.astype(float)

    # 5. Z-score (per module / per column)
    mu       = matrix.mean(axis=0)
    std      = matrix.std(axis=0) + 1e-10
    matrix_z = np.clip((matrix - mu) / std, -3, 3)

    # Filter low-quality modules: fewer than 15 genes or near-zero activation variance
    MIN_GENES = 15
    MIN_STD   = 0.01
    keep_mask = np.array([
        (len(modules[m]) >= MIN_GENES) and (std[i] > MIN_STD)
        for i, m in enumerate(mod_cols)
    ])
    if not keep_mask.all():
        removed = [mod_cols[i] for i, k in enumerate(keep_mask) if not k]
        print(f"  [INFO] Filtered low-quality modules: {removed}")
        mod_cols = [m for m, k in zip(mod_cols, keep_mask) if k]
        matrix_z = matrix_z[:, keep_mask]

    print(f"  Cells: {len(act_df)}  Modules: {len(mod_cols)}")

    # Version 1: ordered by timepoint (already sorted by tps order)
    tp_labels_v1 = act_df["timepoint"].values
    draw_heatmap(
        matrix_z, mod_cols, tp_labels_v1, tp_cmap, tps,
        title=f"{ds_name} — Pseudo-pathway Activation (by timepoint)",
        out_path=os.path.join(OUT_DIR, f"{ds_name}_heatmap_by_timepoint.png"),
        order_mode="timepoint",
    )

    # Version 2: ordered by composite score (high -> low)
    score_order = act_df["composite"].argsort()[::-1].values
    matrix_z_s  = matrix_z[score_order]
    tp_labels_v2 = tp_labels_v1[score_order]
    comp_scores  = act_df["composite"].iloc[score_order].reset_index(drop=True)

    draw_heatmap(
        matrix_z_s, mod_cols, tp_labels_v2, tp_cmap, tps,
        title=f"{ds_name} — Pseudo-pathway Activation (by Composite Score)",
        out_path=os.path.join(OUT_DIR, f"{ds_name}_heatmap_by_score.png"),
        order_mode="score",
        composite_scores=comp_scores,
    )

    del adata, modules, act_df, matrix_z
    gc.collect()

print(f"\nDone! Output dir: {OUT_DIR}\n")
