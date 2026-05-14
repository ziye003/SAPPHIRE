"""
SAPPHIRE — Single-cell Analysis of Pathway Plasticity via
Heterogeneity-Informed Regulatory Entropy
"""

from .core import (
    SAPPHIRE_PARAMS,
    DATASETS_CONFIG,
    load_and_prepare,
    hvg_filter,
    build_network,
    compute_per_cell_metrics,
    compute_composite,
    compute_auc,
    compute_marker_corr,
    shuffle_time_null,
    plot_dataset,
)

__all__ = [
    "SAPPHIRE_PARAMS",
    "DATASETS_CONFIG",
    "load_and_prepare",
    "hvg_filter",
    "build_network",
    "compute_per_cell_metrics",
    "compute_composite",
    "compute_auc",
    "compute_marker_corr",
    "shuffle_time_null",
    "plot_dataset",
]
