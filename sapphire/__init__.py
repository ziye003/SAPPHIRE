"""
SAPPHIRE: Single-cell Analysis of Pathway Plasticity via 
         Heterogeneity-Informed Regulatory Entropy
==========================================================

A network-based framework for quantifying transcriptional plasticity 
in single-cell RNA-seq data.

Main Functions
--------------
compute_pseudo_pathways : Infer pseudo-pathway modules
compute_pathway_activation : Calculate pathway activation scores
compute_sapphire_score : Calculate network entropy
compute_plasticity_decomposition : Decompose into entropy and dispersion
run_sapphire_pipeline : Complete pipeline wrapper

Author: Zi Ye
License: MIT
Version: 1.0.0
"""

from .core import (
    compute_pseudo_pathways,
    compute_pathway_activation,
    compute_sapphire_score,
    compute_plasticity_decomposition,
    infer_time_col,
    get_early_late_cells,
)

from .pipeline import run_sapphire_pipeline
from .utils import plot_sapphire_results, save_results

__version__ = "1.0.0"
__author__ = "Zi Ye"
__license__ = "MIT"

__all__ = [
    "compute_pseudo_pathways",
    "compute_pathway_activation",
    "compute_sapphire_score",
    "compute_plasticity_decomposition",
    "run_sapphire_pipeline",
    "infer_time_col",
    "get_early_late_cells",
    "plot_sapphire_results",
    "save_results",
]
