"""
SAPPHIRE Pipeline
=================

High-level wrapper functions for running complete SAPPHIRE analysis.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import scanpy as sc

from .core import (
    compute_pseudo_pathways,
    compute_pathway_activation,
    compute_sapphire_score,
    compute_plasticity_decomposition,
    infer_time_col,
)


def run_sapphire_pipeline(adata,
                          time_col: Optional[str] = None,
                          params: Optional[Dict] = None,
                          output_dir: Optional[Path] = None,
                          dataset_name: str = "dataset") -> Dict:
    """
    Run complete SAPPHIRE pipeline on single-cell data.
    
    This function executes all SAPPHIRE analysis steps:
    1. Infer pseudo-pathway modules from boundary cells
    2. Compute pathway activation scores
    3. Calculate SAPPHIRE network entropy
    4. Decompose into pathway entropy and network dispersion
    
    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data object with raw counts or log-normalized data
    time_col : str, optional
        Name of time column in adata.obs. If None, auto-detected.
    params : dict, optional
        Parameter dictionary. If None, uses defaults:
        {
            'n_top_genes': 2000,
            'top_k_edges': 10,
            'min_corr': 0.25,
            'leiden_resolution': 1.5,
            'min_module_size': 10,
            'random_state': 0,
        }
    output_dir : Path, optional
        Directory to save results. If None, results not saved to disk.
    dataset_name : str
        Name of dataset (for output filenames)
    
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'modules': pseudo-pathway modules
        - 'edges': network edges
        - 'gene_names': gene names
        - 'activation': pathway activation dataframe
        - 'sapphire_score': SAPPHIRE scores per timepoint
        - 'pathway_entropy': pathway entropy per timepoint
        - 'network_dispersion': network dispersion per timepoint
        - 'adata': input adata (for convenience)
        - 'time_col': time column name
    
    Examples
    --------
    >>> import scanpy as sc
    >>> import sapphire
    >>> 
    >>> # Load data
    >>> adata = sc.read_h5ad("mydata.h5ad")
    >>> 
    >>> # Run SAPPHIRE
    >>> results = sapphire.run_sapphire_pipeline(adata)
    >>> 
    >>> # Access results
    >>> print(results['sapphire_score'])
    >>> print(f"Identified {len(results['modules'])} modules")
    """
    
    # Default parameters
    if params is None:
        params = {
            'n_top_genes': 2000,
            'top_k_edges': 10,
            'min_corr': 0.25,
            'leiden_resolution': 1.5,
            'min_module_size': 10,
            'random_state': 0,
        }
    
    # Auto-detect time column
    if time_col is None:
        time_col = infer_time_col(adata)
        print(f"Auto-detected time column: '{time_col}'")
    
    print(f"\n{'='*70}")
    print(f"Running SAPPHIRE Pipeline: {dataset_name}")
    print(f"{'='*70}\n")
    
    # Step 1: Infer pseudo-pathways
    modules, edges, gene_names = compute_pseudo_pathways(
        adata, time_col, params
    )
    
    # Step 2: Compute pathway activation
    activation_df = compute_pathway_activation(
        adata, modules, gene_names, time_col, params
    )
    
    # Step 3: Compute SAPPHIRE score
    sapphire_score = compute_sapphire_score(
        adata, modules, gene_names, time_col
    )
    
    # Step 4: Plasticity decomposition
    pathway_entropy, network_dispersion = compute_plasticity_decomposition(
        activation_df, adata, time_col
    )
    
    # Compile results
    results = {
        'modules': modules,
        'edges': edges,
        'gene_names': gene_names,
        'activation': activation_df,
        'sapphire_score': sapphire_score,
        'pathway_entropy': pathway_entropy,
        'network_dispersion': network_dispersion,
        'adata': adata,
        'time_col': time_col,
        'params': params,
    }
    
    # Save to disk if requested
    if output_dir is not None:
        from .utils import save_results
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        save_results(results, output_dir, dataset_name)
    
    print(f"\n{'='*70}")
    print(f"SAPPHIRE Pipeline Complete!")
    print(f"{'='*70}\n")
    print(f"Modules identified: {len(modules)}")
    print(f"Network entropy: {sapphire_score.iloc[0]:.3f}")
    
    return results
