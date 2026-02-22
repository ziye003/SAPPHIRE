"""
SAPPHIRE Utilities
==================

Plotting and I/O functions for SAPPHIRE results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional
import json


def plot_sapphire_results(results: Dict, 
                          save_path: Optional[Path] = None,
                          figsize: tuple = (15, 4)):
    """
    Plot SAPPHIRE analysis results.
    
    Creates a three-panel figure showing:
    1. SAPPHIRE score (network entropy) over time
    2. Pathway entropy over time
    3. Network dispersion over time
    
    Parameters
    ----------
    results : dict
        Results dictionary from run_sapphire_pipeline
    save_path : Path, optional
        Path to save figure. If None, displays interactively.
    figsize : tuple
        Figure size (width, height)
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Panel 1: SAPPHIRE score
    ax = axes[0]
    sapphire_score = results['sapphire_score']
    ax.plot(range(len(sapphire_score)), sapphire_score.values, 
            'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax.set_xlabel('Timepoint', fontsize=12)
    ax.set_ylabel('SAPPHIRE Score\n(Network Entropy)', fontsize=12)
    ax.set_title('Network Entropy', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(sapphire_score)))
    ax.set_xticklabels(sapphire_score.index, rotation=45, ha='right')
    ax.grid(alpha=0.3)
    
    # Panel 2: Pathway entropy
    ax = axes[1]
    pathway_entropy = results['pathway_entropy']
    ax.plot(range(len(pathway_entropy)), pathway_entropy.values,
            'o-', linewidth=2, markersize=8, color='#A23B72')
    ax.set_xlabel('Timepoint', fontsize=12)
    ax.set_ylabel('Pathway Entropy', fontsize=12)
    ax.set_title('Within-cell Uncertainty', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(pathway_entropy)))
    ax.set_xticklabels(pathway_entropy.index, rotation=45, ha='right')
    ax.grid(alpha=0.3)
    
    # Panel 3: Network dispersion
    ax = axes[2]
    network_dispersion = results['network_dispersion']
    ax.plot(range(len(network_dispersion)), network_dispersion.values,
            'o-', linewidth=2, markersize=8, color='#F18F01')
    ax.set_xlabel('Timepoint', fontsize=12)
    ax.set_ylabel('Network Dispersion', fontsize=12)
    ax.set_title('Population Divergence', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(network_dispersion)))
    ax.set_xticklabels(network_dispersion.index, rotation=45, ha='right')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    return fig


def save_results(results: Dict, 
                output_dir: Path, 
                dataset_name: str):
    """
    Save SAPPHIRE results to disk.
    
    Saves:
    - modules.json: Pseudo-pathway modules
    - activation.csv: Pathway activation matrix
    - scores.csv: SAPPHIRE scores, pathway entropy, network dispersion
    
    Parameters
    ----------
    results : dict
        Results dictionary from run_sapphire_pipeline
    output_dir : Path
        Output directory
    dataset_name : str
        Dataset name (for filenames)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save modules
    modules_file = output_dir / f"{dataset_name}_modules.json"
    modules_serializable = {
        k: [int(i) for i in v] for k, v in results['modules'].items()
    }
    with open(modules_file, 'w') as f:
        json.dump(modules_serializable, f, indent=2)
    print(f"Modules saved to {modules_file}")
    
    # Save activation matrix
    activation_file = output_dir / f"{dataset_name}_activation.csv"
    results['activation'].to_csv(activation_file)
    print(f"Activation matrix saved to {activation_file}")
    
    # Save scores
    scores_df = pd.DataFrame({
        'SAPPHIRE_score': results['sapphire_score'],
        'pathway_entropy': results['pathway_entropy'],
        'network_dispersion': results['network_dispersion'],
    })
    scores_file = output_dir / f"{dataset_name}_scores.csv"
    scores_df.to_csv(scores_file)
    print(f"Scores saved to {scores_file}")
    
    # Save gene names
    genes_file = output_dir / f"{dataset_name}_genes.txt"
    with open(genes_file, 'w') as f:
        f.write('\n'.join(results['gene_names']))
    print(f"Gene names saved to {genes_file}")


def load_results(output_dir: Path, dataset_name: str) -> Dict:
    """
    Load saved SAPPHIRE results from disk.
    
    Parameters
    ----------
    output_dir : Path
        Output directory
    dataset_name : str
        Dataset name
    
    Returns
    -------
    results : dict
        Results dictionary (without adata)
    """
    output_dir = Path(output_dir)
    
    # Load modules
    modules_file = output_dir / f"{dataset_name}_modules.json"
    with open(modules_file, 'r') as f:
        modules = json.load(f)
    
    # Load activation
    activation_file = output_dir / f"{dataset_name}_activation.csv"
    activation = pd.read_csv(activation_file, index_col=0)
    
    # Load scores
    scores_file = output_dir / f"{dataset_name}_scores.csv"
    scores_df = pd.read_csv(scores_file, index_col=0)
    
    # Load genes
    genes_file = output_dir / f"{dataset_name}_genes.txt"
    with open(genes_file, 'r') as f:
        gene_names = np.array([line.strip() for line in f])
    
    results = {
        'modules': modules,
        'activation': activation,
        'sapphire_score': scores_df['SAPPHIRE_score'],
        'pathway_entropy': scores_df['pathway_entropy'],
        'network_dispersion': scores_df['network_dispersion'],
        'gene_names': gene_names,
    }
    
    return results
