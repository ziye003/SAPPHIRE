"""
SAPPHIRE Basic Example
======================

This script demonstrates basic SAPPHIRE usage on a sample dataset.

Requirements:
- AnnData object with timepoint annotations
- See README.md for data format details
"""

import scanpy as sc
import sapphire
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

# Input data path
DATA_PATH = "path/to/your/data.h5ad"  # Change this to your data

# Output directory
OUTPUT_DIR = Path("sapphire_output")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# SAPPHIRE parameters (optional - these are defaults)
PARAMS = {
    'n_top_genes': 2000,
    'top_k_edges': 10,
    'min_corr': 0.25,
    'leiden_resolution': 1.5,
    'min_module_size': 10,
    'random_state': 0,
}

# ============================================================================
# Load Data
# ============================================================================

print("Loading data...")
adata = sc.read_h5ad(DATA_PATH)

print(f"Loaded: {adata.n_obs} cells × {adata.n_vars} genes")
print(f"Timepoint column: {sapphire.infer_time_col(adata)}")

# ============================================================================
# Run SAPPHIRE Pipeline
# ============================================================================

print("\nRunning SAPPHIRE pipeline...")

results = sapphire.run_sapphire_pipeline(
    adata,
    time_col=None,  # Auto-detect, or specify like "timepoint"
    params=PARAMS,
    output_dir=OUTPUT_DIR,
    dataset_name="example"
)

# ============================================================================
# View Results
# ============================================================================

print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

print(f"\nModules identified: {len(results['modules'])}")

print("\nSAPPHIRE Scores (Network Entropy):")
print(results['sapphire_score'])

print("\nPathway Entropy (Within-cell Uncertainty):")
print(results['pathway_entropy'])

print("\nNetwork Dispersion (Population Divergence):")
print(results['network_dispersion'])

# ============================================================================
# Visualization
# ============================================================================

print("\nGenerating plots...")

# Create visualization
fig = sapphire.plot_sapphire_results(
    results,
    save_path=OUTPUT_DIR / "sapphire_plot.pdf"
)

print(f"\nPlot saved to: {OUTPUT_DIR / 'sapphire_plot.pdf'}")

# ============================================================================
# Inspect Modules
# ============================================================================

print("\nModule sizes:")
for mod_id, genes in results['modules'].items():
    print(f"  {mod_id}: {len(genes)} genes")

# Print genes in first module
first_module = list(results['modules'].keys())[0]
first_module_genes = results['modules'][first_module]
gene_names = results['gene_names'][first_module_genes]

print(f"\nGenes in {first_module} (first 10):")
for i, gene in enumerate(gene_names[:10]):
    print(f"  {i+1}. {gene}")

# ============================================================================
# Access Individual Scores
# ============================================================================

# Access scores for specific timepoint
time_col = results['time_col']
timepoints = sorted(adata.obs[time_col].unique(), 
                   key=sapphire.core.extract_timepoint_number)

print(f"\nScores for {timepoints[-1]} (final timepoint):")
print(f"  SAPPHIRE: {results['sapphire_score'].loc[timepoints[-1]]:.3f}")
print(f"  Pathway Entropy: {results['pathway_entropy'].loc[timepoints[-1]]:.3f}")
print(f"  Network Dispersion: {results['network_dispersion'].loc[timepoints[-1]]:.3f}")

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
print(f"\nResults saved to: {OUTPUT_DIR}")
