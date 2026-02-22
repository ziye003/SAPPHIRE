"""
SAPPHIRE Full Test
"""
import sys
sys.path.insert(0, '/Users/ziye/Documents/sapphire_package')

import scanpy as sc
import sapphire
from pathlib import Path

print("="*70)
print("SAPPHIRE Test")
print("="*70)

# ====================
# 🔧 CHANGE THIS PATH TO YOUR DATA!
# ====================
DATA_PATH = "/Users/ziye/Documents/paper/data/E-MTAB-6268_allTP_merged.h5ad"
# 👆 Replace with your actual data path

OUTPUT_DIR = Path("~/Documents/sapphire_test_results").expanduser()
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# 1. Load data
print("\n[1/4] Loading data...")
adata = sc.read_h5ad(DATA_PATH)
print(f"✅ Data loaded: {adata.n_obs} cells × {adata.n_vars} genes")

# 2. Check structure
print("\n[2/4] Checking data structure...")
time_col = sapphire.infer_time_col(adata)
timepoints = sorted(adata.obs[time_col].unique(), 
                   key=sapphire.core.extract_timepoint_number)
print(f"✅ Time column: '{time_col}'")
print(f"   Timepoints: {timepoints}")

for tp in timepoints:
    n_cells = (adata.obs[time_col] == tp).sum()
    print(f"   - {tp}: {n_cells} cells")

# 3. Quick test (500 cells)
print("\n[3/4] Running quick test (500 cells)...")
print("   (This may take 1-2 minutes...)")
sc.pp.subsample(adata, n_obs=min(500, adata.n_obs))

results = sapphire.run_sapphire_pipeline(
    adata,
    params={
        'n_top_genes': 500,
        'top_k_edges': 10,
        'min_corr': 0.25,
        'leiden_resolution': 1.5,
        'min_module_size': 10,
    },
    output_dir=OUTPUT_DIR,
    dataset_name="cardiomyocyte_test"
)

# 4. Display results
print("\n[4/4] Results Summary...")
print(f"✅ Modules identified: {len(results['modules'])}")

print(f"\nModule sizes:")
for mod_id, genes in results['modules'].items():
    print(f"  {mod_id}: {len(genes)} genes")

print(f"\nSAPPHIRE Scores (Network Entropy):")
for tp, score in results['sapphire_score'].items():
    print(f"  {tp}: {score:.3f}")

print(f"\nPathway Entropy (Within-cell Uncertainty):")
for tp, score in results['pathway_entropy'].items():
    print(f"  {tp}: {score:.3f}")

print(f"\nNetwork Dispersion (Population Divergence):")
for tp, score in results['network_dispersion'].items():
    print(f"  {tp}: {score:.3f}")

# 5. Generate plot
print("\nGenerating visualization...")
sapphire.plot_sapphire_results(
    results,
    save_path=OUTPUT_DIR / "sapphire_test_plot.pdf"
)

print("\n" + "="*70)
print("✅ TEST COMPLETE!")
print("="*70)
print(f"\nResults saved to: {OUTPUT_DIR}")
print(f"\nView plot:")
print(f"  open {OUTPUT_DIR / 'sapphire_test_plot.pdf'}")
print(f"\nOutput files:")
print(f"  - cardiomyocyte_test_modules.json")
print(f"  - cardiomyocyte_test_activation.csv")
print(f"  - cardiomyocyte_test_scores.csv")
print(f"  - cardiomyocyte_test_genes.txt")
print(f"  - sapphire_test_plot.pdf")