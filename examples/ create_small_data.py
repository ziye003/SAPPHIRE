"""
Create small example dataset
"""
import scanpy as sc
from pathlib import Path

# Load full data
print("Loading full data...")
DATA_PATH = "/Users/ziye/Documents/paper/data/E-MTAB-6268_allTP_merged.h5ad"
adata = sc.read_h5ad(DATA_PATH)

print(f"Original: {adata.n_obs} cells × {adata.n_vars} genes")

# Subsample to 500 cells
print("Subsampling to 500 cells...")
sc.pp.subsample(adata, n_obs=500)

# Save
output_path = Path("examples/data/example_small.h5ad")
output_path.parent.mkdir(parents=True, exist_ok=True)

adata.write_h5ad(output_path)
print(f"✅ Saved to: {output_path}")
print(f"Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")