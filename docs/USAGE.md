# SAPPHIRE Installation and Usage Guide

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Data Preparation](#data-preparation)
4. [Running Analysis](#running-analysis)
5. [Understanding Output](#understanding-output)
6. [Troubleshooting](#troubleshooting)

## Installation

### Step 1: Create a conda environment (recommended)

```bash
conda create -n sapphire python=3.9
conda activate sapphire
```

### Step 2: Install SAPPHIRE

**Option A: Install from GitHub**
```bash
git clone https://github.com/ziye003/SAPPHIRE.git
cd SAPPHIRE
pip install -e .
```

**Option B: Install dependencies manually**
```bash
pip install numpy pandas scipy scikit-learn scanpy networkx matplotlib seaborn
```

### Step 3: Verify installation

```python
import sapphire
print(sapphire.__version__)  # Should print: 1.0.0
```

## Quick Start

### Minimal Example

```python
import scanpy as sc
import sapphire

# Load data
adata = sc.read_h5ad("your_data.h5ad")

# Run SAPPHIRE
results = sapphire.run_sapphire_pipeline(adata)

# Plot results
sapphire.plot_sapphire_results(results, save_path="plot.pdf")
```

## Data Preparation

### Required Format

Your data must be an `AnnData` object with:

1. **Expression matrix**: `adata.X`
   - Can be raw counts or log-normalized
   - Shape: (n_cells, n_genes)

2. **Timepoint annotations**: Column in `adata.obs`
   - Accepted names: `"timepoint"`, `"day"`, `"time"`, `"tp_day"`, `"tp_hours"`
   - Format examples: `"D0"`, `"D15"`, `"Day7"`, `"00h"`, `"96h"`

### Example Data Structure

```python
import scanpy as sc
import pandas as pd

# Load data
adata = sc.read_10x_mtx("path/to/matrix/")

# Add timepoint annotations
timepoints = pd.read_csv("timepoints.csv")  # cell_id, timepoint
adata.obs['timepoint'] = timepoints['timepoint'].values

# Basic QC
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# Normalize (optional - SAPPHIRE can handle raw counts)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Save
adata.write_h5ad("preprocessed_data.h5ad")
```

## Running Analysis

### Complete Pipeline

```python
import sapphire

results = sapphire.run_sapphire_pipeline(
    adata,
    time_col="timepoint",  # Specify or use None for auto-detection
    params={
        'n_top_genes': 2000,
        'top_k_edges': 10,
        'min_corr': 0.25,
        'leiden_resolution': 1.5,
        'min_module_size': 10,
    },
    output_dir="sapphire_results",
    dataset_name="my_experiment"
)
```

### Step-by-Step Analysis

```python
# Step 1: Infer pseudo-pathways
modules, edges, gene_names = sapphire.compute_pseudo_pathways(
    adata, 
    time_col="timepoint",
    params=params
)

# Step 2: Compute pathway activation
activation_df = sapphire.compute_pathway_activation(
    adata, modules, gene_names, 
    time_col="timepoint",
    params=params
)

# Step 3: Compute SAPPHIRE score
sapphire_score = sapphire.compute_sapphire_score(
    adata, modules, gene_names,
    time_col="timepoint"
)

# Step 4: Decompose plasticity
pathway_entropy, network_dispersion = sapphire.compute_plasticity_decomposition(
    activation_df, adata,
    time_col="timepoint"
)
```

## Understanding Output

### Results Dictionary

```python
results = {
    'modules': {
        'M0': [0, 15, 23, ...],  # Gene indices in module 0
        'M1': [5, 8, 12, ...],   # Gene indices in module 1
        ...
    },
    'activation': DataFrame,  # Shape: (n_cells, n_modules)
    'sapphire_score': Series,  # Network entropy per timepoint
    'pathway_entropy': Series,  # Within-cell uncertainty
    'network_dispersion': Series,  # Population divergence
    'gene_names': array,  # Gene names
    'adata': AnnData,  # Original data
    'time_col': str,  # Time column name
}
```

### Saved Files

When `output_dir` is specified, SAPPHIRE saves:

```
sapphire_results/
├── my_experiment_modules.json      # Module definitions
├── my_experiment_activation.csv    # Pathway activation matrix
├── my_experiment_scores.csv        # All scores
└── my_experiment_genes.txt         # Gene names
```

### Interpretation

**SAPPHIRE Score (Network Entropy)**
- High: Genes distributed across many modules (high organizational diversity)
- Low: Genes concentrated in few modules (low organizational diversity)
- **Non-monotonic patterns are biologically informative!**

**Pathway Entropy**
- High: Cells have diffuse activation across multiple programs
- Low: Cells have focused activation on specific programs
- Reflects within-cell regulatory uncertainty

**Network Dispersion**
- High: Cells are divergent in activation patterns
- Low: Cells have synchronized activation patterns
- Reflects population-level heterogeneity

## Troubleshooting

### Common Issues

**1. "Cannot find time column"**

```python
# Solution: Specify explicitly
results = sapphire.run_sapphire_pipeline(
    adata,
    time_col="your_column_name"
)
```

**2. Memory error during correlation computation**

```python
# Solution: Reduce number of genes
params = {
    'n_top_genes': 1000,  # Default is 2000
    ...
}
```

**3. "Too few edges" warning**

```python
# Solution: Lower correlation threshold
params = {
    'min_corr': 0.15,  # Default is 0.25
    ...
}
```

**4. No modules identified**

```python
# Solution: Adjust parameters
params = {
    'min_module_size': 5,  # Default is 10
    'leiden_resolution': 1.0,  # Default is 1.5
    ...
}
```

**5. Dataset too small**

SAPPHIRE requires:
- Minimum ~100-200 cells per timepoint
- At least 2 distinct timepoints (early + late)
- At least 500-1000 genes

### Getting Help

If you encounter issues:

1. Check the [GitHub Issues](https://github.com/ziye003/SAPPHIRE/issues)
2. Review example scripts in `examples/`
3. Contact: yexxx399@umn.edu@umn.edu

### Reporting Bugs

When reporting bugs, please include:
- Python version: `python --version`
- Package versions: `pip list`
- Minimal reproducible example
- Error message (full traceback)
- Dataset characteristics (n_cells, n_genes, n_timepoints)

## Advanced Topics

### Custom Preprocessing

```python
import scanpy as sc

# Your custom preprocessing
adata = sc.read_h5ad("data.h5ad")
sc.pp.highly_variable_genes(adata, n_top_genes=3000)
adata = adata[:, adata.var.highly_variable]

# Run SAPPHIRE
results = sapphire.run_sapphire_pipeline(adata)
```

### Batch Processing

```python
from pathlib import Path
import sapphire

datasets = {
    "experiment1": "data/exp1.h5ad",
    "experiment2": "data/exp2.h5ad",
    "experiment3": "data/exp3.h5ad",
}

for name, path in datasets.items():
    print(f"\nProcessing {name}...")
    adata = sc.read_h5ad(path)
    
    results = sapphire.run_sapphire_pipeline(
        adata,
        output_dir=f"results/{name}",
        dataset_name=name
    )
```

### Comparing Multiple Conditions

```python
import matplotlib.pyplot as plt

# Run on control
results_control = sapphire.run_sapphire_pipeline(adata_control)

# Run on treated
results_treated = sapphire.run_sapphire_pipeline(adata_treated)

# Compare
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(results_control['sapphire_score'], label='Control', marker='o')
ax.plot(results_treated['sapphire_score'], label='Treated', marker='s')
ax.legend()
ax.set_ylabel('SAPPHIRE Score')
ax.set_xlabel('Timepoint')
plt.savefig("comparison.pdf")
```

## Performance Tips

1. **For large datasets (>10,000 cells)**:
   - Subsample to ~5,000 cells per timepoint
   - Use `n_top_genes=1500`

2. **For many timepoints**:
   - Focus on boundary timepoints for module inference
   - Apply activation to all intermediate timepoints

3. **For memory-constrained systems**:
   - Reduce `n_top_genes`
   - Process timepoints separately

## Citation

Please cite our paper if you use SAPPHIRE:

```
[Citation information will be added upon publication]
```
