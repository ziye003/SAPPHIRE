# SAPPHIRE

**S**ingle-cell **A**nalysis of **P**athway **P**lasticity via **H**eterogeneity-**I**nformed **R**egulatory **E**ntropy

A network-based framework for quantifying transcriptional plasticity in single-cell RNA-seq data.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

SAPPHIRE addresses a fundamental limitation in plasticity quantification by decomposing transcriptional heterogeneity into mechanistically distinct components:

- **Pathway Entropy**: Within-cell uncertainty (diffuse activation across programs)
- **Network Dispersion**: Population-level divergence (distinct cell states)

Unlike existing methods that conflate these phenomena into scalar scores, SAPPHIRE operates on network-level organization, enabling interpretation of non-monotonic plasticity dynamics.

## Key Features

✨ **Network-based**: Quantifies plasticity through module organization, not gene-level dispersion  
🚀 **Fast**: Optimized correlation computation (~100x faster)  
📊 **Interpretable**: Decomposes plasticity into biological components  
🔬 **Robust**: Handles diverse datasets and edge cases  
📈 **Comprehensive**: Includes visualization and benchmarking tools

## Installation

### From GitHub (recommended)

```bash
git clone https://github.com/ziye003/SAPPHIRE.git
cd SAPPHIRE
pip install -e .
```

### Requirements

- Python ≥ 3.8
- numpy ≥ 1.20
- pandas ≥ 1.3
- scipy ≥ 1.7
- scikit-learn ≥ 1.0
- scanpy ≥ 1.8
- networkx ≥ 2.6
- matplotlib ≥ 3.4
- seaborn ≥ 0.11

## Quick Start

```python
import scanpy as sc
import sapphire

# Load your single-cell data
adata = sc.read_h5ad("your_data.h5ad")

# Run SAPPHIRE pipeline
results = sapphire.run_sapphire_pipeline(
    adata,
    time_col="timepoint",  # or None for auto-detection
    output_dir="sapphire_output"
)

# View results
print(f"Identified {len(results['modules'])} modules")
print(results['sapphire_score'])

# Plot results
sapphire.plot_sapphire_results(
    results,
    save_path="sapphire_plot.pdf"
)
```

## Input Data Format

SAPPHIRE expects an `AnnData` object with:

- **Expression data**: `adata.X` (raw counts or log-normalized)
- **Timepoint annotations**: In `adata.obs` with a column indicating differentiation time
  - Accepted column names: `"timepoint"`, `"tp_day"`, `"tp_hours"`, `"day"`, `"time"`
  - Timepoint format: `"D0"`, `"D15"`, `"Day7"`, `"00h"`, etc.

Example structure:
```python
adata.obs:
    cell_id    timepoint
    cell_001   D0
    cell_002   D0
    cell_003   D15
    ...

adata.X: (n_cells × n_genes) expression matrix
```

## Algorithm Overview

SAPPHIRE performs the following steps:

### 1. Pseudo-pathway Inference
- Extracts early and late boundary cells
- Constructs gene-gene correlation network
- Detects modules via Leiden clustering

### 2. Pathway Activation
- Computes per-cell activation as z-score deviation from time-matched background
- Background = variance-matched random gene sets

### 3. Network Entropy (SAPPHIRE Score)
- Shannon entropy of module size distribution
- High entropy = genes distributed across many modules
- Low entropy = genes concentrated in few modules

### 4. Plasticity Decomposition
- **Pathway Entropy (H)**: Shannon entropy of activation across modules (per cell)
- **Network Dispersion (D)**: Median cosine distance from population centroid

## Parameters

Default parameters work well for most datasets:

```python
params = {
    'n_top_genes': 2000,        # Number of highly variable genes
    'top_k_edges': 10,          # Max edges per gene in network
    'min_corr': 0.25,           # Minimum correlation threshold
    'leiden_resolution': 1.5,    # Leiden clustering resolution
    'min_module_size': 10,      # Minimum genes per module
    'random_state': 0,          # Random seed
}

results = sapphire.run_sapphire_pipeline(adata, params=params)
```

## Output

SAPPHIRE returns a dictionary containing:

```python
results = {
    'modules': dict,              # {module_id: [gene_indices]}
    'edges': list,                # Network edges
    'gene_names': np.ndarray,     # Gene names
    'activation': pd.DataFrame,   # Pathway activation (cells × modules)
    'sapphire_score': pd.Series,  # Network entropy per timepoint
    'pathway_entropy': pd.Series, # Within-cell uncertainty
    'network_dispersion': pd.Series, # Population divergence
    'adata': AnnData,             # Original data
    'time_col': str,              # Time column name
}
```

If `output_dir` is specified, results are saved as:
- `{dataset}_modules.json`: Module definitions
- `{dataset}_activation.csv`: Pathway activation matrix
- `{dataset}_scores.csv`: SAPPHIRE score, entropy, dispersion
- `{dataset}_genes.txt`: Gene names

## Advanced Usage

### Custom Time Column

```python
results = sapphire.run_sapphire_pipeline(
    adata,
    time_col="custom_time_column"
)
```

### Manual Pipeline Steps

```python
import sapphire

# Step 1: Infer modules
modules, edges, gene_names = sapphire.compute_pseudo_pathways(
    adata, time_col="timepoint", params=params
)

# Step 2: Compute activation
activation_df = sapphire.compute_pathway_activation(
    adata, modules, gene_names, time_col="timepoint", params=params
)

# Step 3: Compute SAPPHIRE score
sapphire_score = sapphire.compute_sapphire_score(
    adata, modules, gene_names, time_col="timepoint"
)

# Step 4: Decompose plasticity
pathway_entropy, network_dispersion = sapphire.compute_plasticity_decomposition(
    activation_df, adata, time_col="timepoint"
)
```

### Load Saved Results

```python
from sapphire.utils import load_results

results = load_results(
    output_dir="sapphire_output",
    dataset_name="mydata"
)
```

## Example Datasets

We provide example analyses on seven datasets:
- Cardiomyocyte maturation (Friedman et al., 2018)
- Definitive endoderm (Chu et al., 2016)
- Embryoid body (Tung et al., 2017)
- Kidney organoids (Phipson et al., 2019)
- Neuronal differentiation (Yao et al., 2018)
- Pulmonary fibrosis (Habermann et al., 2020)
- Cross-species development (Pijuan-Sala et al., 2019)

See `examples/` directory for complete workflows.

## Troubleshooting

**Q: "Cannot find time column" error**  
A: Specify time column explicitly: `time_col="your_column_name"`

**Q: "Too few edges" warning**  
A: Dataset may be too small. Try reducing `n_top_genes` or `min_corr`

**Q: Memory error during correlation computation**  
A: Reduce `n_top_genes` to 1000-1500

**Q: No modules identified**  
A: Try lowering `min_corr` to 0.15 or `min_module_size` to 5

## Citation

If you use SAPPHIRE in your research, please cite:

```
[Citation will be added upon publication]
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Issues**: [GitHub Issues](https://github.com/ziye003/SAPPHIRE/issues)
- **Email**: yexxx399@umn.edu@umn.edu

## Acknowledgments

Developed at the University of Minnesota Twin Cities  
Departments of Biomedical Engineering and Bioinformatics and Computational Biology
