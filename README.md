# SAPPHIRE

**S**ingle-cell **A**nalysis of **P**athway **P**lasticity via **H**igh-Resolution **E**ntropy

A label-free, network-based framework for quantifying transcriptional plasticity in single-cell RNA-seq data across differentiation trajectories.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## Overview

SAPPHIRE quantifies transcriptional plasticity at single-cell resolution by decomposing it into two complementary, fully label-free metrics:

- **Pathway Entropy** — per-cell Shannon entropy of module activation distributions. High entropy indicates diffuse, broad activation across co-expression programmes (stem-like, plastic state); low entropy indicates selective, committed activation (differentiated state).
- **Network Dispersion** — per-cell mean cosine distance to the k nearest neighbours in module activation space (kNN density). High dispersion indicates a cell resides in a sparse, locally heterogeneous neighbourhood; low dispersion indicates a densely populated, transcriptionally uniform neighbourhood.

These two metrics are combined into a **Composite Score** via rank averaging, enabling robust early/late separation across biologically diverse differentiation systems.

Unlike CytoTRACE and expression-level entropy, SAPPHIRE operates on the structure of gene co-expression modules rather than raw expression magnitude, enabling detection of transient transcriptional programmes that are invisible to gene-count-based methods.

---

## Key Results

Validated across four human stem cell differentiation datasets:

| Dataset | Pathway Entropy AUC | Network Dispersion AUC | Composite AUC |
|---|---|---|---|
| Cardiomyocyte | 0.928 | 0.739 | 0.732 |
| Endoderm | 0.981 | 0.934 | 0.984 |
| Kidney | 0.554 | 0.902 | 0.844 |
| Neuro | 0.960 | 0.935 | 0.954 |
| **Mean** | **0.856** | **0.878** | **0.879** |

SAPPHIRE mean Composite AUC = **0.879** vs CytoTRACE = 0.656 vs expression entropy = 0.651.

SAPPHIRE identified a transient cardiac commitment module (M4) peaking 1.67× at D2 relative to flanking timepoints (Mann–Whitney p < 2.2×10⁻³⁰⁸), a state invisible to CytoTRACE (AUC = 0.526).

---

## Installation

```bash
git clone https://github.com/ziye003/SAPPHIRE.git
cd SAPPHIRE
pip install -e .
```

### Requirements

```
python >= 3.8
numpy >= 1.20
pandas >= 1.3
scipy >= 1.7
scikit-learn >= 1.0
scanpy >= 1.8
networkx >= 2.6
matplotlib >= 3.4
seaborn >= 0.11
scikit-posthocs >= 0.7
```

---

## Quick Start

```python
import scanpy as sc
from sapphire.core import (
    load_and_prepare,
    hvg_filter,
    build_network,
    compute_per_cell_metrics,
    compute_composite,
)

# Load your data
adata = sc.read_h5ad("your_data.h5ad")

# Filter to top 2000 highly variable genes
adata = hvg_filter(adata, n_top=2000)

# Build label-free gene co-expression network
params = {
    "n_top_genes":      2000,
    "top_k_edges":      10,
    "min_corr":         0.25,
    "leiden_resolution": 1.5,
    "min_module_size":  10,
    "random_state":     0,
}
modules, gene_names = build_network(adata, params)

# Compute per-cell metrics
time_col = "timepoint"   # column in adata.obs with timepoint labels
pc_df = compute_per_cell_metrics(adata, modules, time_col)
pc_df["composite"] = compute_composite(pc_df)

print(pc_df[["timepoint", "pathway_entropy", "network_dispersion", "composite"]].head())
```

---

## Input Format

SAPPHIRE expects an `AnnData` object:

- `adata.X` — log-normalised expression matrix (cells × genes), or raw counts (normalisation applied internally)
- `adata.obs[time_col]` — timepoint labels (e.g. `"D0"`, `"D15"`, `"00h"`, `"Day7"`)

Accepted time column names auto-detected: `timepoint`, `tp_day`, `tp_hours`, `day`, `time`.

---

## Output

`compute_per_cell_metrics()` returns a DataFrame with one row per cell:

| Column | Description |
|---|---|
| `timepoint` | Experimental timepoint label |
| `pathway_entropy` | Per-cell Shannon entropy of module activation (higher = more plastic) |
| `network_dispersion` | Per-cell mean cosine distance to k=30 nearest neighbours in module activation space |
| `composite` | Rank-averaged combination of entropy and dispersion, normalised to [0, 1] |

Results are saved per dataset under `data/sapphire_validation_v2/{Dataset}/`:
- `{Dataset}_per_cell_metrics.csv` — per-cell scores
- `{Dataset}_module_activation.csv` — cells × modules activation matrix
- `{Dataset}_summary.csv` — AUC and marker correlation summary

---

## Repository Structure

```
SAPPHIRE/
├── sapphire/
│   ├── __init__.py
│   └── core.py                  # Core functions: load, filter, build_network,
│                                #   compute_per_cell_metrics, compute_composite
│
├── scripts/                     # Analysis and reproduction scripts
│   ├── run_pipeline.py          # Main validation pipeline (all 4 datasets)
│   ├── ablation.py              # Ablation analysis and violin plots
│   ├── method_comparison.py     # SAPPHIRE vs CytoTRACE vs expression entropy
│   ├── holdout_validation.py    # Strict holdout-cell validation (20 splits)
│   ├── hyperparameter_sensitivity.py
│   ├── resampling_stability.py
│   ├── read_depth_control.py
│   ├── enrichment_analysis.py   # GO/KEGG/Hallmark enrichment via gseapy
│   ├── export_module_genes.py   # Export module gene lists
│   ├── heatmap_pseudopathway.py # Module activation heatmaps
│   ├── umap_plots.py            # Cell UMAP with SAPPHIRE score overlays
│   ├── umap_with_module_overlay.py  # Module activation overlays on UMAP
│   ├── gene_umap.py             # Gene UMAP coloured by module
│   ├── generate_report.py       # Summary PDF report
│   └── run_all.py               # Run all analyses in sequence
│
├── docs/
│   ├── figures/                 # Output figures
│   └── enrichment/              # GO enrichment results and dotplots
│
├── examples/
│   ├── run_example.py
│   └── data/
│       └── example_small.h5ad
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## Reproducing the Paper Results

### 1. Run the main validation pipeline

```bash
conda activate your_env
cd SAPPHIRE
python scripts/run_pipeline.py
```

This runs SAPPHIRE on all four datasets (Cardiomyocyte, Endoderm, Kidney, Neuro) and saves per-cell metrics, module activation matrices, and AUC summaries.

### 2. Run individual analyses

```bash
# Method comparison (vs CytoTRACE, expression entropy)
python scripts/method_comparison.py

# Ablation analysis + violin plots
python scripts/ablation.py

# Holdout-cell validation
python scripts/holdout_validation.py

# Hyperparameter sensitivity
python scripts/hyperparameter_sensitivity.py

# Resampling stability
python scripts/resampling_stability.py

# Read-depth control
python scripts/read_depth_control.py
```

### 3. Generate UMAP figures

```bash
# SAPPHIRE score overlays on cell UMAP
python scripts/umap_plots.py

# Module activation overlays (key figure)
python scripts/umap_with_module_overlay.py
```

### 4. Gene ontology enrichment

```bash
# Export module gene lists first
python scripts/export_module_genes.py --dataset all

# Run enrichment analysis
python scripts/enrichment_analysis.py --dataset all
```

---

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `n_top_genes` | 2000 | Highly variable genes selected by variance |
| `top_k_edges` | 10 | Maximum co-expression edges per gene |
| `min_corr` | 0.25 | Minimum Spearman correlation threshold |
| `leiden_resolution` | 1.5 | Resolution for Leiden community detection |
| `min_module_size` | 10 | Minimum genes per module |
| `knn_k` | 30 | Neighbours for Network Dispersion (kNN density) |
| `random_state` | 0 | Random seed for reproducibility |

---

## Algorithm

### 1. HVG Selection
Top `n_top_genes` highly variable genes selected by expression variance across all cells. No timepoint labels used.

### 2. Label-free Network Construction
Gene co-expression network built from Spearman rank correlation. For each gene, the top `top_k_edges` co-expressed partners (|r| ≥ `min_corr`) are retained as undirected edges. Modules identified by greedy modularity optimisation (Leiden, `leiden_resolution`). No timepoint labels used at any stage.

### 3. Pathway Entropy
For each cell *i*, module activation vector **A**ᵢ ∈ ℝᴹ is computed as the mean log-normalised expression of genes in each module. Pathway Entropy:

```
Hᵢ = −Σₖ pᵢₖ log₂(pᵢₖ)    where    pᵢₖ = |Aᵢₖ| / Σₖ|Aᵢₖ|
```

High entropy = diffuse activation across modules (plastic). Low entropy = selective module activation (committed).

### 4. Network Dispersion
Per-cell mean cosine distance to *k* = 30 nearest neighbours in module activation space:

```
Dᵢ = (1/k) Σⱼ∈Nᵢ cosine_distance(Aᵢ, Aⱼ)
```

Computed via brute-force cosine kNN on L2-normalised activation vectors. No timepoint labels used.

### 5. Composite Score
Rank average of Pathway Entropy and Network Dispersion, normalised to [0, 1]:

```
Cᵢ = (rank(Hᵢ) + rank(Dᵢ)) / (2N)
```

---

## Troubleshooting

**"Cannot find time column"** — specify explicitly: `time_col="your_column_name"`

**"Too few edges" warning** — dataset may be small; try reducing `n_top_genes` to 1000 or `min_corr` to 0.15

**Memory error during correlation** — reduce `n_top_genes` to 1000–1500

**No modules identified** — lower `min_corr` to 0.15 or `min_module_size` to 5

**Inconsistent results across runs** — set `random_state=0` in params

---

## Citation

```
[Citation will be added upon publication]
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

## Contact

- **Issues**: [GitHub Issues](https://github.com/ziye003/SAPPHIRE/issues)
- **Email**: yexxx399@umn.edu

## Acknowledgments

Developed at the University of Minnesota Twin Cities  
Departments of Biomedical Engineering and Bioinformatics and Computational Biology
