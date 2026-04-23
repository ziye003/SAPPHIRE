# SAPPHIRE

**S**ingle-cell **A**nalysis of **P**athway **P**lasticity via **H**eterogeneity-**I**nformed **R**egulatory **E**ntropy

A label-free network-based framework for quantifying transcriptional plasticity from scRNA-seq data.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## Overview

SAPPHIRE quantifies transcriptional plasticity by building gene co-expression networks **without using temporal labels**, then decomposing plasticity into two complementary metrics:

- **Pathway Entropy** — per-cell Shannon entropy of module activation. High entropy = diffuse, stem-like transcriptional state.
- **Network Dispersion** — population-level heterogeneity measured as median cosine distance to the per-timepoint centroid in module-activation space. High dispersion = heterogeneous, plastic population.
- **Composite Score** — rank-average of Pathway Entropy and Network Dispersion.

The key design principle: **timepoint labels are never used during network construction**, only for downstream validation. This eliminates circular reasoning between network inference and trajectory evaluation.

---

## Algorithm

```
1. Load scRNA-seq data (AnnData .h5ad)
2. Normalize and log-transform (if not already done)
3. Select top-N highly variable genes by variance
4. Build label-free gene co-expression network
   - Spearman rank correlation (batched matrix computation)
   - Top-k edges per gene, filtered by min_corr threshold
   - Greedy modularity community detection
5. Compute per-cell metrics
   - Pathway Entropy: Shannon entropy of |activation| / row_sum
   - Network Dispersion: median cosine distance to timepoint centroid
6. Composite: rank-average of the two metrics
```

---

## Validation Results

SAPPHIRE was validated across four human stem-cell differentiation datasets:

| Dataset | Cells | Timepoints | Composite AUC | Dispersion AUC | Entropy AUC |
|---|---|---|---|---|---|
| Cardiomyocyte | 30,000 | D0/D2/D5/D15/D30 | 0.997 | 1.000 | 0.880 |
| Endoderm | 7,580 | 0h/12h/24h/36h/72h/96h | 1.000 | 1.000 | 0.951 |
| Kidney | 5,244 | Day7/Day12/Day19/Day26 | 1.000 | 1.000 | 0.965 |
| Neuro | 30,000 | D11/D30/D52 | 1.000 | 1.000 | 0.981 |

**Method comparison** (mean Composite AUC):

| Method | Mean AUC |
|---|---|
| **SAPPHIRE** | **0.999** |
| CytoTRACE | 0.656 |
| Expression Entropy | 0.651 |

### Robustness

| Test | Result |
|---|---|
| Hyperparameter sensitivity (9 combos × 4 datasets) | mean AUC = 0.994, within-dataset SD = 0.012 |
| Resampling stability (20 × 80% subsample) | mean AUC = 0.953 ± 0.035 |
| Read-depth control (regress log nUMI) | ΔAUC ≤ 0 all datasets |
| Holdout-cell validation (80/20 split, 20 repeats) | Entropy AUC 0.684–0.956; Dispersion Spearman ρ 0.551–1.000 |

The holdout-cell validation is the strictest test: gene modules are inferred from 80% of cells, and the remaining 20% are scored using the fixed module structure without any network reconstruction. This rules out overfitting to the evaluation set.

---

## Installation

```bash
git clone https://github.com/ziye003/SAPPHIRE.git
cd SAPPHIRE
pip install -r requirements.txt
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
```

---

## Usage

### Notebook (recommended)

```python
# Step 1: load core
exec(open("sapphire_core.py").read())

# Step 2: run main validation
exec(open("sapphire_validation_all.py").read())
run_pipeline(dataset="Cardiomyocyte")   # single dataset
run_pipeline()                           # all four datasets
```

### Robustness tests

```python
exec(open("sapphire_core.py").read())

# Hyperparameter sensitivity
exec(open("hyperparameter_sensitivity.py").read())
run_hyperparameter_sensitivity()

# Resampling stability
exec(open("resampling_stability.py").read())
run_resampling_stability()

# Read-depth control
exec(open("read_depth_control.py").read())
run_read_depth_control()

# Method comparison
exec(open("sapphire_validation_all.py").read())
exec(open("method_comparison_all.py").read())
run_method_comparison()

# Holdout-cell validation
exec(open("sapphire_core.py").read())
exec(open("holdout_validation.py").read())
main()
```

---

## Input Format

SAPPHIRE expects an `AnnData` object (`.h5ad`):

```
adata.obs:  must contain a timepoint column (e.g. "timepoint")
            values like "D0", "D15", "Day7", "00h", "96h"

adata.X:    expression matrix (raw counts or log-normalized)
            if raw counts: normalize_total + log1p applied automatically
            if already log1p: set already_log1p=True in DATASETS_CONFIG
```

---

## Configuration

Edit `DATASETS_CONFIG` in `sapphire_core.py` to add your own dataset:

```python
DATASETS_CONFIG = {
    "MyDataset": {
        "file"         : Path("/path/to/data.h5ad"),
        "time_col"     : "timepoint",         # obs column with timepoint labels
        "early_tp"     : "Day0",              # earliest timepoint label
        "late_tp"      : "Day30",             # latest timepoint label
        "already_log1p": False,               # True if data is already log-normalized
        "stem_markers" : ["POU5F1", "SOX2"],  # stemness marker genes
        "diff_markers" : ["GENE1", "GENE2"],  # differentiation marker genes
    },
}
```

Default parameters (adjust if needed):

```python
SAPPHIRE_PARAMS = {
    "n_top_genes"      : 2000,   # HVGs to select
    "top_k_edges"      : 10,     # max edges per gene
    "min_corr"         : 0.25,   # minimum Spearman correlation
    "leiden_resolution": 1.5,    # community detection resolution
    "min_module_size"  : 10,     # minimum genes per module
    "random_state"     : 0,
}
```

---

## Output Files

After running `run_pipeline()`:

```
sapphire_validation_v2/
  {Dataset}/
    {Dataset}_sapphire_metrics.pdf      # trajectory plot
    {Dataset}_per_cell_metrics.csv      # per-cell scores
    {Dataset}_summary.csv               # dataset-level summary
  ALL_datasets_summary.csv              # combined summary table
```

---

## File Structure

```
SAPPHIRE/
  sapphire_core.py               # core functions and configuration
  sapphire_validation_all.py     # main validation pipeline
  hyperparameter_sensitivity.py  # parameter grid search
  resampling_stability.py        # 20x 80% subsampling stability test
  read_depth_control.py          # sequencing depth confound test
  method_comparison_all.py       # comparison vs CytoTRACE, expression entropy
  holdout_validation.py          # strict holdout-cell validation (20 splits)
  generate_report.py             # summary PDF + CSV report
  run_all.py                     # command-line entry point
  requirements.txt
  README.md
```

---

## Troubleshooting

**`NameError: load_and_prepare not found`**  
Run `exec(open("sapphire_core.py").read())` first.

**Memory error during correlation computation**  
Reduce `n_top_genes` to 1000–1500 in `SAPPHIRE_PARAMS`.

**No modules detected**  
Lower `min_corr` to 0.15 or `min_module_size` to 5.

**`__file__` not defined**  
You are running via `exec()` in a notebook — this is expected and handled automatically.

---

## Citation

*Citation to be added upon publication.*

---

## Contact

- Issues: [GitHub Issues](https://github.com/ziye003/SAPPHIRE/issues)
- Email: [yexxx399@umn.edu](mailto:yexxx399@umn.edu)

Developed at the University of Minnesota Twin Cities  
Departments of Biomedical Engineering and Bioinformatics and Computational Biology
