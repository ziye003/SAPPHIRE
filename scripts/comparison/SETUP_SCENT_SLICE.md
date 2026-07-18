# Running the SCENT / SLICE comparison

This adds two baselines that are conceptually distinct from what's already
in Table 3:

| Method | What it actually measures | Already in paper? |
|---|---|---|
| CytoTRACE | gene count per cell | Yes (Table 3) |
| "Expr_Entropy" (current) | naive per-cell Shannon entropy of the expression vector | Yes, but **mislabeled** — see note below |
| **SCENT** (Teschendorff & Enver 2017) | entropy *rate* of a random walk over a protein-protein interaction network, weighted by expression | No — this script adds it |
| **SLICE** (Guo et al. 2017) | entropy over predefined functional gene sets (kappa-statistic gene-gene similarity) | No — this script adds it |

**Important finding:** the paper's current "Expr_Entropy" baseline (cited to
ref 10 / Teschendorff & Enver) is a naive whole-transcriptome Shannon
entropy, not the actual SCENT algorithm (which requires a PPI network and a
random-walk entropy-rate calculation — see `run_scent.R`). `merge_scent_slice_results.py`
renames it to `Naive_Expression_Entropy` when building the combined table so
Table 3 doesn't misattribute it. Once you have real SCENT numbers, decide
whether to keep the naive-entropy row too (it's still a useful third point of
comparison) or drop it.

## 1. Install R packages (needs internet access — do this on your machine, not in this sandbox)

```r
# --- SCENT ---
install.packages(c("Matrix", "mclust", "igraph", "qlcMatrix"))
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install(c("marray", "destiny"))
if (!requireNamespace("devtools", quietly = TRUE)) install.packages("devtools")
devtools::install_github("aet21/SCENT")

# --- SLICE ---
install.packages(c("ggplot2", "igraph", "reshape2", "entropy", "cluster",
                    "princurve", "lmtest", "mgcv"))
BiocManager::install(c("Biobase", "graph", "BioNet"))
devtools::install_github("xu-lab/SLICE")
```

`destiny` and `BioNet` can be slow/fragile to install depending on your R
version — if you hit build errors, that's expected friction with these
older Bioconductor packages, not a mistake in the scripts here.

## 2. Resolve SLICE's human gene-set kappa matrix (manual, needs checking)

SLICE's public demo (`demo/FB.R`) uses a **mouse** kappa-similarity matrix
loaded from the package's bundled data. I could not confirm from package
documentation alone whether a human equivalent ships with the package on
your installed version. Once SLICE is installed, run:

```r
library(SLICE)
data(package = "SLICE")   # look for something like km.human
```

- If a human kappa matrix is bundled, load it as `km` before running
  `run_slice.R` (edit the "REQUIRED MANUAL STEP" block at the top of that
  script to `load()` or `data()` it).
- If not, you'll need to build one following the SLICE vignette's
  kappa-statistic procedure over a human GO Biological Process gene set
  collection (e.g. via `org.Hs.eg.db` + `GSEABase`), then save it as
  `human_km.rda` (must contain an object named `km`).

This is the one step I can't fully script blind — it depends on what's
actually in your installed package version.

## 3. Export SAPPHIRE's benchmark data for R

From inside your existing pipeline (same convention as the other scripts —
exec after `sapphire_validation_all.py`):

```python
exec(open("scripts/comparison/export_for_r.py").read())
export_all(max_cells=8000)   # raise/lower depending on runtime budget
```

This writes `$SAPPHIRE_DATA_ROOT/scent_slice_export/<Dataset>/` with a
genes-x-cells CSV and a metadata CSV per dataset.

## 4. Run SCENT and SLICE per dataset

```bash
export SAPPHIRE_DATA_ROOT=/Users/ziye/Documents/paper/data   # already in your ~/.zshrc

for ds in Cardiomyocyte Endoderm Kidney Neuro; do
  Rscript scripts/comparison/run_scent.R "$ds" \
    "$SAPPHIRE_DATA_ROOT/scent_slice_export/$ds" \
    "$SAPPHIRE_DATA_ROOT/scent_slice_results"

  Rscript scripts/comparison/run_slice.R "$ds" \
    "$SAPPHIRE_DATA_ROOT/scent_slice_export/$ds" \
    "$SAPPHIRE_DATA_ROOT/scent_slice_results"
done
```

Runtime note: `CompSRana` (exact signalling-entropy rate) and SLICE's
bootstrap entropy are both much slower than SAPPHIRE's closed-form metrics.
Start with the smallest dataset (Endoderm, n=758) to sanity-check before
running Cardiomyocyte/Neuro at 8,000 cells.

## 5. Merge results into Table 3's format

```python
exec(open("scripts/comparison/merge_scent_slice_results.py").read())
final_df = build_full_comparison_table()
```

This writes `$SAPPHIRE_DATA_ROOT/scent_slice_results/ALL_method_comparison_with_SCENT_SLICE.csv`
and prints a pivot table (method x dataset, AUC), analogous to Table 3 /
`ALL_method_comparison.csv` from `method_comparison.py`.

## 6. Once you have numbers

Bring the resulting AUC table back and I'll:
- Update Table 3 to include SCENT and SLICE rows
- Rewrite Section 2.2 and the Discussion's SLICE/SCENT paragraph to report
  the real numbers instead of the conceptual-distinction placeholder
- Fix the "Expr_Entropy" mislabeling in Methods 3.7 either way (rename to
  Naive Expression Entropy, or drop it if SCENT supersedes it)
