#!/usr/bin/env Rscript
# run_scent.R
# ===========
# Runs the official SCENT package (Teschendorff & Enver, Nat Commun 2017;
# github.com/aet21/SCENT) on a SAPPHIRE benchmark dataset exported by
# export_for_r.py, and writes per-cell signalling-entropy (SR) scores to CSV.
#
# SCENT is NOT the naive per-cell Shannon entropy of the expression vector
# (that is what SAPPHIRE_paper's current "Expr_Entropy" baseline computes).
# SCENT computes an entropy RATE from a random walk over a protein-protein
# interaction (PPI) network, with edge transition probabilities weighted by
# the expression of the two connected genes (DoIntegPPI + CompSRana).
#
# ---- One-time setup ---------------------------------------------------
#   install.packages(c("Matrix", "mclust", "igraph", "qlcMatrix"))
#   if (!requireNamespace("BiocManager", quietly = TRUE))
#       install.packages("BiocManager")
#   BiocManager::install(c("marray", "destiny"))
#   if (!requireNamespace("devtools", quietly = TRUE)) install.packages("devtools")
#   devtools::install_github("aet21/SCENT")
#
# ---- Usage --------------------------------------------------------------
#   Rscript run_scent.R <dataset_name> <export_dir> <output_dir> [ppi_network]
#
#   dataset_name : e.g. Cardiomyocyte / Endoderm / Kidney / Neuro
#   export_dir   : e.g. $SAPPHIRE_DATA_ROOT/scent_slice_export/<dataset_name>
#   output_dir   : where to write <dataset_name>_scent_SR.csv
#   ppi_network  : "net13Jun12" (Pathway Commons v1, default) or "net17Jan16"
#                  (Pathway Commons v2, larger — ~10,000 proteins)
# -------------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: Rscript run_scent.R <dataset_name> <export_dir> <output_dir> [ppi_network]")
}
dataset_name <- args[1]
export_dir   <- args[2]
output_dir   <- args[3]
ppi_choice   <- ifelse(length(args) >= 4, args[4], "net13Jun12")

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages({
  library(SCENT)
  library(Matrix)
})

expr_path <- file.path(export_dir, paste0(dataset_name, "_expr_linear_genes_x_cells.csv"))
meta_path <- file.path(export_dir, paste0(dataset_name, "_metadata.csv"))

cat(sprintf("Loading expression matrix: %s\n", expr_path))
expr_df <- read.csv(expr_path, row.names = 1, check.names = FALSE)
exp.m <- as.matrix(expr_df)
mode(exp.m) <- "numeric"
cat(sprintf("  %d genes x %d cells\n", nrow(exp.m), ncol(exp.m)))

# Gene identifiers in the SAPPHIRE HVG matrix are gene symbols; SCENT's
# bundled networks (net13Jun12 / net17Jan16) are also indexed by gene
# symbol (annotated from Entrez via the package's own mapping) — see the
# SCENT vignette. If DoIntegPPI reports near-zero overlap, check that
# rownames(exp.m) are HGNC gene symbols (not Ensembl IDs).
data(list = ppi_choice)
ppiA.m <- get(paste0(ppi_choice, ".m"))
cat(sprintf("Loaded PPI network '%s': %d nodes\n", ppi_choice, nrow(ppiA.m)))

cat("Running DoIntegPPI (matching expression matrix to PPI network) ...\n")
integ.l <- DoIntegPPI(exp.m = exp.m, ppiA.m = ppiA.m)
cat(sprintf("  Matched network has %d genes\n", nrow(integ.l$expMC)))

cat("Running CompSRana (signalling entropy rate, exact) ...\n")
sr.o <- CompSRana(integ.l, local = FALSE, mc.cores = max(1, parallel::detectCores() - 1))

out_df <- data.frame(cell = colnames(integ.l$expMC), SCENT_SR = sr.o$SR)
out_path <- file.path(output_dir, paste0(dataset_name, "_scent_SR.csv"))
write.csv(out_df, out_path, row.names = FALSE)
cat(sprintf("Wrote %s (%d cells)\n", out_path, nrow(out_df)))

# Optional faster proxy (Pearson correlation of connectome & transcriptome,
# CCAT) if CompSRana is too slow for the full cell count on your machine:
# ccat.v <- CompCCAT(exp = integ.l$expMC, ppiA = integ.l$adjMC)
# write.csv(data.frame(cell = colnames(integ.l$expMC), SCENT_CCAT = ccat.v),
#           file.path(output_dir, paste0(dataset_name, "_scent_CCAT.csv")),
#           row.names = FALSE)
