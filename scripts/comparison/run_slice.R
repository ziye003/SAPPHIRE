#!/usr/bin/env Rscript
# run_slice.R
# ===========
# Runs the official SLICE package (Guo, Bao, Wagner, Whitsett & Xu, NAR 2017;
# github.com/xu-lab/SLICE) on a SAPPHIRE benchmark dataset exported by
# export_for_r.py, and writes per-cell scEntropy scores to CSV.
#
# SLICE computes entropy over predefined FUNCTIONAL GENE SETS (a
# kappa-statistic gene-gene functional similarity matrix bundled with the
# package), not over raw per-gene expression and not over a PPI network —
# it is a third, distinct baseline from both CytoTRACE (gene count) and
# SCENT (PPI signalling entropy). This is the correct reference to compare
# against SAPPHIRE's module-based (data-driven, not predefined-pathway)
# entropy.
#
# ---- One-time setup ---------------------------------------------------
#   install.packages(c("ggplot2", "igraph", "reshape2", "entropy", "cluster",
#                       "princurve", "lmtest", "mgcv"))
#   if (!requireNamespace("BiocManager", quietly = TRUE))
#       install.packages("BiocManager")
#   BiocManager::install(c("Biobase", "graph", "BioNet"))
#   if (!requireNamespace("devtools", quietly = TRUE)) install.packages("devtools")
#   devtools::install_github("xu-lab/SLICE")
#
#   # Confirm the bundled human gene-set kappa-similarity matrix object name
#   # (this varies by package version/branch — check what's actually shipped):
#   library(SLICE)
#   data(package = "SLICE")     # look for something like km.human / kappa_human
#
# ---- Usage --------------------------------------------------------------
#   Rscript run_slice.R <dataset_name> <export_dir> <output_dir> [B.num] [exp.cutoff]
#
#   B.num      : bootstrap iterations (paper default 100; demo/FB.R value).
#                Reduce (e.g. 20) for a faster first pass on large datasets.
#   exp.cutoff : expression threshold for "detected" gene (default 1, as in
#                SLICE's own demo/FB.R — check units against your input scale).
# -------------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: Rscript run_slice.R <dataset_name> <export_dir> <output_dir> [B.num] [exp.cutoff]")
}
dataset_name <- args[1]
export_dir   <- args[2]
output_dir   <- args[3]
B_num        <- ifelse(length(args) >= 4, as.integer(args[4]), 100L)
exp_cutoff   <- ifelse(length(args) >= 5, as.numeric(args[5]), 1)

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages({
  library(SLICE)
})

expr_path <- file.path(export_dir, paste0(dataset_name, "_expr_linear_genes_x_cells.csv"))
meta_path <- file.path(export_dir, paste0(dataset_name, "_metadata.csv"))

cat(sprintf("Loading expression matrix: %s\n", expr_path))
expr_df <- read.csv(expr_path, row.names = 1, check.names = FALSE)
exp.m <- as.matrix(expr_df)
mode(exp.m) <- "numeric"
cat(sprintf("  %d genes x %d cells\n", nrow(exp.m), ncol(exp.m)))

meta_df <- read.csv(meta_path)
stopifnot(all(colnames(exp.m) == meta_df$cell))

# --- REQUIRED MANUAL STEP -------------------------------------------------
# SLICE's getEntropy() needs a kappa gene-set similarity matrix (km) built
# from a functional gene-set collection (e.g. GO Biological Process). The
# package bundles a mouse one used in demo/FB.R (km, loaded from their data
# folder); for human data you must either (a) load the package's bundled
# human kappa matrix if shipped (check `data(package = "SLICE")`), or
# (b) build one yourself following the SLICE vignette's construct_network /
# kappa-statistic procedure over a human GO BP gene set collection (e.g. via
# the `GSEABase`/`org.Hs.eg.db` packages), then save it as human_km.rda and
# load it here. This step needs a real R session with internet access to
# resolve — it cannot be completed in this sandbox.
#
# load("human_km.rda")   # must define an object called `km`
# -------------------------------------------------------------------------
if (!exists("km")) {
  stop(paste(
    "No kappa gene-set similarity matrix ('km') loaded.",
    "See the REQUIRED MANUAL STEP comment above: load or build the human",
    "functional gene-set kappa matrix before running getEntropy()."
  ))
}

cat("Constructing SLICE object ...\n")
sc <- construct(exprmatrix = exp.m, cellidentity = meta_df$timepoint)

cat(sprintf("Running getEntropy (bootstrap, B.num=%d, exp.cutoff=%s) ...\n", B_num, exp_cutoff))
sc <- getEntropy(
  sc,
  km             = km,
  calculation    = "bootstrap",
  B.num          = B_num,
  exp.cutoff     = exp_cutoff,
  B.size         = 1000,
  clustering.k   = floor(sqrt(1000 / 2)),
  random.seed    = 201602
)

out_df <- data.frame(cell = colnames(exp.m), SLICE_entropy = sc@entropy)
out_path <- file.path(output_dir, paste0(dataset_name, "_slice_entropy.csv"))
write.csv(out_df, out_path, row.names = FALSE)
cat(sprintf("Wrote %s (%d cells)\n", out_path, nrow(out_df)))
