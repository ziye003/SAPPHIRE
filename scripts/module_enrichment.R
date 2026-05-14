# Module enrichment in R
# ALS May 5, 2026
# Updated: run all 4 datasets, save plots automatically

library(tidyverse)
library(clusterProfiler)
library(org.Hs.eg.db)

# Dataset list
datasets <- c("Cardiomyocyte", "Endoderm", "Kidney", "Neuro")

for (ds in datasets) {

  cat("\n========================================\n")
  cat("Processing:", ds, "\n")
  cat("========================================\n")

  # Read wide-format CSV
  filename <- paste0(tolower(ds), "_module_genes_wide.csv")

  if (!file.exists(filename)) {
    cat("File not found, skipping:", filename, "\n")
    next
  }

  df <- read_csv(filename, show_col_types = FALSE)

  # Convert to long format
  df_long <- df %>%
    pivot_longer(
      cols = everything(),
      names_to = "exp",
      values_to = "gene"
    ) %>%
    filter(!is.na(gene) & gene != "")

  # Convert to Entrez IDs
  converted <- bitr(df_long$gene,
                    fromType = "SYMBOL",
                    toType   = "ENTREZID",
                    OrgDb    = org.Hs.eg.db)

  final      <- df_long %>% left_join(converted, by = c("gene" = "SYMBOL"))
  clean_final <- na.omit(final)

  # Group by module
  gene_lists <- split(clean_final$ENTREZID, clean_final$exp)

  # GO BP enrichment
  compare_resultBP <- compareCluster(
    geneCluster  = gene_lists,
    fun          = "enrichGO",
    OrgDb        = org.Hs.eg.db,
    ont          = "BP",
    pvalueCutoff = 0.05
  )

  # Plot
  A <- dotplot(compare_resultBP,
               showCategory = 5,
               title = paste("Gene Ontology Enrichment (BP) —", ds, "Modules"))

  AA <- A +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8)) +
    theme(axis.text.y = element_text(hjust = 1, size = 8))

  # Save plot
  plot_file <- paste0(tolower(ds), "_GO_dotplot.png")
  ggsave(plot_file, plot = AA, width = 12, height = 8, dpi = 150)
  cat("Plot saved:", plot_file, "\n")

  # Save results CSV
  csv_file <- paste0(tolower(ds), "_GO_results.csv")
  write_csv(as.data.frame(compare_resultBP), csv_file)
  cat("Results saved:", csv_file, "\n")
}

cat("\n========================================\n")
cat("All done!\n")
cat("========================================\n")
