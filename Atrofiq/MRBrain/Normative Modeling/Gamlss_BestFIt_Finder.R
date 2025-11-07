library(gamlss)
library(gamlss.dist)
library(readxl)

data_folder <- "C:\Users\saran\OneDrive\Desktop\Normative Modeling\Input Data"
output_directory <- "C:\Users\saran\OneDrive\Desktop\Normative Modeling\output"
output_file <- file.path(output_directory, "statistical_results.txt")

if (!dir.exists(output_directory)) {
  dir.create(output_directory, recursive = TRUE)
  cat(sprintf("Created output directory: %s\n", output_directory))
} else {
  cat(sprintf("Output directory already exists: %s\n", output_directory))
}

candidate_families <- list(
  NO   = NO(),
  GA   = GA(),
  BCCG = BCCG(),
  BCPE = BCPE())

sink(output_file)

cat("Statistical Results for All Files\n")
cat("=================================\n\n")
sink()

excel_files <- list.files(data_folder, pattern = "\\.xlsx$", full.names = TRUE)

for (file_path in excel_files) {
  cat(sprintf("Processing file: %s\n", file_path))
  df_data <- try(read_excel(file_path), silent = TRUE)
  
  if (inherits(df_data, "try-error")) {
    cat(sprintf("Skipping file %s: Unable to read the file\n", file_path))
    next
  }
  
  df_data$Age <- as.numeric(df_data$Age)
  feature_names <- setdiff(names(df_data), "Age")
  if (length(feature_names) != 1) {
    cat(sprintf("Skipping file %s: Data must contain exactly one feature besides Age\n", file_path))
    next
  }
  feature_name <- feature_names[1]

  sink(output_file, append = TRUE)
  cat(sprintf("File: %s\n", basename(file_path)))
  cat(sprintf("Feature: %s\n\n", feature_name))
  sink()
  
  for (fam_name in names(candidate_families)) {
    current_family <- candidate_families[[fam_name]]
    cat(sprintf("Fitting model using '%s' distribution...\n", fam_name))

    current_model <- try(
      gamlss(
        as.formula(paste0("`", feature_name, "` ~ Age")),
        sigma.formula = ~ Age,
        family = current_family,
        data = df_data
      ),
      silent = TRUE
    )
    
    if (!inherits(current_model, "try-error")) {
      global_deviance <- current_model$G.deviance
      aic_val <- AIC(current_model)
      bic_val <- BIC(current_model)
      
      sink(output_file, append = TRUE)
      cat(sprintf("Distribution: %s\n", fam_name))
      cat(sprintf("  Global Deviance: %.4f\n", global_deviance))
      cat(sprintf("  AIC: %.4f\n", aic_val))
      cat(sprintf("  SBC (BIC): %.4f\n\n", bic_val))
      sink()
      
    } else {
      cat(sprintf(" - %s: model failed to converge or incompatible data\n", fam_name))
      sink(output_file, append = TRUE)
      cat(sprintf("Distribution: %s\n", fam_name))
      cat("  Model failed to converge or incompatible data\n\n")
      sink()
    }
  }
}

cat("\nAll statistical results saved successfully.\n")