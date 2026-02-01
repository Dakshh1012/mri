# Set library path to include user-installed packages
user_lib <- "/home/anirudh/R_libs"
if (dir.exists(user_lib)) {
  .libPaths(c(user_lib, .libPaths()))
}

library(gamlss)
library(gamlss.dist)
library(readxl)
library(writexl)
library(parallel)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript normative_modeling.R <input_folder> <output_folder>")
}

input_folder <- args[1]
output_folder <- args[2]

if (!dir.exists(output_folder)) {
  dir.create(output_folder, recursive = TRUE)
}

excel_files <- list.files(input_folder, pattern = "\\.xlsx$", full.names = TRUE)

# Determine number of cores
n_cores <- 14
cat(sprintf("Running in parallel with %d cores\n", n_cores))

process_single_file <- function(file_path) {
  # Load libraries inside worker to ensure environment consistency
  library(gamlss)
  library(gamlss.dist)
  library(readxl)
  library(writexl)
  
  df_data <- try(read_excel(file_path), silent = TRUE)
  if (inherits(df_data, "try-error")) return(NULL)
  
  df_data$Age <- as.numeric(df_data$Age)
  
  # Filter out problematic rows
  df_data <- df_data[!is.na(df_data$Volume) & !is.na(df_data$Age) & !is.na(df_data$TIV), ]
  if (nrow(df_data) < 10) return(NULL)

  # Candidate families as strings
  fam_names <- c("NO", "GA", "BCCG", "BCPE")
  
  best_aic <- Inf
  best_fam_name <- NULL
  
  # Use clean dataframe
  mean_tiv <- mean(df_data$TIV, na.rm=TRUE)
  clean_df <- data.frame(
    y = df_data$Volume,
    Age = df_data$Age,
    TIV_scaled = df_data$TIV / mean_tiv
  )

  # 1. Selection Step
  for (f_name in fam_names) {
    fam_fun <- get(f_name, mode = "function")
    
    # Fit model manually
    model <- try(gamlss(y ~ pb(Age) + TIV_scaled,
                        sigma.formula = ~ pb(Age), 
                        family = fam_fun,
                        data = clean_df, 
                        trace = FALSE), silent = TRUE)
    
    if (!inherits(model, "try-error")) {
        current_aic <- AIC(model)
        if (current_aic < best_aic) {
            best_aic <- current_aic
            best_fam_name <- f_name
        }
    }
  }
  
  # 2. Refit Step (Crucial for Prediction Scoping)
  if (!is.null(best_fam_name)) {
    # We construct the call explicitly using 'bquote' and 'as.name'
    # This ensures the model object contains 'family = NO' (symbol), not a variable reference.
    
    call_expr <- bquote(
      gamlss(y ~ pb(Age) + TIV_scaled,
             sigma.formula = ~ pb(Age), 
             family = .(as.name(best_fam_name)), 
             data = clean_df, 
             trace = FALSE)
    )
    
    best_model <- try(eval(call_expr), silent = TRUE)
                             
    if (inherits(best_model, "try-error")) {
        return(sprintf("%s: Refit failed for %s", basename(file_path), best_fam_name))
    }

    # 3. Prediction Step
    age_range <- 1:100
    percentiles_seq <- seq(0.01, 0.99, by = 0.01)
    
    output_matrix <- matrix(NA, nrow = length(age_range), ncol = length(percentiles_seq))
    colnames(output_matrix) <- paste0(1:99, "th")
    
    for (i in seq_along(age_range)) {
      newdata_i <- data.frame(Age = age_range[i], TIV_scaled = 1.0)
      
      preds <- try({
         mu_p <- predict(best_model, newdata = newdata_i, what = "mu", type = "response", data = clean_df)
         sigma_p <- predict(best_model, newdata = newdata_i, what = "sigma", type = "response", data = clean_df)
         list(mu=mu_p, sigma=sigma_p)
      }, silent = TRUE)
      
      if (inherits(preds, "try-error")) next
      
      mu <- preds$mu
      sigma <- preds$sigma
      
      # Sanity Check: Sigma must be positive
      if (sigma <= 0) next

      # Calculate Quantiles
      q_vals <- try({
        args_list <- list(p = percentiles_seq, mu = mu, sigma = sigma)
        
        # If distribution has extra params (nu, tau), predict them
        if (best_fam_name %in% c("BCCG", "BCPE")) {
             nu_p <- predict(best_model, newdata = newdata_i, what = "nu", type = "response", data = clean_df)
             args_list$nu <- nu_p
             if (best_fam_name == "BCPE") {
                 tau_p <- predict(best_model, newdata = newdata_i, what = "tau", type = "response", data = clean_df)
                 args_list$tau <- tau_p
             }
        }
        
        # Call the quantile function (e.g., qNO, qBCCG)
        q_fun_name <- paste0("q", best_fam_name)
        do.call(q_fun_name, args_list)
        
      }, silent = TRUE)
      
      if (!inherits(q_vals, "try-error")) {
         # Force strictly positive volumes (biological constraint)
         q_vals[q_vals < 0] <- 0
         output_matrix[i, ] <- q_vals
      }
    }
    
    df_res <- as.data.frame(output_matrix)
    df_res$Age <- age_range
    df_res <- df_res[, c("Age", setdiff(names(df_res), "Age"))]
    
    write_xlsx(df_res, path = file.path(output_folder, basename(file_path)))
    return(sprintf("%s: Success (%s)", basename(file_path), best_fam_name))
  } else {
    return(sprintf("%s: Failed (No convergence)", basename(file_path)))
  }
}

# Run in parallel
results <- parallel::mclapply(excel_files, process_single_file, mc.cores = n_cores)
