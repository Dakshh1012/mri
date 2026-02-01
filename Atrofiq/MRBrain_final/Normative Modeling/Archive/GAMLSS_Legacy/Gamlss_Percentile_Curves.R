library(gamlss)
library(gamlss.dist)
library(readxl)
library(writexl)

feature_list <- list(
  c('female_3rd_ventricle', 'BCPE'),
  c('female_4th_ventricle', 'BCPE'),
  c('female_brain-stem', 'BCPE'),
  c('female_csf', 'BCPE'),
  c('female_left_accumbens_area', 'BCPE'),
  c('female_left_amygdala', 'BCPE'),
  c('female_left_caudate', 'BCPE'),
  c('female_left_cerebellum_cortex', 'BCPE'),
  c('female_left_cerebellum_white_matter', 'BCPE'),
  c('female_left_cerebral_cortex', 'BCPE'),
  c('female_left_cerebral_white_matter', 'BCPE'),
  c('female_left_hippocampus', 'BCPE'),
  c('female_left_inferior_lateral_ventricle', 'BCPE'),
  c('female_left_lateral_ventricle', 'BCPE'),
  c('female_left_pallidum', 'BCPE'),
  c('female_left_putamen', 'BCPE'),
  c('female_left_thalamus', 'BCPE'),
  c('female_left_ventral_DC', 'BCPE'),
  c('female_right_accumbens_area', 'BCPE'),
  c('female_right_amygdala', 'BCPE'),
  c('female_right_caudate', 'BCPE'),
  c('female_right_cerebellum_cortex', 'BCPE'),
  c('female_right_cerebellum_white_matter', 'BCPE'),
  c('female_right_cerebral_cortex', 'BCPE'),
  c('female_right_cerebral_white_matter', 'BCPE'),
  c('female_right_hippocampus', 'BCPE'),
  c('female_right_inferior_lateral_ventricle', 'BCPE'),
  c('female_right_lateral_ventricle', 'BCPE'),
  c('female_right_pallidum', 'BCPE'),
  c('female_right_putamen', 'BCPE'),
  c('female_right_thalamus', 'BCPE'),
  c('female_right_ventral_DC', 'BCPE'),
  c('female_total_intracranial', 'BCPE'),
  c('male_3rd_ventricle', 'BCPE'),
  c('male_4th_ventricle', 'BCPE'),
  c('male_brain-stem', 'BCPE'),
  c('male_csf', 'BCPE'),
  c('male_left_accumbens_area', 'BCPE'),
  c('male_left_amygdala', 'BCPE'),
  c('male_left_caudate', 'BCPE'),
  c('male_left_cerebellum_cortex', 'BCPE'),
  c('male_left_cerebellum_white_matter', 'BCPE'),
  c('male_left_cerebral_cortex', 'BCPE'),
  c('male_left_cerebral_white_matter', 'BCPE'),
  c('male_left_hippocampus', 'BCPE'),
  c('male_left_inferior_lateral_ventricle', 'BCPE'),
  c('male_left_lateral_ventricle', 'BCPE'),
  c('male_left_pallidum', 'BCPE'),
  c('male_left_putamen', 'BCPE'),
  c('male_left_thalamus', 'BCPE'),
  c('male_left_ventral_DC', 'BCPE'),
  c('male_right_accumbens_area', 'BCPE'),
  c('male_right_amygdala', 'BCPE'),
  c('male_right_caudate', 'BCPE'),
  c('male_right_cerebellum_cortex', 'BCPE'),
  c('male_right_cerebellum_white_matter', 'BCPE'),
  c('male_right_cerebral_cortex', 'BCPE'),
  c('male_right_cerebral_white_matter', 'BCPE'),
  c('male_right_hippocampus', 'BCPE'),
  c('male_right_inferior_lateral_ventricle', 'BCPE'),
  c('male_right_lateral_ventricle', 'BCPE'),
  c('male_right_pallidum', 'BCPE'),
  c('male_right_putamen', 'BCPE'),
  c('male_right_thalamus', 'BCPE'),
  c('male_right_ventral_DC', 'BCPE'),
  c('male_total_intracranial', 'BCPE')
)

input_folder <- "C:/Users/saran/OneDrive/Desktop/Normative Modeling/Input Data"
output_folder <- "C:/Users/saran/OneDrive/Desktop/Normative Modeling/Percentiles"

if (!dir.exists(output_folder)) {
    dir.create(output_folder, recursive = TRUE)
}

for (item in feature_list) {
  feature_name <- item[1]
  family_name <- item[2]
  adjusted_feature <- sub("^(male_|female_)", "", feature_name)
  
  file_path <- paste0(input_folder, feature_name, ".xlsx")
  df_data <- read_excel(file_path)
  df_data$Age <- as.numeric(df_data$Age)

  if (family_name == "BCCG") family <- BCCG()
  if (family_name == "BCPE") family <- BCPE()
  if (family_name == "NO")   family <- NO()
  if (family_name == "GA")   family <- GA()
  
  model <- gamlss(
    as.formula(paste0("`", adjusted_feature, "` ~ Age")),
    sigma.formula = ~ Age,
    family = family,
    data = df_data
  )
  
  age_range <- 1:100
  percentiles <- seq(0.01, 0.99, length.out = 99)
  
  output_matrix <- matrix(NA, nrow = length(age_range), ncol = length(percentiles))
  colnames(output_matrix) <- paste0(1:length(percentiles), "th")
  rownames(output_matrix) <- age_range
  
  for (i in seq_along(age_range)) {
    newdata_i <- data.frame(Age = age_range[i])
    mu <- predict(model, newdata = newdata_i, what = "mu", type = "response")
    sigma <- predict(model, newdata = newdata_i, what = "sigma", type = "response")
    
    if (model$family[1] == "BCCG") {
      nu <- predict(model, newdata = newdata_i, what = "nu", type = "response")
      q_vals <- qBCCG(p = percentiles, mu = mu, sigma = sigma, nu = nu)
    } else if (model$family[1] == "BCPE") {
      nu  <- predict(model, newdata = newdata_i, what = "nu", type = "response")
      tau <- predict(model, newdata = newdata_i, what = "tau", type = "response")
      q_vals <- qBCPE(p = percentiles, mu = mu, sigma = sigma, nu = nu, tau = tau)
    } else if (model$family[1] == "NO") {
      q_vals <- qNO(p = percentiles, mu = mu, sigma = sigma)
    } else if (model$family[1] == "GA") {
      q_vals <- qGA(p = percentiles, mu = mu, sigma = sigma)
    } else {
      stop("Unsupported family.")
    }
    
    output_matrix[i, ] <- q_vals
  }
  
  df_percentiles <- as.data.frame(output_matrix)
  df_percentiles$Age <- age_range
  
  output_path <- paste0(output_folder, feature_name, ".xlsx")
  write_xlsx(df_percentiles, path = output_path)
}
