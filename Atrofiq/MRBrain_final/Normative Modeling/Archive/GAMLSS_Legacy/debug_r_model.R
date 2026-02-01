
# Set library path
user_lib <- "/home/anirudh/R_libs"
if (dir.exists(user_lib)) {
  .libPaths(c(user_lib, .libPaths()))
}

library(gamlss)
library(gamlss.dist)
library(readxl)

file_path <- "/home/anirudh/Brainagepred/MRBrain/Normative Modeling/Input_Data_TIV/Pre/female_left_hippocampus.xlsx"
df_data <- read_excel(file_path)
df_data$Age <- as.numeric(df_data$Age)
df_data <- df_data[!is.na(df_data$Volume) & !is.na(df_data$Age) & !is.na(df_data$TIV), ]

mean_tiv <- mean(df_data$TIV, na.rm=TRUE)
clean_df <- data.frame(
    y = df_data$Volume,
    Age = df_data$Age,
    TIV_scaled = df_data$TIV / mean_tiv
)

print(head(clean_df))


candidate_families <- list(NO = NO(), GA = GA(), BCCG = BCCG(), BCPE = BCPE())
best_aic <- Inf
best_model <- NULL

print("Evaluating families...")
for (fam_name in names(candidate_families)) {
    print(paste("Fitting", fam_name))
    model <- try(gamlss(y ~ pb(Age) + TIV_scaled,
                        sigma.formula = ~ pb(Age), 
                        family = candidate_families[[fam_name]],
                        data = clean_df, 
                        trace = FALSE), silent = TRUE)
    
    if (!inherits(model, "try-error")) {
        current_aic <- AIC(model)
        print(paste(fam_name, "AIC:", current_aic))
        if (current_aic < best_aic) {
            best_aic <- current_aic
            best_model <- model
        }
    } else {
        print(paste(fam_name, "Failed to fit"))
    }
}

if (is.null(best_model)) {
    stop("All models failed")
}

print(paste("Best model:", best_model$family[1]))

# Predict
newdata_i <- data.frame(Age = 50, TIV_scaled = 1.0)
print("Attempting prediction with Best Model...")
preds <- try({
     mu_p <- predict(best_model, newdata = newdata_i, what = "mu", type = "response", data = clean_df)
     sigma_p <- predict(best_model, newdata = newdata_i, what = "sigma", type = "response", data = clean_df)
     # For BCCG/BCPE check nu/tau
     list(mu=mu_p, sigma=sigma_p)
})

if (inherits(preds, "try-error")) {
    print("Prediction FAILED:")
    print(attr(preds, "condition"))
} else {
    print("Prediction SUCCESS:")
    print(preds)
}
