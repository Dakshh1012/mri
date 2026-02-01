library(gamlss)
library(gamlss.dist)
library(readxl)
library(writexl)

# Set library path
user_lib <- "~/R_libs"
if (dir.exists(user_lib)) {
  .libPaths(c(user_lib, .libPaths()))
}

file_path <- "/home/anirudh/Brainagepred/MRBrain/Normative Modeling/Input_Data_TIV/Pre/female_left_hippocampus.xlsx"
cat("Reading file...\n")
df_data <- read_excel(file_path)
print(head(df_data))

cat("Fitting model...\n")
mean_tiv <- mean(df_data$TIV, na.rm=TRUE)
df_data$TIV_scaled <- df_data$TIV / mean_tiv

model <- try(gamlss(Volume ~ pb(Age) + TIV_scaled,
                    sigma.formula = ~ pb(Age), 
                    family = BCCG(),
                    data = df_data, 
                    trace = TRUE))

if (inherits(model, "try-error")) {
    print(model)
} else {
    print("Model fitted successfully")
    print(summary(model))
    
    cat("Writing output...\n")
    df_out <- data.frame(Age=1:100, Val=1:100)
    write_xlsx(df_out, "test_output.xlsx")
    print("Write success")
}
