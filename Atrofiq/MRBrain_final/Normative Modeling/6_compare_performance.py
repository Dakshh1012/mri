
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

def load_predictions(model_dir, feature):
    """Load y_true, y_pred, variance for a feature."""
    # PCNtoolkit outputs: y_test.txt, yhat_predict.txt, ys2_predict.txt (variance)
    path = os.path.join(model_dir, feature)
    
    try:
        y_true = np.loadtxt(os.path.join(path, "y_test.txt"))
        y_pred = np.loadtxt(os.path.join(path, "yhat_predict.txt"))
        y_var = np.loadtxt(os.path.join(path, "ys2_predict.txt"))
        return y_true, y_pred, y_var
    except IOError:
        return None, None, None

def calculate_metrics(y_true, y_pred, y_var):
    """Compute MSLL, RMSE, Expl_Var."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # MSLL (Mean Standardized Log Loss) - requires variance
    # NLL = 0.5 * log(2*pi*sigma^2) + (y-mu)^2 / (2*sigma^2)
    # We approximate MSLL by comparing to trivial mean model? 
    # Or just report Negative Log Likelihood (NLL). Lower is better.
    nll = 0.5 * np.log(2 * np.pi * y_var) + ((y_true - y_pred)**2) / (2 * y_var)
    msll = np.mean(nll)
    
    # Explained Variance (R2)
    r2 = r2_score(y_true, y_pred)
    
    return rmse, msll, r2

def main():
    base_dir = "/home/anirudh/Brainagepred/MRBrain/Normative Modeling"
    
    dirs = {
        "Raw (Site Covariate)": os.path.join(base_dir, "Models", "PCN_Models", "Pre", "Models"),
        "Harmonized (GLM)": os.path.join(base_dir, "Models_Harmonized", "Pre", "Models")
    }
    
    results = []
    
    # Find common features
    if not os.path.exists(dirs["Raw (Site Covariate)"]):
         print(f"Raw dir not found: {dirs['Raw (Site Covariate)']}")
         return

    feats_1 = os.listdir(dirs["Raw (Site Covariate)"])
    feats_2 = os.listdir(dirs["Harmonized (GLM)"])
    
    common_feats = set(feats_1).intersection(feats_2)
    # Filter only directories
    common_feats = [f for f in common_feats if os.path.isdir(os.path.join(dirs["Raw (Site Covariate)"], f))]
    
    print(f"Comparing {len(common_feats)} features...")
    
    for feat in common_feats:
        row = {'Feature': feat}
        
        for model_name, path in dirs.items():
            yt, yp, yv = load_predictions(path, feat)
            
            if yt is None:
                continue
                
            rmse, msll, r2 = calculate_metrics(yt, yp, yv)
            row[f'{model_name} RMSE'] = rmse
            row[f'{model_name} MSLL'] = msll
            row[f'{model_name} R2'] = r2
            
        if len(row) > 1: # at least one model loaded
            results.append(row)
            
    df_res = pd.DataFrame(results)
    
    if df_res.empty:
        print("No matches found.")
        return

    # Calculate Improvement (Harmonized - Raw)
    # For RMSE/MSLL: Negative is Improvement (Lower is better)
    # For R2: Positive is Improvement (Higher is better)
    
    df_res['RMSE_Diff'] = df_res['Harmonized (GLM) RMSE'] - df_res['Raw (Site Covariate) RMSE']
    df_res['MSLL_Diff'] = df_res['Harmonized (GLM) MSLL'] - df_res['Raw (Site Covariate) MSLL']
    df_res['R2_Diff'] = df_res['Harmonized (GLM) R2'] - df_res['Raw (Site Covariate) R2']
    
    # Save Summary
    out_csv = os.path.join(base_dir, "Comparison_Plots_Trends", "Model_Performance_Comparison.csv")
    df_res.to_csv(out_csv, index=False)
    
    print("\n--- Model Comparison Summary ---")
    print(df_res[['Feature', 'RMSE_Diff', 'MSLL_Diff', 'R2_Diff']].describe())
    
    # Print Top 5 Improvements
    print("\nTop 5 Improved Features (MSLL):")
    print(df_res.sort_values('MSLL_Diff').head(5)[['Feature', 'MSLL_Diff']])

    # Plot R2 Comparison
    plt.figure(figsize=(8, 8))
    plt.scatter(df_res['Raw (Site Covariate) R2'], df_res['Harmonized (GLM) R2'], alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel("Raw R2")
    plt.ylabel("Harmonized R2")
    plt.title("Explained Variance Comparison")
    plt.savefig(os.path.join(base_dir, "Comparison_Plots_Trends", "R2_Select_Scatter.png"))
    
if __name__ == "__main__":
    main()
