
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob

def load_data(dataset_name, base_dir, feature_name):
    # Load Train/Test metadata for plotting (Age, Sex, Site)
    # We saved train_data.csv and test_data.csv in the output dir
    
    out_dir = os.path.join(base_dir, dataset_name)
    train_meta = pd.read_csv(os.path.join(out_dir, "train_data.csv"))
    test_meta = pd.read_csv(os.path.join(out_dir, "test_data.csv"))
    
    # Load Model Predictions
    model_dir = os.path.join(out_dir, "Models", feature_name)
    
    # PCNtoolkit outputs (standard names)
    # yhat_predict.txt -> predicted mean (mu)
    # ys2_predict.txt -> predicted total variance (sigma^2 + noise)
    # y_test.txt -> Actual Y
    
    try:
        yhat = np.loadtxt(os.path.join(model_dir, 'yhat_predict.txt'))
        var = np.loadtxt(os.path.join(model_dir, 'ys2_predict.txt'))
        y_true = np.loadtxt(os.path.join(model_dir, 'y_test.txt'))
        
        # Calculate Sigma
        sigma = np.sqrt(var)
        
        # Calculate Z-scores
        z_scores = (y_true - yhat) / sigma
        
        return train_meta, test_meta, yhat, sigma, y_true, z_scores
        
    except Exception as e:
        print(f"Error loading {feature_name}: {e}")
        return None, None, None, None, None, None

def plot_verification(dataset_name, base_dir, features, suffix=""):
    plot_dir = os.path.join(base_dir, f"../Verification_Plots_PCN{suffix}")
    os.makedirs(plot_dir, exist_ok=True)
    
    for feature in features:
        print(f"Plotting {feature}...")
        _, test_meta, yhat, sigma, y_true, z_scores = load_data(dataset_name, base_dir, feature)
        
        if yhat is None: continue
        
        # Create DataFrame for plotting
        df_plot = test_meta.copy()
        df_plot['Y_True'] = y_true
        df_plot['Y_Pred'] = yhat
        df_plot['Sigma'] = sigma
        df_plot['Z_Score'] = z_scores
        
        # 1. Fan Chart (Scatter + Percentiles)
        plt.figure(figsize=(10, 6))
        
        # Plot Scatter (Test Data)
        # Color by Site if available
        if 'Site' in df_plot.columns:
            sns.scatterplot(data=df_plot, x='Age', y='Y_True', hue='Site', style='Sex_Code', alpha=0.6)
        else:
            sns.scatterplot(data=df_plot, x='Age', y='Y_True', hue='Sex_Code', alpha=0.6)
            
        # Draw Trend Lines (Mean +/- 1.96 Sigma approx 95% CI)
        # Since BLR is effectively linear/spline, we can sort by Age to draw lines
        sort_idx = np.argsort(df_plot['Age'])
        age_sorted = df_plot['Age'].iloc[sort_idx]
        yhat_sorted = yhat[sort_idx]
        sigma_sorted = sigma[sort_idx]
        
        # Percentiles (Approximate for Gaussian Predictive Dist)
        p50 = yhat_sorted
        p025 = yhat_sorted - 1.96 * sigma_sorted
        p975 = yhat_sorted + 1.96 * sigma_sorted
        
        plt.plot(age_sorted, p50, color='black', linewidth=2, label='Mean Prediction')
        plt.fill_between(age_sorted, p025, p975, color='gray', alpha=0.2, label='95% Predictive Interval')
        
        plt.title(f"{dataset_name} ({suffix.strip('_')}): {feature}\nNormative Model Fit")
        plt.ylabel("Volume (TIV Scaled)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{dataset_name}_{feature}_FanPlot.png"))
        plt.close()
        
        # 2. Z-Score Distribution
        plt.figure(figsize=(8, 6))
        sns.histplot(z_scores, kde=True, bins=30)
        plt.axvline(x=-2, color='red', linestyle='--')
        plt.axvline(x=2, color='orange', linestyle='--')
        plt.title(f"{dataset_name} ({suffix.strip('_')}): {feature}\nZ-Score Distribution")
        plt.xlabel("Z-Score")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{dataset_name}_{feature}_ZDist.png"))
        plt.close()
        
        # Check Extremes
        n_atrophy = np.sum(z_scores < -2)
        n_enlarge = np.sum(z_scores > 2)
        pct_atrophy = (n_atrophy / len(z_scores)) * 100
        pct_enlarge = (n_enlarge / len(z_scores)) * 100
        
        print(f"  {feature}: Atrophy (< -2): {pct_atrophy:.1f}%, Enlargement (> 2): {pct_enlarge:.1f}%")

def main():
    base_dir_root = "/home/anirudh/Brainagepred/MRBrain/Normative Modeling"
    
    # Check both Raw and Harmonized model directories
    model_dirs = {
        "Raw": os.path.join(base_dir_root, "Models", "PCN_Models"),
        "Harmonized": os.path.join(base_dir_root, "Models_Harmonized")
    }
    
    features_to_check = [
        'left_hippocampus', 
        'right_hippocampus', 
        'left_lateral_ventricle',
        'brain_stem',
        'left_cerebral_cortex' 
    ]
    
    for model_type, base_dir in model_dirs.items():
        if not os.path.exists(base_dir):
            continue
            
        print(f"\n=== Verifying {model_type} Models ===")
        suffix = f"_{model_type}"
        
        print(f"--- Verifying Pre-Contrast ({model_type}) ---")
        try:
            plot_verification("Pre", base_dir, features_to_check, suffix=suffix)
        except Exception as e:
            print(f"Skipping Pre Verification: {e}")
        
        print(f"--- Verifying Post-Contrast ({model_type}) ---")
        try:
            plot_verification("Post", base_dir, features_to_check, suffix=suffix)
        except Exception as e:
            print(f"Skipping Post Verification: {e}")

if __name__ == "__main__":
    main()
