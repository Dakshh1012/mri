
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import kruskal, f_oneway, norm

def perform_pre_eda(df, output_dir):
    print("\n--- Pre-Contrast EDA (Site Comparison) ---")
    
    # 1. Demographics by Site
    print("Demographics by Site:")
    print(df.groupby('Site')[['Age', 'Sex_Code']].agg(['mean', 'std', 'count']))
    
    # Plot Age Distribution by Site
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x='Age', hue='Site', fill=True, common_norm=False, alpha=0.3)
    plt.title("Age Distribution by Site (Pre-Contrast)")
    plt.savefig(os.path.join(output_dir, "Pre_Age_Dist_by_Site.png"))
    plt.close()
    
    # 2. Structural Comparisons
    # Key regions to check for batch effects
    regions = ['left_hippocampus', 'right_hippocampus', 'left_lateral_ventricle', 'left_cerebral_cortex']
    
    for roi in regions:
        if roi not in df.columns: continue
        
        # Boxplot
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='Site', y=roi, showfliers=False)
        sns.stripplot(data=df, x='Site', y=roi, color='black', alpha=0.3, size=2)
        plt.title(f"Site Comparison: {roi} (Pre-Contrast)")
        plt.savefig(os.path.join(output_dir, f"Pre_Site_Boxplot_{roi}.png"))
        plt.close()
        
        # Stats (Kruskal-Wallis)
        groups = [df[df['Site'] == s][roi].values for s in df['Site'].unique()]
        try:
            stat, p = kruskal(*groups)
            print(f"{roi}: Kruskal-Wallis H={stat:.2f}, p={p:.4e}")
            if p < 0.05:
                print(f"  -> SIGNIFICANT comparisons found for {roi}")
        except ValueError as e:
            print(f"  Could not run stats for {roi}: {e}")

def perform_post_eda(df, output_dir):
    print("\n--- Post-Contrast EDA (Outlier Detection) ---")
    
    # Since only MAX exists, we look for homogeneity
    print(f"Total Subjects: {len(df)}")
    
    regions = ['left_hippocampus', 'left_lateral_ventricle', 'left_cerebral_cortex']
    
    for roi in regions:
        if roi not in df.columns: continue
        
        # 1. Distribution Plot
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=roi, kde=True)
        plt.title(f"Distribution: {roi} (Post-Contrast MAX)")
        plt.savefig(os.path.join(output_dir, f"Post_Dist_{roi}.png"))
        plt.close()
        
        # 2. Z-Score Outlier Detection
        mu, sigma = df[roi].mean(), df[roi].std()
        df[f'{roi}_z'] = (df[roi] - mu) / sigma
        
        outliers = df[np.abs(df[f'{roi}_z']) > 3]
        if not outliers.empty:
            print(f"{roi}: Found {len(outliers)} outliers (>3 SD)")
            # print(outliers[['Age', 'Sex_Code', roi, f'{roi}_z']])
        else:
            print(f"{roi}: No outliers >3 SD found.")

def main():
    base_dir = "/home/anirudh/Brainagepred/MRBrain/Normative Modeling"
    data_dir = os.path.join(base_dir, "Data_Prepared")
    output_dir = os.path.join(base_dir, "Comparison_Plots_Trends", "Site_EDA")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Data
    pre_path = os.path.join(data_dir, "Pre", "Pre_Master.csv")
    post_path = os.path.join(data_dir, "Post", "Post_Master.csv")
    
    if os.path.exists(pre_path):
        df_pre = pd.read_csv(pre_path)
        perform_pre_eda(df_pre, output_dir)
    else:
        print("Pre-Contrast data not found.")
        
    if os.path.exists(post_path):
        df_post = pd.read_csv(post_path)
        perform_post_eda(df_post, output_dir)
    else:
        print("Post-Contrast data not found.")

if __name__ == "__main__":
    main()
