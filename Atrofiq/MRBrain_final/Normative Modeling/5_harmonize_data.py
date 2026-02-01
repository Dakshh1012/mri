
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
import os

def harmonize_dataset(df, features, reference_site='MAX', output_dir=None):
    """
    Harmonize data to a reference site using GLM.
    Model: Y ~ Age + Sex + Site
    Adjustment: Y_adj = Y - (Beta_Site * Site_Dummy)
    effectively shifting other sites to match the Reference Site's intercept.
    """
    df_harm = df.copy()
    
    print(f"Harmonizing {len(features)} features to Reference Site: {reference_site}...")
    
    adjusted_count = 0
    
    for feat in features:
        # Skip if column missing or constant
        if feat not in df.columns or df[feat].std() == 0:
            continue
            
        # Rename temporarily for statsmodels formula (remove special chars if any)
        # Assuming cleaned names from prepare_data are GLM safe (no spaces, etc)
        
        formula = f"{feat} ~ Age + C(Sex_Code) + C(Site, Treatment(reference='{reference_site}'))"
        
        try:
            model = smf.ols(formula, data=df).fit()
            
            # Extract Site coefficients
            # Params look like: Intercept, C(Site)[T.ADNI], C(Site)[T.Batch2], Age, ...
            params = model.params
            
            # Adjust Data
            # For every site that is NOT reference, subtract its coefficient
            # The reference site has coef 0 (implicitly), so no subtraciton.
            
            for term, val in params.items():
                if "C(Site" in term and "Treatment" in term:
                    # Extract site name from term string: "C(Site, Treatment...)[T.ADNI]" -> "ADNI"
                    site_name = term.split("[T.")[1].replace("]", "")
                    
                    # Apply adjustment
                    mask = df['Site'] == site_name
                    df_harm.loc[mask, feat] -= val
            
            adjusted_count += 1
            
        except Exception as e:
            print(f"Failed to harmonize {feat}: {e}")
            
    print(f"Successfully harmonized {adjusted_count} features.")
    return df_harm

def plot_comparison(df_raw, df_harm, feature, output_dir):
    """Generate Before/After Boxplots."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    
    # Before
    sns.boxplot(data=df_raw, x='Site', y=feature, ax=axes[0], showfliers=False)
    sns.stripplot(data=df_raw, x='Site', y=feature, ax=axes[0], color='black', alpha=0.3, size=2)
    axes[0].set_title(f"BEFORE Harmonization: {feature}")
    
    # After
    sns.boxplot(data=df_harm, x='Site', y=feature, ax=axes[1], showfliers=False)
    sns.stripplot(data=df_harm, x='Site', y=feature, ax=axes[1], color='black', alpha=0.3, size=2)
    axes[1].set_title(f"AFTER Harmonization: {feature}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Harmonization_Check_{feature}.png"))
    plt.close()

def main():
    base_dir = "/home/anirudh/Brainagepred/MRBrain/Normative Modeling"
    data_dir = os.path.join(base_dir, "Data_Prepared")
    output_dir = os.path.join(base_dir, "Comparison_Plots_Trends", "Harmonization_Checks")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Pre-contrast
    pre_path = os.path.join(data_dir, "Pre", "Pre_Master.csv")
    if not os.path.exists(pre_path):
        print("Pre-contrast data not found!")
        return
        
    df = pd.read_csv(pre_path)
    
    # Identify Features (excluding meta)
    meta_cols = ['Age', 'Sex', 'Sex_Code', 'Site', 'TIV']
    features = [c for c in df.columns if c not in meta_cols]
    
    # Harmonize
    # Reference = MAX (since Post-contrast is MAX)
    if 'MAX' not in df['Site'].unique():
        print("Warning: 'MAX' site not found in data. Using 'ADNI' as reference?")
        ref = 'ADNI'
    else:
        ref = 'MAX'
        
    df_harm = harmonize_dataset(df, features, reference_site=ref, output_dir=output_dir)
    
    # Verify Key Features
    check_feats = ['left_hippocampus', 'left_lateral_ventricle', 'left_cerebral_cortex']
    for f in check_feats:
        if f in features:
            plot_comparison(df, df_harm, f, output_dir)
            
    # Save Harmonized Data
    out_path = os.path.join(data_dir, "Pre", "Pre_Master_Harmonized.csv")
    df_harm.to_csv(out_path, index=False)
    print(f"Saved harmonized data to {out_path}")

if __name__ == "__main__":
    main()
