import pandas as pd
import numpy as np
import os
import re

def normalize_name(name):
    """Normalize column names."""
    name = str(name).lower().strip()
    name = re.sub(r'[\s\-_]+', '_', name)
    return name.replace('"', '').replace("'", "")

def identify_columns(df):
    """Identify key columns."""
    cols = {normalize_name(c): c for c in df.columns}
    
    tiv_col = next((cols[c] for c in ['total_intracranial', 'estimatedtotalintracranialvol', 'icv', 'tiv'] if c in cols), None)
    age_col = next((cols[c] for c in ['age', 'chronological_age'] if c in cols), None)
    sex_col = next((cols[c] for c in ['sex', 'gender', 'ptgender'] if c in cols), None)
    
    # Subject ID is useful for tracking but not strictly needed for the model matrix X/Y simple export
    # But good to have if we want to join later.
    
    regions = []
    skipped = [tiv_col, age_col, sex_col]
    keywords = ['volume', 'cortex', 'ctx', 'ventricle', 'thalamus', 'caudate', 'putamen', 'pallidum', 'hippocampus', 'amygdala', 'accumbens', 'brain_stem', 'cerebellum', 'dc']
    
    for norm, orig in cols.items():
        if orig in skipped: continue
        if 'thickness' in norm or ('area' in norm and 'accumbens' not in norm): continue
        if pd.api.types.is_numeric_dtype(df[orig]) and any(k in norm for k in keywords):
            regions.append(orig)
            
    return tiv_col, age_col, sex_col, regions

def process_dataset(name, file_paths, output_dir, site_map=None):
    """
    Process a list of files for a specific dataset (Pre or Post).
    site_map: Dictionary mapping filename keywords to Site names (e.g. {'ADNI': 'ADNI', 'MAX': 'MAX'}).
    """
    all_data = []
    
    for fp in file_paths:
        print(f"Reading {fp}...")
        try:
            if fp.endswith('.xlsx'): df = pd.read_excel(fp)
            else: df = pd.read_csv(fp)
        except Exception as e:
            print(f"Error reading {fp}: {e}")
            continue
            
        tiv_c, age_c, sex_c, regions = identify_columns(df)
        if not (tiv_c and age_c and sex_c):
            print(f"Skipping {fp}: Missing columns.")
            continue
            
        # Determine Site
        site = "Unknown"
        if site_map:
            fname = os.path.basename(fp)
            for key, s_name in site_map.items():
                if key in fname:
                    site = s_name
                    break
            # Fallback for Pre-contrast if not matched (e.g. Batch2 -> MAX? or separate?)
            # Assuming Batch2 is MAX based on context, or keep separate. 
            # User mentioned "ADNI and MAX data for pre contrast".
            # Batch2 might be MAX. Let's label 'Batch2' as 'MAX' if user implies only 2 main sites?
            # Or keep 'Batch2' as distinct site 'Batch2'. Safer to keep distinct.
            if site == "Unknown" and "Batch2" in fname:
                site = "Batch2"
        else:
             # Post contrast is all MAX
             site = "MAX"

        # Extract
        sub_df = df[[age_c, sex_c, tiv_c] + regions].copy()
        sub_df = sub_df.rename(columns={age_c: 'Age', sex_c: 'Sex', tiv_c: 'TIV'})
        sub_df['Site'] = site
        
        # Standardize Sex (0=Female, 1=Male or vice versa. BLR needs numeric).
        # Let's map: Female=0, Male=1
        sub_df['Sex_Code'] = sub_df['Sex'].apply(lambda x: 1 if str(x).lower().startswith('m') else 0)
        
        # Normalize Data
        # For PCNtoolkit, we usually provide Covariates (Age, Sex, Site) and Features (Y).
        # We can normalize Y by TIV here, or include TIV in X.
        # User requested: "TIV normalization has to be done as well".
        # User also satisfied with "Covariate Method" (Volume ~ Age + TIV).
        # But commonly for BLR, pre-normalizing (Ratio) or Log-Log regression is used.
        # Given "Covariate Method" success previously, let's keep TIV as a predictor in X?
        # WAIT: User said "Normalize by TIV" in the approved plan.
        # "Normalization will be performed as: Normalized Volume = Raw Volume / Total Intracranial Volume"
        # Okay, we will Ratio Normalize for the features Y.
        
        for r in regions:
            norm_r = normalize_name(r)
            sub_df[norm_r] = sub_df[r] / sub_df['TIV']
            
        # Keep only normalized cols + Age + Sex_Code + Site + TIV (for reference)
        keep_cols = ['Age', 'Sex_Code', 'Site', 'TIV'] + [normalize_name(r) for r in regions]
        all_data.append(sub_df[keep_cols])

    if not all_data:
        return

    full_df = pd.concat(all_data, ignore_index=True)
    full_df = full_df.dropna() # BLR cannot handle NaNs
    
    # Save master file
    os.makedirs(output_dir, exist_ok=True)
    master_path = os.path.join(output_dir, f"{name}_Master.csv")
    full_df.to_csv(master_path, index=False)
    print(f"Saved {master_path} with shape {full_df.shape}")
    print("Sites distribution:")
    print(full_df['Site'].value_counts())

def main():
    base_dir = "/home/anirudh/Brainagepred/MRBrain"
    pre_dir = os.path.join(base_dir, "BrainAge-Prediction_Pre/Data")
    post_file = os.path.join(base_dir, "BrainAge-Prediction/Data/QC_removed_raw_sheet_valid_features.xlsx")
    
    # Update to new 'Data_Prepared' directory
    out_dir = "/home/anirudh/Brainagepred/MRBrain/Normative Modeling/Data_Prepared"
    if os.path.exists(out_dir):
        # Don't delete simply, or ensure we want fresh start.
        # Script logic was to rm tree. OK to keep for fresh data.
        import shutil
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    
    # Pre-Contrast Files
    pre_files = []
    if os.path.exists(pre_dir):
        for f in os.listdir(pre_dir):
            if f.endswith(('.csv', '.xlsx')):
                pre_files.append(os.path.join(pre_dir, f))
                
    # Site Mapping
    # ADNI -> ADNI
    # MAX -> MAX
    # Batch2 -> Batch2 (likely distinct scanner/protocol)
    site_map = {
        'ADNI': 'ADNI',
        'MAX': 'MAX',
        'Batch 2': 'Batch2',
        'Batch2': 'Batch2'
    }
    
    print("\n--- Processing Pre-Contrast ---")
    process_dataset("Pre", pre_files, os.path.join(out_dir, "Pre"), site_map=site_map)
    
    print("\n--- Processing Post-Contrast ---")
    process_dataset("Post", [post_file], os.path.join(out_dir, "Post"), site_map=None) # All MAX

if __name__ == "__main__":
    main()
