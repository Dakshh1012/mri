import os
import pandas as pd
import numpy as np
import re

def normalize_name(name):
    """Normalize column names to a standard format."""
    name = str(name).lower().strip()
    name = re.sub(r'[\s\-_]+', '_', name)
    # Remove quotes if present
    name = name.replace('"', '').replace("'", "")
    return name

def identify_columns(df):
    """Identify TIV, Age, Sex and Regional Volume columns."""
    cols = {normalize_name(c): c for c in df.columns}
    
    tiv_col = None
    age_col = None
    sex_col = None
    
    # Identify TIV
    tiv_candidates = ['total_intracranial', 'estimatedtotalintracranialvol', 'icv', 'tiv']
    for cand in tiv_candidates:
        if cand in cols:
            tiv_col = cols[cand]
            break
            
    # Identify Age
    age_candidates = ['age', 'chronological_age']
    for cand in age_candidates:
        if cand in cols:
            age_col = cols[cand]
            break
            
    # Identify Sex
    sex_candidates = ['sex', 'gender', 'ptgender']
    for cand in sex_candidates:
        if cand in cols:
            sex_col = cols[cand]
            break
            
    # Regional volumes: Focus on volumetric data and cortical volumes
    standard_regions = []
    skipped_cols = [tiv_col, age_col, sex_col]
    
    for norm_name, orig_name in cols.items():
        if orig_name in skipped_cols:
            continue
            
        # Explicitly exclude thicknesses and areas (except accumbens area which is usually volume in these files)
        if 'thickness' in norm_name or ('area' in norm_name and 'accumbens' not in norm_name):
            continue
            
        # Check if it's numeric
        if pd.api.types.is_numeric_dtype(df[orig_name]):
            # Heuristic: include regions containing 'volume', 'cortex', 'ctx', or specific subcortical names
            keywords = ['volume', 'cortex', 'ctx', 'ventricle', 'thalamus', 'caudate', 'putamen', 'pallidum', 'hippocampus', 'amygdala', 'accumbens', 'brain_stem', 'cerebellum', 'dc']
            if any(x in norm_name for x in keywords):
                # Ensure it's not a 'thickness' measure labeled as 'ctx'
                # (Some datasets label them as ctx-lh-region if volume isn't specified, but here we prefer explicit volume if available)
                standard_regions.append(orig_name)
                
    return tiv_col, age_col, sex_col, standard_regions

def process_file(filepath, output_base, dataset_type):
    print(f"Processing {filepath}...")
    if filepath.endswith('.xlsx'):
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)
        
    tiv_col, age_col, sex_col, regions = identify_columns(df)
    
    if not (tiv_col and age_col and sex_col):
        print(f"Warning: Missing required columns in {filepath}. TIV: {tiv_col}, Age: {age_col}, Sex: {sex_col}")
        # Try to find Age/Sex if missing using common patterns if possible, else skip
        if not age_col or not sex_col:
             # Some files might have missing metadata, we check if they are in the same directory
             pass
        return

    # Create output directories
    os.makedirs(output_base, exist_ok=True)
    
    # Process each region
    for region in regions:
        norm_region = normalize_name(region)
        
        # Prepare subset
        sub_df = df[[age_col, sex_col, tiv_col, region]].copy()
        
        # Drop NaNs
        sub_df = sub_df.dropna()
        if sub_df.empty:
            continue
            
        # Normalize by TIV
        sub_df['Normalized_Volume'] = sub_df[region] / sub_df[tiv_col]
        
        # Convert Sex to standard male/female
        sub_df['Sex_Std'] = sub_df[sex_col].apply(lambda x: 'male' if str(x).lower().startswith('m') else 'female')
        
        # Split by gender and save
        for gender in ['male', 'female']:
            gender_df = sub_df[sub_df['Sex_Std'] == gender]
            if gender_df.empty:
                continue
                
            # Final format: Age, Volume (Raw), TIV
            # We explicitly save raw volume because we will handle TIV in the GAMLSS model
            final_df = gender_df[[age_col, region, tiv_col]].rename(columns={age_col: 'Age', region: 'Volume', tiv_col: 'TIV'})
            
            # Save path: {output_base}/{gender}_{norm_region}.xlsx
            out_file = os.path.join(output_base, f"{gender}_{norm_region}.xlsx")
            
            # Since multiple files might contribute to the same region, we append if exists
            if os.path.exists(out_file):
                existing_df = pd.read_excel(out_file)
                final_df = pd.concat([existing_df, final_df], ignore_index=True)
            
            final_df.to_excel(out_file, index=False)

def main():
    pre_data_dir = "/home/anirudh/Brainagepred/MRBrain/BrainAge-Prediction_Pre/Data"
    post_data_file = "/home/anirudh/Brainagepred/MRBrain/BrainAge-Prediction/Data/QC_removed_raw_sheet_valid_features.xlsx"
    
    output_pre = "/home/anirudh/Brainagepred/MRBrain/Normative Modeling/Input_Data_TIV/Pre"
    output_post = "/home/anirudh/Brainagepred/MRBrain/Normative Modeling/Input_Data_TIV/Post"
    
    # Clean output dirs first if they exist
    import shutil
    if os.path.exists("/home/anirudh/Brainagepred/MRBrain/Normative Modeling/Input_Data_TIV"):
        shutil.rmtree("/home/anirudh/Brainagepred/MRBrain/Normative Modeling/Input_Data_TIV")
    
    # Pre-contrast
    if os.path.exists(pre_data_dir):
        for f in os.listdir(pre_data_dir):
            if f.endswith(('.csv', '.xlsx')):
                process_file(os.path.join(pre_data_dir, f), output_pre, "Pre")
                
    # Post-contrast
    if os.path.exists(post_data_file):
        process_file(post_data_file, output_post, "Post")

if __name__ == "__main__":
    main()
