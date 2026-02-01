import os
import pandas as pd
import re

def normalize_name(name):
    """Normalize column names to a standard format."""
    name = str(name).lower().strip()
    name = re.sub(r'[\s\-_]+', '_', name)
    name = name.replace('"', '').replace("'", "")
    return name

def identify_columns(df):
    cols = {normalize_name(c): c for c in df.columns}
    
    tiv_col = None
    age_col = None
    sex_col = None
    
    tiv_candidates = ['total_intracranial', 'estimatedtotalintracranialvol', 'icv', 'tiv']
    for cand in tiv_candidates:
        if cand in cols:
            tiv_col = cols[cand]
            break
            
    age_candidates = ['age', 'chronological_age']
    for cand in age_candidates:
        if cand in cols:
            age_col = cols[cand]
            break
            
    sex_candidates = ['sex', 'gender', 'ptgender']
    for cand in sex_candidates:
        if cand in cols:
            sex_col = cols[cand]
            break
            
    # Include all relevant regions (Volume, Cortex, etc.)
    # User asked for Thickness too, but we didn't find specific columns.
    # We'll stick to 'volume', 'ctx', 'subcortical' keywords.
    standard_regions = []
    skipped_cols = [tiv_col, age_col, sex_col]
    
    for norm_name, orig_name in cols.items():
        if orig_name in skipped_cols:
            continue
            
        # Exclude known metadata or irrelevant columns
        if 'scan' in norm_name or 'subject' in norm_name or 'id' in norm_name:
            continue

        # Keywords for brain features
        keywords = ['volume', 'cortex', 'ctx', 'ventricle', 'thalamus', 'caudate', 'putamen', 'pallidum', 'hippocampus', 'amygdala', 'accumbens', 'brain_stem', 'cerebellum', 'dc']
        if any(x in norm_name for x in keywords):
             standard_regions.append(orig_name)
                 
    return tiv_col, age_col, sex_col, standard_regions

def process_file(filepath, output_base):
    print(f"Processing {filepath}...")
    if filepath.endswith('.xlsx'):
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)
        
    tiv_col, age_col, sex_col, regions = identify_columns(df)
    
    if not (tiv_col and age_col and sex_col):
        print(f"Skipping {filepath}: Missing metadata columns.")
        return

    # Filter for Age > 21 (Adults)
    df = df[df[age_col] > 21]
    
    if df.empty:
        print(f"Skipping {filepath}: No subjects > 21.")
        return

    os.makedirs(output_base, exist_ok=True)
    
    for region in regions:
        norm_region = normalize_name(region)
        
        sub_df = df[[age_col, sex_col, tiv_col, region]].copy()
        sub_df = sub_df.dropna()
        if sub_df.empty:
            continue
            
        sub_df['Sex_Std'] = sub_df[sex_col].apply(lambda x: 'male' if str(x).lower().startswith('m') else 'female')
        
        for gender in ['male', 'female']:
            gender_df = sub_df[sub_df['Sex_Std'] == gender]
            if gender_df.empty:
                continue
                
            # Save Raw Volume and TIV
            final_df = gender_df[[age_col, region, tiv_col]].rename(columns={age_col: 'Age', region: 'Volume', tiv_col: 'TIV'})
            
            out_file = os.path.join(output_base, f"{gender}_{norm_region}.xlsx")
            
            if os.path.exists(out_file):
                existing_df = pd.read_excel(out_file)
                final_df = pd.concat([existing_df, final_df], ignore_index=True)
            
            final_df.to_excel(out_file, index=False)

def main():
    pre_data_dir = "/home/anirudh/Brainagepred/MRBrain/BrainAge-Prediction_Pre/Data"
    post_data_file = "/home/anirudh/Brainagepred/MRBrain/BrainAge-Prediction/Data/QC_removed_raw_sheet_valid_features.xlsx"
    
    # Unified Output Directory
    output_unified = "/home/anirudh/Brainagepred/MRBrain/Normative Modeling/Input_Data_Unified"
    
    import shutil
    if os.path.exists(output_unified):
        shutil.rmtree(output_unified)
    
    # Process Pre
    if os.path.exists(pre_data_dir):
        for f in os.listdir(pre_data_dir):
            if f.endswith(('.csv', '.xlsx')):
                process_file(os.path.join(pre_data_dir, f), output_unified)
                
    # Process Post
    if os.path.exists(post_data_file):
        process_file(post_data_file, output_unified)

if __name__ == "__main__":
    main()
