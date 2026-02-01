
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import OneHotEncoder
import pcntoolkit as pcn

def run_modeling(dataset_name, input_csv, output_dir, cov_cols, feature_cols, site_col=None, harmonized=False):
    """
    Run Bayesian Linear Regression with B-splines.
    """
    print(f"\n=== Running Modeling for {dataset_name} (Harmonized={harmonized}) ===")
    
    df = pd.read_csv(input_csv)
    # ... (split logic remains same) ...

    # If harmonized, we DO NOT use Site in the Design Matrix
    if harmonized:
        print("Harmonized Run: Ignoring Site in Design Matrix")
        site_col = None 
    
    # ... (rest of logic uses site_col correctly) ...
    print(f"Data shape: {df.shape}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Define Split (Stratified if Site provided)
    if site_col and site_col in df.columns:
        print(f"Using Stratified Shuffle Split on '{site_col}'")
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        # We split based on Site indices
        train_idx, test_idx = next(sss.split(df, df[site_col]))
    else:
        print("Using Random Split (80/20)")
        train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42)
        
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)
    
    print(f"Train size: {len(df_train)}, Test size: {len(df_test)}")
    
    # Save split indices/IDs for reproducibility
    df_train.to_csv(os.path.join(output_dir, "train_data.csv"), index=False)
    df_test.to_csv(os.path.join(output_dir, "test_data.csv"), index=False)

    # 2. Prepare Matrices for PCNtoolkit
    # X: Covariates [Age, Sex, Site_1, Site_2...]
    # Y: Features (Brain Volumes)
    
    # Encoding Categorical Variables (Sex is already 0/1, Site might be string)
    def prepare_x(subnet_df, encoder=None):
        # Age
        X_age = subnet_df[['Age']].values
        
        # Sex
        X_sex = subnet_df[['Sex_Code']].values
        
        # Site (One-Hot)
        if site_col and site_col in subnet_df.columns:
            if encoder is None:
                encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                X_site = encoder.fit_transform(subnet_df[[site_col]])
            else:
                X_site = encoder.transform(subnet_df[[site_col]])
            
            # Combine
            X_combined = np.hstack([X_age, X_sex, X_site])
        else:
            X_combined = np.hstack([X_age, X_sex])
            
        return X_combined, encoder

    # Train Encoder on Training Data ONLY
    X_train, site_encoder = prepare_x(df_train)
    X_test, _ = prepare_x(df_test, encoder=site_encoder)
    
    Y_train = df_train[feature_cols].values
    Y_test = df_test[feature_cols].values
    
    print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
    
    # 3. Configure PCNtoolkit Model (BLR)
    # Using Cubic B-splines (spline_order=3) on Age
    
    import patsy
    
    def create_design_matrix(df_subset, site_encoder=None):
        # 1. B-Spline for Age (Cubic, e.g. df=5 or knots)
        # We use patsy to create a B-spline basis 'bs(Age, df=5)'
        # This automatically handles the intercept/basis
        # We center Age to avoid correlation issues? Standard practice is simple scaling.
        # Let's standardize Age first.
        
        age_scaled = (df_subset['Age'] - df_subset['Age'].mean()) / df_subset['Age'].std()
        
        # Create B-spline basis (degree 3) with 5 degrees of freedom (approx 1 knot/20 years)
        # bs() in patsy generates B-splines.
        # We need to ensure the SAME knots are used for Test data.
        # Patsy handles statefulness if we use the returned design info?
        # Simpler: Use fixed knots based on training range.
        
        return age_scaled

    # Let's do it simply:
    # We will use pcn.normative.estimate with a CUSTOM X matrix.
    
    # Define Knots based on Training Age
    tr_age = df_train['Age'].values
    min_age, max_age = tr_age.min(), tr_age.max()
    knots = np.linspace(min_age, max_age, 5)[1:-1] # 3 internal knots
    
    def get_basis(ages):
        # Cubic B-spline with fixed knots
        return patsy.dmatrix(f"bs(x, knots={list(knots)}, degree=3, include_intercept=False) - 1", {"x": ages}, return_type='dataframe').values

    # Construct X_train
    print("Constructing B-spline Basis...")
    B_train = get_basis(df_train['Age'].values)
    B_test = get_basis(df_test['Age'].values)
    
    # Sex
    S_train = df_train[['Sex_Code']].values
    S_test = df_test[['Sex_Code']].values
    
    # Site (if exists)
    if site_col and site_col in df.columns:
        # DROP FIRST to avoid collinearity with Intercept
        enc = OneHotEncoder(sparse=False, handle_unknown='ignore', drop='first')
        Site_train = enc.fit_transform(df_train[[site_col]])
        Site_test = enc.transform(df_test[[site_col]])
        
        X_train = np.hstack([B_train, S_train, Site_train])
        X_test = np.hstack([B_test, S_test, Site_test])
    else:
        X_train = np.hstack([B_train, S_train])
        X_test = np.hstack([B_test, S_test])
        
    # Add Intercept column (BLR usually expects it or adds it, let's add checking docs)
    # PCNtoolkit BLR usually adds intercept if configured, but let's provide clear covariates.
    # We will add a column of 1s just in case, or rely on 'estimate' to add intercept?
    # Safer to add bias column.
    X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
    
    print(f"Design Matrix Shape: {X_train.shape}")
    
    # Save Matrices
    output_path_models = os.path.join(output_dir, "Models")
    os.makedirs(output_path_models, exist_ok=True)
    
    np.savetxt(os.path.join(output_dir, 'covariates_train.txt'), X_train)
    np.savetxt(os.path.join(output_dir, 'covariates_test.txt'), X_test)
    
    # Loop over features
    for i, feature in enumerate(feature_cols):
        print(f"Running BLR for {feature} ({i+1}/{len(feature_cols)})...")
        
        y_tr = Y_train[:, [i]]
        y_te = Y_test[:, [i]]
        
        # Save Y for this feature (toolkit expects files often)
        # We try feeding numpy arrays directly first.
        
        save_path = os.path.join(output_path_models, feature)
        os.makedirs(save_path, exist_ok=True)
        
        # Run Estimate
        # Using BLR
        try:
            # We use the text files we just saved to ensure robustness with toolkit
            # It prefers paths often.
            
            # Temporary single-feature Y files
            np.savetxt(os.path.join(save_path, 'y_train.txt'), y_tr)
            np.savetxt(os.path.join(save_path, 'y_test.txt'), y_te)
            
            # Change CWD to save_path because toolkit writes to CWD
            cwd_backup = os.getcwd()
            os.chdir(save_path)
            
            try:
                pcn.normative.estimate(
                    covfile=os.path.join(output_dir, 'covariates_train.txt'),
                    respfile=os.path.join(save_path, 'y_train.txt'),
                    testcov=os.path.join(output_dir, 'covariates_test.txt'),
                    testresp=os.path.join(save_path, 'y_test.txt'),
                    alg='blr', 
                    savemodel=True,
                    outputsuffix='predict'
                )
            finally:
                os.chdir(cwd_backup)
            
            # Check if outputs exist
            # Toolkit produces: yhat_predict.txt, ys2_predict.txt (based on suffix)
                
            # If attributes differ (e.g. they are in a dict), we might need to inspect.
            # But standard PCN is object based now.
            
        except Exception as e:
            print(f"FAILED {feature}: {e}")


def main():
    base_dir = "/home/anirudh/Brainagepred/MRBrain/Normative Modeling/Data_Prepared"
    
    # Check for Harmonized Data
    harm_csv = os.path.join(base_dir, "Pre", "Pre_Master_Harmonized.csv")
    use_harmonized = False
    
    if os.path.exists(harm_csv):
        print("Found Harmonized Data! Running on Harmonized Data...")
        pre_csv = harm_csv
        out_base = "/home/anirudh/Brainagepred/MRBrain/Normative Modeling/Models_Harmonized"
        use_harmonized = True
    else:
        print("Harmonized Data NOT found. Running on Raw Data...")
        pre_csv = os.path.join(base_dir, "Pre", "Pre_Master.csv")
        out_base = "/home/anirudh/Brainagepred/MRBrain/Normative Modeling/Models"
        
    post_csv = os.path.join(base_dir, "Post", "Post_Master.csv")
    
    # Identify Feature columns (exclude meta)
    df_pre = pd.read_csv(pre_csv, nrows=1)
    meta_cols = ['Age', 'Sex', 'Sex_Code', 'Site', 'TIV']
    feature_cols = [c for c in df_pre.columns if c not in meta_cols]
    
    print(f"Found {len(feature_cols)} features.")
    
    # Run Pre
    run_modeling("Pre", pre_csv, os.path.join(out_base, "Pre"), 
                 cov_cols=['Age', 'Sex_Code'], 
                 feature_cols=feature_cols, 
                 site_col='Site' if not use_harmonized else None,
                 harmonized=use_harmonized)
                 
    # Run Post (No harmonization needed as it's single site)
    # But usually we apply the SAME model or model it separately.
    # Plan says: separate model for Post.
    # We can keep Post in standard Models folder or new one? 
    # Let's put Post in the Harmonized folder too for consistency of the "Run".
    run_modeling("Post", post_csv, os.path.join(out_base, "Post"), 
                 cov_cols=['Age', 'Sex_Code', 'Site'], 
                 feature_cols=feature_cols, 
                 site_col='Site') 

if __name__ == "__main__":
    main()
