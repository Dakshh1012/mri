
import os
import pandas as pd
import numpy as np
import pickle
import patsy
from scipy.stats import norm

def get_basis(ages, knots, min_age, max_age):
    """Reconstruct B-spline basis using the same knots and bounds."""
    # We must strict the bounds to the TRAINING bounds to match the basis
    return patsy.dmatrix(f"bs(x, knots={list(knots)}, lower_bound={min_age}, upper_bound={max_age}, degree=3, include_intercept=False) - 1", {"x": ages}, return_type='dataframe').values

def export_percentiles_for_dataset(dataset_name):
    print(f"\n=== Exporting Percentiles for {dataset_name} ===")
    base_dir = "/home/anirudh/Brainagepred/MRBrain/Normative Modeling"
    models_dir = os.path.join(base_dir, "Models_Harmonized", dataset_name, "Models")
    train_data_path = os.path.join(base_dir, "Models_Harmonized", dataset_name, "train_data.csv")
    
    # Output structure: Percentile_Curves_Excel/Pre/Female, etc.
    output_dir = os.path.join(base_dir, "Percentile_Curves_Excel", dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(train_data_path):
        print(f"Training data not found for {dataset_name}: {train_data_path}")
        return

    # 1. Get Training Parameters (Knots)
    print("Loading training metadata...")
    df_train = pd.read_csv(train_data_path)
    tr_age = df_train['Age'].values
    min_age, max_age = tr_age.min(), tr_age.max()
    knots = np.linspace(min_age, max_age, 5)[1:-1]
    print(f"Knots: {knots} (Bounds: {min_age}-{max_age})")
    
    # 2. Create Dummy Data Grid (Age min_age to 95)
    start_age = int(np.ceil(min_age))
    # Cannot extrapolate beyond training bounds with fixed B-spline basis
    upper_limit_req = 95
    end_age = int(min(upper_limit_req, np.floor(max_age)))
    
    print(f"Prediction Grid: {start_age} to {end_age}")
    age_grid = np.arange(start_age, end_age + 1, 1)
    
    # Basis for Age
    B_grid = get_basis(age_grid, knots, min_age, max_age)
    
    # Intercept
    Intercept = np.ones((len(age_grid), 1))
    
    # Design Matrices for Sex
    # Female: Sex_Code = 0
    # Male: Sex_Code = 1
    # X structure: [Intercept, B_spline, Sex] OR [Intercept, B_spline, Sex, (Site)]
    # In Harmonized Run: X = [Intercept, B_spline, Sex] (Site removed)
    # Important: Post models might HAVE site if they were run differently?
    # Checking 2_run_modeling.py: 
    #   Pre (Harmonized) -> site_col=None -> X=[Intercept, B, Sex] (Correct)
    #   Post -> site_col='Site' -> X=[Intercept, B, Sex, Site] ?
    # Let's check 2_run_modeling.py call for Post.
    # It says: run_modeling("Post", ..., site_col='Site')
    # So Post models EXPECT Site Covariates!
    # If Post models expect Site, we must provide a dummy Site?
    # BUT Post is single site 'MAX'.
    # If Site is 'MAX', then Site_Dummy is ... ? 
    # If OneHotEncoder dropped first, and there is only 1 site, then there are NO site columns (dropped first).
    # If there is >1 site, we have columns.
    # Let's check Post_Master.csv sites. Expected only MAX.
    # If only 1 category, OneHotEncoder with drop='first' produces 0 columns.
    # So X is still [Intercept, B, Sex].
    # We should verify this assumption by trying to predict.
    
    # Female X
    Sex_F = np.zeros((len(age_grid), 1))
    X_Female = np.hstack([Intercept, B_grid, Sex_F])
    
    # Male X
    Sex_M = np.ones((len(age_grid), 1))
    X_Male = np.hstack([Intercept, B_grid, Sex_M])
    
    print(f"Design Matrix Shape: {X_Female.shape}")
    
    # Percentiles of interest: 1% to 99% (Integer steps)
    centiles_int = np.arange(1, 100, 1) # 1, 2, ..., 99
    centiles = centiles_int / 100.0
    
    z_scores = [norm.ppf(c) for c in centiles]
    col_names = [f"Centile_{int(c*100)}" for c in centiles]
    
    features = sorted([d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))])
    print(f"Found {len(features)} features in {dataset_name}.")
    
    for i, feature in enumerate(features):
        print(f"Processing {feature} ({i+1}/{len(features)})...")
        model_path = os.path.join(models_dir, feature, "Models", "NM_0_0_predict.pkl")
        
        if not os.path.exists(model_path):
            print(f"  Model not found: {model_path}")
            continue
            
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                
            # Predict
            # Clip negative predictions to 0
            
            # Predict Female
            try:
                yhat_F, ys2_F = model.predict(X_Female)
            except ValueError as ve:
                print(f"  Prediction shape mismatch for {feature}: {ve}. Trying with extra dummy col?")
                # If Post requires Site column (even if empty/dummy), we might need to add it.
                # Assuming single site MAX, maybe no column.
                continue

            sigma_F = np.sqrt(ys2_F)
            
            df_F = pd.DataFrame({'Age': age_grid})
            for z, name in zip(z_scores, col_names):
                mu = yhat_F.flatten()
                sig = sigma_F.flatten()
                vals = mu + z * sig
                vals[vals < 0] = 0 
                df_F[name] = vals
            
            # Predict Male
            yhat_M, ys2_M = model.predict(X_Male)
            sigma_M = np.sqrt(ys2_M)
            
            df_M = pd.DataFrame({'Age': age_grid})
            for z, name in zip(z_scores, col_names):
                mu = yhat_M.flatten()
                sig = sigma_M.flatten()
                vals = mu + z * sig
                vals[vals < 0] = 0
                df_M[name] = vals
                
            # Save to Separate Excel Files
            # Female
            f_dir = os.path.join(output_dir, "Female")
            os.makedirs(f_dir, exist_ok=True)
            df_F.to_excel(os.path.join(f_dir, f"{feature}.xlsx"), index=False)
            
            # Male
            m_dir = os.path.join(output_dir, "Male")
            os.makedirs(m_dir, exist_ok=True)
            df_M.to_excel(os.path.join(m_dir, f"{feature}.xlsx"), index=False)
                
        except Exception as e:
            print(f"  Error processing {feature}: {e}")

def export_percentiles():
    for dataset in ["Pre", "Post"]:
        try:
            export_percentiles_for_dataset(dataset)
        except Exception as e:
            print(f"Failed to export {dataset}: {e}")

if __name__ == "__main__":
    export_percentiles()
