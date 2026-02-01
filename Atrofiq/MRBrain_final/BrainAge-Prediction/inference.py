#!/usr/bin/env python3
"""
Corrected Brain Age Inference Script
(Feature selection removed — pipeline handles it internally)
"""

import argparse
import sys
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Monkeypatch for legacy pickle loading (numpy version mismatch)
import numpy.random
try:
    import numpy.random._pickle
    original_ctor = numpy.random._pickle.__randomstate_ctor
    def patched_ctor_wrapper(*args, **kwargs):
         if len(args) > 1:
             args = (args[0],)
         return original_ctor(*args, **kwargs)
    
    numpy.random._pickle.__randomstate_ctor = patched_ctor_wrapper
    print("Monkeypatched numpy.random._pickle.__randomstate_ctor for legacy pickle support.")

except Exception as e:
    print(f"Failed to patch numpy.random._pickle: {e}")


class BrainAgePredictor:

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = Path("saved_models") / "brain_age_pipeline.pkl"

        self.model_path = Path(model_path)
        self.pipeline = None
        self.load_model()

    def load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        try:
            import joblib
            self.pipeline = joblib.load(self.model_path)
        except Exception as e:
            print(f"Joblib load failed: {e}. Falling back to pickle.")
            with open(self.model_path, "rb") as f:
                self.pipeline = pickle.load(f)

        print(f"✓ Loaded model: {self.model_path}")

    # -----------------------------------------------------
    def _load_data(self, path):
        path = Path(path)
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)
        print(f"✓ Loaded input: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    # -----------------------------------------------------
    def _preprocess(self, df):
        if "Age" not in df.columns and "age" not in df.columns:
            raise ValueError("Input must contain 'Age' column")
        
        ages = df["Age"].values.astype(float) if "Age" in df.columns else df["age"].values.astype(float)

        # Sex encoding (optional)
        sex_col = None
        for c in ["SEX", "Sex", "sex", "Gender", "gender"]:
            if c in df.columns:
                sex_col = c
                break

        if sex_col:
            le = LabelEncoder()
            df[sex_col] = le.fit_transform(df[sex_col].astype(str))
            print(f"✓ Encoded {sex_col}")

        # Get training-time brain features
        brain_cols = self.pipeline.brain_cols
        missing = [c for c in brain_cols if c not in df.columns]

        if missing:
            print("\n⚠ Missing brain features (filled with 0):")
            for m in missing:
                print("  -", m)

        # Build full feature matrix in correct order
        X = []
        for col in brain_cols:
            if col in df.columns:
                X.append(df[col].values)
            else:
                X.append(np.zeros(len(df)))

        X = np.vstack(X).T  # shape (n_samples, full_feature_count)

        # TIV normalize
        tiv = X.sum(axis=1, keepdims=True)
        tiv[tiv == 0] = 1e-8
        X_norm = X / tiv

        print(f"✓ Preprocessed features: {X_norm.shape[1]} dims (no feature selection applied)")

        return ages, X_norm

    # -----------------------------------------------------
    def predict(self, input_path, output_path=None):
        df = self._load_data(input_path)
        ages, X_full = self._preprocess(df)

        print("\n▶ Generating predictions using pipeline...")

        preds = self.pipeline.predict_corrected(X_full, ages)
        bag = preds - ages

        out_df = pd.DataFrame({
            "Age": ages,
            "Predicted_Age": preds,
            "BAG": bag
        })

        # Save results
        if output_path is None:
            out = Path(input_path).with_name(Path(input_path).stem + "_predictions.csv")
        else:
            out = Path(output_path)

        out_df.to_csv(out, index=False)
        print(f"\n✓ Saved predictions to: {out}")

        print("\nFirst 5 rows:")
        print(out_df.head().to_string(index=False))

        return out_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("-o", "--output", type=str, default=None)
    parser.add_argument("-m", "--model", type=str, default=None)
    args = parser.parse_args()

    pred = BrainAgePredictor(model_path=args.model)
    pred.predict(args.input, args.output)


if __name__ == "__main__":
    main()
