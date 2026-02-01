#!/usr/bin/env python3
"""
Fallback Brain Age Predictor
Provides brain age prediction when the main model fails to load
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
import pickle
import logging

logger = logging.getLogger(__name__)

class FallbackBrainAgePredictor:
    """
    Fallback brain age predictor using a simple linear model
    This is used when the main trained model fails to load
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = LinearRegression()
        self.is_fitted = False
        self.brain_cols = [
            'left_cerebral_white_matter', 'left_cerebral_cortex', 'left_lateral_ventricle',
            'left_thalamus', 'left_caudate', 'left_putamen', 'left_pallidum',
            'left_hippocampus', 'left_amygdala',
            'right_cerebral_white_matter', 'right_cerebral_cortex', 'right_lateral_ventricle',
            'right_thalamus', 'right_caudate', 'right_putamen', 'right_pallidum',
            'right_hippocampus', 'right_amygdala'
        ]
        self._fit_simple_model()
    
    def _fit_simple_model(self):
        """
        Fit a simple model based on typical age-brain volume relationships
        This is a fallback when real training data isn't available
        """
        # Create synthetic training data based on typical age-brain relationships
        np.random.seed(42)  # For reproducibility
        n_samples = 1000
        
        # Generate ages between 18 and 95
        ages = np.random.uniform(18, 95, n_samples)
        
        # Create brain volumes with age-related decline
        features = []
        for age in ages:
            # Base volumes for a 25-year-old
            base_volumes = {
                'left_cerebral_white_matter': 250000,
                'left_cerebral_cortex': 230000,
                'left_lateral_ventricle': 12000,
                'left_thalamus': 8500,
                'left_caudate': 3800,
                'left_putamen': 5200,
                'left_pallidum': 1800,
                'left_hippocampus': 4200,
                'left_amygdala': 1600,
                'right_cerebral_white_matter': 250000,
                'right_cerebral_cortex': 230000,
                'right_lateral_ventricle': 12000,
                'right_thalamus': 8500,
                'right_caudate': 3800,
                'right_putamen': 5200,
                'right_pallidum': 1800,
                'right_hippocampus': 4200,
                'right_amygdala': 1600,
            }
            
            # Apply age-related changes
            age_factor = 1.0 - (max(0, age - 25) * 0.003)  # 0.3% decline per year after 25
            individual_variation = np.random.normal(1.0, 0.1)  # 10% individual variation
            
            feature_row = []
            for col in self.brain_cols:
                volume = base_volumes[col] * age_factor * individual_variation
                feature_row.append(max(0, volume))  # Ensure non-negative
            
            features.append(feature_row)
        
        # Convert to arrays
        X = np.array(features)
        y = ages
        
        # Add some noise to make brain age prediction imperfect
        y_noisy = y + np.random.normal(0, 3, len(y))  # ±3 years noise
        
        # Fit the model
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y_noisy)
        self.is_fitted = True
        
        logger.info("Fallback BrainAge model fitted with synthetic data")
    
    def predict_corrected(self, X, chronological_ages):
        """
        Predict brain age with bias correction
        
        Args:
            X: Feature matrix (n_samples, n_features)
            chronological_ages: Array of chronological ages
            
        Returns:
            Array of predicted brain ages
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get raw predictions
        raw_predictions = self.model.predict(X_scaled)
        
        # Apply simple bias correction
        # Reduce extreme predictions towards chronological age
        corrected_predictions = []
        for raw_pred, chron_age in zip(raw_predictions, chronological_ages):
            # Bias correction: reduce extreme differences
            diff = raw_pred - chron_age
            corrected_diff = diff * 0.7  # Reduce difference by 30%
            corrected_pred = chron_age + corrected_diff
            corrected_predictions.append(corrected_pred)
        
        return np.array(corrected_predictions)

class RobustBrainAgePredictor:
    """
    Robust brain age predictor that tries the main model first, then falls back
    """
    
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = Path("saved_models") / "brain_age_pipeline.pkl"
        
        self.model_path = Path(model_path)
        self.pipeline = None
        self.fallback_predictor = None
        self.use_fallback = False
        
        self.brain_cols = [
            'left_cerebral_white_matter', 'left_cerebral_cortex', 'left_lateral_ventricle',
            'left_thalamus', 'left_caudate', 'left_putamen', 'left_pallidum',
            'left_hippocampus', 'left_amygdala',
            'right_cerebral_white_matter', 'right_cerebral_cortex', 'right_lateral_ventricle',
            'right_thalamus', 'right_caudate', 'right_putamen', 'right_pallidum',
            'right_hippocampus', 'right_amygdala'
        ]
        
        self.load_model()
    
    def load_model(self):
        """Load the main model or fall back to simple model"""
        if not self.model_path.exists():
            logger.warning(f"Model not found: {self.model_path}. Using fallback predictor.")
            self.use_fallback = True
            self.fallback_predictor = FallbackBrainAgePredictor()
            return
        
        try:
            # Try joblib first
            self.pipeline = joblib.load(self.model_path)
            logger.info(f"✓ Loaded model with joblib: {self.model_path}")
            return
        except Exception as e:
            logger.warning(f"Joblib load failed: {e}. Trying pickle...")
        
        try:
            # Try pickle
            with open(self.model_path, "rb") as f:
                self.pipeline = pickle.load(f)
            logger.info(f"✓ Loaded model with pickle: {self.model_path}")
            return
        except Exception as e:
            logger.error(f"Failed to load model with pickle: {e}")
        
        # Fall back to simple model
        logger.warning("Main model failed to load. Using fallback predictor.")
        self.use_fallback = True
        self.fallback_predictor = FallbackBrainAgePredictor()
    
    def _load_data(self, path):
        """Load data from CSV or Excel file"""
        path = Path(path)
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)
        logger.info(f"✓ Loaded input: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def _preprocess(self, df):
        """Preprocess data for prediction"""
        if "Age" not in df.columns and "age" not in df.columns:
            raise ValueError("Input must contain 'Age' column")
        
        ages = df["Age"].values.astype(float) if "Age" in df.columns else df["age"].values.astype(float)
        
        # Get brain features
        missing = [c for c in self.brain_cols if c not in df.columns]
        if missing:
            logger.warning(f"Missing brain features (will use defaults): {missing}")
        
        # Build feature matrix
        X = []
        for col in self.brain_cols:
            if col in df.columns:
                X.append(df[col].values)
            else:
                # Use default values for missing columns
                default_value = 100000 if 'cerebral' in col else 5000  # Rough defaults
                X.append(np.full(len(df), default_value))
        
        X = np.vstack(X).T  # Shape (n_samples, n_features)
        
        # TIV normalize if using main model
        if not self.use_fallback:
            tiv = X.sum(axis=1, keepdims=True)
            tiv[tiv == 0] = 1e-8
            X_norm = X / tiv
            logger.info(f"✓ Preprocessed features: {X_norm.shape[1]} dims (TIV normalized)")
            return ages, X_norm
        else:
            logger.info(f"✓ Preprocessed features: {X.shape[1]} dims (fallback mode)")
            return ages, X
    
    def predict(self, input_path, output_path=None):
        """Make brain age predictions"""
        df = self._load_data(input_path)
        ages, X_processed = self._preprocess(df)
        
        if self.use_fallback:
            logger.info("▶ Generating predictions using fallback model...")
            preds = self.fallback_predictor.predict_corrected(X_processed, ages)
        else:
            logger.info("▶ Generating predictions using main pipeline...")
            try:
                preds = self.pipeline.predict_corrected(X_processed, ages)
            except Exception as e:
                logger.error(f"Main pipeline failed: {e}. Switching to fallback.")
                self.use_fallback = True
                self.fallback_predictor = FallbackBrainAgePredictor()
                preds = self.fallback_predictor.predict_corrected(X_processed, ages)
        
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
        logger.info(f"✓ Saved predictions to: {out}")
        
        logger.info("First 5 rows:")
        logger.info(out_df.head().to_string(index=False))
        
        return out_df

# Create an alias for backwards compatibility
BrainAgePredictor = RobustBrainAgePredictor