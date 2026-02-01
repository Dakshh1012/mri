"""
Brain Age Prediction Model Classes
Separate module for model classes to enable proper pickle serialization
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor, 
                             ExtraTreesRegressor)
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


class SimpleFeatureSelector:
    """Select best features from original features only"""
    
    def __init__(self, n_features=40):
        self.n_features = n_features
        self.selected_features_ = None
        
    def fit(self, X, y):
        print(f"\n▶ Feature Selection (selecting {self.n_features} from {X.shape[1]} features)...")
        
        selector_mi = SelectKBest(mutual_info_regression, k=min(60, X.shape[1]))
        selector_mi.fit(X, y)
        scores_mi = selector_mi.scores_
        
        from sklearn.linear_model import LassoCV
        lasso = LassoCV(cv=5, random_state=42, max_iter=2000)
        lasso.fit(X, y)
        scores_lasso = np.abs(lasso.coef_)
        
        lgb_model = lgb.LGBMRegressor(
            n_estimators=100, max_depth=5, num_leaves=20,
            min_child_samples=15, random_state=42, verbose=-1
        )
        lgb_model.fit(X, y)
        scores_tree = lgb_model.feature_importances_
        
        scores_mi = scores_mi / (scores_mi.max() + 1e-10)
        scores_lasso = scores_lasso / (scores_lasso.max() + 1e-10)
        scores_tree = scores_tree / (scores_tree.max() + 1e-10)
        
        combined_scores = (scores_mi + scores_lasso + scores_tree) / 3
        self.selected_features_ = np.argsort(combined_scores)[-self.n_features:]
        
        print(f"  Selected {len(self.selected_features_)} features")
        return self
    
    def transform(self, X):
        if self.selected_features_ is None:
            raise ValueError("Fit must be called before transform")
        return X[:, self.selected_features_]
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


class AgeStratifiedEnsemble:
    """Separate models for different age groups"""
    
    def __init__(self):
        self.models_before_40 = []
        self.models_after_40 = []
        self.weights_before_40 = []
        self.weights_after_40 = []
        self.scaler = RobustScaler()
        
    def _create_models(self):
        return [
            ('ridge', Ridge(alpha=10.0)),
            ('lasso', Lasso(alpha=1.0, max_iter=3000)),
            ('elastic', ElasticNet(alpha=2.0, l1_ratio=0.5, max_iter=3000)),
            ('lgb', lgb.LGBMRegressor(
                n_estimators=300, learning_rate=0.03, max_depth=6,
                num_leaves=30, min_child_samples=10, subsample=0.8,
                colsample_bytree=0.8, reg_alpha=0.3, reg_lambda=0.3,
                random_state=42, verbose=-1
            )),
            ('xgb', xgb.XGBRegressor(
                n_estimators=300, learning_rate=0.03, max_depth=6,
                min_child_weight=2, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.3, reg_lambda=0.3, random_state=42, verbosity=0
            )),
            ('gb', GradientBoostingRegressor(
                n_estimators=250, learning_rate=0.04, max_depth=6,
                min_samples_split=10, min_samples_leaf=5, subsample=0.8, random_state=42
            )),
            ('rf', RandomForestRegressor(
                n_estimators=300, max_depth=15, min_samples_split=10,
                min_samples_leaf=5, max_features='sqrt', random_state=42, n_jobs=-1
            )),
            ('et', ExtraTreesRegressor(
                n_estimators=300, max_depth=15, min_samples_split=10,
                min_samples_leaf=5, max_features='sqrt', random_state=42, n_jobs=-1
            ))
        ]
    
    def fit(self, X, y):
        print("\n▶ Training Age-Stratified Ensemble...")
        
        X_scaled = self.scaler.fit_transform(X)
        mask_before_40 = y < 40
        mask_after_40 = y >= 40
        
        X_before = X_scaled[mask_before_40]
        y_before = y[mask_before_40]
        X_after = X_scaled[mask_after_40]
        y_after = y[mask_after_40]
        
        print(f"  Before 40: {len(y_before)} samples")
        print(f"  After 40: {len(y_after)} samples")
        
        if len(y_before) > 30:
            print("\n  Training models for AGE < 40:")
            cv_scores_before = []
            models = self._create_models()
            kf = KFold(n_splits=min(5, len(y_before)//10), shuffle=True, random_state=42)
            
            for name, model in models:
                fold_scores = []
                for train_idx, val_idx in kf.split(X_before):
                    X_train, X_val = X_before[train_idx], X_before[val_idx]
                    y_train, y_val = y_before.iloc[train_idx], y_before.iloc[val_idx]
                    model_copy = model.__class__(**model.get_params())
                    model_copy.fit(X_train, y_train)
                    y_pred = model_copy.predict(X_val)
                    fold_scores.append(mean_absolute_error(y_val, y_pred))
                
                avg_mae = np.mean(fold_scores)
                cv_scores_before.append(avg_mae)
                model.fit(X_before, y_before)
                self.models_before_40.append((name, model))
                print(f"    {name}: CV MAE = {avg_mae:.3f}")
            
            cv_scores_before = np.array(cv_scores_before)
            self.weights_before_40 = 1.0 / (cv_scores_before + 1e-6)
            self.weights_before_40 = self.weights_before_40 / self.weights_before_40.sum()
        
        if len(y_after) > 30:
            print("\n  Training models for AGE >= 40:")
            cv_scores_after = []
            models = self._create_models()
            kf = KFold(n_splits=min(5, len(y_after)//10), shuffle=True, random_state=42)
            
            for name, model in models:
                fold_scores = []
                for train_idx, val_idx in kf.split(X_after):
                    X_train, X_val = X_after[train_idx], X_after[val_idx]
                    y_train, y_val = y_after.iloc[train_idx], y_after.iloc[val_idx]
                    model_copy = model.__class__(**model.get_params())
                    model_copy.fit(X_train, y_train)
                    y_pred = model_copy.predict(X_val)
                    fold_scores.append(mean_absolute_error(y_val, y_pred))
                
                avg_mae = np.mean(fold_scores)
                cv_scores_after.append(avg_mae)
                model.fit(X_after, y_after)
                self.models_after_40.append((name, model))
                print(f"    {name}: CV MAE = {avg_mae:.3f}")
            
            cv_scores_after = np.array(cv_scores_after)
            self.weights_after_40 = 1.0 / (cv_scores_after + 1e-6)
            self.weights_after_40 = self.weights_after_40 / self.weights_after_40.sum()
        
        return self
    
    def predict(self, X, ages):
        X_scaled = self.scaler.transform(X)
        predictions = np.zeros(len(X))
        
        mask_before_40 = ages < 40
        mask_after_40 = ages >= 40
        
        if np.sum(mask_before_40) > 0 and len(self.models_before_40) > 0:
            X_before = X_scaled[mask_before_40]
            preds_before = np.zeros((len(X_before), len(self.models_before_40)))
            for i, (name, model) in enumerate(self.models_before_40):
                preds_before[:, i] = model.predict(X_before)
            predictions[mask_before_40] = np.average(preds_before, axis=1, weights=self.weights_before_40)
        
        if np.sum(mask_after_40) > 0 and len(self.models_after_40) > 0:
            X_after = X_scaled[mask_after_40]
            preds_after = np.zeros((len(X_after), len(self.models_after_40)))
            for i, (name, model) in enumerate(self.models_after_40):
                preds_after[:, i] = model.predict(X_after)
            predictions[mask_after_40] = np.average(preds_after, axis=1, weights=self.weights_after_40)
        
        return predictions


class SimpleBiasCorrector:
    """Simple bias correction per age group"""
    
    def __init__(self, strength=0.6):
        self.strength = strength
        self.correction_before_40 = None
        self.correction_after_40 = None
        
    def fit(self, y_true, y_pred):
        print("\n▶ Learning Bias Correction...")
        
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        residuals = y_pred - y_true
        
        mask_before_40 = y_true < 40
        mask_after_40 = y_true >= 40
        
        if np.sum(mask_before_40) > 10:
            self.correction_before_40 = Ridge(alpha=10.0)
            self.correction_before_40.fit(
                y_true[mask_before_40].reshape(-1, 1), 
                residuals[mask_before_40]
            )
            print(f"  Before 40: correction fitted on {np.sum(mask_before_40)} samples")
        
        if np.sum(mask_after_40) > 10:
            self.correction_after_40 = Ridge(alpha=10.0)
            self.correction_after_40.fit(
                y_true[mask_after_40].reshape(-1, 1), 
                residuals[mask_after_40]
            )
            print(f"  After 40: correction fitted on {np.sum(mask_after_40)} samples")
        
        print(f"  Correction strength: {self.strength * 100:.0f}%")
        return self
    
    def correct(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        corrected = y_pred.copy()
        
        mask_before_40 = y_true < 40
        mask_after_40 = y_true >= 40
        
        if self.correction_before_40 is not None and np.sum(mask_before_40) > 0:
            correction = self.correction_before_40.predict(y_true[mask_before_40].reshape(-1, 1))
            corrected[mask_before_40] -= self.strength * correction
        
        if self.correction_after_40 is not None and np.sum(mask_after_40) > 0:
            correction = self.correction_after_40.predict(y_true[mask_after_40].reshape(-1, 1))
            corrected[mask_after_40] -= self.strength * correction
        
        return corrected


class AgeStratifiedPipeline:
    """Simplified pipeline with age stratification"""
    
    def __init__(self, n_features=40):
        self.feature_selector = SimpleFeatureSelector(n_features=n_features)
        self.ensemble = AgeStratifiedEnsemble()
        self.bias_corrector = SimpleBiasCorrector(strength=0.6)
        self.brain_cols = None
        
    def fit(self, X_train, y_train, X_val, y_val):
        print("\n" + "="*70)
        print("AGE-STRATIFIED BRAIN AGE PREDICTION")
        print("="*70)
        
        print(f"\nDataset: Train={len(y_train)}, Val={len(y_val)}")
        print(f"Original features: {X_train.shape[1]}")
        
        X_train_sel = self.feature_selector.fit_transform(X_train, y_train)
        X_val_sel = self.feature_selector.transform(X_val)
        
        print(f"Final feature dimension: {X_train_sel.shape[1]}")
        
        self.ensemble.fit(X_train_sel, y_train)
        
        y_train_pred = self.ensemble.predict(X_train_sel, y_train)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        
        y_val_pred = self.ensemble.predict(X_val_sel, y_val)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        
        print(f"\n  Training MAE: {train_mae:.3f}")
        print(f"  Validation MAE (before correction): {val_mae:.3f}")
        
        self.bias_corrector.fit(y_val, y_val_pred)
        y_val_corrected = self.bias_corrector.correct(y_val, y_val_pred)
        val_mae_corrected = mean_absolute_error(y_val, y_val_corrected)
        print(f"  Validation MAE (after correction): {val_mae_corrected:.3f}")
        
        return self
    
    def predict(self, X_test, ages):
        X_sel = self.feature_selector.transform(X_test)
        y_pred = self.ensemble.predict(X_sel, ages)
        return y_pred
    
    def predict_corrected(self, X_test, y_test):
        y_pred = self.predict(X_test, y_test)
        y_pred_corrected = self.bias_corrector.correct(y_test, y_pred)
        return y_pred_corrected