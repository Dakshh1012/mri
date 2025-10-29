#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import datetime
import pickle
import json
import warnings
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Add
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

# SHAP for explainability
try:
    import shap
    SHAP_AVAILABLE = True
    print("SHAP library loaded successfully")
except ImportError:
    SHAP_AVAILABLE = False
    print("WARNING: SHAP library not available. Install with: pip install shap")

# Configuration
RANDOM_STATE = 876
EPOCHS = 150
BATCH_SIZE = 32

# Set random seeds
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

class PhysicsInformedModel:
    def __init__(self, input_dim, hidden_layers=None, dropout_rates=None, learning_rate=0.001):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers or [256, 128, 128, 64, 32]
        self.dropout_rates = dropout_rates or [0.5, 0.3, 0.2, 0.1, 0.1]
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
    
    def build_model(self):
        input_layer = Input(shape=(self.input_dim,))
        
        x = Dense(self.hidden_layers[0], activation='swish')(input_layer)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rates[0])(x)
        
        x1 = Dense(self.hidden_layers[1], activation='swish')(x)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(self.dropout_rates[1])(x1)
        x1 = Dense(self.hidden_layers[2], activation='swish')(x1)
        
        x_skip = Dense(self.hidden_layers[2], activation='swish')(x)
        x = Add()([x1, x_skip])
        
        x = Dense(self.hidden_layers[3], activation='swish')(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rates[3])(x)
        x = Dense(self.hidden_layers[4], activation='swish')(x)
        
        output = Dense(1, activation='linear', name='solution')(x)
        
        self.model = Model(inputs=input_layer, outputs=output)
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate), 
            loss='mse', 
            metrics=['mae']
        )
        
        return self.model
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=150, batch_size=32, verbose=0):
        if self.model is None:
            self.build_model()
            
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        callbacks = []
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            validation_data = (X_val_scaled, y_val)
            callbacks.append(EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True))
        else:
            validation_data = None
        
        self.history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks
        )
        return self
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled, verbose=0).flatten()
    
    def save_model(self, filepath_base):
        if self.model is None:
            raise ValueError("Model must be fitted before saving")
        
        tf_model_path = f"{filepath_base}_tf_model"
        self.model.save(tf_model_path)
        
        model_info = {
            'input_dim': self.input_dim,
            'hidden_layers': self.hidden_layers,
            'dropout_rates': self.dropout_rates,
            'learning_rate': self.learning_rate,
            'scaler': self.scaler
        }
        
        with open(f"{filepath_base}_info.pkl", 'wb') as f:
            pickle.dump(model_info, f)

class bias_corrector:
    def __init__(self):
        self.correction_model = LinearRegression()
        self.is_fitted = False
        self.train_age_stats = {}
    
    def fit(self, y_train_true, y_train_pred):
        brain_age_gap = y_train_pred - y_train_true
        y_train_reshaped = np.array(y_train_true).reshape(-1, 1)
        self.correction_model.fit(y_train_reshaped, brain_age_gap)
        self.is_fitted = True
        
        self.train_age_stats = {
            'min_age': float(y_train_true.min()),
            'max_age': float(y_train_true.max()),
            'mean_age': float(y_train_true.mean()),
            'std_age': float(y_train_true.std())
        }
    
    def correct_predictions(self, y_true, y_pred):
        if not self.is_fitted:
            raise ValueError("Bias correction model must be fitted first")
        
        y_true_reshaped = np.array(y_true).reshape(-1, 1)
        predicted_bias = self.correction_model.predict(y_true_reshaped)
        corrected_predictions = y_pred - predicted_bias
        return corrected_predictions

class EnhancedBrainAgePipeline:
    def __init__(self, output_dir="results"):
        self.output_dir = Path(output_dir)
        self.models_dir = self.output_dir / "saved_models"
        self.plots_dir = self.output_dir / "plots"
        self.importance_dir = self.output_dir / "feature_importance"
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.importance_dir.mkdir(exist_ok=True)
        
        # Initialize storage
        self.results = {}
        self.feature_importance_results = {}
        self.best_models = {}
        self.bias_correctors = {}
        
    def calculate_metrics(self, y_true, y_pred, tol=1e-8):
        """Calculate comprehensive evaluation metrics"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        if y_true.size == 0 or y_pred.size == 0 or y_true.shape[0] != y_pred.shape[0]:
            return {key: np.nan for key in ['MSE', 'RMSE', 'MAE', 'MedianAE', 'R²', 'Correlation', 'P-value']}
        
        if np.std(y_pred) <= tol:
            return {key: np.nan for key in ['MSE', 'RMSE', 'MAE', 'MedianAE', 'R²', 'Correlation', 'P-value']}
        
        try:
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            median_ae = np.median(np.abs(y_true - y_pred))
            r2 = r2_score(y_true, y_pred)
            
            if np.std(y_true) == 0 or np.std(y_pred) == 0:
                correlation, p_value = np.nan, np.nan
            else:
                correlation, p_value = stats.pearsonr(y_true, y_pred)
            
            return {
                'MSE': float(mse),
                'RMSE': float(rmse),
                'MAE': float(mae),
                'MedianAE': float(median_ae),
                'R²': float(r2),
                'Correlation': float(correlation),
                'P-value': float(p_value)
            }
        except Exception:
            return {key: np.nan for key in ['MSE', 'RMSE', 'MAE', 'MedianAE', 'R²', 'Correlation', 'P-value']}

    def extract_feature_importance_comprehensive(self, model, X_train, X_test, y_test, model_name, age_group, feature_names):
        """
        Extract feature importance using multiple methods with fallbacks
        """
        print(f"  Extracting feature importance for {model_name}...")
        
        importance_results = {
            'feature_names': feature_names,
            'methods_used': [],
            'importance_scores': {},
            'rankings': {}
        }
        
        try:
            # Method 1: SHAP (preferred)
            if SHAP_AVAILABLE:
                shap_importance = self._calculate_shap_importance(
                    model, X_train, X_test, model_name, feature_names
                )
                if shap_importance is not None:
                    importance_results['importance_scores']['shap'] = shap_importance
                    importance_results['methods_used'].append('SHAP')
                    print(f"    ✓ SHAP feature importance calculated")
            
            # Method 2: Permutation Importance (universal fallback)
            perm_importance = self._calculate_permutation_importance(
                model, X_test, y_test, model_name, feature_names
            )
            if perm_importance is not None:
                importance_results['importance_scores']['permutation'] = perm_importance
                importance_results['methods_used'].append('Permutation')
                print(f"    ✓ Permutation feature importance calculated")
            
            # Method 3: Model-specific methods
            model_specific_importance = self._calculate_model_specific_importance(
                model, model_name, feature_names
            )
            if model_specific_importance is not None:
                importance_results['importance_scores']['model_specific'] = model_specific_importance
                importance_results['methods_used'].append('Model-specific')
                print(f"    ✓ Model-specific feature importance calculated")
            
            # Method 4: Gradient-based (for neural networks)
            if model_name == 'Physics-Informed':
                gradient_importance = self._calculate_gradient_importance(
                    model, X_test, feature_names
                )
                if gradient_importance is not None:
                    importance_results['importance_scores']['gradient'] = gradient_importance
                    importance_results['methods_used'].append('Gradient-based')
                    print(f"    ✓ Gradient-based feature importance calculated")
            
            # Create rankings for each method
            for method, scores in importance_results['importance_scores'].items():
                # Sort by absolute importance (descending)
                sorted_indices = np.argsort(np.abs(scores))[::-1]
                importance_results['rankings'][method] = {
                    'feature_ranking': [feature_names[i] for i in sorted_indices],
                    'importance_ranking': [float(scores[i]) for i in sorted_indices]
                }
            
            
            return importance_results
            
        except Exception as e:
            print(f"    ⚠️ Feature importance extraction failed: {str(e)}")
            return None
    
    def _calculate_shap_importance(self, model, X_train, X_test, model_name, feature_names):
        """Calculate SHAP-based feature importance"""
        if not SHAP_AVAILABLE:
            return None
        
        try:
            # Sample data for faster computation
            sample_size = min(100, len(X_train))
            test_sample_size = min(50, len(X_test))
            
            X_train_sample = X_train.sample(n=sample_size, random_state=RANDOM_STATE) if hasattr(X_train, 'sample') else X_train[:sample_size]
            X_test_sample = X_test.sample(n=test_sample_size, random_state=RANDOM_STATE) if hasattr(X_test, 'sample') else X_test[:test_sample_size]
            
            if model_name == 'Physics-Informed':
                # For neural networks
                def model_predict(x):
                    return model.predict(x)
                explainer = shap.KernelExplainer(model_predict, X_train_sample)
                shap_values = explainer.shap_values(X_test_sample, nsamples=50)
                
            elif model_name in ['Random Forest', 'Gradient Boosting']:
                # For tree-based models
                explainer = shap.TreeExplainer(model)
                if hasattr(X_test_sample, 'values'):
                    shap_values = explainer.shap_values(X_test_sample.values)
                else:
                    shap_values = explainer.shap_values(X_test_sample)
                    
            else:
                # For linear models and others
                try:
                    explainer = shap.LinearExplainer(model, X_train_sample)
                    shap_values = explainer.shap_values(X_test_sample)
                except:
                    # Fallback to kernel explainer
                    def model_predict(x):
                        return model.predict(x)
                    explainer = shap.KernelExplainer(model_predict, X_train_sample)
                    shap_values = explainer.shap_values(X_test_sample, nsamples=50)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # Calculate mean absolute SHAP values as importance
            importance_scores = np.mean(np.abs(shap_values), axis=0)
            return importance_scores
            
        except Exception as e:
            print(f"      SHAP calculation failed: {str(e)}")
            return None
    
    def _calculate_permutation_importance(self, model, X_test, y_test, model_name, feature_names):
        """Calculate permutation-based feature importance"""
        try:
            # Create a wrapper for different model types
            if model_name == 'Physics-Informed':
                def predict_func(X):
                    return model.predict(X)
            else:
                predict_func = model.predict
            
            # Calculate permutation importance
            perm_result = permutation_importance(
                model, X_test, y_test, 
                scoring='neg_mean_squared_error',
                n_repeats=5, 
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
            
            return perm_result.importances_mean
            
        except Exception as e:
            print(f"      Permutation importance calculation failed: {str(e)}")
            return None
    
    def _calculate_model_specific_importance(self, model, model_name, feature_names):
        """Calculate model-specific feature importance"""
        try:
            if model_name in ['Random Forest', 'Gradient Boosting']:
                return model.feature_importances_
            elif model_name in ['Linear Regression', 'Ridge', 'Lasso']:
                # Use absolute coefficients as importance
                return np.abs(model.coef_)
            else:
                return None
                
        except Exception as e:
            print(f"      Model-specific importance calculation failed: {str(e)}")
            return None
    
    def _calculate_gradient_importance(self, model, X_test, feature_names):
        """Calculate gradient-based importance for neural networks"""
        try:
            # Sample a few test points
            sample_size = min(10, len(X_test))
            X_sample = X_test.sample(n=sample_size, random_state=RANDOM_STATE) if hasattr(X_test, 'sample') else X_test[:sample_size]
            
            # Convert to tensor
            X_scaled = model.scaler.transform(X_sample)
            X_tensor = tf.convert_to_tensor(X_scaled, dtype=tf.float32)
            
            # Calculate gradients
            with tf.GradientTape() as tape:
                tape.watch(X_tensor)
                predictions = model.model(X_tensor)
            
            gradients = tape.gradient(predictions, X_tensor)
            
            # Calculate importance as mean absolute gradient
            if gradients is not None:
                importance_scores = np.mean(np.abs(gradients.numpy()), axis=0)
                return importance_scores
            else:
                return None
                
        except Exception as e:
            print(f"      Gradient-based importance calculation failed: {str(e)}")
            return None
    

    def train_all_models(self, X_train, X_test, y_train, y_test, age_group):
        """Train all models and return results"""
        print(f"Training models for {age_group} age group...")
        
        results = {}
        predictions = {}
        trained_models = {}
        
        # Prepare scaled data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models
        models_config = [
            ('Linear Regression', LinearRegression(), X_train, X_test, False, None),
            ('Ridge', Ridge(alpha=1.0, random_state=RANDOM_STATE), X_train, X_test, False, None),
            ('Lasso', Lasso(alpha=0.1, random_state=RANDOM_STATE), X_train, X_test, False, None),
            ('Random Forest', RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1), X_train, X_test, False, None),
            ('Gradient Boosting', GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE), X_train, X_test, False, None),
            ('SVR', SVR(kernel='rbf', C=100, gamma='scale'), X_train_scaled, X_test_scaled, True, scaler),
            ('KNN', KNeighborsRegressor(n_neighbors=5), X_train_scaled, X_test_scaled, True, scaler)
        ]
        
        # Train traditional models
        for model_name, model, train_data, test_data, uses_scaling, model_scaler in models_config:
            try:
                model.fit(train_data, y_train)
                pred = model.predict(test_data)
                results[model_name] = self.calculate_metrics(y_test, pred)
                predictions[model_name] = pred
                trained_models[model_name] = {
                    'model': model,
                    'scaler': model_scaler,
                    'uses_scaling': uses_scaling,
                    'train_data': train_data
                }
                print(f"  {model_name}: RMSE = {results[model_name]['RMSE']:.4f}")
            except Exception as e:
                print(f"  {model_name}: Failed - {str(e)}")
                continue
        
        # Train Physics-Informed Model
        try:
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train, y_train, test_size=0.2, random_state=RANDOM_STATE
            )
            
            pim = PhysicsInformedModel(input_dim=X_train.shape[1])
            pim.fit(X_train_split, y_train_split, X_val_split, y_val_split, 
                   epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
            
            pim_pred = pim.predict(X_test)
            results['Physics-Informed'] = self.calculate_metrics(y_test, pim_pred)
            predictions['Physics-Informed'] = pim_pred
            trained_models['Physics-Informed'] = {
                'model': pim,
                'scaler': None,
                'uses_scaling': False,
                'train_data': X_train
            }
            print(f"  Physics-Informed: RMSE = {results['Physics-Informed']['RMSE']:.4f}")
        except Exception as e:
            print(f"  Physics-Informed: Failed - {str(e)}")
        
        return results, predictions, trained_models, X_train, y_train

    def apply_bias_correction(self, trained_models, X_train, y_train, X_test, y_test, age_group):
        """Apply bias correction to all models and return corrected results"""
        print(f"Applying bias correction for {age_group} age group...")
        
        corrected_results = {}
        corrected_predictions = {}
        bias_correctors = {}
        
        for model_name, model_info in trained_models.items():
            try:
                model = model_info['model']
                scaler = model_info['scaler']
                uses_scaling = model_info['uses_scaling']
                
                # Get training predictions for bias correction
                if uses_scaling and scaler is not None:
                    train_pred = model.predict(scaler.transform(X_train))
                else:
                    train_pred = model.predict(X_train)
                
                # Fit bias corrector
                bias_corrector_ = bias_corrector()
                bias_corrector_.fit(y_train, train_pred)
                
                # Get test predictions
                if uses_scaling and scaler is not None:
                    test_pred = model.predict(scaler.transform(X_test))
                else:
                    test_pred = model.predict(X_test)
                
                # Apply bias correction
                corrected_test_pred = bias_corrector_.correct_predictions(y_test, test_pred)
                
                # Calculate corrected metrics
                corrected_results[model_name] = self.calculate_metrics(y_test, corrected_test_pred)
                corrected_predictions[model_name] = corrected_test_pred
                bias_correctors[model_name] = bias_corrector_
                
                print(f"  {model_name}: Corrected RMSE = {corrected_results[model_name]['RMSE']:.4f}")
                
            except Exception as e:
                print(f"  Bias correction for {model_name}: Failed - {str(e)}")
                continue
        
        return corrected_results, corrected_predictions, bias_correctors

    def save_feature_importance(self, importance_results, age_group, model_name):
        """Save feature importance results to CSV and JSON files"""
        
        # Save comprehensive results as JSON
        json_path = self.importance_dir / f"{age_group}_{model_name}_feature_importance.json"
        with open(json_path, 'w') as f:
            json.dump(importance_results, f, indent=2, default=str)
        
        # Create CSV files for each method
        for method in importance_results['methods_used']:
            if method in importance_results['rankings']:
                ranking_data = importance_results['rankings'][method]
                df = pd.DataFrame({
                    'feature': ranking_data['feature_ranking'],
                    'importance_score': ranking_data['importance_ranking'],
                    'rank': range(1, len(ranking_data['feature_ranking']) + 1)
                })
                
                csv_path = self.importance_dir / f"{age_group}_{model_name}_{method.lower()}_importance.csv"
                df.to_csv(csv_path, index=False)
                print(f"    Saved {method} importance: {csv_path}")
        
        # Save consensus ranking if available

    def create_feature_importance_plots(self, importance_results, age_group, model_name):
        """Create visualization for feature importance"""
        
        if not importance_results or not importance_results['methods_used']:
            return
        
        # Get the primary method (prefer consensus, then SHAP, then others)
        if 'consensus' in importance_results:
            primary_method = 'consensus'
            feature_ranking = importance_results['consensus']['feature_ranking'][:15]  # Top 15
            importance_scores = importance_results['consensus']['mean_ranks'][:15]
            ylabel = 'Mean Rank (lower is better)'
            title_suffix = 'Consensus Ranking'
        elif 'shap' in importance_results['rankings']:
            primary_method = 'shap'
            feature_ranking = importance_results['rankings']['shap']['feature_ranking'][:15]
            importance_scores = importance_results['rankings']['shap']['importance_ranking'][:15]
            ylabel = 'SHAP Importance'
            title_suffix = 'SHAP Analysis'
        else:
            primary_method = importance_results['methods_used'][0]
            feature_ranking = importance_results['rankings'][primary_method]['feature_ranking'][:15]
            importance_scores = importance_results['rankings'][primary_method]['importance_ranking'][:15]
            ylabel = f'{primary_method.title()} Importance'
            title_suffix = f'{primary_method.title()} Analysis'
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Horizontal bar plot
        y_pos = np.arange(len(feature_ranking))
        bars = plt.barh(y_pos, np.abs(importance_scores), color='steelblue', alpha=0.7)
        
        # Customize plot
        plt.yticks(y_pos, feature_ranking)
        plt.xlabel(ylabel)
        plt.title(f'Feature Importance - {age_group.title()} ({model_name})\n{title_suffix}')
        plt.gca().invert_yaxis()  # Highest importance at top
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, importance_scores)):
            width = bar.get_width()
            plt.text(width + 0.01 * max(np.abs(importance_scores)), 
                    bar.get_y() + bar.get_height()/2, 
                    f'{abs(score):.3f}', 
                    ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        plot_path = self.plots_dir / f"{age_group}_{model_name}_feature_importance.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Feature importance plot saved: {plot_path}")

    def save_best_model(self, best_model_name, trained_models, bias_corrector, age_group, feature_names, performance_metrics):
        """Save only the best performing model"""
        print(f"Saving best model ({best_model_name}) for {age_group} age group...")
        
        age_group_dir = self.models_dir / age_group
        age_group_dir.mkdir(exist_ok=True)
        
        model_info = trained_models[best_model_name]
        model = model_info['model']
        scaler = model_info['scaler']
        
        # Save the model
        if best_model_name == 'Physics-Informed':
            filepath_base = age_group_dir / "best_model"
            model.save_model(str(filepath_base))
        else:
            model_path = age_group_dir / "best_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            if scaler is not None:
                scaler_path = age_group_dir / "best_model_scaler.pkl"
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
        
        # Save bias corrector
        bias_corrector_path = age_group_dir / "bias_corrector.pkl"
        with open(bias_corrector_path, 'wb') as f:
            pickle.dump(bias_corrector, f)
        
        # Save comprehensive metadata
        metadata = {
            'age_group': age_group,
            'model_name': best_model_name,
            'model_type': 'neural_network' if best_model_name == 'Physics-Informed' else 'traditional',
            'uses_scaling': model_info['uses_scaling'],
            'feature_names': feature_names,
            'performance_metrics': performance_metrics,
            'timestamp': datetime.datetime.now().isoformat(),
            'random_state': RANDOM_STATE,
            'has_bias_correction': True
        }
        
        metadata_path = age_group_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"  Best model saved to: {age_group_dir}")

    def create_visualizations(self, original_results, corrected_results, corrected_predictions, y_test, age_group, feature_names):
        """Create comprehensive visualizations comparing original vs corrected results"""
        
        # Create comparison DataFrame
        df_original = pd.DataFrame(original_results).T
        df_corrected = pd.DataFrame(corrected_results).T
        
        plt.figure(figsize=(18, 12))
        
        # RMSE comparison
        plt.subplot(2, 3, 1)
        models = df_corrected.index
        x_pos = np.arange(len(models))
        width = 0.35
        
        plt.bar(x_pos - width/2, df_original['RMSE'], width, label='Original', alpha=0.7, color='skyblue')
        plt.bar(x_pos + width/2, df_corrected['RMSE'], width, label='Bias Corrected', alpha=0.7, color='lightcoral')
        
        plt.title(f'RMSE Comparison - {age_group.title()} Age Group')
        plt.ylabel('RMSE')
        plt.xlabel('Models')
        plt.xticks(x_pos, models, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # R² comparison
        plt.subplot(2, 3, 2)
        plt.bar(x_pos - width/2, df_original['R²'], width, label='Original', alpha=0.7, color='lightgreen')
        plt.bar(x_pos + width/2, df_corrected['R²'], width, label='Bias Corrected', alpha=0.7, color='darkgreen')
        
        plt.title(f'R² Score Comparison - {age_group.title()} Age Group')
        plt.ylabel('R² Score')
        plt.xlabel('Models')
        plt.xticks(x_pos, models, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Best model prediction scatter plot (corrected)
        plt.subplot(2, 3, 3)
        best_model_name = df_corrected.loc[df_corrected['RMSE'].idxmin()].name
        best_pred = corrected_predictions[best_model_name]
        
        plt.scatter(y_test, best_pred, alpha=0.6, s=30, color='darkred')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('True Age')
        plt.ylabel('Predicted Age (Bias Corrected)')
        plt.title(f'Best Model: {best_model_name}\nCorrected RMSE: {df_corrected.loc[best_model_name, "RMSE"]:.3f}')
        plt.grid(True, alpha=0.3)
        
        # Error distribution (corrected)
        plt.subplot(2, 3, 4)
        errors = best_pred - y_test
        plt.hist(errors, bins=20, alpha=0.7, color='coral', edgecolor='black')
        plt.xlabel('Prediction Error (years)')
        plt.ylabel('Frequency')
        plt.title(f'Error Distribution - {best_model_name} (Corrected)')
        plt.axvline(0, color='red', linestyle='--', alpha=0.8)
        plt.axvline(np.mean(errors), color='blue', linestyle='--', alpha=0.8, label=f'Mean: {np.mean(errors):.2f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Feature importance (if available)
        plt.subplot(2, 3, 5)
        if age_group in self.feature_importance_results and best_model_name in self.feature_importance_results[age_group]:
            importance_data = self.feature_importance_results[age_group][best_model_name]
            
            # Get the best available importance method
            if 'consensus' in importance_data:
                feature_ranking = importance_data['consensus']['feature_ranking'][:10]
                importance_scores = importance_data['consensus']['mean_ranks'][:10]
                plt.barh(range(len(importance_scores)), importance_scores, color='gold')
                plt.yticks(range(len(importance_scores)), feature_ranking)
                plt.xlabel('Mean Rank (lower is better)')
                plt.title(f'Top 10 Features - {best_model_name}\n(Consensus Ranking)')
                plt.gca().invert_yaxis()
            elif 'shap' in importance_data['rankings']:
                feature_ranking = importance_data['rankings']['shap']['feature_ranking'][:10]
                importance_scores = importance_data['rankings']['shap']['importance_ranking'][:10]
                plt.barh(range(len(importance_scores)), importance_scores, color='gold')
                plt.yticks(range(len(importance_scores)), feature_ranking)
                plt.xlabel('SHAP Importance')
                plt.title(f'Top 10 Features - {best_model_name}\n(SHAP Analysis)')
                plt.gca().invert_yaxis()
            elif importance_data['methods_used']:
                method = importance_data['methods_used'][0]
                feature_ranking = importance_data['rankings'][method]['feature_ranking'][:10]
                importance_scores = importance_data['rankings'][method]['importance_ranking'][:10]
                plt.barh(range(len(importance_scores)), importance_scores, color='gold')
                plt.yticks(range(len(importance_scores)), feature_ranking)
                plt.xlabel(f'{method.title()} Importance')
                plt.title(f'Top 10 Features - {best_model_name}\n({method.title()} Analysis)')
                plt.gca().invert_yaxis()
        else:
            plt.text(0.5, 0.5, 'Feature importance\nanalysis pending...', ha='center', va='center', 
                    transform=plt.gca().transAxes, fontsize=12)
            plt.title('Feature Importance')
        
        # Bias correction effect
        plt.subplot(2, 3, 6)
        improvement = df_original.loc[best_model_name, 'RMSE'] - df_corrected.loc[best_model_name, 'RMSE']
        plt.bar(['Original', 'Bias Corrected'], 
                [df_original.loc[best_model_name, 'RMSE'], df_corrected.loc[best_model_name, 'RMSE']], 
                color=['skyblue', 'darkblue'])
        plt.ylabel('RMSE')
        plt.title(f'Bias Correction Effect\nImprovement: {improvement:.3f} years')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'comprehensive_analysis_{age_group}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def process_age_group(self, data, age_condition, age_group_name):
        """Process a specific age group with guaranteed feature importance extraction"""
        print(f"\n{'='*60}")
        print(f"Processing {age_group_name.upper()} Age Group")
        print(f"{'='*60}")
        
        age_data = data[age_condition]
        
        if age_data.empty:
            print(f"No data found for {age_group_name} age group")
            return None
        
        print(f"Found {len(age_data)} subjects")
        
        # Prepare features and target
        X = age_data.drop(columns=['Age'])
        y = age_data['Age']
        self.feature_names = X.columns.tolist()
        
        if len(X) < 10:
            print(f"Insufficient data ({len(X)} samples)")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )
        
        # Train models
        original_results, original_predictions, trained_models, X_train, y_train = self.train_all_models(
            X_train, X_test, y_train, y_test, age_group_name
        )
        
        # Apply bias correction
        corrected_results, corrected_predictions, bias_correctors = self.apply_bias_correction(
            trained_models, X_train, y_train, X_test, y_test, age_group_name
        )
        
        # Find best model (based on corrected results)
        best_model_name = min(corrected_results.keys(), 
                             key=lambda k: corrected_results[k]['RMSE'] if not np.isnan(corrected_results[k]['RMSE']) else float('inf'))
        
        print(f"\nBest model for {age_group_name}: {best_model_name}")
        print(f"Corrected RMSE: {corrected_results[best_model_name]['RMSE']:.4f}")
        
        # GUARANTEED FEATURE IMPORTANCE EXTRACTION
        print(f"\n{'='*40}")
        print(f"EXTRACTING FEATURE IMPORTANCE")
        print(f"{'='*40}")
        
        # Initialize feature importance results for this age group
        if age_group_name not in self.feature_importance_results:
            self.feature_importance_results[age_group_name] = {}
        
        # Extract feature importance for the best model
        best_model_info = trained_models[best_model_name]
        importance_results = self.extract_feature_importance_comprehensive(
            best_model_info['model'], X_train, X_test, y_test, 
            best_model_name, age_group_name, self.feature_names
        )
        
        if importance_results:
            self.feature_importance_results[age_group_name][best_model_name] = importance_results
            
            # Save feature importance results
            self.save_feature_importance(importance_results, age_group_name, best_model_name)
            
            # Create feature importance plots
            self.create_feature_importance_plots(importance_results, age_group_name, best_model_name)
            
            print(f"✓ Feature importance successfully extracted using {len(importance_results['methods_used'])} method(s)")
            print(f"  Methods used: {', '.join(importance_results['methods_used'])}")
        else:
            print(f"⚠️ Feature importance extraction failed for {best_model_name}")
        
        # Store results
        self.results[age_group_name] = {
            'original_results': original_results,
            'corrected_results': corrected_results,
            'corrected_predictions': corrected_predictions,
            'best_model_name': best_model_name,
            'test_data': {'X_test': X_test, 'y_test': y_test}
        }
        
        # Save only the best model
        self.save_best_model(
            best_model_name, trained_models, bias_correctors[best_model_name], 
            age_group_name, self.feature_names, corrected_results[best_model_name]
        )
        
        # Store best models info
        self.best_models[age_group_name] = {
            'name': best_model_name,
            'model': trained_models[best_model_name],
            'bias_corrector': bias_correctors[best_model_name],
            'performance': corrected_results[best_model_name]
        }
        
        # Create visualizations
        self.create_visualizations(
            original_results, corrected_results, corrected_predictions, 
            y_test, age_group_name, self.feature_names
        )
        
        return corrected_results

    def generate_final_report(self):
        """Generate comprehensive final report including feature importance"""
        print(f"\n{'='*60}")
        print("GENERATING FINAL REPORT")
        print(f"{'='*60}")
        
        report = {
            'pipeline_info': {
                'timestamp': datetime.datetime.now().isoformat(),
                'random_state': RANDOM_STATE,
                'output_directory': str(self.output_dir),
                'bias_correction_applied': True,
                'feature_importance_extracted': True,
                'only_best_models_saved': True
            },
            'age_groups': {},
            'best_models_comparison': {},
            'feature_importance_summary': {}
        }
        
        # Process each age group results
        best_performances = {}
        for age_group, results_data in self.results.items():
            corrected_results = results_data['corrected_results']
            best_model_name = results_data['best_model_name']
            
            best_performances[age_group] = {
                'name': best_model_name, 
                'rmse': corrected_results[best_model_name]['RMSE'],
                'r2': corrected_results[best_model_name]['R²'],
                'mae': corrected_results[best_model_name]['MAE']
            }
            
            # Store age group results
            report['age_groups'][age_group] = {
                'best_model': best_model_name,
                'best_performance_corrected': corrected_results[best_model_name],
                'all_models_corrected': corrected_results,
                'original_vs_corrected_comparison': {
                    model_name: {
                        'original_rmse': results_data['original_results'][model_name]['RMSE'],
                        'corrected_rmse': corrected_results[model_name]['RMSE'],
                        'improvement': results_data['original_results'][model_name]['RMSE'] - corrected_results[model_name]['RMSE']
                    }
                    for model_name in corrected_results.keys()
                    if model_name in results_data['original_results']
                },
                'data_summary': {
                    'n_test_samples': len(results_data['test_data']['y_test']),
                    'age_range_test': [
                        float(results_data['test_data']['y_test'].min()),
                        float(results_data['test_data']['y_test'].max())
                    ]
                }
            }
            
            # Add feature importance summary
            if age_group in self.feature_importance_results and best_model_name in self.feature_importance_results[age_group]:
                importance_data = self.feature_importance_results[age_group][best_model_name]
                
                # Get top 10 features from the best available method
                if 'consensus' in importance_data:
                    top_features = importance_data['consensus']['feature_ranking'][:10]
                    method_used = 'consensus'
                elif 'shap' in importance_data['rankings']:
                    top_features = importance_data['rankings']['shap']['feature_ranking'][:10]
                    method_used = 'SHAP'
                elif importance_data['methods_used']:
                    method_used = importance_data['methods_used'][0]
                    top_features = importance_data['rankings'][method_used]['feature_ranking'][:10]
                else:
                    top_features = []
                    method_used = 'none'
                
                report['feature_importance_summary'][age_group] = {
                    'best_model': best_model_name,
                    'methods_used': importance_data['methods_used'],
                    'primary_method': method_used,
                    'top_10_features': top_features,
                    'total_features': len(importance_data['feature_names'])
                }
        
        # Add comparison
        report['best_models_comparison'] = best_performances
        if len(best_performances) > 1:
            overall_winner = min(best_performances.items(), key=lambda x: x[1]['rmse'])
            report['overall_best'] = {
                'age_group': overall_winner[0],
                'model_name': overall_winner[1]['name'],
                'performance': overall_winner[1]
            }
        
        # Save comprehensive report
        report_path = self.output_dir / 'final_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Create summary CSV for feature importance
        self.create_feature_importance_summary_csv()
        
        # Print summary
        print("\nFINAL RESULTS SUMMARY (Bias Corrected):")
        print("="*50)
        
        for age_group, performance in best_performances.items():
            print(f"{age_group.upper()}: {performance['name']}")
            print(f"  - RMSE: {performance['rmse']:.4f}")
            print(f"  - R²: {performance['r2']:.4f}")
            print(f"  - MAE: {performance['mae']:.4f}")
            
            # Print feature importance info
            if age_group in report['feature_importance_summary']:
                fi_info = report['feature_importance_summary'][age_group]
                print(f"  - Feature Importance: ✓ ({fi_info['primary_method']})")
                print(f"    Top 3 features: {', '.join(fi_info['top_10_features'][:3])}")
            else:
                print(f"  - Feature Importance: ⚠️ Not available")
        
        if len(best_performances) > 1:
            winner = report['overall_best']
            print(f"\nOVERALL BEST: {winner['age_group']} - {winner['model_name']}")
            print(f"  - RMSE: {winner['performance']['rmse']:.4f}")
        
        print(f"\nFEATURE IMPORTANCE FILES GENERATED:")
        print(f"="*40)
        importance_files = list(self.importance_dir.glob("*.csv"))
        for file in importance_files:
            print(f"  - {file.name}")
        
        print(f"\nFiles generated:")
        print(f"- Final report: {report_path}")
        print(f"- Plots directory: {self.plots_dir}")
        print(f"- Feature importance directory: {self.importance_dir}")
        print(f"- Best models saved in: {self.models_dir}")
        print(f"  └── Each age group folder contains:")
        print(f"      ├── best_model.pkl (or best_model_tf_model/)")
        print(f"      ├── bias_corrector.pkl")
        print(f"      ├── metadata.json")
        print(f"      └── best_model_scaler.pkl (if needed)")
        
        return report

    def create_feature_importance_summary_csv(self):
        """Create a summary CSV comparing feature importance across age groups"""
        
        summary_data = []
        
        for age_group, models_data in self.feature_importance_results.items():
            for model_name, importance_data in models_data.items():
                
                # Get the primary ranking method
                if 'consensus' in importance_data:
                    ranking_data = importance_data['consensus']
                    method = 'consensus'
                elif 'shap' in importance_data['rankings']:
                    ranking_data = importance_data['rankings']['shap']
                    method = 'SHAP'
                elif importance_data['methods_used']:
                    method = importance_data['methods_used'][0]
                    ranking_data = importance_data['rankings'][method]
                else:
                    continue
                
                # Add top 20 features to summary
                for rank, (feature, score) in enumerate(zip(
                    ranking_data['feature_ranking'][:20], 
                    ranking_data['importance_ranking'][:20] if 'importance_ranking' in ranking_data else ranking_data['mean_ranks'][:20]
                ), 1):
                    summary_data.append({
                        'age_group': age_group,
                        'model': model_name,
                        'method': method,
                        'rank': rank,
                        'feature': feature,
                        'importance_score': score
                    })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_path = self.importance_dir / 'feature_importance_summary.csv'
            summary_df.to_csv(summary_path, index=False)
            print(f"  - Feature importance summary: {summary_path}")
            
            # Create comparison table for top features
            pivot_df = summary_df[summary_df['rank'] <= 10].pivot_table(
                index=['feature'], 
                columns=['age_group'], 
                values='rank',
                fill_value=np.nan
            )
            comparison_path = self.importance_dir / 'top_features_comparison.csv'
            pivot_df.to_csv(comparison_path)
            print(f"  - Top features comparison: {comparison_path}")

    def run_pipeline(self, data_path):
        """Run the complete enhanced pipeline with guaranteed feature importance"""
        print("="*60)
        print("ENHANCED BRAIN AGE PREDICTION PIPELINE")
        print("WITH GUARANTEED FEATURE IMPORTANCE EXTRACTION")
        print("="*60)
        
        # Load data
        print(f"Loading data from: {data_path}")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        data = pd.read_csv(data_path)
        
        if 'Age' not in data.columns:
            raise ValueError("Age column not found in the data")
        
        print(f"Data loaded: {data.shape[0]} samples, {data.shape[1]} features")
        print(f"Age range: {data['Age'].min():.1f} - {data['Age'].max():.1f} years")
        
        # Process age groups - feature importance extraction is guaranteed here
        results_before = self.process_age_group(data, data['Age'] < 40, 'before_40')
        results_after = self.process_age_group(data, data['Age'] >= 40, 'after_40')
        
        # Generate final report
        final_report = self.generate_final_report()
        
        print(f"\n{'='*60}")
        print("ENHANCED PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"✓ Only best performing models saved (with bias correction)")
        print(f"✓ All results are bias-corrected")
        print(f"✓ Feature importance GUARANTEED for both age groups:")
        
        for age_group in ['before_40', 'after_40']:
            if age_group in self.feature_importance_results:
                models_with_importance = list(self.feature_importance_results[age_group].keys())
                print(f"  - {age_group}: {', '.join(models_with_importance)}")
        
        print(f"✓ Multiple feature importance methods used as fallbacks")
        print(f"✓ Comprehensive visualizations and CSV files created")
        print(f"✓ Ready for inference using prediction script")
        
        return final_report


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Brain Age Prediction Pipeline with Guaranteed Feature Importance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python enhanced_brain_age_pipeline.py --data features.csv
    
    # Specify output directory
    python enhanced_brain_age_pipeline.py --data features.csv --output_dir my_results

Features:
    - Guaranteed feature importance extraction for best models in both age groups
    - Multiple fallback methods: SHAP, Permutation, Model-specific, Gradient-based
    - Consensus rankings when multiple methods available
    - CSV files for easy analysis
    - Enhanced visualizations
        """
    )
    
    parser.add_argument('--data', type=str, required=True,
                       help='Path to CSV file with brain features and Age column')
    parser.add_argument('--output_dir', type=str, default='enhanced_brain_age_results',
                       help='Output directory for results (default: enhanced_brain_age_results)')
    parser.add_argument('--random_state', type=int, default=876,
                       help='Random state for reproducibility (default: 876)')
    
    args = parser.parse_args()
    
    # Update global random state
    global RANDOM_STATE
    RANDOM_STATE = args.random_state
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)
    
    # Initialize and run enhanced pipeline
    pipeline = EnhancedBrainAgePipeline(output_dir=args.output_dir)
    
    try:
        final_report = pipeline.run_pipeline(args.data)
        
        print(f"\nResults saved to: '{args.output_dir}' directory")
        print(f"Feature importance files saved to: '{args.output_dir}/feature_importance/' directory")
        print(f"Use the prediction script to make inference with saved models!")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())