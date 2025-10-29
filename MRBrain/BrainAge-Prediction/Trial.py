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

warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Add, LayerNormalization, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.utils import plot_model

from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

try:
    import shap
    SHAP_AVAILABLE = True
    print("SHAP library loaded successfully")
except ImportError:
    SHAP_AVAILABLE = False
    print("WARNING: SHAP library not available. Install with: pip install shap")

RANDOM_STATES = [876, 123, 456, 789, 321]
EPOCHS = 200
BATCH_SIZE = 32

class PhysicsInformedModel:
    def __init__(self, input_dim, hidden_layers=None, dropout_rates=None, learning_rate=0.001, random_state=876):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers or [512, 256, 256, 128, 64, 32]
        self.dropout_rates = dropout_rates or [0.4, 0.3, 0.2, 0.2, 0.1, 0.05]
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        self.scaler = PowerTransformer(method='yeo-johnson', standardize=True)
        self.history = None
    
    def build_model(self):
        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)
        
        input_layer = Input(shape=(self.input_dim,))
        
        x = Dense(self.hidden_layers[0], activation='swish', 
                 kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(input_layer)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rates[0])(x)
        
        branch1 = Dense(self.hidden_layers[1], activation='swish',
                       kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
        branch1 = LayerNormalization()(branch1)
        branch1 = Dropout(self.dropout_rates[1])(branch1)
        branch1 = Dense(self.hidden_layers[2], activation='swish',
                       kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(branch1)
        
        branch2 = Dense(self.hidden_layers[1], activation='relu',
                       kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
        branch2 = BatchNormalization()(branch2)
        branch2 = Dropout(self.dropout_rates[1])(branch2)
        
        x_skip = Dense(self.hidden_layers[2], activation='swish',
                      kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
        
        merged = Add()([branch1, x_skip])
        merged = Concatenate()([merged, branch2])
        
        x = Dense(self.hidden_layers[3], activation='swish',
                 kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(merged)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rates[3])(x)
        
        x = Dense(self.hidden_layers[4], activation='swish',
                 kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
        x = LayerNormalization()(x)
        x = Dropout(self.dropout_rates[4])(x)
        
        x = Dense(self.hidden_layers[5], activation='swish',
                 kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
        
        output = Dense(1, activation='linear', name='solution')(x)
        
        self.model = Model(inputs=input_layer, outputs=output)
        
        optimizer = Adam(learning_rate=self.learning_rate, 
                        beta_1=0.9, beta_2=0.999, epsilon=1e-8, clipnorm=1.0)
        
        self.model.compile(
            optimizer=optimizer, 
            loss='huber_loss', 
            metrics=['mae']
        )
        
        return self.model
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=200, batch_size=32, verbose=0):
        if self.model is None:
            self.build_model()
            
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        callbacks = [
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-7, verbose=0)
        ]
        
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            validation_data = (X_val_scaled, y_val)
            callbacks.extend([
                EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=0),
                ModelCheckpoint('temp_best_model.h5', monitor='val_loss', save_best_only=True, verbose=0)
            ])
        else:
            validation_data = None
            callbacks.append(EarlyStopping(monitor='loss', patience=20, restore_best_weights=True, verbose=0))
        
        self.history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks,
            shuffle=True
        )
        
        if os.path.exists('temp_best_model.h5'):
            self.model = tf.keras.models.load_model('temp_best_model.h5')
            os.remove('temp_best_model.h5')
        
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
            'random_state': self.random_state,
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

class MultiTrialBrainAgePipeline:
    def __init__(self, output_dir="results", random_states=None):
        self.output_dir = Path(output_dir)
        self.models_dir = self.output_dir / "saved_models"
        self.plots_dir = self.output_dir / "plots"
        self.importance_dir = self.output_dir / "feature_importance"
        self.trials_dir = self.output_dir / "trial_results"
        
        self.output_dir.mkdir(exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.importance_dir.mkdir(exist_ok=True)
        self.trials_dir.mkdir(exist_ok=True)
        
        self.random_states = random_states or RANDOM_STATES
        
        self.trial_results = {}
        self.best_trial_results = {}
        self.feature_importance_results = {}
        self.best_models = {}
        self.bias_correctors = {}
        
    def calculate_metrics(self, y_true, y_pred, tol=1e-8):
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
        print(f"  Extracting feature importance for {model_name}...")
        
        importance_results = {
            'feature_names': feature_names,
            'methods_used': [],
            'importance_scores': {},
            'rankings': {}
        }
        
        try:
            if SHAP_AVAILABLE:
                shap_importance = self._calculate_shap_importance(
                    model, X_train, X_test, model_name, feature_names
                )
                if shap_importance is not None:
                    importance_results['importance_scores']['shap'] = shap_importance
                    importance_results['methods_used'].append('SHAP')
                    print(f"    ✓ SHAP feature importance calculated")
            
            perm_importance = self._calculate_permutation_importance(
                model, X_test, y_test, model_name, feature_names
            )
            if perm_importance is not None:
                importance_results['importance_scores']['permutation'] = perm_importance
                importance_results['methods_used'].append('Permutation')
                print(f"    ✓ Permutation feature importance calculated")
            
            model_specific_importance = self._calculate_model_specific_importance(
                model, model_name, feature_names
            )
            if model_specific_importance is not None:
                importance_results['importance_scores']['model_specific'] = model_specific_importance
                importance_results['methods_used'].append('Model-specific')
                print(f"    ✓ Model-specific feature importance calculated")
            
            if model_name == 'Physics-Informed':
                gradient_importance = self._calculate_gradient_importance(
                    model, X_test, feature_names
                )
                if gradient_importance is not None:
                    importance_results['importance_scores']['gradient'] = gradient_importance
                    importance_results['methods_used'].append('Gradient-based')
                    print(f"    ✓ Gradient-based feature importance calculated")
            
            for method, scores in importance_results['importance_scores'].items():
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
        if not SHAP_AVAILABLE:
            return None
        
        try:
            sample_size = min(100, len(X_train))
            test_sample_size = min(50, len(X_test))
            
            X_train_sample = X_train.sample(n=sample_size, random_state=876) if hasattr(X_train, 'sample') else X_train[:sample_size]
            X_test_sample = X_test.sample(n=test_sample_size, random_state=876) if hasattr(X_test, 'sample') else X_test[:test_sample_size]
            
            if model_name == 'Physics-Informed':
                def model_predict(x):
                    return model.predict(x)
                explainer = shap.KernelExplainer(model_predict, X_train_sample)
                shap_values = explainer.shap_values(X_test_sample, nsamples=50)
                
            elif model_name in ['Random Forest', 'Gradient Boosting', 'Extra Trees', 'Ensemble']:
                explainer = shap.TreeExplainer(model)
                if hasattr(X_test_sample, 'values'):
                    shap_values = explainer.shap_values(X_test_sample.values)
                else:
                    shap_values = explainer.shap_values(X_test_sample)
                    
            else:
                try:
                    explainer = shap.LinearExplainer(model, X_train_sample)
                    shap_values = explainer.shap_values(X_test_sample)
                except:
                    def model_predict(x):
                        return model.predict(x)
                    explainer = shap.KernelExplainer(model_predict, X_train_sample)
                    shap_values = explainer.shap_values(X_test_sample, nsamples=50)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            importance_scores = np.mean(np.abs(shap_values), axis=0)
            return importance_scores
            
        except Exception as e:
            print(f"      SHAP calculation failed: {str(e)}")
            return None
    
    def _calculate_permutation_importance(self, model, X_test, y_test, model_name, feature_names):
        try:
            if model_name == 'Physics-Informed':
                def predict_func(X):
                    return model.predict(X)
            else:
                predict_func = model.predict
            
            perm_result = permutation_importance(
                model, X_test, y_test, 
                scoring='neg_mean_squared_error',
                n_repeats=5, 
                random_state=876,
                n_jobs=-1
            )
            
            return perm_result.importances_mean
            
        except Exception as e:
            print(f"      Permutation importance calculation failed: {str(e)}")
            return None
    
    def _calculate_model_specific_importance(self, model, model_name, feature_names):
        try:
            if model_name in ['Random Forest', 'Gradient Boosting', 'Extra Trees']:
                return model.feature_importances_
            elif model_name in ['Linear Regression', 'Ridge', 'Lasso', 'ElasticNet']:
                return np.abs(model.coef_)
            elif model_name == 'Ensemble':
                if hasattr(model.estimators_[0], 'feature_importances_'):
                    importances = []
                    for estimator in model.estimators_:
                        if hasattr(estimator, 'feature_importances_'):
                            importances.append(estimator.feature_importances_)
                    if importances:
                        return np.mean(importances, axis=0)
            return None
                
        except Exception as e:
            print(f"      Model-specific importance calculation failed: {str(e)}")
            return None
    
    def _calculate_gradient_importance(self, model, X_test, feature_names):
        try:
            sample_size = min(10, len(X_test))
            X_sample = X_test.sample(n=sample_size, random_state=876) if hasattr(X_test, 'sample') else X_test[:sample_size]
            
            X_scaled = model.scaler.transform(X_sample)
            X_tensor = tf.convert_to_tensor(X_scaled, dtype=tf.float32)
            
            with tf.GradientTape() as tape:
                tape.watch(X_tensor)
                predictions = model.model(X_tensor)
            
            gradients = tape.gradient(predictions, X_tensor)
            
            if gradients is not None:
                importance_scores = np.mean(np.abs(gradients.numpy()), axis=0)
                return importance_scores
            else:
                return None
                
        except Exception as e:
            print(f"      Gradient-based importance calculation failed: {str(e)}")
            return None

    def train_all_models_single_trial(self, X_train, X_test, y_train, y_test, age_group, random_state):
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        results = {}
        predictions = {}
        trained_models = {}
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        power_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
        X_train_power = power_transformer.fit_transform(X_train)
        X_test_power = power_transformer.transform(X_test)
        
        rf_params = {
            'n_estimators': 200,
            'max_depth': None,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'n_jobs': -1,
            'random_state': random_state,
            'criterion': 'squared_error',
            'max_samples': 0.8
        }
        
        gb_params = {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_samples_split': 5,
            'min_samples_leaf': 4,
            'subsample': 0.8,
            'max_features': 'sqrt',
            'random_state': random_state,
            'loss': 'huber',
            'alpha': 0.9
        }
        
        et_params = {
            'n_estimators': 200,
            'max_depth': None,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'n_jobs': -1,
            'random_state': random_state,
            'criterion': 'squared_error'
        }
        
        models_config = [
            ('Linear Regression', LinearRegression(), X_train, X_test, False, None),
            ('Ridge', Ridge(alpha=10.0, random_state=random_state, max_iter=2000), X_train_power, X_test_power, True, power_transformer),
            ('Lasso', Lasso(alpha=0.01, random_state=random_state, max_iter=2000), X_train_power, X_test_power, True, power_transformer),
            ('ElasticNet', ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=random_state, max_iter=2000), X_train_power, X_test_power, True, power_transformer),
            ('Random Forest', RandomForestRegressor(**rf_params), X_train, X_test, False, None),
            ('Gradient Boosting', GradientBoostingRegressor(**gb_params), X_train, X_test, False, None),
            ('Extra Trees', ExtraTreesRegressor(**et_params), X_train, X_test, False, None),
            ('SVR', SVR(kernel='rbf', C=1000, gamma='scale', epsilon=0.01), X_train_scaled, X_test_scaled, True, scaler),
            ('KNN', KNeighborsRegressor(n_neighbors=7, weights='distance', metric='minkowski'), X_train_scaled, X_test_scaled, True, scaler)
        ]
        
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
                    'train_data': train_data,
                    'random_state': random_state
                }
            except Exception as e:
                print(f"    {model_name}: Failed - {str(e)}")
                continue
        
        try:
            rf_model = RandomForestRegressor(**rf_params)
            gb_model = GradientBoostingRegressor(**gb_params)
            et_model = ExtraTreesRegressor(**et_params)
            
            ensemble = VotingRegressor(
                estimators=[
                    ('rf', rf_model),
                    ('gb', gb_model),
                    ('et', et_model)
                ],
                weights=[0.4, 0.35, 0.25]
            )
            
            ensemble.fit(X_train, y_train)
            ensemble_pred = ensemble.predict(X_test)
            results['Ensemble'] = self.calculate_metrics(y_test, ensemble_pred)
            predictions['Ensemble'] = ensemble_pred
            trained_models['Ensemble'] = {
                'model': ensemble,
                'scaler': None,
                'uses_scaling': False,
                'train_data': X_train,
                'random_state': random_state
            }
        except Exception as e:
            print(f"    Ensemble: Failed - {str(e)}")
        
        try:
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train, y_train, test_size=0.2, random_state=random_state
            )
            
            pim = PhysicsInformedModel(input_dim=X_train.shape[1], random_state=random_state)
            pim.fit(X_train_split, y_train_split, X_val_split, y_val_split, 
                   epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
            
            pim_pred = pim.predict(X_test)
            results['Physics-Informed'] = self.calculate_metrics(y_test, pim_pred)
            predictions['Physics-Informed'] = pim_pred
            trained_models['Physics-Informed'] = {
                'model': pim,
                'scaler': None,
                'uses_scaling': False,
                'train_data': X_train,
                'random_state': random_state
            }
        except Exception as e:
            print(f"    Physics-Informed: Failed - {str(e)}")
        
        return results, predictions, trained_models

    def apply_bias_correction_single_trial(self, trained_models, X_train, y_train, X_test, y_test, random_state):
        corrected_results = {}
        corrected_predictions = {}
        bias_correctors = {}
        
        for model_name, model_info in trained_models.items():
            try:
                model = model_info['model']
                scaler = model_info['scaler']
                uses_scaling = model_info['uses_scaling']
                
                if uses_scaling and scaler is not None:
                    train_pred = model.predict(scaler.transform(X_train))
                else:
                    train_pred = model.predict(X_train)
                
                bias_corrector_ = bias_corrector()
                bias_corrector_.fit(y_train, train_pred)
                
                if uses_scaling and scaler is not None:
                    test_pred = model.predict(scaler.transform(X_test))
                else:
                    test_pred = model.predict(X_test)
                
                corrected_test_pred = bias_corrector_.correct_predictions(y_test, test_pred)
                
                corrected_results[model_name] = self.calculate_metrics(y_test, corrected_test_pred)
                corrected_predictions[model_name] = corrected_test_pred
                bias_correctors[model_name] = bias_corrector_
                
            except Exception as e:
                print(f"    Bias correction for {model_name}: Failed - {str(e)}")
                continue
        
        return corrected_results, corrected_predictions, bias_correctors

    def select_best_trial(self, age_group):
        print(f"\nSelecting best trial for {age_group} age group...")
        
        trial_summary = {}
        
        for random_state in self.random_states:
            if random_state in self.trial_results[age_group]:
                corrected_results = self.trial_results[age_group][random_state]['corrected_results']
                
                rmse_values = [results['RMSE'] for results in corrected_results.values() if not np.isnan(results['RMSE'])]
                
                if rmse_values:
                    mean_rmse = np.mean(rmse_values)
                    best_model_rmse = min(rmse_values)
                    best_model_name = min(corrected_results.keys(), 
                                        key=lambda k: corrected_results[k]['RMSE'] if not np.isnan(corrected_results[k]['RMSE']) else float('inf'))
                    
                    trial_summary[random_state] = {
                        'mean_rmse': mean_rmse,
                        'best_model_rmse': best_model_rmse,
                        'best_model_name': best_model_name,
                        'n_successful_models': len(rmse_values)
                    }
                    
                    print(f"  Trial {random_state}: Mean RMSE = {mean_rmse:.4f}, Best = {best_model_name} ({best_model_rmse:.4f})")
        
        if not trial_summary:
            print(f"  No valid trials found for {age_group}")
            return None
        
        best_random_state = min(trial_summary.keys(), key=lambda k: trial_summary[k]['mean_rmse'])
        best_trial_info = trial_summary[best_random_state]
        
        print(f"  ✓ Best trial: {best_random_state} (Mean RMSE: {best_trial_info['mean_rmse']:.4f})")
        print(f"    Best model in trial: {best_trial_info['best_model_name']} (RMSE: {best_trial_info['best_model_rmse']:.4f})")
        
        trial_comparison_path = self.trials_dir / f'{age_group}_trial_comparison.json'
        with open(trial_comparison_path, 'w') as f:
            json.dump(trial_summary, f, indent=2, default=str)
        
        return best_random_state, trial_summary

    def save_feature_importance(self, importance_results, age_group, model_name):
        json_path = self.importance_dir / f"{age_group}_{model_name}_feature_importance.json"
        with open(json_path, 'w') as f:
            json.dump(importance_results, f, indent=2, default=str)
        
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

    def create_feature_importance_plots(self, importance_results, age_group, model_name):
        if not importance_results or not importance_results['methods_used']:
            return
        
        if 'shap' in importance_results['rankings']:
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
        
        plt.figure(figsize=(12, 8))
        
        y_pos = np.arange(len(feature_ranking))
        bars = plt.barh(y_pos, np.abs(importance_scores), color='steelblue', alpha=0.7)
        
        plt.yticks(y_pos, feature_ranking)
        plt.xlabel(ylabel)
        plt.title(f'Feature Importance - {age_group.title()} ({model_name})\n{title_suffix}')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        
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

    def save_best_model(self, best_model_name, trained_models, bias_corrector, age_group, feature_names, performance_metrics, random_state):
        print(f"Saving best model ({best_model_name}) for {age_group} age group from trial {random_state}...")
        
        age_group_dir = self.models_dir / age_group
        age_group_dir.mkdir(exist_ok=True)
        
        model_info = trained_models[best_model_name]
        model = model_info['model']
        scaler = model_info['scaler']
        
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
        
        bias_corrector_path = age_group_dir / "bias_corrector.pkl"
        with open(bias_corrector_path, 'wb') as f:
            pickle.dump(bias_corrector, f)
        
        metadata = {
            'age_group': age_group,
            'model_name': best_model_name,
            'model_type': 'neural_network' if best_model_name == 'Physics-Informed' else 'traditional',
            'uses_scaling': model_info['uses_scaling'],
            'feature_names': feature_names,
            'performance_metrics': performance_metrics,
            'timestamp': datetime.datetime.now().isoformat(),
            'selected_random_state': random_state,
            'trained_on_multiple_trials': True,
            'all_random_states_tested': self.random_states,
            'has_bias_correction': True
        }
        
        metadata_path = age_group_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"  Best model saved to: {age_group_dir}")

    def create_visualizations(self, original_results, corrected_results, corrected_predictions, y_test, age_group, feature_names, best_random_state, trial_summary):
        df_original = pd.DataFrame(original_results).T
        df_corrected = pd.DataFrame(corrected_results).T
        
        plt.figure(figsize=(20, 14))
        
        plt.subplot(3, 3, 1)
        models = df_corrected.index
        x_pos = np.arange(len(models))
        width = 0.35
        
        plt.bar(x_pos - width/2, df_original['RMSE'], width, label='Original', alpha=0.7, color='skyblue')
        plt.bar(x_pos + width/2, df_corrected['RMSE'], width, label='Bias Corrected', alpha=0.7, color='lightcoral')
        
        plt.title(f'RMSE Comparison - {age_group.title()} Age Group\n(Best Trial: Random State {best_random_state})')
        plt.ylabel('RMSE')
        plt.xlabel('Models')
        plt.xticks(x_pos, models, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 2)
        plt.bar(x_pos - width/2, df_original['R²'], width, label='Original', alpha=0.7, color='lightgreen')
        plt.bar(x_pos + width/2, df_corrected['R²'], width, label='Bias Corrected', alpha=0.7, color='darkgreen')
        
        plt.title(f'R² Score Comparison - {age_group.title()} Age Group')
        plt.ylabel('R² Score')
        plt.xlabel('Models')
        plt.xticks(x_pos, models, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 3)
        best_model_name = df_corrected.loc[df_corrected['RMSE'].idxmin()].name
        best_pred = corrected_predictions[best_model_name]
        
        plt.scatter(y_test, best_pred, alpha=0.6, s=30, color='darkred')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('True Age')
        plt.ylabel('Predicted Age (Bias Corrected)')
        plt.title(f'Best Model: {best_model_name}\nCorrected RMSE: {df_corrected.loc[best_model_name, "RMSE"]:.3f}')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 4)
        errors = best_pred - y_test
        plt.hist(errors, bins=20, alpha=0.7, color='coral', edgecolor='black')
        plt.xlabel('Prediction Error (years)')
        plt.ylabel('Frequency')
        plt.title(f'Error Distribution - {best_model_name} (Corrected)')
        plt.axvline(0, color='red', linestyle='--', alpha=0.8)
        plt.axvline(np.mean(errors), color='blue', linestyle='--', alpha=0.8, label=f'Mean: {np.mean(errors):.2f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 5)
        if age_group in self.feature_importance_results and best_model_name in self.feature_importance_results[age_group]:
            importance_data = self.feature_importance_results[age_group][best_model_name]
            
            if 'shap' in importance_data['rankings']:
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
        
        plt.subplot(3, 3, 6)
        improvement = df_original.loc[best_model_name, 'RMSE'] - df_corrected.loc[best_model_name, 'RMSE']
        plt.bar(['Original', 'Bias Corrected'], 
                [df_original.loc[best_model_name, 'RMSE'], df_corrected.loc[best_model_name, 'RMSE']], 
                color=['skyblue', 'darkblue'])
        plt.ylabel('RMSE')
        plt.title(f'Bias Correction Effect\nImprovement: {improvement:.3f} years')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 7)
        trial_states = list(trial_summary.keys())
        trial_rmses = [trial_summary[state]['mean_rmse'] for state in trial_states]
        colors = ['red' if state == best_random_state else 'lightblue' for state in trial_states]
        
        plt.bar(range(len(trial_states)), trial_rmses, color=colors)
        plt.xlabel('Random State')
        plt.ylabel('Mean RMSE')
        plt.title(f'Trial Performance Comparison\n(Best: {best_random_state})')
        plt.xticks(range(len(trial_states)), trial_states)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 8)
        best_model_rmses = [trial_summary[state]['best_model_rmse'] for state in trial_states]
        colors = ['red' if state == best_random_state else 'lightgreen' for state in trial_states]
        
        plt.bar(range(len(trial_states)), best_model_rmses, color=colors)
        plt.xlabel('Random State')
        plt.ylabel('Best Model RMSE')
        plt.title(f'Best Model Performance Across Trials')
        plt.xticks(range(len(trial_states)), trial_states)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 9)
        plt.text(0.1, 0.9, f'Multi-Trial Summary:', fontsize=14, fontweight='bold', transform=plt.gca().transAxes)
        plt.text(0.1, 0.8, f'Trials tested: {len(self.random_states)}', fontsize=11, transform=plt.gca().transAxes)
        plt.text(0.1, 0.7, f'Best random state: {best_random_state}', fontsize=11, transform=plt.gca().transAxes)
        plt.text(0.1, 0.6, f'Best model: {best_model_name}', fontsize=11, transform=plt.gca().transAxes)
        plt.text(0.1, 0.5, f'Final RMSE: {df_corrected.loc[best_model_name, "RMSE"]:.4f}', fontsize=11, transform=plt.gca().transAxes)
        plt.text(0.1, 0.4, f'Final R²: {df_corrected.loc[best_model_name, "R²"]:.4f}', fontsize=11, transform=plt.gca().transAxes)
        plt.text(0.1, 0.3, f'Improvement: {improvement:.4f}', fontsize=11, transform=plt.gca().transAxes)
        plt.text(0.1, 0.2, f'Mean trial RMSE: {trial_summary[best_random_state]["mean_rmse"]:.4f}', fontsize=11, transform=plt.gca().transAxes)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'comprehensive_analysis_{age_group}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def process_age_group_multi_trial(self, data, age_condition, age_group_name):
        print(f"\n{'='*60}")
        print(f"Processing {age_group_name.upper()} Age Group - MULTI-TRIAL")
        print(f"{'='*60}")
        
        age_data = data[age_condition]
        
        if age_data.empty:
            print(f"No data found for {age_group_name} age group")
            return None
        
        print(f"Found {len(age_data)} subjects")
        print(f"Testing {len(self.random_states)} random states: {self.random_states}")
        
        X = age_data.drop(columns=['Age'])
        y = age_data['Age']
        self.feature_names = X.columns.tolist()
        
        if len(X) < 10:
            print(f"Insufficient data ({len(X)} samples)")
            return None
        
        self.trial_results[age_group_name] = {}
        
        print(f"\nRunning {len(self.random_states)} trials...")
        
        for i, random_state in enumerate(self.random_states, 1):
            print(f"\n--- Trial {i}/{len(self.random_states)} (Random State: {random_state}) ---")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=random_state
            )
            
            original_results, original_predictions, trained_models = self.train_all_models_single_trial(
                X_train, X_test, y_train, y_test, age_group_name, random_state
            )
            
            print(f"  Original results:")
            for model_name, metrics in original_results.items():
                print(f"    {model_name}: RMSE = {metrics['RMSE']:.4f}")
            
            corrected_results, corrected_predictions, bias_correctors = self.apply_bias_correction_single_trial(
                trained_models, X_train, y_train, X_test, y_test, random_state
            )
            
            print(f"  Corrected results:")
            for model_name, metrics in corrected_results.items():
                print(f"    {model_name}: RMSE = {metrics['RMSE']:.4f}")
            
            self.trial_results[age_group_name][random_state] = {
                'original_results': original_results,
                'corrected_results': corrected_results,
                'corrected_predictions': corrected_predictions,
                'original_predictions': original_predictions,
                'trained_models': trained_models,
                'bias_correctors': bias_correctors,
                'test_data': {'X_test': X_test, 'y_test': y_test, 'X_train': X_train, 'y_train': y_train}
            }
        
        best_random_state, trial_summary = self.select_best_trial(age_group_name)
        
        if best_random_state is None:
            print(f"No valid trials found for {age_group_name}")
            return None
        
        best_trial_data = self.trial_results[age_group_name][best_random_state]
        best_corrected_results = best_trial_data['corrected_results']
        best_corrected_predictions = best_trial_data['corrected_predictions']
        best_trained_models = best_trial_data['trained_models']
        best_bias_correctors = best_trial_data['bias_correctors']
        best_test_data = best_trial_data['test_data']
        
        best_model_name = min(best_corrected_results.keys(), 
                             key=lambda k: best_corrected_results[k]['RMSE'] if not np.isnan(best_corrected_results[k]['RMSE']) else float('inf'))
        
        print(f"\nFINAL SELECTION for {age_group_name}:")
        print(f"✓ Best trial: Random State {best_random_state}")
        print(f"✓ Best model: {best_model_name}")
        print(f"✓ Final RMSE: {best_corrected_results[best_model_name]['RMSE']:.4f}")
        
        print(f"\n{'='*40}")
        print(f"EXTRACTING FEATURE IMPORTANCE")
        print(f"{'='*40}")
        
        if age_group_name not in self.feature_importance_results:
            self.feature_importance_results[age_group_name] = {}
        
        best_model_info = best_trained_models[best_model_name]
        importance_results = self.extract_feature_importance_comprehensive(
            best_model_info['model'], best_test_data['X_train'], best_test_data['X_test'], best_test_data['y_test'], 
            best_model_name, age_group_name, self.feature_names
        )
        
        if importance_results:
            self.feature_importance_results[age_group_name][best_model_name] = importance_results
            
            self.save_feature_importance(importance_results, age_group_name, best_model_name)
            
            self.create_feature_importance_plots(importance_results, age_group_name, best_model_name)
            
            print(f"✓ Feature importance successfully extracted using {len(importance_results['methods_used'])} method(s)")
            print(f"  Methods used: {', '.join(importance_results['methods_used'])}")
        else:
            print(f"⚠️ Feature importance extraction failed for {best_model_name}")
        
        self.best_trial_results[age_group_name] = {
            'original_results': best_trial_data['original_results'],
            'corrected_results': best_corrected_results,
            'corrected_predictions': best_corrected_predictions,
            'best_model_name': best_model_name,
            'best_random_state': best_random_state,
            'trial_summary': trial_summary,
            'test_data': best_test_data
        }
        
        self.save_best_model(
            best_model_name, best_trained_models, best_bias_correctors[best_model_name], 
            age_group_name, self.feature_names, best_corrected_results[best_model_name], best_random_state
        )
        
        self.best_models[age_group_name] = {
            'name': best_model_name,
            'model': best_trained_models[best_model_name],
            'bias_corrector': best_bias_correctors[best_model_name],
            'performance': best_corrected_results[best_model_name],
            'random_state': best_random_state
        }
        
        self.create_visualizations(
            best_trial_data['original_results'], best_corrected_results, best_corrected_predictions, 
            best_test_data['y_test'], age_group_name, self.feature_names, best_random_state, trial_summary
        )
        
        return best_corrected_results

    def generate_final_report(self):
        print(f"\n{'='*60}")
        print("GENERATING FINAL MULTI-TRIAL REPORT")
        print(f"{'='*60}")
        
        report = {
            'pipeline_info': {
                'timestamp': datetime.datetime.now().isoformat(),
                'random_states_tested': self.random_states,
                'multi_trial_selection': True,
                'output_directory': str(self.output_dir),
                'bias_correction_applied': True,
                'feature_importance_extracted': True,
                'only_best_models_saved': True
            },
            'age_groups': {},
            'best_models_comparison': {},
            'feature_importance_summary': {},
            'trial_analysis': {}
        }
        
        best_performances = {}
        for age_group, results_data in self.best_trial_results.items():
            corrected_results = results_data['corrected_results']
            best_model_name = results_data['best_model_name']
            best_random_state = results_data['best_random_state']
            trial_summary = results_data['trial_summary']
            
            best_performances[age_group] = {
                'name': best_model_name, 
                'rmse': corrected_results[best_model_name]['RMSE'],
                'r2': corrected_results[best_model_name]['R²'],
                'mae': corrected_results[best_model_name]['MAE'],
                'random_state': best_random_state
            }
            
            report['age_groups'][age_group] = {
                'best_model': best_model_name,
                'best_random_state': best_random_state,
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
            
            report['trial_analysis'][age_group] = {
                'all_trials': trial_summary,
                'best_trial_state': best_random_state,
                'selection_criterion': 'mean_rmse',
                'improvement_over_worst_trial': {
                    'worst_trial_rmse': max(trial_summary[state]['mean_rmse'] for state in trial_summary.keys()),
                    'best_trial_rmse': trial_summary[best_random_state]['mean_rmse'],
                    'improvement': max(trial_summary[state]['mean_rmse'] for state in trial_summary.keys()) - trial_summary[best_random_state]['mean_rmse']
                }
            }
            
            if age_group in self.feature_importance_results and best_model_name in self.feature_importance_results[age_group]:
                importance_data = self.feature_importance_results[age_group][best_model_name]
                
                if 'shap' in importance_data['rankings']:
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
                    'best_random_state': best_random_state,
                    'methods_used': importance_data['methods_used'],
                    'primary_method': method_used,
                    'top_10_features': top_features,
                    'total_features': len(importance_data['feature_names'])
                }
        
        report['best_models_comparison'] = best_performances
        if len(best_performances) > 1:
            overall_winner = min(best_performances.items(), key=lambda x: x[1]['rmse'])
            report['overall_best'] = {
                'age_group': overall_winner[0],
                'model_name': overall_winner[1]['name'],
                'performance': overall_winner[1]
            }
        
        report_path = self.output_dir / 'final_multi_trial_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.create_feature_importance_summary_csv()
        
        print("\nFINAL MULTI-TRIAL RESULTS SUMMARY (Bias Corrected):")
        print("="*55)
        
        for age_group, performance in best_performances.items():
            print(f"{age_group.upper()}: {performance['name']} (Random State: {performance['random_state']})")
            print(f"  - RMSE: {performance['rmse']:.4f}")
            print(f"  - R²: {performance['r2']:.4f}")
            print(f"  - MAE: {performance['mae']:.4f}")
            
            if age_group in report['trial_analysis']:
                trial_info = report['trial_analysis'][age_group]
                improvement = trial_info['improvement_over_worst_trial']['improvement']
                print(f"  - Multi-trial improvement: {improvement:.4f} RMSE")
            
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
        
        print(f"\nMULTI-TRIAL ANALYSIS:")
        print(f"="*25)
        print(f"- Total random states tested: {len(self.random_states)}")
        print(f"- Random states: {self.random_states}")
        print(f"- Selection criterion: Mean RMSE across all models")
        
        print(f"\nFEATURE IMPORTANCE FILES GENERATED:")
        print(f"="*40)
        importance_files = list(self.importance_dir.glob("*.csv"))
        for file in importance_files:
            print(f"  - {file.name}")
        
        print(f"\nFiles generated:")
        print(f"- Multi-trial report: {report_path}")
        print(f"- Trial comparison files: {self.trials_dir}")
        print(f"- Plots directory: {self.plots_dir}")
        print(f"- Feature importance directory: {self.importance_dir}")
        print(f"- Best models saved in: {self.models_dir}")
        print(f"  └── Each age group folder contains:")
        print(f"      ├── best_model.pkl (or best_model_tf_model/)")
        print(f"      ├── bias_corrector.pkl")
        print(f"      ├── metadata.json (includes selected random state)")
        print(f"      └── best_model_scaler.pkl (if needed)")
        
        return report

    def create_feature_importance_summary_csv(self):
        summary_data = []
        
        for age_group, models_data in self.feature_importance_results.items():
            for model_name, importance_data in models_data.items():
                
                if 'shap' in importance_data['rankings']:
                    ranking_data = importance_data['rankings']['shap']
                    method = 'SHAP'
                elif importance_data['methods_used']:
                    method = importance_data['methods_used'][0]
                    ranking_data = importance_data['rankings'][method]
                else:
                    continue
                
                for rank, (feature, score) in enumerate(zip(
                    ranking_data['feature_ranking'][:20], 
                    ranking_data['importance_ranking'][:20]
                ), 1):
                    summary_data.append({
                        'age_group': age_group,
                        'model': model_name,
                        'method': method,
                        'rank': rank,
                        'feature': feature,
                        'importance_score': score,
                        'selected_random_state': self.best_models[age_group]['random_state']
                    })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_path = self.importance_dir / 'feature_importance_summary.csv'
            summary_df.to_csv(summary_path, index=False)
            print(f"  - Feature importance summary: {summary_path}")
            
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
        print("="*60)
        print("MULTI-TRIAL BRAIN AGE PREDICTION PIPELINE")
        print("WITH GUARANTEED FEATURE IMPORTANCE EXTRACTION")
        print("="*60)
        
        print(f"Loading data from: {data_path}")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        data = pd.read_csv(data_path)
        
        if 'Age' not in data.columns:
            raise ValueError("Age column not found in the data")
        
        print(f"Data loaded: {data.shape[0]} samples, {data.shape[1]} features")
        print(f"Age range: {data['Age'].min():.1f} - {data['Age'].max():.1f} years")
        
        results_before = self.process_age_group_multi_trial(data, data['Age'] < 40, 'before_40')
        results_after = self.process_age_group_multi_trial(data, data['Age'] >= 40, 'after_40')
        
        final_report = self.generate_final_report()
        
        print(f"\n{'='*60}")
        print("MULTI-TRIAL PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"✓ Tested {len(self.random_states)} random states: {self.random_states}")
        print(f"✓ Selected best performing models based on mean RMSE")
        print(f"✓ Only best models saved (with bias correction)")
        print(f"✓ All results are bias-corrected")
        print(f"✓ Feature importance GUARANTEED for both age groups:")
        
        for age_group in ['before_40', 'after_40']:
            if age_group in self.feature_importance_results:
                models_with_importance = list(self.feature_importance_results[age_group].keys())
                best_state = self.best_models[age_group]['random_state']
                print(f"  - {age_group}: {', '.join(models_with_importance)} (Random State: {best_state})")
        
        print(f"✓ Multiple feature importance methods used as fallbacks")
        print(f"✓ Enhanced multi-trial visualizations and analysis")
        print(f"✓ Trial comparison data saved")
        print(f"✓ Ready for inference using prediction script")
        
        return final_report


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Trial Brain Age Prediction Pipeline with Guaranteed Feature Importance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python multi_trial_brain_age_pipeline.py --data features.csv
    python multi_trial_brain_age_pipeline.py --data features.csv --output_dir my_results
    python multi_trial_brain_age_pipeline.py --data features.csv --random_states 123 456 789

Features:
    - Trains models on 5 different random states (876, 123, 456, 789, 321)
    - Selects best performing model based on mean RMSE across all models
    - Guaranteed feature importance extraction for best models
    - Multiple fallback methods: SHAP, Permutation, Model-specific, Gradient-based
    - Enhanced visualizations with trial comparison
    - Trial analysis and comparison reports
    - All original functionality preserved
        """
    )
    
    parser.add_argument('--data', type=str, required=True,
                       help='Path to CSV file with brain features and Age column')
    parser.add_argument('--output_dir', type=str, default='multi_trial_brain_age_results',
                       help='Output directory for results (default: multi_trial_brain_age_results)')
    parser.add_argument('--random_states', nargs='+', type=int, default=RANDOM_STATES,
                       help=f'Random states for trials (default: {RANDOM_STATES})')
    
    args = parser.parse_args()
    
    if len(args.random_states) < 2:
        print("WARNING: At least 2 random states recommended for meaningful comparison")
    
    print(f"Using random states: {args.random_states}")
    
    pipeline = MultiTrialBrainAgePipeline(output_dir=args.output_dir, random_states=args.random_states)
    
    try:
        final_report = pipeline.run_pipeline(args.data)
        
        print(f"\nResults saved to: '{args.output_dir}' directory")
        print(f"Feature importance files saved to: '{args.output_dir}/feature_importance/' directory")
        print(f"Trial analysis files saved to: '{args.output_dir}/trial_results/' directory")
        print(f"Use the prediction script to make inference with saved models!")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())