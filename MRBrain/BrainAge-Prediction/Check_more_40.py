import pandas as pd
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import datetime

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

RANDOM_STATE = 434343

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

class PhysicsInformedModel:
    def __init__(self, input_dim, hidden_layers=None, dropout_rates=None, learning_rate=0.001):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers or [256, 256, 128, 64, 32]
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
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32, verbose=1):
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
        return self.model.predict(X_scaled).flatten()

class BiasCorrector:
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
            'min_age': y_train_true.min(),
            'max_age': y_train_true.max(),
            'mean_age': y_train_true.mean(),
            'std_age': y_train_true.std()
        }
        
        print(f"Bias correction fitted - Slope: {self.correction_model.coef_[0]:.4f}, "
              f"Intercept: {self.correction_model.intercept_:.4f}")
    
    def correct_predictions(self, y_true, y_pred):
        if not self.is_fitted:
            raise ValueError("Bias correction model must be fitted first")
        
        margin = max(5, 2 * self.train_age_stats['std_age'])
        if y_true.min() < self.train_age_stats['min_age'] - margin or \
           y_true.max() > self.train_age_stats['max_age'] + margin:
            print("Warning: Test ages outside training range, bias correction may be unreliable")
        
        y_true_reshaped = np.array(y_true).reshape(-1, 1)
        predicted_bias = self.correction_model.predict(y_true_reshaped)
        
        corrected_predictions = y_pred - predicted_bias
        return corrected_predictions

def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    if y_true.size == 0 or y_pred.size == 0 or y_true.shape[0] != y_pred.shape[0]:
        return {key: np.nan for key in ['MSE', 'RMSE', 'MAE', 'R²', 'MAPE', 'Correlation', 'P-value']}
    
    try:
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        epsilon = np.finfo(np.float64).eps
        mask = np.abs(y_true) > epsilon
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.nan
        
        correlation, p_value = stats.pearsonr(y_true, y_pred)
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'MAPE': mape,
            'Correlation': correlation,
            'P-value': p_value
        }
    except Exception:
        return {key: np.nan for key in ['MSE', 'RMSE', 'MAE', 'R²', 'MAPE', 'Correlation', 'P-value']}

def train_all_models(X_train, X_test, y_train, y_test):
    results = {}
    predictions = {}
    all_models = {}
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model_configs = [
        ('Linear Regression', LinearRegression(), X_train, X_test, False),
        ('Ridge', Ridge(alpha=1.0, random_state=RANDOM_STATE), X_train, X_test, False),
        ('Lasso', Lasso(alpha=0.1, random_state=RANDOM_STATE), X_train, X_test, False),
        ('Random Forest', RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1), X_train, X_test, False),
        ('Gradient Boosting', GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE), X_train, X_test, False),
        ('SVR', SVR(kernel='rbf', C=100, gamma='scale'), X_train_scaled, X_test_scaled, True),
        ('KNN', KNeighborsRegressor(n_neighbors=5), X_train_scaled, X_test_scaled, True)
    ]
    
    for model_name, model, train_data, test_data, needs_scaling in model_configs:
        print(f"Training {model_name}...")
        model.fit(train_data, y_train)
        pred = model.predict(test_data)
        results[model_name] = calculate_metrics(y_test, pred)
        predictions[model_name] = pred
        all_models[model_name] = (model, train_data, test_data, needs_scaling, scaler if needs_scaling else None)
        print(f"   RMSE: {results[model_name]['RMSE']:.4f}")
    
    print("Training Physics-Informed Model...")
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE
    )
    
    pim = PhysicsInformedModel(input_dim=X_train.shape[1])
    pim.fit(X_train_split, y_train_split, X_val_split, y_val_split, 
            epochs=150, batch_size=32, verbose=1)
    
    pim_pred = pim.predict(X_test)
    results['Physics-Informed'] = calculate_metrics(y_test, pim_pred)
    predictions['Physics-Informed'] = pim_pred
    all_models['Physics-Informed'] = (pim, X_train, X_test, False, None)
    print(f"   RMSE: {results['Physics-Informed']['RMSE']:.4f}")
    
    return results, predictions, all_models

def apply_bias_correction(results, predictions, all_models, y_train, y_test):
    print("\nApplying bias correction to all models...")
    
    for model_name, (model, train_data, test_data, needs_scaling, scaler_obj) in all_models.items():
        print(f"{model_name}:")
        
        if needs_scaling and scaler_obj is not None:
            model_train_pred = model.predict(scaler_obj.transform(train_data))
        else:
            model_train_pred = model.predict(train_data)
        
        bias_corrector = BiasCorrector()
        bias_corrector.fit(y_train, model_train_pred)
        
        original_pred = predictions[model_name]
        corrected_pred = bias_corrector.correct_predictions(y_test, original_pred)
        
        corrected_name = f'{model_name} (Bias Corrected)'
        results[corrected_name] = calculate_metrics(y_test, corrected_pred)
        predictions[corrected_name] = corrected_pred

def create_visualizations(results, predictions, y_test, X_train, y_train, excel_path):
    df_results = pd.DataFrame(results).T.round(4)
    
    dataset_name = os.path.basename(excel_path).replace('.xlsx', '').replace(' ', '_')
    timestamp = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    csv_filename = f"{dataset_name}_BrainAgePrediction_above_40_rs{RANDOM_STATE}_{timestamp}.csv"
    df_results.to_csv(csv_filename)
    
    for i, (model_name, row) in enumerate(df_results.iterrows(), 1):
        print(f"{i:2d}. {model_name:30s} - RMSE: {row['RMSE']:7.4f}, MAE: {row['MAE']:7.4f}, R²: {row['R²']:7.4f}")
    
    base_models = ['Linear Regression', 'Ridge', 'Lasso', 'Random Forest', 
                   'Gradient Boosting', 'SVR', 'KNN', 'Physics-Informed']
    
    bias_improvements = {}
    for model_name in base_models:
        if model_name in results and f'{model_name} (Bias Corrected)' in results:
            original_rmse = results[model_name]['RMSE']
            corrected_rmse = results[f'{model_name} (Bias Corrected)']['RMSE']
            improvement = original_rmse - corrected_rmse
            bias_improvements[model_name] = improvement
    
    print("\nBias Correction Improvements:")
    for model, improvement in sorted(bias_improvements.items(), key=lambda x: x[1], reverse=True):
        print(f"{model:20s}: {improvement:+7.4f}")
    
    n_models = len(predictions)
    n_cols = 4
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for i, (model_name, pred) in enumerate(predictions.items()):
        if i < len(axes):
            ax = axes[i]
            ax.scatter(y_test, pred, alpha=0.6, s=20)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax.set_xlabel('True Age')
            ax.set_ylabel('Predicted Age')
            ax.set_title(f'{model_name}\nRMSE: {results[model_name]["RMSE"]:.3f}')
            ax.grid(True, alpha=0.3)
    
    for i in range(len(predictions), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig(f"{RANDOM_STATE}_scatter_plots_above_40.png", dpi=300)
    plt.show()
    
    n_error_models = min(9, len(predictions))
    n_cols_error = 3
    n_rows_error = (n_error_models + n_cols_error - 1) // n_cols_error
    
    fig, axes = plt.subplots(n_rows_error, n_cols_error, figsize=(15, 5*n_rows_error))
    if n_rows_error == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for i, (model_name, pred) in enumerate(list(predictions.items())[:n_error_models]):
        ax = axes[i]
        errors = pred - y_test
        ax.hist(errors, bins=30, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Prediction Error (years)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{model_name} - Error Distribution\nMean Error: {np.mean(errors):.3f}')
        ax.axvline(0, color='red', linestyle='--', alpha=0.8)
        ax.grid(True, alpha=0.3)
    
    for i in range(n_error_models, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig(f"{RANDOM_STATE}_error_distributions_above_40.png", dpi=300)
    plt.show()
    
    metrics_df = pd.DataFrame(results).T
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0,0].bar(range(len(metrics_df)), metrics_df['RMSE'], color='skyblue')
    axes[0,0].set_title('Root Mean Square Error (Lower is Better)')
    axes[0,0].set_ylabel('RMSE')
    axes[0,0].set_xticks(range(len(metrics_df)))
    axes[0,0].set_xticklabels(metrics_df.index, rotation=45, ha='right')
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].bar(range(len(metrics_df)), metrics_df['R²'], color='lightgreen')
    axes[0,1].set_title('R² Score (Higher is Better)')
    axes[0,1].set_ylabel('R²')
    axes[0,1].set_xticks(range(len(metrics_df)))
    axes[0,1].set_xticklabels(metrics_df.index, rotation=45, ha='right')
    axes[0,1].grid(True, alpha=0.3)
    
    axes[1,0].bar(range(len(metrics_df)), metrics_df['MAE'], color='coral')
    axes[1,0].set_title('Mean Absolute Error (Lower is Better)')
    axes[1,0].set_ylabel('MAE')
    axes[1,0].set_xticks(range(len(metrics_df)))
    axes[1,0].set_xticklabels(metrics_df.index, rotation=45, ha='right')
    axes[1,0].grid(True, alpha=0.3)
    
    axes[1,1].bar(range(len(metrics_df)), metrics_df['Correlation'], color='gold')
    axes[1,1].set_title('Correlation (Higher is Better)')
    axes[1,1].set_ylabel('Correlation')
    axes[1,1].set_xticks(range(len(metrics_df)))
    axes[1,1].set_xticklabels(metrics_df.index, rotation=45, ha='right')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{RANDOM_STATE}_performance_metrics_above_40.png", dpi=300)
    plt.show()
    
    rf_importance = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
    rf_importance.fit(X_train, y_train)
    
    importance_df = pd.DataFrame({
        'Feature': X_train.columns.tolist(),
        'Importance': rf_importance.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(20)
    plt.barh(top_features['Feature'], top_features['Importance'])
    plt.xlabel('Feature Importance')
    plt.title('Top 20 Feature Importances (Random Forest)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    return df_results

def main():
    parser = argparse.ArgumentParser(description="Brain Age Prediction (>40)")
    parser.add_argument('--data', type=str, required=True, help='Path to Excel data file')
    args = parser.parse_args()

    if not os.path.isfile(args.data):
        raise FileNotFoundError(f"Data file not found: {args.data}")

    new_data = pd.read_csv(args.data)
    valid_features = new_data.columns.tolist()

    corr_matrix = new_data.loc[new_data['Age'] > 39][valid_features].corr()
    age_corr = corr_matrix['Age'].abs().sort_values(ascending=False)
    top10_features = age_corr.index[1:11]
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(new_data.loc[new_data['Age'] > 39][list(top10_features) + ['Age']].corr(), 
                annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap - Top 10 Features (Age > 39)')
    plt.show()
    
    data_above_40 = new_data[new_data['Age'] > 39]
    X = data_above_40[valid_features].drop(columns=['Age'])
    y = data_above_40['Age']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    results, predictions, all_models = train_all_models(X_train, X_test, y_train, y_test)
    apply_bias_correction(results, predictions, all_models, y_train, y_test)
    df_results = create_visualizations(results, predictions, y_test, X_train, y_train, args.data)
    
    return df_results

if __name__ == "__main__":
    main()