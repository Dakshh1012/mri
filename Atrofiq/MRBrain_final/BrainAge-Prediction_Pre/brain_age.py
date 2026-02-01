#!/usr/bin/env python3
"""
Brain Age Model Training Script with Visualization - Pre-Contrast Alignment
Aligned with Post-Contrast methodology from MRBrain/BrainAge-Prediction
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import model classes from separate module
from brain_age_models import AgeStratifiedPipeline

# =====================================================
# CONFIGURATION & DIRECTORIES
# =====================================================
MODEL_DIR = Path("saved_models")
MODEL_DIR.mkdir(exist_ok=True)

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

# Dataset paths
DATA_ADNI = Path('Data/ADNI_volumes.csv')
DATA_MAX = Path('Data/MAX_volumes.csv')
DATA_BATCH2 = Path('Data/Batch2_volumes.csv')

print(f"Models will be saved to: {MODEL_DIR.absolute()}")
print(f"Plots will be saved to: {PLOTS_DIR.absolute()}")

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def plot_predictions(y_true, y_pred, title, filename, show_stats=True):
    fig, ax = plt.subplots(figsize=(8, 8))
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    r, _ = pearsonr(y_true, y_pred)
    
    ax.scatter(y_true, y_pred, alpha=0.5, s=30, edgecolors='k', linewidth=0.5)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Identity')
    
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    ax.plot(y_true, p(y_true), 'b-', linewidth=2, alpha=0.7, label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')
    
    ax.set_xlabel('Chronological Age (years)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Brain Age (years)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    if show_stats:
        stats_text = f'MAE = {mae:.2f} years\nRMSE = {rmse:.2f} years\nR² = {r2:.3f}\nr = {r:.3f}\nn = {len(y_true)}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {filename}")


def plot_residuals(y_true, y_pred, title, filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    residuals = y_pred - y_true
    ax.scatter(y_true, residuals, alpha=0.5, s=30, edgecolors='k', linewidth=0.5)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    std = residuals.std()
    mean = residuals.mean()
    ax.axhline(y=mean + 2*std, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='±2 SD')
    ax.axhline(y=mean - 2*std, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axhline(y=mean, color='blue', linestyle='-', linewidth=1.5, alpha=0.7, label=f'Mean = {mean:.2f}')
    
    ax.set_xlabel('Chronological Age (years)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Residual (Predicted - Actual)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    stats_text = f'Mean = {mean:.2f} years\nSD = {std:.2f} years\nRange = [{residuals.min():.2f}, {residuals.max():.2f}]'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {filename}")


def plot_error_distribution(y_true, y_pred, title, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    
    ax1.hist(errors, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    ax1.axvline(x=errors.mean(), color='orange', linestyle='-', linewidth=2, label=f'Mean = {errors.mean():.2f}')
    ax1.set_xlabel('Prediction Error (years)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('Signed Prediction Error Distribution', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    ax2.hist(abs_errors, bins=30, color='coral', alpha=0.7, edgecolor='black')
    mae = abs_errors.mean()
    ax2.axvline(x=mae, color='darkred', linestyle='-', linewidth=2, label=f'MAE = {mae:.2f}')
    ax2.set_xlabel('Absolute Prediction Error (years)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax2.set_title('Absolute Prediction Error Distribution', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    within_2 = np.sum(abs_errors <= 2) / len(abs_errors) * 100
    within_3 = np.sum(abs_errors <= 3) / len(abs_errors) * 100
    within_5 = np.sum(abs_errors <= 5) / len(abs_errors) * 100
    
    stats_text = f'Within ±2y: {within_2:.1f}%\nWithin ±3y: {within_3:.1f}%\nWithin ±5y: {within_5:.1f}%'
    ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes, 
             fontsize=9, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {filename}")


def plot_age_stratified_performance(y_true, y_pred, title, filename, n_bins=5):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    bins = np.percentile(y_true, np.linspace(0, 100, n_bins + 1))
    bin_indices = np.digitize(y_true, bins[1:-1])
    
    bin_centers, bin_maes, bin_stds, bin_counts = [], [], [], []
    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_y_true = y_true[mask]
            bin_y_pred = y_pred[mask]
            errors = bin_y_pred - bin_y_true
            bin_centers.append(bin_y_true.mean())
            bin_maes.append(np.abs(errors).mean())
            bin_stds.append(errors.std())
            bin_counts.append(np.sum(mask))
    
    ax1.bar(range(len(bin_centers)), bin_maes, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.errorbar(range(len(bin_centers)), bin_maes, yerr=bin_stds, fmt='none', 
                 color='red', capsize=5, capthick=2, label='±1 SD')
    ax1.set_xlabel('Age Bin', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Mean Absolute Error (years)', fontsize=11, fontweight='bold')
    ax1.set_title('MAE by Age Group', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(bin_centers)))
    ax1.set_xticklabels([f'{c:.0f}y\n(n={n})' for c, n in zip(bin_centers, bin_counts)])
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    scatter = ax2.scatter(y_true, y_pred, c=y_true, cmap='viridis', 
                         alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Identity')
    ax2.set_xlabel('Chronological Age (years)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Predicted Brain Age (years)', fontsize=11, fontweight='bold')
    ax2.set_title('Predictions Colored by Age', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2).set_label('Age (years)', fontsize=10, fontweight='bold')
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {filename}")


def plot_bland_altman(y_true, y_pred, title, filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    mean_val = (y_true + y_pred) / 2
    diff = y_pred - y_true
    md, sd = diff.mean(), diff.std()
    
    ax.scatter(mean_val, diff, alpha=0.5, s=30, edgecolors='k', linewidth=0.5)
    ax.axhline(md, color='blue', linestyle='-', linewidth=2, label=f'Mean Diff = {md:.2f}')
    ax.axhline(md + 1.96*sd, color='red', linestyle='--', linewidth=2, label=f'+1.96 SD = {md + 1.96*sd:.2f}')
    ax.axhline(md - 1.96*sd, color='red', linestyle='--', linewidth=2, label=f'-1.96 SD = {md - 1.96*sd:.2f}')
    
    ax.set_xlabel('Mean of Predicted and Actual Age (years)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Difference (Predicted - Actual)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    within_loa = np.sum((diff >= md - 1.96*sd) & (diff <= md + 1.96*sd)) / len(diff) * 100
    stats_text = f'Within LoA: {within_loa:.1f}%\nSD = {sd:.2f} years'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {filename}")


def plot_comparison_summary(train_metrics, val_metrics, test_metrics, filename):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    metrics = ['MAE', 'RMSE', 'R²', 'Correlation']
    colors = ['steelblue', 'coral', 'mediumseagreen']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        values = [train_metrics[metric], val_metrics[metric], test_metrics[metric]]
        bars = ax.bar(['Train', 'Validation', 'Test'], values, color=colors, alpha=0.7, edgecolor='black')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.set_ylabel(metric, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        if metric in ['R²', 'Correlation']: ax.set_ylim([0, 1.1])
    
    plt.suptitle('Model Performance Summary', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {filename}")


def main():
    """Main execution - Pre-Contrast Multi-Dataset Training"""
    
    print("\n" + "="*70)
    print("COMBINED PRE-CONTRAST DATASET PREPARATION")
    print("="*70)
    
    # 1. Load Datasets
    print(f"Loading ADNI: {DATA_ADNI}")
    adni_df = pd.read_csv(DATA_ADNI)
    print(f"Loading MAX: {DATA_MAX}")
    max_df = pd.read_csv(DATA_MAX)
    print(f"Loading Batch 2: {DATA_BATCH2}")
    batch2_df = pd.read_csv(DATA_BATCH2)
    
    # 2. Harmonize
    if 'age' in max_df.columns: max_df = max_df.rename(columns={'age': 'Age'})
    if 'age' in batch2_df.columns: batch2_df = batch2_df.rename(columns={'age': 'Age'})
    
    metadata_cols = ['SubjectID', 'ScanID', 'Age', 'SEX', 'Sex', 'sex', 'Gender', 'Date', 'Filename', 'FilePath', 'JoinKey', 'Unnamed: 0']
    brain_cols_adni = [c for c in adni_df.columns if c not in metadata_cols]
    brain_cols_max = [c for c in max_df.columns if c not in metadata_cols]
    brain_cols_b2 = [c for c in batch2_df.columns if c not in metadata_cols]
    
    brain_cols = sorted(list(set(brain_cols_adni) & set(brain_cols_max) & set(brain_cols_b2)))
    print(f"Common brain features: {len(brain_cols)}")
    
    df = pd.concat([adni_df[['Age'] + brain_cols], max_df[['Age'] + brain_cols], batch2_df[['Age'] + brain_cols]], ignore_index=True)
    df = df.dropna(subset=['Age'])
    
    # 3. Filter Age > 56 (User Request)
    print("Filtering for Age > 56...")
    df = df[df['Age'] > 56].copy()
    print(f"Retained samples: {len(df)}")
    
    # 4. TIV Normalization (Methodology Alignment: Sum features)
    print("Calculating TIV (Sum of brain features)...")
    df['TIV'] = df[brain_cols].sum(axis=1)
    for col in brain_cols:
        df[col] = df[col] / df['TIV']
        
    # 5. Outlier Removal (Methodology Alignment: 3SD)
    original_size = len(df)
    for col in brain_cols:
        mean, std = df[col].mean(), df[col].std()
        df = df[(df[col] >= mean - 3*std) & (df[col] <= mean + 3*std)]
    print(f"After outlier removal (3SD): {len(df)} ({original_size - len(df)} removed)")
    
    # 6. Splits
    X = df[brain_cols].values
    y = df['Age'].values
    
    age_groups = pd.qcut(y, q=5, labels=False, duplicates='drop')
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=age_groups)
    
    age_groups_temp = pd.qcut(y_temp, q=4, labels=False, duplicates='drop')
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, random_state=42, stratify=age_groups_temp)
    
    y_train = pd.Series(y_train)
    y_val = pd.Series(y_val)
    y_test = pd.Series(y_test)
    
    # 7. Training
    pipeline = AgeStratifiedPipeline(n_features=40)
    pipeline.brain_cols = brain_cols
    pipeline.fit(X_train, y_train, X_val, y_val)
    
    # 8. Predictions & Calibration
    y_train_pred = pipeline.predict(X_train, y_train)
    y_val_pred = pipeline.predict(X_val, y_val)
    y_test_pred = pipeline.predict(X_test, y_test)
    
    y_train_corrected = pipeline.bias_corrector.correct(y_train, y_train_pred)
    y_val_corrected = pipeline.bias_corrector.correct(y_val, y_val_pred)
    y_test_corrected = pipeline.bias_corrector.correct(y_test, y_test_pred)
    
    # 9. Metrics
    def calc_metrics(y_t, y_p):
        err = y_p - y_t
        return {
            'MAE': mean_absolute_error(y_t, y_p),
            'RMSE': np.sqrt(mean_squared_error(y_t, y_p)),
            'R²': r2_score(y_t, y_p),
            'Correlation': pearsonr(y_t, y_p)[0],
            'SD': np.std(err)
        }
    
    train_m = calc_metrics(y_train, y_train_corrected)
    val_m = calc_metrics(y_val, y_val_corrected)
    test_m = calc_metrics(y_test, y_test_corrected)
    
    # 10. Visualization
    print("\nGenerating Plots...")
    
    # --- Training Set Plots ---
    print("  Training Set Plots...")
    plot_predictions(y_train, y_train_corrected, 'Training Set: Predicted vs Actual Brain Age', 'train_predictions.png')
    plot_residuals(y_train, y_train_corrected, 'Training Set: Residual Analysis', 'train_residuals.png')
    plot_error_distribution(y_train, y_train_corrected, 'Training Set: Error Distribution', 'train_error_distribution.png')
    plot_age_stratified_performance(y_train, y_train_corrected, 'Training Set: Age-Stratified Performance', 'train_age_stratified.png')
    plot_bland_altman(y_train, y_train_corrected, 'Training Set: Bland-Altman Plot', 'train_bland_altman.png')

    # --- Validation Set Plots ---
    print("  Validation Set Plots...")
    plot_predictions(y_val, y_val_corrected, 'Validation Set: Predicted vs Actual Brain Age', 'val_predictions.png')
    plot_residuals(y_val, y_val_corrected, 'Validation Set: Residual Analysis', 'val_residuals.png')
    plot_error_distribution(y_val, y_val_corrected, 'Validation Set: Error Distribution', 'val_error_distribution.png')
    plot_age_stratified_performance(y_val, y_val_corrected, 'Validation Set: Age-Stratified Performance', 'val_age_stratified.png')
    plot_bland_altman(y_val, y_val_corrected, 'Validation Set: Bland-Altman Plot', 'val_bland_altman.png')

    # --- Test Set Plots ---
    print("  Test Set Plots...")
    plot_predictions(y_test, y_test_corrected, 'Test Set: Predicted vs Actual Brain Age', 'test_predictions.png')
    plot_residuals(y_test, y_test_corrected, 'Test Set: Residual Analysis', 'test_residuals.png')
    plot_error_distribution(y_test, y_test_corrected, 'Test Set: Error Distribution', 'test_error_distribution.png')
    plot_age_stratified_performance(y_test, y_test_corrected, 'Test Set: Age-Stratified Performance', 'test_age_stratified.png')
    plot_bland_altman(y_test, y_test_corrected, 'Test Set: Bland-Altman Plot', 'test_bland_altman.png')
    
    plot_comparison_summary(train_m, val_m, test_m, 'performance_comparison.png')
    
    # 11. Reporting
    print(f"\nFINAL TEST RESULTS")
    print(f"  MAE:  {test_m['MAE']:.3f} years")
    print(f"  RMSE: {test_m['RMSE']:.3f} years")
    print(f"  R²:   {test_m['R²']:.3f}")
    print(f"  r:    {test_m['Correlation']:.3f}")
    
    # Save Pipeline
    model_path = MODEL_DIR / "brain_age_pipeline.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"\nPipeline saved to: {model_path}")
    print("Training complete!")


if __name__ == "__main__":
    main()