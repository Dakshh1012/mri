#!/usr/bin/env python3
"""
Brain Age Model Training Script with Visualization
Uses brain_age_models.py classes for proper pickle serialization
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
# MODEL DIRECTORY
# =====================================================
MODEL_DIR = Path("saved_models")
MODEL_DIR.mkdir(exist_ok=True)

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

print(f"Models will be saved to: {MODEL_DIR.absolute()}")
print(f"Plots will be saved to: {PLOTS_DIR.absolute()}")

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def plot_predictions(y_true, y_pred, title, filename, show_stats=True):
    """
    Create scatter plot of predicted vs actual age with regression line.
    
    Args:
        y_true: Actual chronological ages
        y_pred: Predicted brain ages
        title: Plot title
        filename: Output filename
        show_stats: Whether to show statistics on plot
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    r, _ = pearsonr(y_true, y_pred)
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.5, s=30, edgecolors='k', linewidth=0.5)
    
    # Identity line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Identity')
    
    # Regression line
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    ax.plot(y_true, p(y_true), 'b-', linewidth=2, alpha=0.7, label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')
    
    ax.set_xlabel('Chronological Age (years)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Brain Age (years)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add statistics text box
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
    """
    Create residual plot (error vs age).
    
    Args:
        y_true: Actual chronological ages
        y_pred: Predicted brain ages
        title: Plot title
        filename: Output filename
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    residuals = y_pred - y_true
    
    # Scatter plot
    ax.scatter(y_true, residuals, alpha=0.5, s=30, edgecolors='k', linewidth=0.5)
    
    # Zero line
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    
    # ±2 SD lines
    std = residuals.std()
    mean = residuals.mean()
    ax.axhline(y=mean + 2*std, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='±2 SD')
    ax.axhline(y=mean - 2*std, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    
    # Mean line
    ax.axhline(y=mean, color='blue', linestyle='-', linewidth=1.5, alpha=0.7, label=f'Mean = {mean:.2f}')
    
    ax.set_xlabel('Chronological Age (years)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Residual (Predicted - Actual)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Statistics text
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
    """
    Create histogram of prediction errors.
    
    Args:
        y_true: Actual chronological ages
        y_pred: Predicted brain ages
        title: Plot title
        filename: Output filename
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    
    # Signed errors histogram
    ax1.hist(errors, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    ax1.axvline(x=errors.mean(), color='orange', linestyle='-', linewidth=2, label=f'Mean = {errors.mean():.2f}')
    ax1.set_xlabel('Prediction Error (years)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('Signed Prediction Error Distribution', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Absolute errors histogram
    ax2.hist(abs_errors, bins=30, color='coral', alpha=0.7, edgecolor='black')
    mae = abs_errors.mean()
    ax2.axvline(x=mae, color='darkred', linestyle='-', linewidth=2, label=f'MAE = {mae:.2f}')
    ax2.set_xlabel('Absolute Prediction Error (years)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax2.set_title('Absolute Prediction Error Distribution', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
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
    """
    Plot performance metrics across age bins.
    
    Args:
        y_true: Actual chronological ages
        y_pred: Predicted brain ages
        title: Plot title
        filename: Output filename
        n_bins: Number of age bins
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Create age bins
    bins = np.percentile(y_true, np.linspace(0, 100, n_bins + 1))
    bin_indices = np.digitize(y_true, bins[1:-1])
    
    # Calculate metrics per bin
    bin_centers = []
    bin_maes = []
    bin_stds = []
    bin_counts = []
    
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
    
    # Plot MAE by age bin
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
    
    # Plot prediction scatter colored by age bin
    scatter = ax2.scatter(y_true, y_pred, c=y_true, cmap='viridis', 
                         alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
    
    # Identity line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Identity')
    
    ax2.set_xlabel('Chronological Age (years)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Predicted Brain Age (years)', fontsize=11, fontweight='bold')
    ax2.set_title('Predictions Colored by Age', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Age (years)', fontsize=10, fontweight='bold')
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {filename}")


def plot_bland_altman(y_true, y_pred, title, filename):
    """
    Create Bland-Altman plot for agreement analysis.
    
    Args:
        y_true: Actual chronological ages
        y_pred: Predicted brain ages
        title: Plot title
        filename: Output filename
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    mean = (y_true + y_pred) / 2
    diff = y_pred - y_true
    
    md = diff.mean()
    sd = diff.std()
    
    # Scatter plot
    ax.scatter(mean, diff, alpha=0.5, s=30, edgecolors='k', linewidth=0.5)
    
    # Mean difference line
    ax.axhline(md, color='blue', linestyle='-', linewidth=2, label=f'Mean Diff = {md:.2f}')
    
    # Limits of agreement (±1.96 SD)
    ax.axhline(md + 1.96*sd, color='red', linestyle='--', linewidth=2, label=f'+1.96 SD = {md + 1.96*sd:.2f}')
    ax.axhline(md - 1.96*sd, color='red', linestyle='--', linewidth=2, label=f'-1.96 SD = {md - 1.96*sd:.2f}')
    
    ax.set_xlabel('Mean of Predicted and Actual Age (years)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Difference (Predicted - Actual)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add statistics
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
    """
    Create summary comparison plot of train/val/test metrics.
    
    Args:
        train_metrics: Dict with train metrics
        val_metrics: Dict with val metrics
        test_metrics: Dict with test metrics
        filename: Output filename
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    metrics = ['MAE', 'RMSE', 'R²', 'Correlation']
    colors = ['steelblue', 'coral', 'mediumseagreen']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        values = [train_metrics[metric], val_metrics[metric], test_metrics[metric]]
        bars = ax.bar(['Train', 'Validation', 'Test'], values, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel(metric, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Set y-axis limits for better visualization
        if metric in ['R²', 'Correlation']:
            ax.set_ylim([0, 1.1])
    
    plt.suptitle('Model Performance Summary', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {filename}")


def main(filepath):
    """Main execution"""
    
    df = pd.read_excel(filepath)
    print(f"Data loaded: {df.shape}")
    
    brain_cols = [col for col in df.columns if col not in ['Age', 'SEX', 'Sex', 'sex']]
    print(f"Brain features: {len(brain_cols)}")
    
    df_clean = df.dropna()
    print(f"After removing NA: {df_clean.shape}")
    
    # TIV normalization
    df_clean['TIV'] = df_clean[brain_cols].sum(axis=1)
    for col in brain_cols:
        df_clean[col] = df_clean[col] / df_clean['TIV']
    
    # Remove outliers
    original_size = len(df_clean)
    for col in brain_cols:
        mean, std = df_clean[col].mean(), df_clean[col].std()
        df_clean = df_clean[(df_clean[col] >= mean - 3*std) & 
                           (df_clean[col] <= mean + 3*std)]
    
    print(f"After outlier removal: {df_clean.shape} ({original_size - len(df_clean)} removed)")
    
    X = df_clean[brain_cols].values
    y = df_clean['Age'].values
    
    print(f"\nAge range: {y.min():.1f} - {y.max():.1f} years")
    print(f"Age mean ± std: {y.mean():.1f} ± {y.std():.1f} years")
    
    age_groups = pd.qcut(y, q=5, labels=False, duplicates='drop')
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=age_groups
    )
    
    age_groups_temp = pd.qcut(y_temp, q=4, labels=False, duplicates='drop')
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=age_groups_temp
    )
    
    y_train = pd.Series(y_train, index=range(len(y_train)))
    y_val = pd.Series(y_val, index=range(len(y_val)))
    y_test = pd.Series(y_test, index=range(len(y_test)))
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(y_train)} samples (Age: {y_train.min():.1f}-{y_train.max():.1f})")
    print(f"  Validation: {len(y_val)} samples (Age: {y_val.min():.1f}-{y_val.max():.1f})")
    print(f"  Test: {len(y_test)} samples (Age: {y_test.min():.1f}-{y_test.max():.1f})")
    
    # Train pipeline
    pipeline = AgeStratifiedPipeline(n_features=40)
    pipeline.brain_cols = brain_cols
    pipeline.fit(X_train, y_train, X_val, y_val)
    
    # Generate predictions
    y_train_pred = pipeline.predict(X_train, y_train)
    y_val_pred = pipeline.predict(X_val, y_val)
    y_test_pred = pipeline.predict(X_test, y_test)
    
    # Apply bias correction
    y_train_corrected = pipeline.bias_corrector.correct(y_train, y_train_pred)
    y_val_corrected = pipeline.bias_corrector.correct(y_val, y_val_pred)
    y_test_corrected = pipeline.bias_corrector.correct(y_test, y_test_pred)
    
    # Calculate metrics
    train_metrics = {
        'MAE': mean_absolute_error(y_train, y_train_corrected),
        'RMSE': np.sqrt(mean_squared_error(y_train, y_train_corrected)),
        'R²': r2_score(y_train, y_train_corrected),
        'Correlation': pearsonr(y_train, y_train_corrected)[0]
    }
    
    val_metrics = {
        'MAE': mean_absolute_error(y_val, y_val_corrected),
        'RMSE': np.sqrt(mean_squared_error(y_val, y_val_corrected)),
        'R²': r2_score(y_val, y_val_corrected),
        'Correlation': pearsonr(y_val, y_val_corrected)[0]
    }
    
    test_metrics = {
        'MAE': mean_absolute_error(y_test, y_test_corrected),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_test_corrected)),
        'R²': r2_score(y_test, y_test_corrected),
        'Correlation': pearsonr(y_test, y_test_corrected)[0]
    }
    
    print(f"\n{'='*70}")
    print(f"GENERATING VISUALIZATION PLOTS")
    print(f"{'='*70}\n")
    
    # Training set plots
    print("Training Set Plots:")
    plot_predictions(y_train, y_train_corrected, 
                    'Training Set: Predicted vs Actual Brain Age',
                    'train_predictions.png')
    plot_residuals(y_train, y_train_corrected,
                  'Training Set: Residual Analysis',
                  'train_residuals.png')
    plot_error_distribution(y_train, y_train_corrected,
                           'Training Set: Error Distribution',
                           'train_error_distribution.png')
    plot_age_stratified_performance(y_train, y_train_corrected,
                                   'Training Set: Age-Stratified Performance',
                                   'train_age_stratified.png')
    plot_bland_altman(y_train, y_train_corrected,
                     'Training Set: Bland-Altman Plot',
                     'train_bland_altman.png')
    
    # Validation set plots
    print("\nValidation Set Plots:")
    plot_predictions(y_val, y_val_corrected,
                    'Validation Set: Predicted vs Actual Brain Age',
                    'val_predictions.png')
    plot_residuals(y_val, y_val_corrected,
                  'Validation Set: Residual Analysis',
                  'val_residuals.png')
    plot_error_distribution(y_val, y_val_corrected,
                           'Validation Set: Error Distribution',
                           'val_error_distribution.png')
    plot_age_stratified_performance(y_val, y_val_corrected,
                                   'Validation Set: Age-Stratified Performance',
                                   'val_age_stratified.png')
    plot_bland_altman(y_val, y_val_corrected,
                     'Validation Set: Bland-Altman Plot',
                     'val_bland_altman.png')
    
    # Test set plots
    print("\nTest Set Plots:")
    plot_predictions(y_test, y_test_corrected,
                    'Test Set: Predicted vs Actual Brain Age',
                    'test_predictions.png')
    plot_residuals(y_test, y_test_corrected,
                  'Test Set: Residual Analysis',
                  'test_residuals.png')
    plot_error_distribution(y_test, y_test_corrected,
                           'Test Set: Error Distribution',
                           'test_error_distribution.png')
    plot_age_stratified_performance(y_test, y_test_corrected,
                                   'Test Set: Age-Stratified Performance',
                                   'test_age_stratified.png')
    plot_bland_altman(y_test, y_test_corrected,
                     'Test Set: Bland-Altman Plot',
                     'test_bland_altman.png')
    
    # Comparison summary
    print("\nComparison Plots:")
    plot_comparison_summary(train_metrics, val_metrics, test_metrics,
                          'performance_comparison.png')
    
    # Print final metrics
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS SUMMARY")
    print(f"{'='*70}\n")
    
    print("Training Set:")
    print(f"  MAE:  {train_metrics['MAE']:.3f} years")
    print(f"  RMSE: {train_metrics['RMSE']:.3f} years")
    print(f"  R²:   {train_metrics['R²']:.3f}")
    print(f"  r:    {train_metrics['Correlation']:.3f}")
    
    print("\nValidation Set:")
    print(f"  MAE:  {val_metrics['MAE']:.3f} years")
    print(f"  RMSE: {val_metrics['RMSE']:.3f} years")
    print(f"  R²:   {val_metrics['R²']:.3f}")
    print(f"  r:    {val_metrics['Correlation']:.3f}")
    
    print("\nTest Set:")
    print(f"  MAE:  {test_metrics['MAE']:.3f} years")
    print(f"  RMSE: {test_metrics['RMSE']:.3f} years")
    print(f"  R²:   {test_metrics['R²']:.3f}")
    print(f"  r:    {test_metrics['Correlation']:.3f}")
    
    within_2 = np.sum(np.abs(y_test_corrected - y_test) <= 2) / len(y_test) * 100
    within_3 = np.sum(np.abs(y_test_corrected - y_test) <= 3) / len(y_test) * 100
    within_5 = np.sum(np.abs(y_test_corrected - y_test) <= 5) / len(y_test) * 100
    
    print(f"\nTest Set Accuracy:")
    print(f"  Within ±2 years: {within_2:.1f}%")
    print(f"  Within ±3 years: {within_3:.1f}%")
    print(f"  Within ±5 years: {within_5:.1f}%")
    
    # Save pipeline
    print(f"\n{'='*70}")
    print("SAVING MODEL")
    print(f"{'='*70}")
    
    model_path = MODEL_DIR / "brain_age_pipeline.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    
    print(f"✓ Pipeline saved to: {model_path}")
    print(f"✓ All plots saved to: {PLOTS_DIR.absolute()}")
    print("\n✓ Training complete!")
    
    return pipeline


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = "Data/QC_removed_raw_sheet_valid_features.xlsx"
    
    print(f"Using data file: {filepath}")
    pipeline = main(filepath)