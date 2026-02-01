#!/usr/bin/env python3
"""
Plot normative curves from results.json with chronological age marker.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path


def load_results(results_file: str) -> dict:
    """Load the results JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)


def plot_region_curves(region_data: dict, participant_age: int, region_name: str):
    """Plot percentile curves for a single brain region."""
    ages = region_data['ages']
    curves = region_data['percentile_curves']
    
    plt.figure(figsize=(10, 6))
    
    # Plot percentile curves
    colors = {'1': '#ff0000', '5': '#ff6600', '10': '#ff9900', 
             '25': '#ffcc00', '50': '#00cc00', '75': '#0099ff', 
             '90': '#0066ff', '95': '#0033ff', '99': '#0000ff'}
    
    for percentile, values in curves.items():
        color = colors.get(percentile, 'gray')
        label = f'{percentile}th percentile'
        plt.plot(ages, values, color=color, linewidth=2, label=label)
    
    # Plot chronological age marker
    y_min, y_max = plt.ylim()
    plt.axvline(x=participant_age, color='red', linestyle='--', linewidth=3, 
                label=f'Chronological Age ({participant_age})')
    
    # Add a red dot at the age line (middle of y-axis)
    y_mid = (y_min + y_max) / 2
    plt.plot(participant_age, y_mid, 'ro', markersize=10, 
             label=f'Age Marker')
    
    plt.xlabel('Age (years)')
    plt.ylabel('Volume')
    plt.title(f'Normative Curves for {region_name.replace("_", " ").title()}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_all_regions(results: dict, output_dir: str = None):
    """Plot curves for all regions in the results."""
    participant_info = results['participant_info']
    participant_age = participant_info['chronological_age']
    participant_id = participant_info['participant_id']
    
    region_analyses = results['region_analyses']
    
    if not region_analyses:
        print("No region analyses found in results!")
        return
    
    # Create output directory if specified
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Plotting curves for participant {participant_id} (age {participant_age})")
    print(f"Found {len(region_analyses)} regions to plot")
    
    for region_name, region_data in region_analyses.items():
        print(f"Plotting {region_name}...")
        
        plot_region_curves(region_data, participant_age, region_name)
        
        if output_dir:
            # Save individual plots
            filename = f"{participant_id}_{region_name}_curves.png"
            filepath = Path(output_dir) / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        plt.show()


def create_summary_plot(results: dict, output_dir: str = None):
    """Create a summary plot with multiple regions in subplots."""
    participant_info = results['participant_info']
    participant_age = participant_info['chronological_age']
    participant_id = participant_info['participant_id']
    
    region_analyses = results['region_analyses']
    
    if not region_analyses:
        print("No region analyses found in results!")
        return
    
    n_regions = len(region_analyses)
    n_cols = 3
    n_rows = (n_regions + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    fig.suptitle(f'Normative Curves Summary - {participant_id} (Age {participant_age})', 
                fontsize=16, fontweight='bold')
    
    # Flatten axes array for easier indexing
    if n_regions == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    colors = {'50': '#00cc00', '25': '#ffcc00', '75': '#0099ff', 
             '10': '#ff9900', '90': '#0066ff'}
    
    for idx, (region_name, region_data) in enumerate(region_analyses.items()):
        ax = axes[idx]
        
        ages = region_data['ages']
        curves = region_data['percentile_curves']
        
        # Plot key percentiles only for summary
        for percentile in ['10', '25', '50', '75', '90']:
            if percentile in curves:
                color = colors.get(percentile, 'gray')
                ax.plot(ages, curves[percentile], color=color, 
                       linewidth=2, label=f'{percentile}th')
        
        # Plot age marker
        y_min, y_max = ax.get_ylim() if ax.get_ylim() != (0, 1) else (0, 100)
        ax.axvline(x=participant_age, color='red', linestyle='--', linewidth=2)
        y_mid = (y_min + y_max) / 2
        ax.plot(participant_age, y_mid, 'ro', markersize=8)
        
        ax.set_title(region_name.replace("_", " ").title(), fontsize=12)
        ax.set_xlabel('Age (years)')
        ax.set_ylabel('Volume')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    # Hide extra subplots
    for idx in range(n_regions, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if output_dir:
        filename = f"{participant_id}_summary_curves.png"
        filepath = Path(output_dir) / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved summary plot: {filepath}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot normative curves from results.json")
    parser.add_argument("--results", "-r", required=True, 
                       help="Path to results.json file")
    parser.add_argument("--output-dir", "-o", 
                       help="Output directory for saving plots")
    parser.add_argument("--summary-only", "-s", action="store_true",
                       help="Only create summary plot (not individual plots)")
    
    args = parser.parse_args()
    
    try:
        # Load results
        results = load_results(args.results)
        
        if args.summary_only:
            create_summary_plot(results, args.output_dir)
        else:
            plot_all_regions(results, args.output_dir)
            create_summary_plot(results, args.output_dir)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())