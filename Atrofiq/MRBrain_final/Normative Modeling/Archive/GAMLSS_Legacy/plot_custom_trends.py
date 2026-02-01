import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

def plot_percentile_trends(region, gender, percentiles_folder, output_dir, dataset_name):
    """
    Plots lines for 25th, 50th, 75th, and 99th percentiles.
    Hue varies by percentile.
    """
    file_name = f"{gender}_{region}.xlsx"
    file_path = os.path.join(percentiles_folder, file_name)
    
    if not os.path.exists(file_path):
        print(f"Skipping {region} ({gender}): File not found in {dataset_name}")
        return

    df = pd.read_excel(file_path)
    if df.empty or len(df.columns) < 2:
        print(f"Skipping {region}: Empty or invalid")
        return

    # User requested 25, 50, 75, 100 (using 99th as proxy for 100th boundary)
    cols_to_plot = {
        '25th': {'color': '#4daf4a', 'label': '25%', 'style': '--'},  # Green
        '50th': {'color': '#377eb8', 'label': '50% (Median)', 'style': '-'}, # Blue, Bold
        '75th': {'color': '#ff7f00', 'label': '75%', 'style': '--'},  # Orange
        '99th': {'color': '#e41a1c', 'label': '100% (Boundary)', 'style': ':'}  # Red
    }
    
    plt.figure(figsize=(10, 6))
    
    for col, props in cols_to_plot.items():
        if col in df.columns:
            linewidth = 2.5 if col == '50th' else 1.5
            plt.plot(df['Age'], df[col], 
                     color=props['color'], 
                     linestyle=props['style'], 
                     linewidth=linewidth, 
                     label=props['label'])
            
    plt.title(f"Normative Trends ({dataset_name}): {region} ({gender})")
    plt.xlabel("Age (Years)")
    plt.ylabel("Volume (mm³, Mean TIV Adjusted)")
    plt.legend(title="Percentile Volume")
    plt.grid(True, alpha=0.3)
    plt.xlim(10, 100) # Enforce age range preference
    
    # Save
    safe_region = region.replace("ctx-lh-", "Left ").replace("ctx-rh-", "Right ").replace("_", " ").title()
    out_filename = f"trend_{dataset_name.lower()}_{gender}_{region}.png"
    out_path = os.path.join(output_dir, out_filename)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")

def main():
    base_dir = "/home/anirudh/Brainagepred/MRBrain/Normative Modeling"
    
    # Define datasets
    datasets = {
        "Pre-Contrast": os.path.join(base_dir, "Percentiles_TIV/Pre"),
        "Post-Contrast": os.path.join(base_dir, "Percentiles_TIV/Post")
    }
    
    output_dir = os.path.join(base_dir, "Comparison_Plots_Trends")
    os.makedirs(output_dir, exist_ok=True)
    
    # Key regions of interest
    regions = [
        "left_hippocampus", "right_hippocampus",
        "left_lateral_ventricle", "right_lateral_ventricle",
        "left_cerebral_cortex", "right_cerebral_cortex",
        "total_intracranial",
        "brain_stem" # Added for verification
    ]
    
    for name, folder in datasets.items():
        if not os.path.exists(folder):
            print(f"Folder not found: {folder}")
            continue
            
        print(f"Generating plots for {name}...")
        for gender in ["male", "female"]:
            for region in regions:
                 plot_percentile_trends(region, gender, folder, output_dir, name)

if __name__ == "__main__":
    main()
