import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_comparison_unified(region, gender, unified_folder, raw_unified_folder, output_dir):
    file_name = f"{gender}_{region}.xlsx"
    unified_file = os.path.join(unified_folder, file_name)
    raw_file = os.path.join(raw_unified_folder, file_name)
    
    if not os.path.exists(unified_file):
        print(f"Missing unified file: {unified_file}")
        return

    df_unified = pd.read_excel(unified_file)
    
    plt.figure(figsize=(10, 6))
    
    # Plot Raw Data
    if os.path.exists(raw_file):
        df_raw = pd.read_excel(raw_file)
        plt.scatter(df_raw['Age'], df_raw['Volume'], color='gray', alpha=0.3, s=10, label='Raw Data (Adjusted, >21)')
    
    # Plot Unified Curve
    plt.plot(df_unified['Age'], df_unified['50th'], label='Unified Model (Median)', color='black', linewidth=2.5)
    plt.fill_between(df_unified['Age'], df_unified['5th'], df_unified['95th'], color='black', alpha=0.15, label='5th-95th Percentile')
    
    plt.title(f"Adult Normative Curve (Age > 21): {region} ({gender})")
    plt.xlabel("Age")
    plt.ylabel("Volume (mm³)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"unified_adult_{gender}_{region}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot to {out_path}")

def main():
    unified_folder = "/home/anirudh/Brainagepred/MRBrain/Normative Modeling/Percentiles_Unified"
    raw_unified_folder = "/home/anirudh/Brainagepred/MRBrain/Normative Modeling/Input_Data_Unified"
    output_dir = "/home/anirudh/Brainagepred/MRBrain/Normative Modeling/Comparison_Plots_Unified"
    
    # Compare key regions
    key_regions = ['left_hippocampus', 'right_hippocampus', 'left_lateral_ventricle', 'right_lateral_ventricle', 'left_cerebral_cortex', 'right_cerebral_cortex', 'total_intracranial']
    
    # Attempt to find actual regions from folder
    if os.path.exists(unified_folder):
        files = [f for f in os.listdir(unified_folder) if f.endswith('.xlsx')]
        # Extract unique regions (remove gender prefix)
        regions = set()
        for f in files:
            parts = f.replace('.xlsx', '').split('_', 1)
            if len(parts) == 2:
                regions.add(parts[1])
        
        # Plot for all found keys if in interesting list or just plot all? 
        # Let's plot the key ones + specific requests
        interesting = ['left_hippocampus', 'right_hippocampus', 'left_lateral_ventricle', 'left_cerebral_cortex']
        for region in interesting:
            if region in regions:
                for gender in ['male', 'female']:
                     plot_comparison_unified(region, gender, unified_folder, raw_unified_folder, output_dir)
            else:
                 # Try plotting 'ctx-lh-hippocampus' if standard naming differs
                 pass

if __name__ == "__main__":
    main()
