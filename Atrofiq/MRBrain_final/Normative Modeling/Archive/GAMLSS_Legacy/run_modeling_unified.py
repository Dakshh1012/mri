import os
import subprocess
import shutil

def run_gamlss(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Call the R script
    # We pass input_dir and output_dir as arguments
    r_script = "/home/anirudh/Brainagepred/MRBrain/Normative Modeling/normative_modeling.R"
    
    print(f"\n--- Running R-based Modeling for: {os.path.basename(input_dir)} ---")
    
    try:
        # We need to ensure the R_libs path is passed or handled in the R script
        # Stream output directly to allow viewing progress
        result = subprocess.run(
            ["Rscript", r_script, input_dir, output_dir],
            check=False # Don't raise exception, just finish
        )
        if result.returncode != 0:
            print(f"R script finished with return code {result.returncode}")
            
    except Exception as e:
        print(f"Failed to run R script: {e}")

def main():
    base_dir = "/home/anirudh/Brainagepred/MRBrain/Normative Modeling"
    
    # Unified (Pre + Post, Age > 21)
    input_unified = os.path.join(base_dir, "Input_Data_Unified")
    output_unified = os.path.join(base_dir, "Percentiles_Unified")
    
    # We can clean the output dir to ensure a fresh start
    if os.path.exists(output_unified):
        shutil.rmtree(output_unified)
    
    if os.path.exists(input_unified):
        # We need to run prepare_unified_data.py but it is run manually or we assume it's done
        # Let's just run the modeling part
        run_gamlss(input_unified, output_unified)
    else:
        print(f"Input directory not found: {input_unified}")
        print("Please run prepare_unified_data.py first.")

if __name__ == "__main__":
    main()
