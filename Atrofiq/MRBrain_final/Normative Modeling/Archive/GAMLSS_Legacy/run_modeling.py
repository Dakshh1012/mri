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
    input_base = os.path.join(base_dir, "Input_Data_TIV")
    output_base = os.path.join(base_dir, "Percentiles_TIV")
    
    # We can clean the output dir to ensure a fresh start
    # if os.path.exists(output_base):
    #     shutil.rmtree(output_base)
    
    # Pre-contrast
    pre_input = os.path.join(input_base, "Pre")
    pre_output = os.path.join(output_base, "Pre")
    if os.path.exists(pre_input):
        run_gamlss(pre_input, pre_output)
    
    # Post-contrast
    post_input = os.path.join(input_base, "Post")
    post_output = os.path.join(output_base, "Post")
    if os.path.exists(post_input):
        run_gamlss(post_input, post_output)

if __name__ == "__main__":
    main()
