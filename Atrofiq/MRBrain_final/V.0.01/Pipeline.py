#!/usr/bin/env python3
"""
Integrated MRI Processing Pipeline (MRBrain V.0.01)
---------------------------------------------------
Features:
1. Smart Preprocessing:
   - Auto-detects 2D vs 3D (slice count)
   - Auto-detects Pre vs Post contrast (DICOM tags)
   - Converts DICOM -> NIfTI
2. Generation (2D -> 3D):
   - Only for 2D Pre-Contrast inputs
3. Segmentation:
   - SynthSeg-based segmentation
4. Brain Age Prediction:
   - Predicts brain age and BAG
5. Normative Modeling:
   - Calculates percentiles for key regions

Usage:
    from Pipeline import MRPipeline
    pipeline = MRPipeline()
    results = pipeline.run(input_path, age, gender)
"""

import os
import sys
import argparse
import subprocess
import glob
import pandas as pd
import numpy as np
import nibabel as nib
import joblib
import pickle
import json
import shutil
import pydicom
from pathlib import Path
from datetime import datetime

# =============================================================================
# CONFIGURATION & PATHS
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))  # Back to /home/anirudh/Brainagepred

# 1. GENERATION
GEN_SCRIPT_DIR = os.path.join(PROJECT_ROOT, "MRBrain", "2D-3D")
GEN_CHECKPOINT = os.path.join(GEN_SCRIPT_DIR, "All_Checkpoints_Benchmark", "best_model.pth")

# 2. SEGMENTATION
SEG_SCRIPT = os.path.join(PROJECT_ROOT, "MRBrain", "Segmentation", "mri_pipeline_pytorch.py")

# 3. BRAIN AGE
AGE_SCRIPT_DIR_PRE = os.path.join(PROJECT_ROOT, "MRBrain", "BrainAge-Prediction_Pre")
AGE_MODEL_PRE = os.path.join(AGE_SCRIPT_DIR_PRE, "saved_models", "brain_age_pipeline.pkl")

AGE_SCRIPT_DIR_POST = os.path.join(PROJECT_ROOT, "MRBrain", "BrainAge-Prediction")
AGE_MODEL_POST = os.path.join(AGE_SCRIPT_DIR_POST, "saved_models", "brain_age_pipeline.pkl")

# 4. NORMATIVE
NORM_DIR = os.path.join(PROJECT_ROOT, "MRBrain", "Normative Modeling", "Percentile_Curves_Excel")

# PREPROCESSING THRESHOLDS
SLICE_THRESHOLD_3D = 50  # If slices > 50, treat as 3D

# =============================================================================
# UTILS: MONKEYPATCH FOR LEGACY PICKLE
# =============================================================================
import numpy.random
try:
    import numpy.random._pickle
    original_ctor = numpy.random._pickle.__randomstate_ctor
    def patched_ctor_wrapper(*args, **kwargs):
         if len(args) > 1:
             args = (args[0],)
         return original_ctor(*args, **kwargs)
    numpy.random._pickle.__randomstate_ctor = patched_ctor_wrapper
except Exception:
    pass

# =============================================================================
# PIPELINE CLASS
# =============================================================================

class MRPipeline:
    def __init__(self, device="cuda:0", seg_device="cuda:2"):
        self.device = device
        self.seg_device = seg_device
        self.age_pipeline = None
        self.current_model_mode = None # 'Pre' or 'Post'
        
        # Add necessary paths to sys.path
        if GEN_SCRIPT_DIR not in sys.path:
            sys.path.append(GEN_SCRIPT_DIR)
        # We might need both or swap? Typically simple append is fine if imports distinct.
        if AGE_SCRIPT_DIR_PRE not in sys.path:
            sys.path.append(AGE_SCRIPT_DIR_PRE)
        if AGE_SCRIPT_DIR_POST not in sys.path:
            sys.path.append(AGE_SCRIPT_DIR_POST)

        # Import Generation Model locally now that path is added
        try:
             # Make sure 2D-3D is in path
             if GEN_SCRIPT_DIR not in sys.path:
                 sys.path.append(GEN_SCRIPT_DIR)
             from Inference import MRIInference, preprocess_2d_volume
             self.MRIInference = MRIInference
             self.preprocess_2d_volume = preprocess_2d_volume
             print("✓ 2D-3D Generation Module Imported")
        except ImportError as e:
             print(f"✗ Failed to import 2D-3D Generation Module: {e}")
             self.MRIInference = None


    def load_models(self, contrast_mode="Pre"):
        """
        Lazy load brain age model for specific contrast mode.
        contrast_mode: 'Pre' or 'Post'
        """
        if self.age_pipeline is not None and self.current_model_mode == contrast_mode:
            return # Already loaded correct model
            
        target_model_path = AGE_MODEL_PRE if contrast_mode == "Pre" else AGE_MODEL_POST
        
        try:
            print(f"Loading Brain Age Model ({contrast_mode}) from: {target_model_path}")
            if os.path.exists(target_model_path):
                # Clean up old if exists?
                self.age_pipeline = None
                
                try:
                    self.age_pipeline = joblib.load(target_model_path)
                except:
                    with open(target_model_path, "rb") as f:
                        self.age_pipeline = pickle.load(f)
                
                self.current_model_mode = contrast_mode
                print(f"✓ Brain Age Model ({contrast_mode}) Loaded")
            else:
                print(f"✗ Brain Age Model not found at {target_model_path}")
        except Exception as e:
            print(f"✗ Failed to load Brain Age Model: {e}")

    def analyze_dicom_series(self, dicom_folder):
        """
        Analyze a DICOM series to determine:
        1. Modality (MR)
        2. 2D vs 3D (based on number of files/slices)
        3. Contrast (Pre vs Post)
        Returns: dict with 'is_3d', 'is_contrast', 'num_slices'
        """
        dcm_files = sorted(glob.glob(os.path.join(dicom_folder, "*.dcm")))
        if not dcm_files:
            # Try without extension
            dcm_files = sorted([f for f in glob.glob(os.path.join(dicom_folder, "*")) if os.path.isfile(f) and not f.endswith('.json')])
            
        if not dcm_files:
            return None

        try:
            # Scan a few files to be sure (sometimes localizers are 2D but next is 3D)
            # But usually a series folder is one series.
            dcm = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)
            
            # --- Check Contrast ---
            is_contrast = False
            contrast_reason = []
            
            # 1. Explicit Tag: (0018, 0010) Contrast/Bolus Agent
            agent = getattr(dcm, 'ContrastBolusAgent', '')
            if agent and str(agent).lower() not in ['none', 'no', '', 'nan']:
                is_contrast = True
                contrast_reason.append(f"Tag: {agent}")
                
            # 2. Series Description: (0008, 103E)
            desc = str(getattr(dcm, 'SeriesDescription', '')).lower()
            if any(x in desc for x in ['+c', 'gad', 'post', 'contrast', 't1_c', 't1+']):
                is_contrast = True
                contrast_reason.append(f"Desc: {desc}")
                
            # 3. Sequence Name: (0018, 0024)
            seq_name = str(getattr(dcm, 'SequenceName', '')).lower()
            if any(x in seq_name for x in ['+c', 'gad', 'post']):
                is_contrast = True
                contrast_reason.append(f"Seq: {seq_name}")
                
            # --- Check Dimensions ---
            num_slices = len(dcm_files)
            # Heuristic: If num_slices > threshold, it's 3D volume.
            # Localizers/Scouts are usually < 10 slices.
            # 2D T1s can be 20-30 slices. 3D T1s are usually > 100.
            is_3d = num_slices > SLICE_THRESHOLD_3D
            
            # Special check: If 2D but many slices (e.g. fMRI or DTI), it's 4D. 
            # But for T1, usually it's just slices.
            
            return {
                'is_3d': is_3d, 
                'is_contrast': is_contrast, 
                'num_slices': num_slices,
                'contrast_info': ", ".join(contrast_reason) if contrast_reason else "None"
            }
        except Exception as e:
            print(f"Error analyzing DICOM: {e}")
            return None

    def convert_dicom_to_nifti(self, dicom_dir, output_dir):
        """Convert DICOM to NIfTI using dcm2niix."""
        os.makedirs(output_dir, exist_ok=True)
        filename_base = "converted"
        cmd = ["dcm2niix", "-z", "y", "-f", filename_base, "-o", output_dir, dicom_dir]
        
        try:
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"dcm2niix failed: {e}")
            return None

        # Find largest file
        nifti_files = glob.glob(os.path.join(output_dir, f"{filename_base}*.nii.gz"))
        if not nifti_files:
            return None
        return max(nifti_files, key=os.path.getsize)

    def get_normative_data(self, region_name, sex, contrast_mode="Pre"):
        """
        Retrieve the full dataframe for a specific region's normative curve.
        contrast_mode: 'Pre' or 'Post'
        """
        sex_folder = "Male" if str(sex).lower().startswith('m') else "Female"
        # Select subfolder based on contrast
        sub_dir = "Post" if contrast_mode == "Post" else "Pre"
        
        norm_path = os.path.join(NORM_DIR, sub_dir, sex_folder)
        
        # Normalize region name to match file
        # Files are like "left_hippocampus.xlsx"
        # Input might be "Left-Hippocampus" etc.
        region_clean = region_name.lower().replace('-', '_').replace(' ', '_')
        
        # Try direct or clean
        chart_path = os.path.join(norm_path, f"{region_name}.xlsx")
        if not os.path.exists(chart_path):
            chart_path = os.path.join(norm_path, f"{region_clean}.xlsx")
            
        if os.path.exists(chart_path):
            return pd.read_excel(chart_path)
        return None

        if os.path.exists(chart_path):
            return pd.read_excel(chart_path)
        return None

    def ensure_canonical_orientation(self, nifti_path):
        """
        Check if NIfTI is RAS (canonical). If not, reorient and save.
        Returns path to reoriented file (or original if already RAS).
        """
        try:
            img = nib.load(nifti_path)
            # Check orientation
            axcodes = nib.aff2axcodes(img.affine)
            if axcodes != ('R', 'A', 'S'):
                print(f"Detected non-canonical orientation {axcodes}. Reorienting to RAS...")
                canonical_img = nib.as_closest_canonical(img)
                
                # Save as new file to avoid overwriting original input if needed, 
                # or better yet, save to a 'processed' location.
                # Here we will overwrite or save with suffix to be safe.
                # Actually, safe to overwrite internal processing files, but if it's user input...
                # Let's save to a temp or modified name in the same dir.
                
                base, ext = os.path.splitext(nifti_path)
                if ext == '.gz': 
                    base, ext2 = os.path.splitext(base)
                    ext = ext2 + ext
                
                out_name = f"{base}_RAS{ext}"
                nib.save(canonical_img, out_name)
                print(f"Reoriented volume saved to: {out_name}")
                return out_name
            return nifti_path
        except Exception as e:
            print(f"Warning: Failed to check/reorient NIfTI: {e}")
            return nifti_path

    def analyze_nifti_header(self, nifti_path):
        """
        Analyze NIfTI header to guess if 2D or 3D.
        Try to find JSON sidecar or use filename heuristics for Contrast.
        """
        try:
            nifti_path = os.path.abspath(nifti_path)
            img = nib.load(nifti_path)
            shape = img.shape
            
            is_3d = False
            if len(shape) >= 3 and shape[2] > SLICE_THRESHOLD_3D:
                is_3d = True
            
            # Check Contrast
            is_contrast = False
            contrast_reason = []
            
            # 1. JSON Sidecar (BIDS or dcm2niix)
            # Look for .json with same basename
            json_path = nifti_path.replace('.nii.gz', '.json').replace('.nii', '.json')
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        meta = json.load(f)
                        # Check keys
                        keys_to_check = ['ContrastBolusAgent', 'ContrastBolusIngredient']
                        for k in keys_to_check:
                            if k in meta and meta[k]:
                                val = str(meta[k]).lower()
                                if val not in ['none', 'no', '']:
                                    is_contrast = True
                                    contrast_reason.append(f"JSON {k}: {val}")
                        
                        # Check SeriesDescription in JSON
                        if 'SeriesDescription' in meta:
                            desc = str(meta['SeriesDescription']).lower()
                            if any(x in desc for x in ['+c', 'gad', 'post', 'contrast', 't1_c']):
                                is_contrast = True
                                contrast_reason.append(f"JSON Desc: {desc}")
                except Exception as e:
                    print(f"Warning: Failed to parse sidecar JSON: {e}")
            
            # 2. Filename Heuristics (if no JSON or JSON inconclusive)
            filename = os.path.basename(nifti_path).lower()
            if any(x in filename for x in ['+c', 'gad', 'post', 'contrast']):
                is_contrast = True
                contrast_reason.append(f"Filename: {filename}")

            return {
                'is_3d': is_3d, 
                'is_contrast': is_contrast,
                'shape': shape,
                'contrast_info': ", ".join(contrast_reason) if contrast_reason else "None"
            }
        except Exception as e:
            print(f"Error analyzing NIfTI: {e}")
            return None

    def run(self, input_path, age, gender, output_dir=None, force_3d=False, force_contrast=False):
        """
        Run the full pipeline.
        input_path: Path to DICOM folder or NIfTI file.
        age: float
        gender: "Male" or "Female" (or "M"/"F")
        force_3d: Boolean override to treat input as 3D.
        force_contrast: Boolean override to treat input as Post-Contrast.
        """
        self.load_models()
        
        # Prepare output directory
        if output_dir is None:
            output_dir = os.path.join("results", datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(output_dir, exist_ok=True)
        
        results = {
            'patient_id': 'Unknown',
            'chronological_age': age,
            'gender': gender,
            'status': 'Started',
            'warnings': []
        }
        
        print(f"\n=== Starting Pipeline for Subject (Age: {age}, Sex: {gender}) ===")
        
        # ---------------------------------------------------------------------
        # STEP 1: PREPROCESSING & CLASSIFICATION
        # ---------------------------------------------------------------------
        print("\n--- Step 1: Preprocessing & Routing ---")
        
        input_path = os.path.abspath(input_path)
        processing_nifti = None
        is_dicom = os.path.isdir(input_path)
        
        is_3d = True  # Default
        is_contrast = False # Default
        
        if is_dicom:
            results['patient_id'] = os.path.basename(input_path)
            print(f"Input detected as DICOM folder: {input_path}")
            
            # 1. Analyze
            analysis = self.analyze_dicom_series(input_path)
            if analysis:
                is_3d = analysis['is_3d']
                is_contrast = analysis['is_contrast']
                print(f"DICOM Analysis: 3D={is_3d} (Slices={analysis['num_slices']}), Contrast={is_contrast} ({analysis['contrast_info']})")
            
            # 2. Convert
            print("Converting DICOM to NIfTI...")
            converted = self.convert_dicom_to_nifti(input_path, output_dir)
            if not converted:
                print("Error: DICOM conversion failed.")
                return None
            processing_nifti = converted
            
        else:
            # File input
            results['patient_id'] = os.path.basename(input_path).replace('.nii.gz', '').replace('.nii', '')
            print(f"Input detected as NIfTI file: {input_path}")
            processing_nifti = input_path
            
            # Analyze NIfTI
            analysis = self.analyze_nifti_header(processing_nifti)
            if analysis:
                is_3d = analysis['is_3d']
                is_contrast = analysis.get('is_contrast', False) # Updated analyze_nifti_header returns this
                
                print(f"NIfTI Analysis: 3D={is_3d}, Contrast={is_contrast} ({analysis.get('contrast_info', 'None')})")
        
        # APPLY OVERRIDES
        if force_3d:
            print("(!) MANUAL OVERRIDE: Forcing 3D Mode")
            is_3d = True
        
        if force_contrast:
            print("(!) MANUAL OVERRIDE: Forcing Post-Contrast Mode")
            is_contrast = True
        
        results['input_nifti'] = processing_nifti
        results['is_3d'] = is_3d
        results['is_contrast'] = is_contrast

        # Set Mode and Load Model
        contrast_mode = "Post" if is_contrast else "Pre"
        self.load_models(contrast_mode)
        
        # ---------------------------------------------------------------------
        # STEP 2: ROUTING & GENERATION
        # ---------------------------------------------------------------------
        
        final_volume_path = processing_nifti
        
        # Rule 1: Post-Contrast -> Skip Generation, maybe warn?
        if is_contrast:
            print("(!) DETECTED CONTRAST: Skipping Generation. Proceeding directly to Segmentation.")
            results['warnings'].append("Contrast detected (or forced). Generation skipped.")
            
        # Rule 2: 2D Pre-Contrast -> Run Generator (2D-to-3D)
        elif not is_3d:
            print("(!) DETECTED 2D PRE-CONTRAST: Running 2D-to-3D Generation...")
            
            gen_out_dir = os.path.join(output_dir, "01_Generated")
            os.makedirs(gen_out_dir, exist_ok=True)
            
            # Direct Generation Execution
            print("(!) DETECTED 2D PRE-CONTRAST: Running 2D-to-3D Generation (Direct Mode)...")
            
            gen_out_dir = os.path.join(output_dir, "01_Generated")
            os.makedirs(gen_out_dir, exist_ok=True)
            
            try:
                if self.MRIInference is None:
                    raise ImportError("Generation module not loaded properly")
                
                # 1. Preprocess
                print("  Preprocessing...")
                lr_vol_np, lr_info = self.preprocess_2d_volume(processing_nifti, target_slices=25, target_size=256)
                
                # 2. Inference
                print("  Inference...")
                inferencer = self.MRIInference(GEN_CHECKPOINT, device=self.device)
                gen_vol_np = inferencer.generate_3d_volume(lr_vol_np)
                
                # 3. Save
                base_name = os.path.basename(processing_nifti).replace('.nii.gz', '').replace('.nii', '')
                out_name = f"{base_name}_generated_3d.nii.gz"
                output_path_gen = os.path.join(gen_out_dir, out_name)
                
                inferencer.save_nifti(gen_vol_np, output_path_gen, lr_info)
                
                final_volume_path = output_path_gen
                print(f"Generation Successful: {final_volume_path}")
                
            except Exception as e:
                print(f"Generation failed with error: {e}")
                results['warnings'].append(f"Generation error: {e}")
                import traceback
                traceback.print_exc()


        else:
            print("(!) DETECTED 3D PRE-CONTRAST: proceeding directly to Segmentation.")

        # --- Reorient to Canonical (RAS) before Segmentation ---
        # This fixes visualization issues (axial vs sagittal) and ensures SynthSeg consistency
        final_volume_path = self.ensure_canonical_orientation(final_volume_path)
        results['processed_volume'] = final_volume_path

        # ---------------------------------------------------------------------
        # STEP 3: SEGMENTATION
        # ---------------------------------------------------------------------
        print("\n--- Step 3: Segmentation ---")
        
        seg_output_dir = os.path.join(output_dir, "02_Segmentation")
        os.makedirs(seg_output_dir, exist_ok=True)
        volumes_csv = os.path.join(seg_output_dir, "volumes.csv")
        seg_nifti_dir = os.path.join(seg_output_dir, "Segmentation")
        
        # Check if already segmented (for restart capability)
        if os.path.exists(volumes_csv):
            print("Using existing volumes.csv")
        else:
            # Run Segmentation
            cmd_seg = [
                sys.executable, SEG_SCRIPT,
                "--i", final_volume_path,
                "--o", seg_nifti_dir,
                "--vol", volumes_csv,
                "--parc",
                "--device", self.seg_device
            ]
            print(f"Running Segmentation (this may take a while)...")
            try:
                subprocess.check_call(cmd_seg)
            except Exception as e:
                print(f"Segmentation failed: {e}")
                return results

        results['volumes_csv'] = volumes_csv
        
        # Find segmented output file
        # Usually in seg_nifti_dir/<name>_synthseg.nii.gz
        seg_files = glob.glob(os.path.join(seg_nifti_dir, "*synthseg*.nii.gz")) + \
                    glob.glob(os.path.join(seg_nifti_dir, "*seg*.nii.gz"))
                    
        if seg_files:
            results['segmented_nifti'] = seg_files[0]
        
        # ---------------------------------------------------------------------
        # STEP 4: BRAIN AGE PREDICTION
        # ---------------------------------------------------------------------
        print("\n--- Step 4: Prediction ---")
        
        if os.path.exists(volumes_csv) and self.age_pipeline:
            try:
                # 1. Prepare Features
                df_vol = pd.read_csv(volumes_csv)
                
                # Check for required columns
                brain_cols = self.age_pipeline.brain_cols
                
                # Normalize sex: Pipeline usually expects 0 for Female, 1 for Male (or vice versa, check underlying model)
                # Based on previous scripts: Sex_F=0, Sex_M=1 seems standard for this codebase
                sex_val = 0 if str(gender).lower().startswith('f') else 1
                
                # Map features
                input_row = {}
                for col in brain_cols:
                    if col in df_vol.columns:
                        input_row[col] = df_vol.iloc[0][col]
                    else:
                        input_row[col] = 0.0 # Missing
                
                # Create Feature Matrix X
                X = np.array([list(input_row.values())])
                
                # TIV Normalization (Sum of all brain volumes)
                tiv = np.sum(X, axis=1, keepdims=True)
                tiv[tiv == 0] = 1e-8
                X_norm = X / tiv
                
                # Predict
                if hasattr(self.age_pipeline, 'predict_corrected'):
                    pred_age = self.age_pipeline.predict_corrected(X_norm, np.array([float(age)]))[0]
                else:
                    pred_age = self.age_pipeline.predict(X_norm)[0]
                
                bag = pred_age - float(age)
                
                results['predicted_age'] = round(pred_age, 2)
                results['bag'] = round(bag, 2)
                
                print(f"Predicted Age: {pred_age:.2f}")
                print(f"BAG: {bag:.2f}")
                
            except Exception as e:
                print(f"Prediction failed: {e}")
                import traceback
                traceback.print_exc()

        # ---------------------------------------------------------------------
        # STEP 5: NORMATIVE MODELING
        # ---------------------------------------------------------------------
        print("\n--- Step 5: Normative Analysis ---")
        
        if os.path.exists(volumes_csv):
             norm_results = self.get_normative_percentiles(volumes_csv, float(age), gender, contrast_mode=contrast_mode)
             if norm_results:
                 results['normative'] = norm_results
                 # Generate CSV for Normative
                 norm_df = pd.DataFrame(norm_results)
                 norm_csv_path = os.path.join(output_dir, "normative_analysis.csv")
                 norm_df.to_csv(norm_csv_path, index=False)
                 print(f"Normative analysis saved to {norm_csv_path}")

        # Final Save
        final_results_csv = os.path.join(output_dir, f"{results['patient_id']}_brain_age_results.csv")
        
        # Create a simple DF for the main result
        simple_res = {
            'PatientID': results['patient_id'],
            'Chronological_Age': results['chronological_age'],
            'Gender': results['gender'],
            'Predicted_Brain_Age': results.get('predicted_age', 'N/A'),
            'Brain_Age_Gap': results.get('bag', 'N/A'),
            'Modality': '2D' if not is_3d else '3D',
            'Contrast': 'Yes' if is_contrast else 'No'
        }
        pd.DataFrame([simple_res]).to_csv(final_results_csv, index=False)
        
        print("\n=== Pipeline Completed ===")
        return results

    def get_normative_percentiles(self, volumes_csv, age, sex, contrast_mode="Pre"):
        """
        Calculate percentiles for key regions.
        """
        try:
            df_vol = pd.read_csv(volumes_csv)
            sex_folder = "Male" if str(sex).lower().startswith('m') else "Female"
            # Select subfolder based on contrast
            sub_dir = "Post" if contrast_mode == "Post" else "Pre"
            norm_path = os.path.join(NORM_DIR, sub_dir, sex_folder)
            
            if not os.path.exists(norm_path):
                print(f"Normative charts not found at {norm_path}")
                return None
            
            # Map standard synthseg names to our normative file names
            # Normative files are like "left_hippocampus.xlsx"
            # SynthSeg columns are like "left_hippocampus" (usually)
            
            # Interesting regions to report
            key_regions = [
                'left_hippocampus', 'right_hippocampus', 
                'left_lateral_ventricle', 'right_lateral_ventricle',
                'left_thalamus', 'right_thalamus',
                'left_amygdala', 'right_amygdala',
                'brain_stem', 
                'left_cerebral_cortex', 'right_cerebral_cortex' # Total GM essentially
            ]
            
            results = []
            
            # We need to map synthseg column names to file names
            # SynthSeg typically uses standard names, but let's check exact matches or normalized
            # Normalize volumes (Divide by TIV * 1000 or similar? Or just raw?)
            # Usually Normative models are built on normalized or raw data.
            # ASSUMPTION: The normative models provided are likely on Normalized data (Volume / TIV) or specific metrics.
            # Wait, the user script previously just did a lookup. Let's look at the excel files again.
            # The excel files have 'Age' as index (rows) and 'Centile_1' to 'Centile_99' as columns.
            # This implies the VALUE in the cell is the VOLUME.
            # So we look up the row for the Age, and find where our Subject's Volume falls in the percentiles.
            
            for region in key_regions:
                # 1. Load Chart
                chart_path = os.path.join(norm_path, f"{region}.xlsx")
                if not os.path.exists(chart_path):
                    continue
                    
                chart = pd.read_excel(chart_path)
                
                # 2. Find matching column in volumes
                # Try exact match first
                vol = None
                
                # Common mapping variations between SynthSeg and typical filenames
                # E.g. "Left-Hippocampus" vs "left_hippocampus"
                # We'll normalize both to compare
                
                # Find the column in df_vol that matches 'region' normalized
                for col in df_vol.columns:
                    norm_col = col.lower().replace('-', '_').replace(' ', '_')
                    if norm_col == region:
                        # Normalize by TIV
                        # Check if 'total intracranial' exists
                        tiv_col = [c for c in df_vol.columns if 'total intracranial' in c.lower()]
                        if tiv_col:
                            tiv = df_vol.iloc[0][tiv_col[0]]
                            if tiv > 0:
                                vol = df_vol.iloc[0][col] / tiv
                            else:
                                vol = df_vol.iloc[0][col] # Fallback
                        else:
                            vol = df_vol.iloc[0][col] # Fallback
                        break
                
                if vol is None:
                    continue
                
                # 3. Lookup Age row
                # Chart rows are ages. We assume 'Age' column exists (it does from previous `head` check)
                # Find precise or closest age row
                age_int = int(round(age))
                row = chart[chart['Age'] == age_int]
                
                if row.empty:
                    # User out of range? Clamp?
                    if age_int < chart['Age'].min():
                        row = chart[chart['Age'] == chart['Age'].min()]
                    elif age_int > chart['Age'].max():
                        row = chart[chart['Age'] == chart['Age'].max()]
                
                if row.empty:
                    continue
                    
                # 4. Find Percentile
                # The row values (Centile_1 ... Centile_99) are volumes
                # We want to find p such that Centile_p approx vol
                
                # Extract centile values
                centile_cols = [c for c in chart.columns if c.startswith('Centile_')]
                # Sort just in case
                centile_cols.sort(key=lambda x: int(x.split('_')[1]))
                
                centile_values = row.iloc[0][centile_cols].values
                centile_ints = [int(c.split('_')[1]) for c in centile_cols]
                
                # Interpolate
                percentile = np.interp(vol, centile_values, centile_ints)
                
                results.append({
                    'Region': region,
                    'Volume': vol,
                    'Percentile': round(percentile, 1)
                })
                
            return results
            
        except Exception as e:
            print(f"Normative analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    parser = argparse.ArgumentParser(description="Integrated MRBrain Pipeline")
    parser.add_argument("--input", required=True, help="Input DICOM directory or NIfTI file")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--age", type=float, required=True, help="Chronological Age")
    parser.add_argument("--gender", required=True, help="Gender (M/F)")
    
    args = parser.parse_args()
    
    pipeline = MRPipeline()
    pipeline.run(args.input, args.age, args.gender, args.output)

if __name__ == "__main__":
    main()