"""
Simple volume extraction from NIfTI files without requiring full SynthSeg models.
This extracts basic volumetric information that can be used for analysis.
"""
import os
import csv
import nibabel as nib
import numpy as np
from typing import Dict, Tuple, List

def extract_basic_volumes(nifti_path: str) -> Dict[str, float]:
    """
    Extract basic volumetric information from a NIfTI file.
    Returns volumes for different intensity-based regions.
    """
    try:
        # Load the NIfTI file
        img = nib.load(nifti_path)
        data = img.get_fdata()
        voxel_volume = np.prod(img.header.get_zooms()[:3])  # mm³ per voxel
        
        # Basic intensity-based segmentation
        # These thresholds are approximations for different brain tissues
        volumes = {}
        
        # Total brain volume (non-zero voxels)
        non_zero_mask = data > 0
        volumes['total_brain'] = np.sum(non_zero_mask) * voxel_volume
        
        # Get intensity statistics
        non_zero_data = data[non_zero_mask]
        if len(non_zero_data) > 0:
            mean_intensity = np.mean(non_zero_data)
            std_intensity = np.std(non_zero_data)
            
            # Approximate tissue segmentation based on intensity
            # CSF (low intensity)
            csf_mask = (data > 0) & (data < mean_intensity - 0.5 * std_intensity)
            volumes['csf'] = np.sum(csf_mask) * voxel_volume
            
            # Gray matter (medium intensity)
            gm_mask = (data >= mean_intensity - 0.5 * std_intensity) & (data < mean_intensity + 0.3 * std_intensity)
            volumes['gray_matter'] = np.sum(gm_mask) * voxel_volume
            
            # White matter (high intensity)
            wm_mask = data >= mean_intensity + 0.3 * std_intensity
            volumes['white_matter'] = np.sum(wm_mask) * voxel_volume
            
            # Brain regions approximation (very rough)
            # Left/Right hemisphere split (middle plane)
            mid_sagittal = data.shape[0] // 2
            left_hemisphere = data[:mid_sagittal, :, :] > 0
            right_hemisphere = data[mid_sagittal:, :, :] > 0
            volumes['left_hemisphere'] = np.sum(left_hemisphere) * voxel_volume
            volumes['right_hemisphere'] = np.sum(right_hemisphere) * voxel_volume
            
            # Additional approximated regions
            volumes['frontal_approximation'] = volumes['gray_matter'] * 0.25  # ~25% of GM
            volumes['parietal_approximation'] = volumes['gray_matter'] * 0.20  # ~20% of GM
            volumes['temporal_approximation'] = volumes['gray_matter'] * 0.22  # ~22% of GM
            volumes['occipital_approximation'] = volumes['gray_matter'] * 0.15  # ~15% of GM
            volumes['cerebellum_approximation'] = volumes['gray_matter'] * 0.18  # ~18% of GM
            
            # Subcortical structures (very rough estimates)
            volumes['caudate_approximation'] = volumes['gray_matter'] * 0.03
            volumes['putamen_approximation'] = volumes['gray_matter'] * 0.04
            volumes['pallidum_approximation'] = volumes['gray_matter'] * 0.015
            volumes['hippocampus_approximation'] = volumes['gray_matter'] * 0.025
            volumes['amygdala_approximation'] = volumes['gray_matter'] * 0.01
            volumes['thalamus_approximation'] = volumes['gray_matter'] * 0.045
            
            # Ventricular system
            volumes['lateral_ventricles_approximation'] = volumes['csf'] * 0.6
            volumes['third_ventricle_approximation'] = volumes['csf'] * 0.1
            volumes['fourth_ventricle_approximation'] = volumes['csf'] * 0.05
            
        else:
            # Fallback values if no valid data
            volumes = {key: 0.0 for key in [
                'total_brain', 'csf', 'gray_matter', 'white_matter',
                'left_hemisphere', 'right_hemisphere',
                'frontal_approximation', 'parietal_approximation', 'temporal_approximation',
                'occipital_approximation', 'cerebellum_approximation',
                'caudate_approximation', 'putamen_approximation', 'pallidum_approximation',
                'hippocampus_approximation', 'amygdala_approximation', 'thalamus_approximation',
                'lateral_ventricles_approximation', 'third_ventricle_approximation', 'fourth_ventricle_approximation'
            ]}
            
        return volumes
        
    except Exception as e:
        print(f"Error extracting volumes: {str(e)}")
        return {}

def save_volumes_to_csv(volumes: Dict[str, float], output_path: str, subject_id: str = "subject") -> bool:
    """
    Save volumes to CSV file in the format expected by the pipeline.
    """
    try:
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Header
            header = ['subject'] + list(volumes.keys())
            writer.writerow(header)
            
            # Data
            row = [subject_id] + list(volumes.values())
            writer.writerow(row)
            
        return True
    except Exception as e:
        print(f"Error saving CSV: {str(e)}")
        return False

def extract_and_save_volumes(nifti_path: str, csv_output_path: str, subject_id: str = None) -> Tuple[bool, Dict[str, float]]:
    """
    Extract volumes from NIfTI file and save to CSV.
    Returns (success, volumes_dict)
    """
    if subject_id is None:
        subject_id = os.path.splitext(os.path.basename(nifti_path))[0]
        if subject_id.endswith('.nii'):
            subject_id = subject_id[:-4]
    
    volumes = extract_basic_volumes(nifti_path)
    
    if volumes:
        success = save_volumes_to_csv(volumes, csv_output_path, subject_id)
        return success, volumes
    else:
        return False, {}

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python simple_volume_extractor.py <input_nifti> <output_csv> [subject_id]")
        sys.exit(1)
    
    nifti_path = sys.argv[1]
    csv_path = sys.argv[2]
    subject_id = sys.argv[3] if len(sys.argv) > 3 else None
    
    success, volumes = extract_and_save_volumes(nifti_path, csv_path, subject_id)
    
    if success:
        print(f"Successfully extracted volumes to {csv_path}")
        print(f"Total brain volume: {volumes.get('total_brain', 0):.2f} mm³")
    else:
        print("Failed to extract volumes")