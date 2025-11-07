#!/usr/bin/env python3
"""
Preprocessing pipeline for building Alignment_LR_HR_slices.

- Input:
    --input_2d_dir : directory with 2D anisotropic MRIs (≈25 slices per subject)
    --input_3d_dir : directory with 3D isotropic MRIs (≈250+ slices per subject)
- Output:
    A single directory with paired slices:
        SUBJECT_slice{idx}_lr.npy   (from 2D scan)
        SUBJECT_slice{idx}_hr.npy   (from 3D scan)

Note: Counts are not the same (25 vs 250+). That’s intentional.
The dataset class (MRISliceDatasetProgressive) aligns LR into sparse volumes
and compares against HR full volumes.
"""

import os
import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

# ------------------------
# Utils
# ------------------------
def normalize_minmax(img):
    """Normalize image to [0,1]."""
    vmin, vmax = np.min(img), np.max(img)
    if vmax > vmin:
        return (img - vmin) / (vmax - vmin)
    else:
        return np.zeros_like(img)

def resize_slice(slice_2d, target_shape=(256, 256)):
    """Resize 2D slice to target shape using bilinear interpolation."""
    zoom_factors = (target_shape[0] / slice_2d.shape[0],
                    target_shape[1] / slice_2d.shape[1])
    return zoom(slice_2d, zoom_factors, order=1)

def save_slices_from_volume(nifti_path, output_dir, tag, target_shape=(256, 256)):
    """
    Convert NIfTI volume into .npy slices.
    tag = 'lr' or 'hr'
    """
    nii = nib.load(nifti_path)
    vol = nii.get_fdata()

    subj_id = os.path.basename(nifti_path).replace(".nii.gz", "").replace(".nii", "")
    print(f"[{tag.upper()}] Processing {subj_id} with shape {vol.shape}")

    if vol.ndim != 3:
        raise ValueError(f"{nifti_path}: Expected 3D NIfTI, got {vol.shape}")

    D = vol.shape[2]
    for idx in range(D):
        slice_2d = vol[:, :, idx]
        slice_2d = resize_slice(slice_2d, target_shape)
        slice_2d = normalize_minmax(slice_2d).astype(np.float32)

        fname = f"{subj_id}.nii_slice{idx}_{tag}.npy"
        np.save(os.path.join(output_dir, fname), slice_2d)

    print(f"  -> Saved {D} {tag.upper()} slices for {subj_id}")

# ------------------------
# Main
# ------------------------
def main():

    parser = argparse.ArgumentParser(description="Build Alignment_LR_HR_slices dataset")
    parser.add_argument("--mode", type=str, choices=["both", "lr", "hr"], default="both",
                        help="Which slices to generate: both, lr, or hr")
    parser.add_argument("--input_2d_dir", type=str,
                        help="Directory with 2D MRIs (low-res, ~25 slices)")
    parser.add_argument("--input_3d_dir", type=str,
                        help="Directory with 3D MRIs (high-res, ~250 slices)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for LR/HR slices")
    parser.add_argument("--target_shape", type=str, default="256,256",
                        help="In-plane shape, e.g. 256,256")
    args = parser.parse_args()

    # Validate required inputs based on mode
    if args.mode in ("both", "lr") and not args.input_2d_dir:
        parser.error("--input_2d_dir is required when mode is 'both' or 'lr'")
    if args.mode in ("both", "hr") and not args.input_3d_dir:
        parser.error("--input_3d_dir is required when mode is 'both' or 'hr'")

    os.makedirs(args.output_dir, exist_ok=True)
    target_shape = tuple(map(int, args.target_shape.split(",")))

    if args.mode in ("both", "lr"):
        for f in sorted(os.listdir(args.input_2d_dir)):
            if f.endswith(".nii") or f.endswith(".nii.gz"):
                save_slices_from_volume(os.path.join(args.input_2d_dir, f),
                                        args.output_dir, tag="lr", target_shape=target_shape)

    if args.mode in ("both", "hr"):
        for f in sorted(os.listdir(args.input_3d_dir)):
            if f.endswith(".nii") or f.endswith(".nii.gz"):
                save_slices_from_volume(os.path.join(args.input_3d_dir, f),
                                        args.output_dir, tag="hr", target_shape=target_shape)

    print("\n✅ Preprocessing complete!")
    print("Slices saved to:", args.output_dir)

if __name__ == "__main__":
    main()
