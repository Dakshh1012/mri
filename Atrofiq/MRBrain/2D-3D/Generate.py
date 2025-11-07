#!/usr/bin/env python3
"""
Generate.py
-----------
Generate a full 3D MRI volume from LR 2D slices using the trained Generator3D.
This mimics the MRISliceDatasetProgressive preprocessing steps.
"""

import os
import glob
import argparse
import numpy as np
import torch
import nibabel as nib
from scipy.ndimage import zoom

from Main import Generator3D  # model
from Eval import ProgressiveModelEvaluator  # for generate_full_volume


# -------------------------------
# Helpers
# -------------------------------
def map_indices(slice_idx_list, target_depth):
    """Map original slice indices to target depth range [0, target_depth-1]."""
    max_idx = max(slice_idx_list)
    return [
        min(int((i * (target_depth - 1)) / max_idx), target_depth - 1)
        for i in slice_idx_list
    ]


def add_linear_interpolation(volume, known_indices):
    """Fill missing slices with linear interpolation (as in dataset)."""
    known_indices = sorted(set(known_indices))
    for a, b in zip(known_indices[:-1], known_indices[1:]):
        if b - a > 1:
            s0, s1 = volume[a], volume[b]
            for j in range(a + 1, b):
                alpha = (j - a) / (b - a)
                volume[j] = ((1 - alpha) * s0 + alpha * s1) * 0.5
    return volume


def normalize_volume(volume):
    """Normalize to [0,1]."""
    vmin, vmax = volume.min(), volume.max()
    return (volume - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(volume)


def build_sparse_volume(slice_dir, participant_id, target_depth=256, target_size=256):
    """
    Build sparse 3D volume from LR slices (mimics MRISliceDatasetProgressive).
    Returns normalized (D,H,W) numpy array.
    """
    lr_files = glob.glob(os.path.join(slice_dir, f"{participant_id}_slice*_lr.npy"))
    if not lr_files:
        lr_files = glob.glob(os.path.join(slice_dir, f"{participant_id}.nii_slice*_lr.npy"))
        if not lr_files:
            raise FileNotFoundError(f"No LR slice files found for {participant_id} in {slice_dir}")

    lr_slices = {}
    slice_nums = []
    for file in lr_files:
        basename = os.path.basename(file)
        slice_num = int(basename.split("slice")[1].split("_")[0])
        lr_data = np.load(file).squeeze().astype(np.float32)
        lr_slices[slice_num] = lr_data
        slice_nums.append(slice_num)

    mapped = map_indices(slice_nums, target_depth)
    volume = np.zeros((target_depth, target_size, target_size), dtype=np.float32)

    placed = []
    for (sn, z) in zip(slice_nums, mapped):
        sl = lr_slices[sn]
        if sl.shape != (target_size, target_size):
            factors = (target_size / sl.shape[0], target_size / sl.shape[1])
            sl = zoom(sl, factors, order=1)
        volume[z] = sl
        placed.append(z)

    # Interpolation + normalization
    volume = add_linear_interpolation(volume, placed)
    volume = normalize_volume(volume)

    print(f"Sparse volume built: shape={volume.shape}, "
          f"min={volume.min():.4f}, max={volume.max():.4f}, mean={volume.mean():.4f}")

    return volume


def save_as_nifti(volume, output_path, spacing=(1.0, 1.0, 1.0)):
    """Save (D,H,W) volume as NIfTI file."""
    vol = np.transpose(volume, (1, 2, 0))  # -> (H,W,D)
    affine = np.eye(4)
    affine[0, 0], affine[1, 1], affine[2, 2] = spacing
    nib.save(nib.Nifti1Image(vol, affine), output_path)


# -------------------------------
# Main
# -------------------------------
def main():

    parser = argparse.ArgumentParser(description="Generate 3D MRI from LR slices")
    parser.add_argument("--model", required=True, help="Path to best_progressive_model.pth")
    parser.add_argument("--slices", required=True, help="Directory containing LR slice .npy files")
    parser.add_argument("--num_participants", type=int, required=True, help="Number of participants to process")
    parser.add_argument("--output_dir", default="generated_volumes", help="Directory to save output NIfTI files")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find all unique participant prefixes in the slice directory
    lr_files = glob.glob(os.path.join(args.slices, "*_slice*_lr.npy"))
    if not lr_files:
        lr_files = glob.glob(os.path.join(args.slices, "*.nii_slice*_lr.npy"))
    if not lr_files:
        raise FileNotFoundError(f"No LR slice files found in {args.slices}")

    # Extract participant IDs from filenames
    def extract_prefix(path):
        base = os.path.basename(path)
        if base.endswith(".npy"):
            return base.split("_slice")[0]
        return None
    all_participants = sorted(set(filter(None, (extract_prefix(f) for f in lr_files))))
    if not all_participants:
        raise RuntimeError("Could not extract any participant IDs from slice files.")

    selected_participants = all_participants[:args.num_participants]
    print(f"Processing {len(selected_participants)} participants: {selected_participants}")

    evaluator = ProgressiveModelEvaluator(args.model, device=args.device)

    for participant_id in selected_participants:
        try:
            sparse_np = build_sparse_volume(args.slices, participant_id,
                                            target_depth=evaluator.config.get("full_volume_slices", 256),
                                            target_size=evaluator.config["target_size"])
            sparse_tensor = torch.from_numpy(sparse_np).unsqueeze(0)  # [1,1,D,H,W]
            generated = evaluator.generate_full_volume(sparse_tensor)
            print(f"Generated volume for {participant_id}: shape={generated.shape}, min={generated.min():.4f}, max={generated.max():.4f}")
            output_path = os.path.join(args.output_dir, f"{participant_id}_generated_volume.nii.gz")
            save_as_nifti(generated, output_path)
            print(f"Saved NIfTI: {output_path}")
        except Exception as e:
            print(f"Error processing {participant_id}: {e}")


if __name__ == "__main__":
    main()
