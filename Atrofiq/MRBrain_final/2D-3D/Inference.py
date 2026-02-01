#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import json
import warnings
warnings.filterwarnings("ignore")


# ============================================================================
# MODEL ARCHITECTURE (from training script)
# ============================================================================

class EdgeAwareConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.edge_conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, dilation=1)
        self.norm = nn.GroupNorm(min(32, out_ch), out_ch)
        
    def forward(self, x):
        main = self.conv(x)
        edge = self.edge_conv(x)
        combined = main + 0.3 * edge
        return F.gelu(self.norm(combined))


class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        v = self.value(x).view(B, -1, H * W)
        
        attn = torch.softmax(torch.bmm(q, k), dim=-1)
        out = torch.bmm(v, attn.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        
        return self.gamma * out + x


class AnatomyAwareEncoder(nn.Module):
    def __init__(self, base_ch=64):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv3d(1, base_ch, 3, padding=1),
            nn.GroupNorm(8, base_ch),
            nn.GELU()
        )
        
        self.down1 = nn.Sequential(
            nn.Conv3d(base_ch, base_ch*2, 3, stride=2, padding=1),
            nn.GroupNorm(16, base_ch*2),
            nn.GELU(),
            nn.Conv3d(base_ch*2, base_ch*2, 3, padding=1),
            nn.GroupNorm(16, base_ch*2),
            nn.GELU()
        )
        
        self.down2 = nn.Sequential(
            nn.Conv3d(base_ch*2, base_ch*4, 3, stride=2, padding=1),
            nn.GroupNorm(32, base_ch*4),
            nn.GELU(),
            nn.Conv3d(base_ch*4, base_ch*4, 3, padding=1),
            nn.GroupNorm(32, base_ch*4),
            nn.GELU()
        )
        
        self.down3 = nn.Sequential(
            nn.Conv3d(base_ch*4, base_ch*8, 3, stride=2, padding=1),
            nn.GroupNorm(32, base_ch*8),
            nn.GELU(),
            nn.Conv3d(base_ch*8, base_ch*8, 3, padding=1),
            nn.GroupNorm(32, base_ch*8),
            nn.GELU()
        )
        
        self.final = nn.Sequential(
            nn.Conv3d(base_ch*8, base_ch*8, 3, padding=1),
            nn.GroupNorm(32, base_ch*8),
            nn.GELU()
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        return self.final(x)


class AnatomyDecoder(nn.Module):
    def __init__(self, in_ch=512):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 512, 3, padding=1),
            nn.GroupNorm(32, 512),
            nn.GELU()
        )
        
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            EdgeAwareConv(512, 256),
            EdgeAwareConv(256, 256)
        )
        self.attn1 = SpatialAttention(256)
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            EdgeAwareConv(256, 128),
            EdgeAwareConv(128, 128)
        )
        self.attn2 = SpatialAttention(128)
        
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            EdgeAwareConv(128, 64),
            EdgeAwareConv(64, 64)
        )
        
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.stem(x)
        x = self.up1(x)
        x = self.attn1(x)
        x = self.up2(x)
        x = self.attn2(x)
        x = self.up3(x)
        return self.final(x)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AnatomyAwareEncoder(base_ch=64)
        self.decoder = AnatomyDecoder(in_ch=512)
        
    def sample_slice(self, volume_features, t):
        B, C, D, H, W = volume_features.shape
        
        if not torch.is_tensor(t):
            t = torch.tensor([t], device=volume_features.device).repeat(B)
        elif t.dim() == 0:
            t = t.unsqueeze(0).repeat(B)
            
        z = 2 * t - 1
        z = z.view(B, 1, 1, 1)
        
        y = torch.linspace(-1, 1, H, device=volume_features.device)
        x = torch.linspace(-1, 1, W, device=volume_features.device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)
        z = z.expand(-1, 1, H, W)
        
        grid = torch.stack([grid_x, grid_y, z], dim=-1)
        
        sampled = F.grid_sample(volume_features, grid, align_corners=True, 
                               mode='bilinear', padding_mode='border')
        
        return sampled.squeeze(2)

    def forward(self, sparse_volume, t):
        features = self.encoder(sparse_volume)
        slice_features = self.sample_slice(features, t)
        return self.decoder(slice_features)


# ============================================================================
# PREPROCESSING FUNCTIONS (from preprocessing script)
# ============================================================================

def load_nifti_with_orientation(filepath):
    try:
        img = nib.load(filepath)
        img_ras = nib.as_closest_canonical(img)
        data = img_ras.get_fdata().astype(np.float32)
        
        return {
            'data': data,
            'affine': img_ras.affine,
            'header': img_ras.header,
            'voxel_sizes': np.abs(np.diag(img_ras.affine)[:3]),
            'shape': data.shape
        }
    except Exception as e:
        print(f"ERROR loading {filepath}: {e}")
        return None


def force_axial_orientation(data, affine):
    voxel_sizes = np.abs(np.diag(affine)[:3])
    slice_axis = np.argmax(voxel_sizes)
    
    if slice_axis == 2:
        return np.transpose(data, (2, 0, 1)), 'Axial'
    elif slice_axis == 0:
        return np.transpose(data, (2, 1, 0)), 'Sagittal->Axial'
    elif slice_axis == 1:
        return np.transpose(data, (2, 0, 1)), 'Coronal->Axial'
    
    return data, 'Unknown'


def normalize_simple(volume):
    nonzero = volume[volume > 0]
    
    if len(nonzero) == 0:
        return volume
    
    p1, p99 = np.percentile(nonzero, [1, 99])
    
    if p99 <= p1:
        return volume
    
    volume_norm = np.clip(volume, p1, p99)
    volume_norm = (volume_norm - p1) / (p99 - p1)
    
    return volume_norm.astype(np.float32)


def resize_spatial_dims(volume, target_size=256):
    if volume.shape[1] == target_size and volume.shape[2] == target_size:
        return volume
    
    zoom_factors = [1.0, target_size / volume.shape[1], target_size / volume.shape[2]]
    return zoom(volume, zoom_factors, order=1).astype(np.float32)


def resample_slices(volume, target_slices):
    if volume.shape[0] == target_slices:
        return volume
    
    zoom_factor = target_slices / volume.shape[0]
    zoom_factors = [zoom_factor, 1.0, 1.0]
    return zoom(volume, zoom_factors, order=1).astype(np.float32)


def preprocess_2d_volume(nifti_path, target_slices=25, target_size=256):
    """Preprocess 2D NIfTI volume for inference."""
    print(f"\n  Loading: {Path(nifti_path).name}")
    
    # Load NIfTI
    lr_info = load_nifti_with_orientation(nifti_path)
    if lr_info is None:
        raise ValueError(f"Failed to load {nifti_path}")
    
    print(f"    Original shape: {lr_info['shape']}")
    print(f"    Voxel sizes: {lr_info['voxel_sizes']}")
    
    # Force axial orientation
    lr_data, orientation = force_axial_orientation(lr_info['data'], lr_info['affine'])
    print(f"    Orientation: {orientation} -> shape {lr_data.shape}")
    
    # Normalize
    lr_data = normalize_simple(lr_data)
    print(f"    Normalized")
    
    # Resize spatial dimensions
    lr_data = resize_spatial_dims(lr_data, target_size)
    print(f"    Resized spatial: {lr_data.shape}")
    
    # Resample slices
    lr_data = resample_slices(lr_data, target_slices)
    print(f"    Final shape: {lr_data.shape}")
    
    return lr_data, lr_info


def preprocess_sparse_npy(subject_dir, target_slices=25, target_size=256):
    """Preprocess sparse_volume.npy for inference."""
    subject_dir = Path(subject_dir)
    sparse_path = subject_dir / 'sparse_volume.npy'
    
    if not sparse_path.exists():
        raise FileNotFoundError(f"sparse_volume.npy not found in {subject_dir}")
        
    print(f"\n  Loading sparse NPY: {sparse_path.name}")
    lr_data = np.load(sparse_path).astype(np.float32)
    print(f"    Loaded shape: {lr_data.shape}")
    
    # The sparse_volume.npy from Preprocessing_pipeline should already be normalized 
    # and resized/resampled, but we'll ensure it matches the target constraints.
    
    # Resize spatial if needed
    if lr_data.shape[1] != target_size or lr_data.shape[2] != target_size:
        lr_data = resize_spatial_dims(lr_data, target_size)
        print(f"    Resized spatial to {target_size}: {lr_data.shape}")
        
    # Resample slices if needed
    if lr_data.shape[0] != target_slices:
        lr_data = resample_slices(lr_data, target_slices)
        print(f"    Resampled slices to {target_slices}: {lr_data.shape}")
        
    # Create dummy lr_info as npy doesn't have affine
    lr_info = {
        'affine': np.eye(4),
        'shape': lr_data.shape,
        'voxel_sizes': (1.0, 1.0, 1.0)
    }
    
    return lr_data, lr_info


# ============================================================================
# INFERENCE ENGINE
# ============================================================================

class MRIInference:
    def __init__(self, checkpoint_path, device='cuda', target_slices_output=256):
        # Parse device(s)
        if isinstance(device, str):
            if ',' in device:
                self.device_ids = [int(d.strip()) for d in device.split(',') if d.strip().isdigit()]
                if not self.device_ids: # Fallback if no valid IDs are parsed
                    self.device_ids = [0] if torch.cuda.is_available() else []
                self.device = torch.device(f'cuda:{self.device_ids[0]}' if self.device_ids else 'cpu')
            else:
                self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
                self.device_ids = [self.device.index] if self.device.type == 'cuda' else []
        else: # Assume it's already a torch.device object
            self.device = device
            self.device_ids = [self.device.index] if self.device.type == 'cuda' else []

        self.target_slices_output = target_slices_output
        
        print(f"\nInitializing inference on {self.device} (GPUs: {self.device_ids if self.device_ids else 'None'})")
        print(f"Loading checkpoint: {checkpoint_path}")
        
        # Load model
        self.model = Generator()
        checkpoint = torch.load(checkpoint_path, map_location='cpu') # Load to CPU first
        
        if 'generator' in checkpoint:
            self.model.load_state_dict(checkpoint['generator'])
            print(f"  Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"  Best PSNR: {checkpoint.get('best_psnr', 'N/A')}")
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()

        # Move model to device(s)
        if self.device_ids and len(self.device_ids) > 1:
            print(f"  Using DataParallel on devices: {self.device_ids}")
            self.model = nn.DataParallel(self.model, device_ids=self.device_ids)
            self.model.to(self.device) # Move to primary device
        elif self.device.type == 'cuda':
            self.model.to(self.device)
        
        print("✓ Model loaded successfully")
    
    @torch.no_grad()
    def generate_3d_volume(self, lr_volume_np, batch_size=8):
        """Generate 3D volume from 2D input."""
        print(f"\n  Generating {self.target_slices_output} slices...")
        
        # Convert to tensor
        lr_tensor = torch.from_numpy(lr_volume_np).unsqueeze(0).unsqueeze(0)
        lr_tensor = lr_tensor.to(self.device)
        
        # Generate slices
        generated_slices = []
        t_values = np.linspace(0, 1, self.target_slices_output)
        
        for i in tqdm(range(0, len(t_values), batch_size), desc="    Generating"):
            batch_t = t_values[i:i+batch_size]
            batch_slices = []
            
            for t in batch_t:
                t_tensor = torch.tensor([t], device=self.device, dtype=torch.float32)
                pred_slice = self.model(lr_tensor, t_tensor)
                batch_slices.append(pred_slice.squeeze().cpu().numpy())
            
            generated_slices.extend(batch_slices)
        
        # Stack into volume
        output_volume = np.stack(generated_slices, axis=0)
        print(f"    Generated volume shape: {output_volume.shape}")
        
        return output_volume
    
    def save_nifti(self, volume, output_path, reference_info, denormalize=True):
        """Save generated volume as NIfTI."""
        
        # Optionally denormalize (scale back to original intensity range)
        if denormalize:
            # The volume is in [0, 1] range from sigmoid output
            # Keep it in this range or scale to reference
            pass
        
        # Create affine matrix for output
        # Use isotropic voxel spacing for the generated high-res volume
        ref_affine = reference_info['affine']
        output_voxel_size = 1.0  # 1mm isotropic
        
        output_affine = np.eye(4)
        output_affine[0, 0] = output_voxel_size
        output_affine[1, 1] = output_voxel_size
        output_affine[2, 2] = output_voxel_size
        
        # Create NIfTI image
        nifti_img = nib.Nifti1Image(volume, output_affine)
        
        # Copy some header information from reference
        nifti_img.header['descrip'] = b'Generated by MRI 2D-3D GAN'
        
        # Save
        nib.save(nifti_img, output_path)
        print(f"    Saved: {output_path}")
    
    def create_visualization(self, lr_volume, generated_volume, output_path):
        """Create visualization comparing input and output."""
        fig = plt.figure(figsize=(20, 8))
        
        # Sample slices
        n_lr = lr_volume.shape[0]
        n_gen = generated_volume.shape[0]
        positions = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for i, pos in enumerate(positions):
            # Input LR slices
            lr_idx = int(pos * (n_lr - 1))
            ax = plt.subplot(2, 5, i + 1)
            ax.imshow(lr_volume[lr_idx], cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'Input 2D: {lr_idx}/{n_lr}', fontsize=10)
            ax.axis('off')
            
            # Generated HR slices
            gen_idx = int(pos * (n_gen - 1))
            ax = plt.subplot(2, 5, i + 6)
            ax.imshow(generated_volume[gen_idx], cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'Generated 3D: {gen_idx}/{n_gen}', fontsize=10)
            ax.axis('off')
        
        plt.suptitle(f'2D Input ({lr_volume.shape}) → 3D Generated ({generated_volume.shape})', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Visualization saved: {output_path}")


# ============================================================================
# MAIN INFERENCE PIPELINE
# ============================================================================

def run_inference(input_path, output_dir, checkpoint_path, device='cuda',
                 target_slices_lr=25, target_slices_hr=256, target_size=256,
                 batch_size=8, save_visualization=True):
    """Run complete inference pipeline."""
    
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract subject ID
    subject_id = input_path.name if input_path.is_dir() else input_path.stem.replace('.nii', '').replace('.gz', '')
    print(f"\n{'='*80}")
    print(f"PROCESSING: {subject_id}")
    print(f"{'='*80}")
    
    try:
        # Step 1: Preprocess
        if input_path.is_dir():
            print("\nStep 1: Loading sparse NPY from directory...")
            lr_volume, lr_info = preprocess_sparse_npy(
                input_path, 
                target_slices=target_slices_lr, 
                target_size=target_size
            )
        else:
            print("\nStep 1: Preprocessing 2D NIfTI input...")
            lr_volume, lr_info = preprocess_2d_volume(
                input_path, 
                target_slices=target_slices_lr, 
                target_size=target_size
            )
        
        # Step 2: Initialize model
        print("\nStep 2: Loading model...")
        inferencer = MRIInference(
            checkpoint_path, 
            device=device,
            target_slices_output=target_slices_hr
        )
        
        # Step 3: Generate 3D volume
        print("\nStep 3: Generating 3D volume...")
        generated_volume = inferencer.generate_3d_volume(lr_volume, batch_size=batch_size)
        
        # Step 4: Save output
        print("\nStep 4: Saving results...")
        output_nifti = output_dir / f"{subject_id}_generated_3d.nii.gz"
        inferencer.save_nifti(generated_volume, output_nifti, lr_info)
        
        # Step 5: Create visualization
        if save_visualization:
            print("\nStep 5: Creating visualization...")
            vis_path = output_dir / f"{subject_id}_comparison.png"
            inferencer.create_visualization(lr_volume, generated_volume, vis_path)
        
        # Save metadata
        metadata = {
            'subject_id': subject_id,
            'input_file': str(input_path),
            'output_file': str(output_nifti),
            'input_shape': list(lr_volume.shape),
            'output_shape': list(generated_volume.shape),
            'status': 'success'
        }
        
        metadata_path = output_dir / f"{subject_id}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"✓ SUCCESS: {subject_id}")
        print(f"  Input:  {lr_volume.shape}")
        print(f"  Output: {generated_volume.shape}")
        print(f"  Saved:  {output_nifti}")
        print(f"{'='*80}")
        
        return metadata
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"✗ FAILED: {subject_id}")
        print(f"  Error: {e}")
        print(f"{'='*80}")
        
        import traceback
        traceback.print_exc()
        
        return {
            'subject_id': subject_id,
            'input_file': str(input_path),
            'status': 'failed',
            'error': str(e)
        }


def batch_inference(input_dir, output_dir, checkpoint_path, device='cuda',
                   target_slices_lr=25, target_slices_hr=256, target_size=256,
                   batch_size=8, save_visualization=True):
    """Run inference on all NIfTI files in a directory."""
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Find all NIfTI files and subject directories with sparse_volume.npy
    input_items = []
    
    # Check for NIfTI files
    input_items += list(input_dir.glob('*.nii')) + list(input_dir.glob('*.nii.gz'))
    
    # Check for subject directories
    for item in input_dir.iterdir():
        if item.is_dir() and (item / 'sparse_volume.npy').exists():
            input_items.append(item)
    
    if not input_items:
        print(f"No NIfTI files or subject directories (with sparse_volume.npy) found in {input_dir}")
        return
    
    print(f"\n{'='*80}")
    print(f"BATCH INFERENCE")
    print(f"{'='*80}")
    print(f"Found {len(input_items)} candidates")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Model:  {checkpoint_path}")
    print(f"{'='*80}")
    
    results = []
    
    for item in input_items:
        result = run_inference(
            item, output_dir, checkpoint_path, device,
            target_slices_lr, target_slices_hr, target_size,
            batch_size, save_visualization
        )
        results.append(result)
    
    # Save summary
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    summary = {
        'total': len(results),
        'successful': len(successful),
        'failed': len(failed),
        'results': results
    }
    
    summary_path = output_dir / 'inference_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"BATCH COMPLETE")
    print(f"{'='*80}")
    print(f"Total:      {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed:     {len(failed)}")
    print(f"Summary:    {summary_path}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description='MRI 2D to 3D Inference Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file inference
  python inference.py --input scan.nii.gz --output ./results --checkpoint weights/best_model.pth
  
  # Batch inference on directory
  python inference.py --input_dir ./2d_scans/ --output ./results --checkpoint weights/best_model.pth
        """
    )
    
    # Input/Output
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', type=str, help='Single input NIfTI file or subject directory')
    input_group.add_argument('--input_dir', type=str, help='Directory with NIfTI files or subject subdirectories')
    
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pth file)')
    
    # Preprocessing parameters
    parser.add_argument('--target_slices_lr', type=int, default=25,
                       help='Target number of input slices (default: 25)')
    parser.add_argument('--target_slices_hr', type=int, default=256,
                       help='Target number of output slices (default: 256)')
    parser.add_argument('--target_size', type=int, default=256,
                       help='Target spatial size (H, W) (default: 256)')
    
    # Inference parameters
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for slice generation (default: 8)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu, default: cuda)')
    parser.add_argument('--no_visualization', action='store_true',
                       help='Skip creating visualization images')
    
    args = parser.parse_args()
    
    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    
    # Run inference
    if args.input:
        # Single file
        if not Path(args.input).exists():
            parser.error(f"Input file not found: {args.input}")
        
        run_inference(
            args.input, args.output, args.checkpoint, args.device,
            args.target_slices_lr, args.target_slices_hr, args.target_size,
            args.batch_size, not args.no_visualization
        )
    else:
        # Batch processing
        if not Path(args.input_dir).exists():
            parser.error(f"Input directory not found: {args.input_dir}")
        
        batch_inference(
            args.input_dir, args.output, args.checkpoint, args.device,
            args.target_slices_lr, args.target_slices_hr, args.target_size,
            args.batch_size, not args.no_visualization
        )


if __name__ == '__main__':
    main()