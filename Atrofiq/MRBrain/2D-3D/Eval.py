#!/usr/bin/env python3
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import nibabel as nib
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics.pairwise import cosine_similarity
import json
import argparse
from tqdm import tqdm
import gc
from scipy.ndimage import zoom

# Import your existing components
from Better_trainer import MRISliceDatasetProgressive, test_model_progressive
from Main import Generator3D, clear_memory, get_memory_usage


class QualityDiscriminator(nn.Module):
    """Standalone discriminator for slice quality assessment"""
    
    def __init__(self, input_channels=2, base_features=64):
        super(QualityDiscriminator, self).__init__()
        
        # Takes concatenated [generated_slice, hr_slice] as input
        self.features = nn.Sequential(
            # Input: 2 x 256 x 256
            nn.Conv2d(input_channels, base_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64 x 128 x 128
            nn.Conv2d(base_features, base_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128 x 64 x 64
            nn.Conv2d(base_features * 2, base_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 256 x 32 x 32
            nn.Conv2d(base_features * 4, base_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 512 x 16 x 16
            nn.Conv2d(base_features * 8, base_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 512 x 8 x 8
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten()
        )
        
        # Quality score head
        self.quality_head = nn.Sequential(
            nn.Linear(base_features * 8 * 4 * 4, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Quality score between 0 and 1
        )
        
        # Similarity head
        self.similarity_head = nn.Sequential(
            nn.Linear(base_features * 8 * 4 * 4, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Similarity score between 0 and 1
        )
        
    def forward(self, generated_slice, hr_slice):
        # Concatenate generated and HR slices
        x = torch.cat([generated_slice.unsqueeze(1), hr_slice.unsqueeze(1)], dim=1)
        features = self.features(x)
        
        quality_score = self.quality_head(features)
        similarity_score = self.similarity_head(features)
        
        return quality_score, similarity_score


class SliceQualityMetrics:
    """Comprehensive slice quality assessment"""
    
    @staticmethod
    def compute_ssim(generated, target, data_range=1.0):
        """Compute SSIM between generated and target slices"""
        if generated.shape != target.shape:
            return 0.0
        return ssim(generated, target, data_range=data_range)
    
    @staticmethod
    def compute_psnr(generated, target, max_val=1.0):
        """Compute PSNR between generated and target slices"""
        mse = np.mean((generated - target) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(max_val / np.sqrt(mse))
    
    @staticmethod
    def compute_cosine_similarity(generated, target):
        """Compute cosine similarity between flattened slices"""
        gen_flat = generated.flatten().reshape(1, -1)
        target_flat = target.flatten().reshape(1, -1)
        return cosine_similarity(gen_flat, target_flat)[0, 0]
    
    @staticmethod
    def compute_mae(generated, target):
        """Compute Mean Absolute Error"""
        return np.mean(np.abs(generated - target))
    
    @staticmethod
    def compute_gradient_similarity(generated, target):
        """Compute gradient-based similarity"""
        # Sobel gradients
        from scipy.ndimage import sobel
        
        grad_gen_x = sobel(generated, axis=1)
        grad_gen_y = sobel(generated, axis=0)
        grad_target_x = sobel(target, axis=1)
        grad_target_y = sobel(target, axis=0)
        
        grad_gen = np.sqrt(grad_gen_x**2 + grad_gen_y**2)
        grad_target = np.sqrt(grad_target_x**2 + grad_target_y**2)
        
        return SliceQualityMetrics.compute_cosine_similarity(grad_gen, grad_target)
    
    @classmethod
    def compute_comprehensive_metrics(cls, generated, target):
        """Compute all quality metrics for a slice pair"""
        return {
            'ssim': cls.compute_ssim(generated, target),
            'psnr': cls.compute_psnr(generated, target),
            'cosine_sim': cls.compute_cosine_similarity(generated, target),
            'mae': cls.compute_mae(generated, target),
            'gradient_sim': cls.compute_gradient_similarity(generated, target)
        }


class ProgressiveModelEvaluator:
    """Main evaluation pipeline for progressive MRI GAN"""
    
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model_path = model_path
        
        # Load the progressive model
        print(f"Loading progressive model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        self.config = checkpoint['config']
        
        # Initialize generator
        self.generator = Generator3D(base_features=self.config['base_features']).to(device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.generator.eval()
        
        # Initialize quality discriminator
        self.quality_discriminator = QualityDiscriminator().to(device)
        self.quality_discriminator.eval()
        
        print("Models loaded successfully")
        print(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"Quality discriminator parameters: {sum(p.numel() for p in self.quality_discriminator.parameters()):,}")
    
    def generate_full_volume(self, sparse_input):
        """Generate 256 slices from sparse input"""
        with torch.no_grad(), torch.cuda.amp.autocast():
            sparse_input_gpu = sparse_input.unsqueeze(0).to(self.device)
            generated_volume = self.generator(sparse_input_gpu)
            return generated_volume.squeeze().cpu().numpy()
    
    def evaluate_slice_quality(self, generated_volume, hr_volume, discriminator_threshold=0.7):
        """Evaluate quality of each generated slice against HR counterpart"""
        num_slices = generated_volume.shape[0]
        slice_metrics = []
        discriminator_scores = []
        
        print("Evaluating slice quality...")
        
        # Normalize volumes for discriminator
        gen_norm = (generated_volume - generated_volume.min()) / (generated_volume.max() - generated_volume.min() + 1e-8)
        hr_norm = (hr_volume - hr_volume.min()) / (hr_volume.max() - hr_volume.min() + 1e-8)
        
        with torch.no_grad():
            for i in tqdm(range(num_slices), desc="Evaluating slices"):
                gen_slice = generated_volume[i]
                hr_slice = hr_volume[i]
                
                # Compute traditional metrics
                metrics = SliceQualityMetrics.compute_comprehensive_metrics(gen_slice, hr_slice)
                
                # Compute discriminator scores
                gen_tensor = torch.from_numpy(gen_norm[i]).float().to(self.device)
                hr_tensor = torch.from_numpy(hr_norm[i]).float().to(self.device)
                
                quality_score, similarity_score = self.quality_discriminator(
                    gen_tensor.unsqueeze(0), hr_tensor.unsqueeze(0)
                )
                
                disc_metrics = {
                    'quality_score': quality_score.item(),
                    'similarity_score': similarity_score.item(),
                    'combined_score': (quality_score.item() + similarity_score.item()) / 2
                }
                
                # Combine all metrics
                combined_metrics = {**metrics, **disc_metrics}
                slice_metrics.append(combined_metrics)
                discriminator_scores.append(disc_metrics['combined_score'])
        
        return slice_metrics, discriminator_scores
    
    def select_good_slices(self, generated_volume, hr_volume, slice_metrics, 
                          ssim_threshold=0.7, cosine_threshold=0.8, 
                          discriminator_threshold=0.6, psnr_threshold=20.0):
        """Select high-quality slices based on multiple criteria"""
        
        good_slice_indices = []
        selection_criteria = []
        
        for i, metrics in enumerate(slice_metrics):
            criteria = {
                'ssim_good': metrics['ssim'] >= ssim_threshold,
                'cosine_good': metrics['cosine_sim'] >= cosine_threshold,
                'disc_good': metrics['combined_score'] >= discriminator_threshold,
                'psnr_good': metrics['psnr'] >= psnr_threshold
            }
            
            # A slice is considered good if it meets at least 3 out of 4 criteria
            good_count = sum(criteria.values())
            is_good = good_count >= 3
            
            if is_good:
                good_slice_indices.append(i)
            
            selection_criteria.append({**criteria, 'is_good': is_good, 'score_count': good_count})
        
        print(f"Selected {len(good_slice_indices)} out of {len(slice_metrics)} slices as high quality")
        
        return good_slice_indices, selection_criteria
    
    def create_hybrid_volume(self, generated_volume, hr_volume, good_slice_indices, strategy='best_available'):
        """Create hybrid volume using best available slices"""
        
        hybrid_volume = np.zeros_like(hr_volume)
        slice_sources = []  
        
        for i in range(generated_volume.shape[0]):
            if i in good_slice_indices:
                hybrid_volume[i] = generated_volume[i]
                slice_sources.append('generated')
            else:
                if strategy == 'best_available':
                    hybrid_volume[i] = hr_volume[i]
                    slice_sources.append('hr_fallback')
                elif strategy == 'interpolation':
                    hybrid_volume[i] = self._interpolate_slice(generated_volume, good_slice_indices, i)
                    slice_sources.append('interpolated')
                else:
                    hybrid_volume[i] = generated_volume[i]  
                    slice_sources.append('generated_kept')
        
        return hybrid_volume, slice_sources
    
    def _interpolate_slice(self, volume, good_indices, target_idx):
        """Interpolate slice from nearest good slices"""
        if not good_indices:
            return volume[target_idx]
        
        # Find nearest good slices
        distances = [abs(idx - target_idx) for idx in good_indices]
        nearest_idx = good_indices[np.argmin(distances)]
        
        return volume[nearest_idx]  # Simple nearest neighbor for now
    
    def save_as_nifti(self, volume, affine, output_path, description=""):
        """Save volume as NIfTI file"""
        nifti_img = nib.Nifti1Image(volume, affine)
        nib.save(nifti_img, output_path)
        print(f"Saved {description} volume: {output_path}")
    
    def create_3d_visualization(self, generated_volume, hr_volume, hybrid_volume, 
                               slice_metrics, good_slice_indices, output_prefix):
        """Create comprehensive 3D visualization across all axes"""
        
        # Get volume dimensions
        depth, height, width = generated_volume.shape
        
        # Create figure with subplots for different views
        fig = plt.figure(figsize=(20, 24))
        
        # Define slice positions for visualization
        axial_slice = depth // 2
        coronal_slice = height // 2
        sagittal_slice = width // 2
        
        views = [
            ('Axial', axial_slice, lambda v, s: v[s, :, :], 0),
            ('Coronal', coronal_slice, lambda v, s: v[:, s, :], 1),
            ('Sagittal', sagittal_slice, lambda v, s: v[:, :, s], 2)
        ]
        
        for view_idx, (view_name, slice_pos, slice_func, axis) in enumerate(views):
            # Extract slices for this view
            gen_slice = slice_func(generated_volume, slice_pos)
            hr_slice = slice_func(hr_volume, slice_pos)
            hybrid_slice = slice_func(hybrid_volume, slice_pos)
            
            # Create subplot grid for this view
            row_start = view_idx * 2
            
            # Generated volume
            ax1 = plt.subplot2grid((6, 4), (row_start, 0))
            im1 = ax1.imshow(gen_slice, cmap='gray', aspect='auto')
            ax1.set_title(f'{view_name} - Generated\n(Slice {slice_pos})')
            ax1.axis('off')
            plt.colorbar(im1, ax=ax1, fraction=0.046)
            
            # HR volume
            ax2 = plt.subplot2grid((6, 4), (row_start, 1))
            im2 = ax2.imshow(hr_slice, cmap='gray', aspect='auto')
            ax2.set_title(f'{view_name} - Ground Truth\n(Slice {slice_pos})')
            ax2.axis('off')
            plt.colorbar(im2, ax=ax2, fraction=0.046)
            
            # Hybrid volume
            ax3 = plt.subplot2grid((6, 4), (row_start, 2))
            im3 = ax3.imshow(hybrid_slice, cmap='gray', aspect='auto')
            ax3.set_title(f'{view_name} - Hybrid\n(Slice {slice_pos})')
            ax3.axis('off')
            plt.colorbar(im3, ax=ax3, fraction=0.046)
            
            # Difference map
            ax4 = plt.subplot2grid((6, 4), (row_start, 3))
            diff_slice = np.abs(gen_slice - hr_slice)
            im4 = ax4.imshow(diff_slice, cmap='hot', aspect='auto')
            ax4.set_title(f'{view_name} - |Generated - GT|\n(Slice {slice_pos})')
            ax4.axis('off')
            plt.colorbar(im4, ax=ax4, fraction=0.046)
            
            # Quality indicators
            if axis == 0:  # For axial view, show slice quality along depth
                ax5 = plt.subplot2grid((6, 4), (row_start + 1, 0), colspan=4)
                
                # Plot quality metrics
                slice_indices = range(len(slice_metrics))
                ssim_scores = [m['ssim'] for m in slice_metrics]
                cosine_scores = [m['cosine_sim'] for m in slice_metrics]
                disc_scores = [m['combined_score'] for m in slice_metrics]
                
                ax5.plot(slice_indices, ssim_scores, label='SSIM', alpha=0.7)
                ax5.plot(slice_indices, cosine_scores, label='Cosine Sim', alpha=0.7)
                ax5.plot(slice_indices, disc_scores, label='Discriminator', alpha=0.7)
                
                # Highlight good slices
                for idx in good_slice_indices:
                    ax5.axvline(x=idx, color='green', alpha=0.3, linestyle='--')
                
                ax5.axvline(x=slice_pos, color='red', linewidth=2, label='Current Slice')
                ax5.set_xlabel('Slice Index')
                ax5.set_ylabel('Quality Score')
                ax5.set_title('Slice Quality Metrics Along Depth')
                ax5.legend()
                ax5.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_3d_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create summary statistics plot
        self._create_quality_summary_plot(slice_metrics, good_slice_indices, output_prefix)
    
    def _create_quality_summary_plot(self, slice_metrics, good_slice_indices, output_prefix):
        """Create summary plot of quality metrics"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        metrics_names = ['ssim', 'psnr', 'cosine_sim', 'mae', 'quality_score', 'similarity_score']
        good_set = set(good_slice_indices)
        
        for i, metric_name in enumerate(metrics_names):
            row, col = i // 3, i % 3
            ax = axes[row, col]
            
            # Separate good and bad slices
            good_values = [slice_metrics[idx][metric_name] for idx in range(len(slice_metrics)) if idx in good_set]
            bad_values = [slice_metrics[idx][metric_name] for idx in range(len(slice_metrics)) if idx not in good_set]
            
            # Create histogram
            ax.hist(bad_values, bins=20, alpha=0.7, color='red', label='Poor Quality', density=True)
            ax.hist(good_values, bins=20, alpha=0.7, color='green', label='Good Quality', density=True)
            
            ax.set_xlabel(metric_name.upper())
            ax.set_ylabel('Density')
            ax.set_title(f'{metric_name.upper()} Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_quality_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Quality summary saved: {output_prefix}_quality_summary.png")
    
    def evaluate_participant(self, data_dir, participant_idx, output_dir='evaluation_results'):
        """Complete evaluation pipeline for a single participant"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Evaluating participant {participant_idx}")
        
        # Load data
        dataset = MRISliceDatasetProgressive(
            data_dir,
            target_slices=256,  # Full volume
            full_volume_slices=self.config.get('full_volume_slices', 256),
            target_size=self.config['target_size']
        )
        
        if participant_idx >= len(dataset):
            raise ValueError(f"Participant index {participant_idx} out of range (max: {len(dataset)-1})")
        
        sparse_input, hr_target, participant_id = dataset[participant_idx]
        
        print(f"Processing participant: {participant_id}")
        print(f"Input shape: {sparse_input.shape}, Target shape: {hr_target.shape}")
        
        # Generate full volume
        print("Generating 256 slices...")
        generated_volume = self.generate_full_volume(sparse_input)
        
        hr_volume = hr_target.squeeze().numpy()
        
        # Evaluate slice quality
        slice_metrics, discriminator_scores = self.evaluate_slice_quality(generated_volume, hr_volume)
        
        # Select good slices
        good_slice_indices, selection_criteria = self.select_good_slices(generated_volume, hr_volume, slice_metrics)
        
        # Create hybrid volume
        hybrid_volume, slice_sources = self.create_hybrid_volume(generated_volume, hr_volume, good_slice_indices)
        
        # Save results
        participant_output_dir = os.path.join(output_dir, f'participant_{participant_id}')
        os.makedirs(participant_output_dir, exist_ok=True)
        
        # Get original affine transformation (assuming identity for now)
        affine = np.eye(4)
        affine[0:3, 0:3] *= 2.0  # Adjust voxel size if needed
        
        # Save volumes as NIfTI
        self.save_as_nifti(generated_volume, affine, 
                          os.path.join(participant_output_dir, 'generated_volume.nii.gz'),
                          "Generated")
        
        self.save_as_nifti(hr_volume, affine,
                          os.path.join(participant_output_dir, 'ground_truth_volume.nii.gz'),
                          "Ground Truth")
        
        self.save_as_nifti(hybrid_volume, affine,
                          os.path.join(participant_output_dir, 'hybrid_volume.nii.gz'),
                          "Hybrid")
        
        # Save evaluation results
        results = {
            'participant_id': participant_id,
            'total_slices': len(slice_metrics),
            'good_slices_count': len(good_slice_indices),
            'good_slices_ratio': len(good_slice_indices) / len(slice_metrics),
            'slice_metrics': slice_metrics,
            'good_slice_indices': good_slice_indices,
            'selection_criteria': selection_criteria,
            'slice_sources': slice_sources,
            'overall_metrics': {
                'avg_ssim': np.mean([m['ssim'] for m in slice_metrics]),
                'avg_psnr': np.mean([m['psnr'] for m in slice_metrics]),
                'avg_cosine_sim': np.mean([m['cosine_sim'] for m in slice_metrics]),
                'avg_discriminator_score': np.mean([m['combined_score'] for m in slice_metrics])
            }
        }
        
        # with open(os.path.join(participant_output_dir, 'evaluation_results.json'), 'w') as f:
        #     json.dump(results, f, indent=2)
        
        # Create visualizations
        output_prefix = os.path.join(participant_output_dir, participant_id)
        self.create_3d_visualization(generated_volume, hr_volume, hybrid_volume,
                                   slice_metrics, good_slice_indices, output_prefix)
        
        print(f"Evaluation completed for participant {participant_id}")
        print(f"Good slices: {len(good_slice_indices)}/{len(slice_metrics)} ({100*len(good_slice_indices)/len(slice_metrics):.1f}%)")
        print(f"Average SSIM: {results['overall_metrics']['avg_ssim']:.4f}")
        print(f"Average PSNR: {results['overall_metrics']['avg_psnr']:.2f}")
        print(f"Results saved in: {participant_output_dir}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Progressive MRI GAN Evaluation Pipeline')
    parser.add_argument('--model_path', type=str, default='best_model_progressive.pth',
                       help='Path to trained progressive model')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to data directory')
    parser.add_argument('--participant_idx', type=int, default=0,
                       help='Participant index to evaluate')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto/cuda/cpu)')
    parser.add_argument('--ssim_threshold', type=float, default=0.7,
                       help='SSIM threshold for good slices')
    parser.add_argument('--cosine_threshold', type=float, default=0.8,
                       help='Cosine similarity threshold for good slices')
    parser.add_argument('--discriminator_threshold', type=float, default=0.6,
                       help='Discriminator score threshold for good slices')
    parser.add_argument('--psnr_threshold', type=float, default=20.0,
                       help='PSNR threshold for good slices')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    if device == 'cuda' and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    
    try:
        # Initialize evaluator
        evaluator = ProgressiveModelEvaluator(args.model_path, device)
        
        # Run evaluation
        results = evaluator.evaluate_participant(
            args.data_dir, 
            args.participant_idx,
            args.output_dir
        )
        
        print("\nEvaluation Summary:")
        print(f"Participant: {results['participant_id']}")
        print(f"Total slices: {results['total_slices']}")
        print(f"Good quality slices: {results['good_slices_count']} ({results['good_slices_ratio']:.2%})")
        print(f"Average SSIM: {results['overall_metrics']['avg_ssim']:.4f}")
        print(f"Average PSNR: {results['overall_metrics']['avg_psnr']:.2f} dB")
        print(f"Average Cosine Similarity: {results['overall_metrics']['avg_cosine_sim']:.4f}")
        print(f"Average Discriminator Score: {results['overall_metrics']['avg_discriminator_score']:.4f}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()