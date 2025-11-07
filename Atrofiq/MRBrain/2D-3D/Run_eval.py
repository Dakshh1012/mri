#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import json
from typing import List, Dict, Tuple
import nibabel as nib
from tqdm import tqdm

# Import the main evaluator
from Eval import ProgressiveModelEvaluator


class BatchEvaluationManager:
    """Manage batch evaluation of multiple participants"""
    
    def __init__(self, model_path, device='cuda'):
        self.evaluator = ProgressiveModelEvaluator(model_path, device)
        self.batch_results = []
    
    def evaluate_all_participants(self, data_dir, output_dir='batch_evaluation_results',
                                 max_participants=None):
        """Evaluate all participants in the dataset"""
        
        # Get dataset to find total participants
        from Better_trainer import MRISliceDatasetProgressive
        dataset = MRISliceDatasetProgressive(
            data_dir,
            target_slices=256,
            full_volume_slices=self.evaluator.config.get('full_volume_slices', 256),
            target_size=self.evaluator.config['target_size']
        )
        
        total_participants = len(dataset)
        if max_participants:
            total_participants = min(total_participants, max_participants)
        
        print(f"Evaluating {total_participants} participants...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Evaluate each participant
        for participant_idx in tqdm(range(total_participants), desc="Evaluating participants"):
            try:
                print(f"\n--- Processing Participant {participant_idx} ---")
                result = self.evaluator.evaluate_participant(data_dir, participant_idx, output_dir)
                self.batch_results.append(result)
                
                # Save intermediate results
                if (participant_idx + 1) % 5 == 0:
                    self.save_batch_summary(output_dir)
                    
            except Exception as e:
                print(f"Failed to evaluate participant {participant_idx}: {e}")
                continue
        
        # Save final results and create comprehensive analysis
        self.save_batch_summary(output_dir)
        self.create_comprehensive_analysis(output_dir)
        
        return self.batch_results
    
    def save_batch_summary(self, output_dir):
        """Save summary of all batch results"""
        summary_data = []
        
        for result in self.batch_results:
            summary_data.append({
                'participant_id': result['participant_id'],
                'total_slices': result['total_slices'],
                'good_slices_count': result['good_slices_count'],
                'good_slices_ratio': result['good_slices_ratio'],
                'avg_ssim': result['overall_metrics']['avg_ssim'],
                'avg_psnr': result['overall_metrics']['avg_psnr'],
                'avg_cosine_sim': result['overall_metrics']['avg_cosine_sim'],
                'avg_discriminator_score': result['overall_metrics']['avg_discriminator_score']
            })
        
        # Save as CSV
        df = pd.DataFrame(summary_data)
        df.to_csv(os.path.join(output_dir, 'batch_evaluation_summary.csv'), index=False)
        
        # Save detailed JSON
        with open(os.path.join(output_dir, 'batch_evaluation_detailed.json'), 'w') as f:
            json.dump(self.batch_results, f, indent=2)
        
        print(f"Batch summary saved: {len(self.batch_results)} participants evaluated")
    
    def create_comprehensive_analysis(self, output_dir):
        """Create comprehensive analysis plots"""
        if not self.batch_results:
            return
        
        # Create analysis plots
        self._create_quality_distribution_plot(output_dir)
        self._create_participant_comparison_plot(output_dir)
        self._create_slice_position_analysis(output_dir)
        self._create_correlation_analysis(output_dir)
        self._create_failure_mode_analysis(output_dir)
    
    def _create_quality_distribution_plot(self, output_dir):
        """Create distribution plots for quality metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Collect all metrics
        all_ssim = []
        all_psnr = []
        all_cosine = []
        all_disc = []
        good_ratios = []
        
        for result in self.batch_results:
            metrics = result['slice_metrics']
            all_ssim.extend([m['ssim'] for m in metrics])
            all_psnr.extend([m['psnr'] for m in metrics])
            all_cosine.extend([m['cosine_sim'] for m in metrics])
            all_disc.extend([m['combined_score'] for m in metrics])
            good_ratios.append(result['good_slices_ratio'])
        
        # Plot distributions
        axes[0, 0].hist(all_ssim, bins=50, alpha=0.7, color='blue')
        axes[0, 0].set_title('SSIM Distribution')
        axes[0, 0].set_xlabel('SSIM')
        axes[0, 0].axvline(np.mean(all_ssim), color='red', linestyle='--', label=f'Mean: {np.mean(all_ssim):.3f}')
        axes[0, 0].legend()
        
        axes[0, 1].hist(all_psnr, bins=50, alpha=0.7, color='green')
        axes[0, 1].set_title('PSNR Distribution')
        axes[0, 1].set_xlabel('PSNR (dB)')
        axes[0, 1].axvline(np.mean(all_psnr), color='red', linestyle='--', label=f'Mean: {np.mean(all_psnr):.1f}')
        axes[0, 1].legend()
        
        axes[0, 2].hist(all_cosine, bins=50, alpha=0.7, color='orange')
        axes[0, 2].set_title('Cosine Similarity Distribution')
        axes[0, 2].set_xlabel('Cosine Similarity')
        axes[0, 2].axvline(np.mean(all_cosine), color='red', linestyle='--', label=f'Mean: {np.mean(all_cosine):.3f}')
        axes[0, 2].legend()
        
        axes[1, 0].hist(all_disc, bins=50, alpha=0.7, color='purple')
        axes[1, 0].set_title('Discriminator Score Distribution')
        axes[1, 0].set_xlabel('Discriminator Score')
        axes[1, 0].axvline(np.mean(all_disc), color='red', linestyle='--', label=f'Mean: {np.mean(all_disc):.3f}')
        axes[1, 0].legend()
        
        axes[1, 1].hist(good_ratios, bins=20, alpha=0.7, color='cyan')
        axes[1, 1].set_title('Good Slices Ratio Distribution')
        axes[1, 1].set_xlabel('Good Slices Ratio')
        axes[1, 1].axvline(np.mean(good_ratios), color='red', linestyle='--', label=f'Mean: {np.mean(good_ratios):.3f}')
        axes[1, 1].legend()
        
        # Box plot comparing all metrics
        metrics_data = [all_ssim, np.array(all_psnr)/50, all_cosine, all_disc]  # Normalize PSNR for comparison
        axes[1, 2].boxplot(metrics_data, labels=['SSIM', 'PSNR/50', 'Cosine', 'Disc'])
        axes[1, 2].set_title('Metrics Comparison (Box Plot)')
        axes[1, 2].set_ylabel('Normalized Score')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'quality_distribution_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_participant_comparison_plot(self, output_dir):
        """Create participant-wise comparison plot"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        participant_ids = [r['participant_id'] for r in self.batch_results]
        good_ratios = [r['good_slices_ratio'] for r in self.batch_results]
        avg_ssim = [r['overall_metrics']['avg_ssim'] for r in self.batch_results]
        avg_psnr = [r['overall_metrics']['avg_psnr'] for r in self.batch_results]
        avg_disc = [r['overall_metrics']['avg_discriminator_score'] for r in self.batch_results]
        
        # Good slices ratio by participant
        axes[0, 0].bar(range(len(participant_ids)), good_ratios, color='lightblue')
        axes[0, 0].set_title('Good Slices Ratio by Participant')
        axes[0, 0].set_xlabel('Participant Index')
        axes[0, 0].set_ylabel('Good Slices Ratio')
        axes[0, 0].axhline(np.mean(good_ratios), color='red', linestyle='--', label=f'Mean: {np.mean(good_ratios):.3f}')
        axes[0, 0].legend()
        
        # Average SSIM by participant
        axes[0, 1].bar(range(len(participant_ids)), avg_ssim, color='lightgreen')
        axes[0, 1].set_title('Average SSIM by Participant')
        axes[0, 1].set_xlabel('Participant Index')
        axes[0, 1].set_ylabel('Average SSIM')
        axes[0, 1].axhline(np.mean(avg_ssim), color='red', linestyle='--', label=f'Mean: {np.mean(avg_ssim):.3f}')
        axes[0, 1].legend()
        
        # Average PSNR by participant
        axes[1, 0].bar(range(len(participant_ids)), avg_psnr, color='lightyellow')
        axes[1, 0].set_title('Average PSNR by Participant')
        axes[1, 0].set_xlabel('Participant Index')
        axes[1, 0].set_ylabel('Average PSNR (dB)')
        axes[1, 0].axhline(np.mean(avg_psnr), color='red', linestyle='--', label=f'Mean: {np.mean(avg_psnr):.1f}')
        axes[1, 0].legend()
        
        # Correlation plot
        axes[1, 1].scatter(avg_ssim, good_ratios, alpha=0.7, color='purple')
        axes[1, 1].set_xlabel('Average SSIM')
        axes[1, 1].set_ylabel('Good Slices Ratio')
        axes[1, 1].set_title('SSIM vs Good Slices Ratio')
        
        # Add correlation coefficient
        correlation = np.corrcoef(avg_ssim, good_ratios)[0, 1]
        axes[1, 1].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                       transform=axes[1, 1].transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'participant_comparison_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_slice_position_analysis(self, output_dir):
        """Analyze quality metrics by slice position"""
        # Collect metrics by slice position
        position_metrics = {}
        
        for result in self.batch_results:
            slice_metrics = result['slice_metrics']
            good_indices = set(result['good_slice_indices'])
            
            for i, metrics in enumerate(slice_metrics):
                if i not in position_metrics:
                    position_metrics[i] = {'ssim': [], 'psnr': [], 'cosine_sim': [], 'is_good': []}
                
                position_metrics[i]['ssim'].append(metrics['ssim'])
                position_metrics[i]['psnr'].append(metrics['psnr'])
                position_metrics[i]['cosine_sim'].append(metrics['cosine_sim'])
                position_metrics[i]['is_good'].append(1 if i in good_indices else 0)
        
        # Calculate average metrics by position
        positions = sorted(position_metrics.keys())
        avg_ssim_by_pos = [np.mean(position_metrics[pos]['ssim']) for pos in positions]
        avg_psnr_by_pos = [np.mean(position_metrics[pos]['psnr']) for pos in positions]
        good_ratio_by_pos = [np.mean(position_metrics[pos]['is_good']) for pos in positions]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # SSIM by slice position
        axes[0, 0].plot(positions, avg_ssim_by_pos, 'b-', alpha=0.7)
        axes[0, 0].fill_between(positions, avg_ssim_by_pos, alpha=0.3)
        axes[0, 0].set_title('Average SSIM by Slice Position')
        axes[0, 0].set_xlabel('Slice Position')
        axes[0, 0].set_ylabel('Average SSIM')
        axes[0, 0].grid(True, alpha=0.3)
        
        # PSNR by slice position
        axes[0, 1].plot(positions, avg_psnr_by_pos, 'g-', alpha=0.7)
        axes[0, 1].fill_between(positions, avg_psnr_by_pos, alpha=0.3)
        axes[0, 1].set_title('Average PSNR by Slice Position')
        axes[0, 1].set_xlabel('Slice Position')
        axes[0, 1].set_ylabel('Average PSNR (dB)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Good ratio by slice position
        axes[1, 0].plot(positions, good_ratio_by_pos, 'r-', alpha=0.7)
        axes[1, 0].fill_between(positions, good_ratio_by_pos, alpha=0.3)
        axes[1, 0].set_title('Good Slices Ratio by Position')
        axes[1, 0].set_xlabel('Slice Position')
        axes[1, 0].set_ylabel('Good Slices Ratio')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Heatmap of quality across positions
        quality_matrix = np.zeros((len(self.batch_results), len(positions)))
        for i, result in enumerate(self.batch_results):
            for j, pos in enumerate(positions):
                if pos < len(result['slice_metrics']):
                    quality_matrix[i, j] = result['slice_metrics'][pos]['ssim']
        
        im = axes[1, 1].imshow(quality_matrix, cmap='viridis', aspect='auto', interpolation='nearest')
        axes[1, 1].set_title('SSIM Heatmap (Participants vs Slice Position)')
        axes[1, 1].set_xlabel('Slice Position')
        axes[1, 1].set_ylabel('Participant Index')
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'slice_position_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_correlation_analysis(self, output_dir):
        """Create correlation analysis between different metrics"""
        # Collect all metrics for correlation analysis
        all_metrics_data = []
        
        for result in self.batch_results:
            for metrics in result['slice_metrics']:
                all_metrics_data.append([
                    metrics['ssim'],
                    metrics['psnr'],
                    metrics['cosine_sim'],
                    metrics['mae'],
                    metrics['quality_score'],
                    metrics['similarity_score'],
                    metrics['combined_score']
                ])
        
        df_metrics = pd.DataFrame(all_metrics_data, columns=[
            'SSIM', 'PSNR', 'Cosine_Sim', 'MAE', 'Quality_Score', 'Similarity_Score', 'Combined_Score'
        ])
        
        # Create correlation matrix
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Correlation heatmap
        correlation_matrix = df_metrics.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=axes[0], fmt='.3f')
        axes[0].set_title('Metrics Correlation Matrix')
        
        # Pairplot for key metrics
        key_metrics = df_metrics[['SSIM', 'PSNR', 'Cosine_Sim', 'Combined_Score']].sample(n=min(1000, len(df_metrics)))
        
        # Scatter plot matrix
        scatter_data = key_metrics.values
        n_metrics = scatter_data.shape[1]
        
        for i in range(n_metrics):
            for j in range(n_metrics):
                if i != j:
                    axes[1].scatter(scatter_data[:, i], scatter_data[:, j], alpha=0.1, s=1)
        
        axes[1].set_title('Metrics Scatter Plot Matrix (Sample)')
        axes[1].set_xlabel('Various Metrics')
        axes[1].set_ylabel('Various Metrics')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save correlation matrix as CSV
        correlation_matrix.to_csv(os.path.join(output_dir, 'metrics_correlation_matrix.csv'))
    
    def _create_failure_mode_analysis(self, output_dir):
        """Analyze common failure modes"""
        # Identify slices with poor quality
        poor_slices_analysis = {
            'low_ssim': [],
            'low_psnr': [],
            'low_cosine': [],
            'low_discriminator': [],
            'position_distribution': []
        }
        
        ssim_threshold = 0.5
        psnr_threshold = 15.0
        cosine_threshold = 0.6
        disc_threshold = 0.4
        
        for result in self.batch_results:
            good_indices = set(result['good_slice_indices'])
            
            for i, metrics in enumerate(result['slice_metrics']):
                if i not in good_indices:  # Poor quality slice
                    poor_slices_analysis['position_distribution'].append(i)
                    
                    if metrics['ssim'] < ssim_threshold:
                        poor_slices_analysis['low_ssim'].append(metrics)
                    if metrics['psnr'] < psnr_threshold:
                        poor_slices_analysis['low_psnr'].append(metrics)
                    if metrics['cosine_sim'] < cosine_threshold:
                        poor_slices_analysis['low_cosine'].append(metrics)
                    if metrics['combined_score'] < disc_threshold:
                        poor_slices_analysis['low_discriminator'].append(metrics)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Position distribution of poor slices
        axes[0, 0].hist(poor_slices_analysis['position_distribution'], bins=50, alpha=0.7, color='red')
        axes[0, 0].set_title('Distribution of Poor Quality Slices by Position')
        axes[0, 0].set_xlabel('Slice Position')
        axes[0, 0].set_ylabel('Count')
        
        # Failure mode counts
        failure_counts = [
            len(poor_slices_analysis['low_ssim']),
            len(poor_slices_analysis['low_psnr']),
            len(poor_slices_analysis['low_cosine']),
            len(poor_slices_analysis['low_discriminator'])
        ]
        failure_labels = ['Low SSIM', 'Low PSNR', 'Low Cosine', 'Low Discriminator']
        
        axes[0, 1].bar(failure_labels, failure_counts, color=['red', 'orange', 'yellow', 'purple'])
        axes[0, 1].set_title('Failure Mode Distribution')
        axes[0, 1].set_ylabel('Count of Poor Slices')
        plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)
        
        # Quality distribution for good vs poor slices
        all_good_ssim = []
        all_poor_ssim = []
        
        for result in self.batch_results:
            good_indices = set(result['good_slice_indices'])
            for i, metrics in enumerate(result['slice_metrics']):
                if i in good_indices:
                    all_good_ssim.append(metrics['ssim'])
                else:
                    all_poor_ssim.append(metrics['ssim'])
        
        axes[1, 0].hist(all_poor_ssim, bins=30, alpha=0.7, color='red', label='Poor Quality')
        axes[1, 0].hist(all_good_ssim, bins=30, alpha=0.7, color='green', label='Good Quality')
        axes[1, 0].set_title('SSIM Distribution: Good vs Poor Slices')
        axes[1, 0].set_xlabel('SSIM')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].legend()
        
        # Summary statistics
        stats_text = f"""Failure Analysis Summary:
        Total Poor Slices: {len(poor_slices_analysis['position_distribution'])}
        Low SSIM Cases: {len(poor_slices_analysis['low_ssim'])}
        Low PSNR Cases: {len(poor_slices_analysis['low_psnr'])}
        Low Cosine Cases: {len(poor_slices_analysis['low_cosine'])}
        Low Discriminator Cases: {len(poor_slices_analysis['low_discriminator'])}
        
        Average SSIM (Good): {np.mean(all_good_ssim):.3f}
        Average SSIM (Poor): {np.mean(all_poor_ssim):.3f}
        """
        
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
        axes[1, 1].set_title('Failure Analysis Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'failure_mode_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save detailed failure analysis
        with open(os.path.join(output_dir, 'failure_analysis.json'), 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_analysis = {
                'total_poor_slices': len(poor_slices_analysis['position_distribution']),
                'failure_mode_counts': dict(zip(failure_labels, failure_counts)),
                'position_distribution_summary': {
                    'mean_position': np.mean(poor_slices_analysis['position_distribution']),
                    'std_position': np.std(poor_slices_analysis['position_distribution']),
                    'min_position': int(np.min(poor_slices_analysis['position_distribution'])),
                    'max_position': int(np.max(poor_slices_analysis['position_distribution']))
                },
                'quality_comparison': {
                    'good_slices_avg_ssim': np.mean(all_good_ssim),
                    'poor_slices_avg_ssim': np.mean(all_poor_ssim),
                    'quality_gap': np.mean(all_good_ssim) - np.mean(all_poor_ssim)
                }
            }
            json.dump(serializable_analysis, f, indent=2)


class NiftiVolumeComparer:
    """Advanced NIfTI volume comparison and visualization"""
    
    @staticmethod
    def create_orthogonal_view_comparison(generated_path, hr_path, hybrid_path, output_path):
        """Create orthogonal view comparison of three volumes"""
        
        # Load NIfTI files
        gen_img = nib.load(generated_path)
        hr_img = nib.load(hr_path)
        hybrid_img = nib.load(hybrid_path)
        
        gen_data = gen_img.get_fdata()
        hr_data = hr_img.get_fdata()
        hybrid_data = hybrid_img.get_fdata()
        
        # Get volume dimensions
        depth, height, width = gen_data.shape
        
        # Define slice positions (center slices)
        axial_slice = depth // 2
        coronal_slice = height // 2
        sagittal_slice = width // 2
        
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        
        views = [
            ('Axial', axial_slice, lambda v, s: v[s, :, :]),
            ('Coronal', coronal_slice, lambda v, s: v[:, s, :]),
            ('Sagittal', sagittal_slice, lambda v, s: v[:, :, s])
        ]
        
        for row, (view_name, slice_pos, slice_func) in enumerate(views):
            # Extract slices
            gen_slice = slice_func(gen_data, slice_pos)
            hr_slice = slice_func(hr_data, slice_pos)
            hybrid_slice = slice_func(hybrid_data, slice_pos)
            
            # Generated
            axes[row, 0].imshow(gen_slice, cmap='gray', aspect='equal')
            axes[row, 0].set_title(f'{view_name} - Generated')
            axes[row, 0].axis('off')
            
            # Ground Truth
            axes[row, 1].imshow(hr_slice, cmap='gray', aspect='equal')
            axes[row, 1].set_title(f'{view_name} - Ground Truth')
            axes[row, 1].axis('off')
            
            # Hybrid
            axes[row, 2].imshow(hybrid_slice, cmap='gray', aspect='equal')
            axes[row, 2].set_title(f'{view_name} - Hybrid')
            axes[row, 2].axis('off')
            
            # Difference map
            diff_slice = np.abs(gen_slice - hr_slice)
            im = axes[row, 3].imshow(diff_slice, cmap='hot', aspect='equal')
            axes[row, 3].set_title(f'{view_name} - |Gen - GT|')
            axes[row, 3].axis('off')
            plt.colorbar(im, ax=axes[row, 3], fraction=0.046)
        
        plt.suptitle('Orthogonal Volume Comparison', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Orthogonal view comparison saved: {output_path}")


def run_comprehensive_evaluation():
    """Run the complete evaluation pipeline with example usage"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Progressive MRI GAN Evaluation')
    parser.add_argument('--model_path', type=str, default='best_model_progressive.pth',
                       help='Path to trained progressive model')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to data directory')
    parser.add_argument('--mode', type=str, choices=['single', 'batch'], default='single',
                       help='Evaluation mode')
    parser.add_argument('--participant_idx', type=int, default=0,
                       help='Participant index for single evaluation')
    parser.add_argument('--max_participants', type=int, default=10,
                       help='Maximum participants for batch evaluation')
    parser.add_argument('--output_dir', type=str, default='comprehensive_evaluation',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto/cuda/cpu)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Running comprehensive evaluation on {device}")
    
    if args.mode == 'single':
        # Single participant evaluation
        evaluator = ProgressiveModelEvaluator(args.model_path, device)
        result = evaluator.evaluate_participant(args.data_dir, args.participant_idx, args.output_dir)
        
        print("\n=== Single Participant Evaluation Complete ===")
        print(f"Results saved in: {args.output_dir}/participant_{result['participant_id']}")
        
    else:
        # Batch evaluation
        batch_manager = BatchEvaluationManager(args.model_path, device)
        results = batch_manager.evaluate_all_participants(
            args.data_dir, args.output_dir, args.max_participants
        )
        
        print("\n=== Batch Evaluation Complete ===")
        print(f"Evaluated {len(results)} participants")
        print(f"Results saved in: {args.output_dir}")
        print("\nGenerated Analysis Files:")
        print("- batch_evaluation_summary.csv")
        print("- quality_distribution_analysis.png")
        print("- participant_comparison_analysis.png")
        print("- slice_position_analysis.png")
        print("- correlation_analysis.png")
        print("- failure_mode_analysis.png")


if __name__ == "__main__":
    run_comprehensive_evaluation()