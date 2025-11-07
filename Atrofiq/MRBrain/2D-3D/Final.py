#!/usr/bin/env python3
"""
Progressive MRI GAN Evaluation Runner
=====================================

This script provides easy-to-use functions for running the comprehensive evaluation pipeline.
"""

import os
import sys
import torch
import argparse
import json
from datetime import datetime

# Add current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from Eval import ProgressiveModelEvaluator
    from Run_eval import BatchEvaluationManager, NiftiVolumeComparer
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required files are in the same directory:")
    print("- Run_eval.py")
    print("- batch_evaluation_utils.py")
    print("- BetterTrainer.py")
    print("- Main.py")
    sys.exit(1)


class EvaluationRunner:
    """Main runner class for progressive MRI GAN evaluation"""
    
    def __init__(self):
        self.config = self._load_default_config()
    
    def _load_default_config(self):
        """Load default configuration"""
        return {
            'quality_thresholds': {
                'ssim_threshold': 0.7,
                'cosine_threshold': 0.8,
                'discriminator_threshold': 0.6,
                'psnr_threshold': 20.0
            },
            'evaluation_settings': {
                'device': 'cpu',
                'batch_size': 1,
                'save_intermediate': True
            },
            'visualization': {
                'create_3d_views': True,
                'create_quality_plots': True,
                'save_nifti': True
            }
        }
    
    def single_participant_evaluation(self, model_path, data_dir, participant_idx, output_dir=None):
        """
        Run evaluation for a single participant
        
        Args:
            model_path (str): Path to the trained progressive model (.pth file)
            data_dir (str): Path to the data directory
            participant_idx (int): Index of participant to evaluate
            output_dir (str): Output directory (optional)
        
        Returns:
            dict: Evaluation results
        """
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"evaluation_single_{timestamp}"
        
        print(f"🚀 Starting single participant evaluation...")
        print(f"Model: {model_path}")
        print(f"Data directory: {data_dir}")
        print(f"Participant index: {participant_idx}")
        print(f"Output directory: {output_dir}")
        
        # Set device
        device = self._get_device()
        print(f"Using device: {device}")
        
        # Create evaluator
        evaluator = ProgressiveModelEvaluator(model_path, device)
        
        # Run evaluation
        result = evaluator.evaluate_participant(
            data_dir=data_dir,
            participant_idx=participant_idx,
            output_dir=output_dir
        )
        
        # Print summary
        self._print_single_evaluation_summary(result)
        
        return result
    
    def batch_evaluation(self, model_path, data_dir, max_participants=None, output_dir=None):
        """
        Run batch evaluation for multiple participants
        
        Args:
            model_path (str): Path to the trained progressive model (.pth file)
            data_dir (str): Path to the data directory
            max_participants (int): Maximum number of participants to evaluate
            output_dir (str): Output directory (optional)
        
        Returns:
            list: List of evaluation results
        """
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"evaluation_batch_{timestamp}"
        
        print(f"🚀 Starting batch evaluation...")
        print(f"Model: {model_path}")
        print(f"Data directory: {data_dir}")
        print(f"Max participants: {max_participants or 'All'}")
        print(f"Output directory: {output_dir}")
        
        # Set device
        device = self._get_device()
        print(f"Using device: {device}")
        
        # Create batch manager
        batch_manager = BatchEvaluationManager(model_path, device)
        
        # Run batch evaluation
        results = batch_manager.evaluate_all_participants(
            data_dir=data_dir,
            output_dir=output_dir,
            max_participants=max_participants
        )
        
        # Print summary
        self._print_batch_evaluation_summary(results, output_dir)
        
        return results
    
    def compare_volumes(self, generated_path, hr_path, hybrid_path, output_path):
        """
        Create orthogonal view comparison of volumes
        
        Args:
            generated_path (str): Path to generated volume NIfTI file
            hr_path (str): Path to ground truth volume NIfTI file
            hybrid_path (str): Path to hybrid volume NIfTI file
            output_path (str): Output path for comparison image
        """
        print(f"📊 Creating volume comparison...")
        print(f"Generated: {generated_path}")
        print(f"Ground Truth: {hr_path}")
        print(f"Hybrid: {hybrid_path}")
        print(f"Output: {output_path}")
        
        NiftiVolumeComparer.create_orthogonal_view_comparison(
            generated_path, hr_path, hybrid_path, output_path
        )
        
        print("✅ Volume comparison completed!")
    
    def _get_device(self):
        """Get appropriate device"""
        device_setting = self.config['evaluation_settings']['device']
        
        if device_setting == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                print(f"🔥 GPU detected: {torch.cuda.get_device_name(0)}")
                torch.backends.cudnn.benchmark = True
            else:
                device = 'cpu'
                print("💻 Using CPU")
        else:
            device = device_setting
        
        return device
    
    def _print_single_evaluation_summary(self, result):
        """Print summary for single evaluation"""
        print("\n" + "="*60)
        print("📈 EVALUATION SUMMARY")
        print("="*60)
        print(f"Participant ID: {result['participant_id']}")
        print(f"Total Slices: {result['total_slices']}")
        print(f"Good Quality Slices: {result['good_slices_count']} ({result['good_slices_ratio']:.1%})")
        print("\nQuality Metrics:")
        print(f"  Average SSIM: {result['overall_metrics']['avg_ssim']:.4f}")
        print(f"  Average PSNR: {result['overall_metrics']['avg_psnr']:.2f} dB")
        print(f"  Average Cosine Similarity: {result['overall_metrics']['avg_cosine_sim']:.4f}")
        print(f"  Average Discriminator Score: {result['overall_metrics']['avg_discriminator_score']:.4f}")
        print("\n✅ Single participant evaluation completed!")
        print("="*60)
    
    def _print_batch_evaluation_summary(self, results, output_dir):
        """Print summary for batch evaluation"""
        if not results:
            print("❌ No results to summarize")
            return
        
        # Calculate aggregate statistics
        total_slices = sum(r['total_slices'] for r in results)
        total_good_slices = sum(r['good_slices_count'] for r in results)
        avg_good_ratio = total_good_slices / total_slices if total_slices > 0 else 0
        
        avg_ssim = sum(r['overall_metrics']['avg_ssim'] for r in results) / len(results)
        avg_psnr = sum(r['overall_metrics']['avg_psnr'] for r in results) / len(results)
        avg_cosine = sum(r['overall_metrics']['avg_cosine_sim'] for r in results) / len(results)
        avg_disc = sum(r['overall_metrics']['avg_discriminator_score'] for r in results) / len(results)
        
        print("\n" + "="*60)
        print("📊 BATCH EVALUATION SUMMARY")
        print("="*60)
        print(f"Participants Evaluated: {len(results)}")
        print(f"Total Slices Processed: {total_slices:,}")
        print(f"Overall Good Quality Slices: {total_good_slices:,} ({avg_good_ratio:.1%})")
        print("\nAggregate Quality Metrics:")
        print(f"  Average SSIM: {avg_ssim:.4f}")
        print(f"  Average PSNR: {avg_psnr:.2f} dB")
        print(f"  Average Cosine Similarity: {avg_cosine:.4f}")
        print(f"  Average Discriminator Score: {avg_disc:.4f}")
        print(f"\n📁 Results saved in: {output_dir}")
        print("\n📈 Generated Analysis Files:")
        print("  • batch_evaluation_summary.csv - Participant-wise summary")
        print("  • quality_distribution_analysis.png - Quality metrics distributions")
        print("  • participant_comparison_analysis.png - Cross-participant comparison")
        print("  • slice_position_analysis.png - Quality by slice position")
        print("  • correlation_analysis.png - Metrics correlation analysis")
        print("  • failure_mode_analysis.png - Failure pattern analysis")
        print("\n✅ Batch evaluation completed!")
        print("="*60)
    
    def save_config(self, config_path):
        """Save current configuration"""
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"Configuration saved: {config_path}")
    
    def load_config(self, config_path):
        """Load configuration from file"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        print(f"Configuration loaded: {config_path}")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description='Progressive MRI GAN Evaluation Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single participant evaluation
  python evaluation_runner.py --mode single --model best_model_progressive.pth --data /path/to/data --participant 0

  # Batch evaluation (first 5 participants)
  python evaluation_runner.py --mode batch --model best_model_progressive.pth --data /path/to/data --max_participants 5

  # Compare volumes
  python evaluation_runner.py --mode compare --gen generated.nii.gz --hr ground_truth.nii.gz --hybrid hybrid.nii.gz
        """
    )
    
    parser.add_argument('--mode', type=str, 
                       choices=['single', 'batch', 'compare'],
                       required=True,
                       help='Evaluation mode')
    
    # Common arguments
    parser.add_argument('--model', type=str,
                       help='Path to trained progressive model (.pth file)')
    parser.add_argument('--data', type=str,
                       help='Path to data directory')
    parser.add_argument('--output', type=str,
                       help='Output directory')
    
    # Single mode arguments
    parser.add_argument('--participant', type=int, default=0,
                       help='Participant index for single evaluation')
    
    # Batch mode arguments
    parser.add_argument('--max_participants', type=int,
                       help='Maximum participants for batch evaluation')
    
    # Compare mode arguments
    parser.add_argument('--gen', type=str,
                       help='Path to generated volume NIfTI file')
    parser.add_argument('--hr', type=str,
                       help='Path to ground truth volume NIfTI file')
    parser.add_argument('--hybrid', type=str,
                       help='Path to hybrid volume NIfTI file')
    parser.add_argument('--compare_output', type=str,
                       help='Output path for comparison image')
    
    # Configuration
    parser.add_argument('--config', type=str,
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Create runner
    runner = EvaluationRunner()
    
    # Load configuration if provided
    if args.config and os.path.exists(args.config):
        runner.load_config(args.config)
    
    try:
        if args.mode == 'single':
            if not all([args.model, args.data]):
                print("❌ Error: --model and --data are required for single mode")
                sys.exit(1)
            
            result = runner.single_participant_evaluation(
                model_path=args.model,
                data_dir=args.data,
                participant_idx=args.participant,
                output_dir=args.output
            )
            
        elif args.mode == 'batch':
            if not all([args.model, args.data]):
                print("❌ Error: --model and --data are required for batch mode")
                sys.exit(1)
            
            results = runner.batch_evaluation(
                model_path=args.model,
                data_dir=args.data,
                max_participants=args.max_participants,
                output_dir=args.output
            )
            
        elif args.mode == 'compare':
            if not all([args.gen, args.hr, args.hybrid]):
                print("❌ Error: --gen, --hr, and --hybrid are required for compare mode")
                sys.exit(1)
            
            output_path = args.compare_output or 'volume_comparison.png'
            runner.compare_volumes(
                generated_path=args.gen,
                hr_path=args.hr,
                hybrid_path=args.hybrid,
                output_path=output_path
            )
    
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        sys.exit(1)


# Easy-to-use functions for interactive use
def quick_single_eval(model_path, data_dir, participant_idx=0):
    """Quick single participant evaluation"""
    runner = EvaluationRunner()
    return runner.single_participant_evaluation(model_path, data_dir, participant_idx)

def quick_batch_eval(model_path, data_dir, max_participants=5):
    """Quick batch evaluation"""
    runner = EvaluationRunner()
    return runner.batch_evaluation(model_path, data_dir, max_participants)

def quick_compare(generated_path, hr_path, hybrid_path, output_path='comparison.png'):
    """Quick volume comparison"""
    runner = EvaluationRunner()
    return runner.compare_volumes(generated_path, hr_path, hybrid_path, output_path)


if __name__ == "__main__":
    main()