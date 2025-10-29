#!/usr/bin/env python3
import imageio
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse
import json
from torch.utils.data import DataLoader
from torch.optim import Adam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc

from Main import (
    MRISliceDataset, Generator3D, Discriminator3D, GANLoss,
    clear_memory, get_memory_usage, get_model_config
)


class MRISliceDatasetProgressive(MRISliceDataset):
    
    def __init__(self, data_dir: str, target_slices: int = 64, full_volume_slices: int = 256, 
                 target_size: int = 256, cache_size: int = 3, debug: bool = False):
        self.full_volume_slices = full_volume_slices
        super().__init__(data_dir, target_slices, target_size, cache_size, debug)
        print(f"Progressive dataset: {target_slices} slices (from {full_volume_slices} total)")
    
    def _create_volume(self, slices_dict: dict, target_depth: int):
        volume = np.zeros((target_depth, self.target_size, self.target_size), dtype=np.float32)
        
        if not slices_dict:
            return volume
        
        slice_indices = sorted(slices_dict.keys())
        max_slice_idx = max(slice_indices)
        
        for slice_idx in slice_indices:
            slice_data = slices_dict[slice_idx]
            mapped_idx = int((slice_idx * (target_depth - 1)) / max_slice_idx)
            mapped_idx = min(mapped_idx, target_depth - 1)
            
            if slice_data.shape != (self.target_size, self.target_size):
                from scipy.ndimage import zoom
                zoom_factors = (self.target_size / slice_data.shape[0], 
                              self.target_size / slice_data.shape[1])
                slice_data = zoom(slice_data, zoom_factors, order=1)
            
            volume[mapped_idx] = slice_data
        
        return volume
    
    def _create_sparse_input(self, lr_slices: dict, target_depth: int):
        sparse_volume = np.zeros((target_depth, self.target_size, self.target_size), dtype=np.float32)
        placed_indices = []
        
        if not lr_slices:
            return sparse_volume, placed_indices
        
        slice_indices = sorted(lr_slices.keys())
        max_slice_idx = max(slice_indices)
        
        for slice_idx in slice_indices:
            slice_data = lr_slices[slice_idx]
            mapped_idx = int((slice_idx * (target_depth - 1)) / max_slice_idx)
            mapped_idx = min(mapped_idx, target_depth - 1)
            
            if slice_data.shape != (self.target_size, self.target_size):
                from scipy.ndimage import zoom
                zoom_factors = (self.target_size / slice_data.shape[0], 
                              self.target_size / slice_data.shape[1])
                slice_data = zoom(slice_data, zoom_factors, order=1)
            
            sparse_volume[mapped_idx] = slice_data
            placed_indices.append(mapped_idx)
        
        self._add_linear_interpolation(sparse_volume, placed_indices)
        
        return sparse_volume, placed_indices


class ProgressiveTrainer:
    
    def __init__(self, config):
        self.config = config
        self.device = config['device']
        self.full_volume_slices = config.get('full_volume_slices', 256)
        
        print("Initializing Progressive MRI GAN Trainer...")
        
        self.generator = Generator3D(base_features=config['base_features']).to(self.device)
        self.discriminator = Discriminator3D(base_features=config['base_features']).to(self.device)
        
        self.g_optimizer = Adam(self.generator.parameters(), 
                               lr=config['lr_g'], betas=(0.5, 0.999))
        self.d_optimizer = Adam(self.discriminator.parameters(), 
                               lr=config['lr_d'], betas=(0.5, 0.999))
        
        self.criterion = GANLoss(
            lambda_l1=config.get('lambda_l1', 100.0),
            lambda_gradient=config.get('lambda_gradient', 10.0)
        )
        
        self.scaler_g = torch.cuda.amp.GradScaler()
        self.scaler_d = torch.cuda.amp.GradScaler()
        
        self.history = {'g_loss': [], 'd_loss': [], 'l1_loss': [], 'slice_counts': []}
        
        print(f"Generator params: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"Discriminator params: {sum(p.numel() for p in self.discriminator.parameters()):,}")
    
    def train_epoch(self, dataloader, epoch, current_slice_count):
        self.generator.train()
        self.discriminator.train()
        
        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_l1_loss = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} ({current_slice_count} slices)')
        
        for batch_idx, (sparse_input, real_target, _) in enumerate(pbar):
            batch_size = sparse_input.size(0)
            sparse_input = sparse_input.to(self.device, non_blocking=True)
            real_target = real_target.to(self.device, non_blocking=True)
            
            if batch_idx % 3 == 0:
                self.d_optimizer.zero_grad(set_to_none=True)
                
                with torch.cuda.amp.autocast():
                    real_pred = self.discriminator(real_target)
                    
                    with torch.no_grad():
                        fake_output = self.generator(sparse_input)
                    fake_pred = self.discriminator(fake_output.detach())
                    
                    d_loss = self.criterion.discriminator_loss(real_pred, fake_pred)
                
                self.scaler_d.scale(d_loss).backward()
                self.scaler_d.unscale_(self.d_optimizer)
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
                self.scaler_d.step(self.d_optimizer)
                self.scaler_d.update()
                
                epoch_d_loss += d_loss.item()
            
            self.g_optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast():
                fake_output = self.generator(sparse_input)
                fake_pred = self.discriminator(fake_output)
                
                g_loss, adv_loss, l1_loss, grad_loss = self.criterion.generator_loss(
                    fake_pred, fake_output, real_target)
            
            self.scaler_g.scale(g_loss).backward()
            self.scaler_g.unscale_(self.g_optimizer)
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
            self.scaler_g.step(self.g_optimizer)
            self.scaler_g.update()
            
            epoch_g_loss += g_loss.item()
            epoch_l1_loss += l1_loss.item()
            
            pbar.set_postfix({
                'G': f'{g_loss.item():.3f}',
                'D': f'{d_loss.item() if batch_idx % 3 == 0 else 0:.3f}',
                'L1': f'{l1_loss.item():.3f}'
            })
            
            if batch_idx % 5 == 0:
                del sparse_input, real_target, fake_output
                if batch_idx % 3 == 0:
                    del real_pred, fake_pred
                torch.cuda.empty_cache()
                gc.collect()
        
        num_batches = len(dataloader)
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / max(num_batches // 3, 1)
        avg_l1_loss = epoch_l1_loss / num_batches
        
        return avg_g_loss, avg_d_loss, avg_l1_loss
    
    def calculate_progressive_stages(self, start_slices, increment_ratio, max_slices, total_epochs):
        stages = []
        current_slices = start_slices
        
        while current_slices < max_slices:
            next_slices = min(int(current_slices * (1 + increment_ratio)), max_slices)
            
            if current_slices == max_slices:
                break
                
            epochs_for_stage = max(1, total_epochs // 6)
            if next_slices == max_slices:
                epochs_for_stage = total_epochs - sum(stage[1] for stage in stages)
            
            stages.append((current_slices, epochs_for_stage))
            current_slices = next_slices
        
        if not stages or stages[-1][0] != max_slices:
            remaining_epochs = total_epochs - sum(stage[1] for stage in stages)
            stages.append((max_slices, max(1, remaining_epochs)))
        
        total_stage_epochs = sum(stage[1] for stage in stages)
        if total_stage_epochs != total_epochs:
            stages[-1] = (stages[-1][0], stages[-1][1] + (total_epochs - total_stage_epochs))
        
        return stages
    
    def progressive_train(self, data_dir, total_epochs, batch_size, start_slices=128, increment_ratio=0.2):
        stages = self.calculate_progressive_stages(start_slices, increment_ratio, 
                                                 self.full_volume_slices, total_epochs)
        
        print(f"Progressive training stages: {stages}")
        
        best_loss = float('inf')
        total_epochs_completed = 0
        
        for stage_idx, (slice_count, stage_epochs) in enumerate(stages):
            print(f"\n{'='*60}")
            print(f"STAGE {stage_idx + 1}: Training with {slice_count} slices for {stage_epochs} epochs")
            print(f"{'='*60}")
            
            dataset = MRISliceDatasetProgressive(
                data_dir,
                target_slices=slice_count,
                full_volume_slices=self.full_volume_slices,
                target_size=self.config['target_size'],
                cache_size=1 if slice_count > 128 else 2
            )
            
            stage_batch_size = 1
            
            dataloader = DataLoader(
                dataset, batch_size=stage_batch_size, shuffle=True,
                num_workers=0, pin_memory=True
            )
            
            print(f"Stage {stage_idx + 1}: {len(dataset)} participants, batch_size={stage_batch_size}")
            
            for epoch in range(stage_epochs):
                g_loss, d_loss, l1_loss = self.train_epoch(dataloader, total_epochs_completed, slice_count)
                
                self.history['g_loss'].append(g_loss)
                self.history['d_loss'].append(d_loss)
                self.history['l1_loss'].append(l1_loss)
                self.history['slice_counts'].append(slice_count)
                
                memory_info = get_memory_usage()
                print(f'Stage {stage_idx + 1}, Epoch {epoch+1}/{stage_epochs} (Global: {total_epochs_completed+1}): '
                      f'G={g_loss:.4f}, D={d_loss:.4f}, L1={l1_loss:.4f}, Slices={slice_count}')
                print(f'GPU: {memory_info["gpu_allocated"]:.2f}GB')
                
                if l1_loss < best_loss:
                    best_loss = l1_loss
                    self.save_model('best_model_progressive')
                
                if (total_epochs_completed + 1) % 10 == 0:
                    self.save_model(f'checkpoint_epoch_{total_epochs_completed+1}')
                    self.save_training_plot(f'training_progress_epoch_{total_epochs_completed+1}.png')
                
                total_epochs_completed += 1
            
            self.save_model(f'stage_{stage_idx + 1}_complete_{slice_count}slices')
            print(f"Stage {stage_idx + 1} completed. Current best L1 loss: {best_loss:.6f}")
            
            torch.cuda.empty_cache()
            gc.collect()
        
        self.save_model('final_model_progressive')
        self.save_training_plot('final_progressive_training_plot.png')
        print(f"\nProgressive training completed! Trained up to {self.full_volume_slices} slices")
        
        return total_epochs_completed
    
    def save_model(self, prefix):
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'config': self.config,
            'history': self.history,
            'scaler_g': self.scaler_g.state_dict(),
            'scaler_d': self.scaler_d.state_dict()
        }, f'{prefix}.pth')
        print(f"Model saved: {prefix}.pth")
    
    def load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        
        if 'scaler_g' in checkpoint:
            self.scaler_g.load_state_dict(checkpoint['scaler_g'])
            self.scaler_d.load_state_dict(checkpoint['scaler_d'])
        
        if 'history' in checkpoint:
            old_history = checkpoint['history']
            self.history['g_loss'] = old_history.get('g_loss', [])
            self.history['d_loss'] = old_history.get('d_loss', [])
            self.history['l1_loss'] = old_history.get('l1_loss', [])
            self.history['slice_counts'] = old_history.get('slice_counts', [])
        
        print(f"Model loaded from: {checkpoint_path}")
    
    def save_training_plot(self, filename):
        if not self.history['g_loss']:
            return
            
        epochs = range(1, len(self.history['g_loss']) + 1)
        slice_counts = self.history['slice_counts']
        
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 3, 1)
        plt.plot(epochs, self.history['g_loss'], 'b-', label='Generator', alpha=0.7)
        plt.plot(epochs, self.history['d_loss'], 'r-', label='Discriminator', alpha=0.7)
        plt.title('GAN Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        stage_changes = []
        prev_count = slice_counts[0] if slice_counts else 64
        for i, count in enumerate(slice_counts):
            if count != prev_count:
                stage_changes.append(i)
                prev_count = count
        
        for change_point in stage_changes:
            plt.axvline(x=change_point, color='gray', linestyle='--', alpha=0.5)
        
        plt.subplot(1, 3, 2)
        plt.plot(epochs, self.history['l1_loss'], 'g-')
        plt.title('L1 Reconstruction Loss')
        plt.xlabel('Epoch')
        plt.ylabel('L1 Loss')
        plt.grid(True, alpha=0.3)
        
        for change_point in stage_changes:
            plt.axvline(x=change_point, color='gray', linestyle='--', alpha=0.5)
        
        plt.subplot(1, 3, 3)
        last_10_indices = range(max(0, len(slice_counts) - 10), len(slice_counts))
        plt.plot([epochs[i] for i in last_10_indices], [slice_counts[i] for i in last_10_indices], 'purple', linewidth=2)
        plt.title('Progressive Slice Count')
        plt.xlabel('Epoch')
        plt.ylabel('Number of Slices')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Progressive training plot saved: {filename}")


def test_model_progressive(checkpoint_path, data_dir, target_slices=256, device='cuda', 
                         participant_idx=0, num_slices=5):
    print(f"Testing progressive model: {checkpoint_path}")
    print(f"Target slices: {target_slices}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    generator = Generator3D(base_features=config['base_features']).to(device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    
    dataset = MRISliceDatasetProgressive(
        data_dir,
        target_slices=target_slices,
        full_volume_slices=config.get('full_volume_slices', 256),
        target_size=config['target_size']
    )
    
    if participant_idx >= len(dataset):
        participant_idx = 0
        print(f"Invalid participant index, using 0")
    
    sparse_input, real_target, participant_id = dataset[participant_idx]
    
    print(f"Data shapes - Input: {sparse_input.shape}, Target: {real_target.shape}")
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        sparse_input_gpu = sparse_input.unsqueeze(0).to(device)
        generated_output = generator(sparse_input_gpu)
        generated_output = generated_output.squeeze().cpu().numpy()
    
    sparse_np = sparse_input.squeeze().numpy()
    target_np = real_target.squeeze().numpy()
    
    mse = np.mean((generated_output - target_np) ** 2)
    mae = np.mean(np.abs(generated_output - target_np))
    psnr = 20 * np.log10(1.0 / np.sqrt(mse + 1e-8))
    
    print(f"Progressive Test Results for {participant_id} ({target_slices} slices):")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  PSNR: {psnr:.2f} dB")
    
    depth = generated_output.shape[0]
    slice_indices = np.linspace(0, depth - 1, num_slices, dtype=int)
    
    fig, axes = plt.subplots(3, num_slices, figsize=(5*num_slices, 15))
    if num_slices == 1:
        axes = axes.reshape(3, 1)
    
    fig.suptitle(f'Progressive Test Results - {participant_id} ({target_slices} slices)\n'
                 f'MSE: {mse:.6f}, PSNR: {psnr:.2f}dB', fontsize=14)
    
    for i, slice_idx in enumerate(slice_indices):
        has_input_data = np.any(sparse_np[slice_idx] > 0.01)
        
        axes[0, i].imshow(sparse_np[slice_idx], cmap='gray', vmin=0, vmax=1)
        input_type = "Original" if has_input_data else "Interpolated"
        axes[0, i].set_title(f'Input - Slice {slice_idx}\n({input_type})', fontsize=10)
        axes[0, i].axis('off')
        
        axes[1, i].imshow(generated_output[slice_idx], cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f'Generated - Slice {slice_idx}', fontsize=10)
        axes[1, i].axis('off')
        
        axes[2, i].imshow(target_np[slice_idx], cmap='gray', vmin=0, vmax=1)
        axes[2, i].set_title(f'Ground Truth - Slice {slice_idx}', fontsize=10)
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'progressive_test_{target_slices}slices_{num_slices}views.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Progressive test visualization saved: progressive_test_{target_slices}slices_{num_slices}views.png")
    
    return {'mse': mse, 'mae': mae, 'psnr': psnr, 'target_slices': target_slices}


def main():
    parser = argparse.ArgumentParser(description='Progressive MRI GAN Trainer')
    parser.add_argument('--data_dir', type=str, required=True, 
                       help='Path to data directory')
    parser.add_argument('--mode', type=str, choices=['train', 'progressive_train', 'test', 'progressive_test', 'explore'], 
                       default='progressive_train', help='Training mode')
    parser.add_argument('--epochs', type=int, default=100, 
                       help='Total number of training epochs')
    parser.add_argument('--batch_size', type=int, default=0, 
                       help='Batch size (0 for auto)')
    parser.add_argument('--lr_g', type=float, default=2e-4, 
                       help='Generator learning rate')
    parser.add_argument('--lr_d', type=float, default=1e-4, 
                       help='Discriminator learning rate')
    parser.add_argument('--base_features', type=int, default=0, 
                       help='Base features (0 for auto)')
    parser.add_argument('--target_slices', type=int, default=64, 
                       help='Initial target slices (for progressive training)')
    parser.add_argument('--full_volume_slices', type=int, default=256, 
                       help='Full volume slice count (final target)')
    parser.add_argument('--target_size', type=int, default=256, 
                       help='Target image size')
    parser.add_argument('--device', type=str, default='auto', 
                       help='Device (auto/cuda/cpu)')
    parser.add_argument('--checkpoint', type=str, default='', 
                       help='Checkpoint path for testing')
    parser.add_argument('--participant_idx', type=int, default=0, 
                       help='Participant index for testing')
    parser.add_argument('--num_slices', type=int, default=5, 
                        help='Number of slices to visualize in test mode')
    parser.add_argument('--test_slices', type=int, default=256, 
                        help='Number of slices to test with (for progressive_test)')
    parser.add_argument('--start_slices', type=int, default=64, 
                        help='Starting number of slices for progressive training')
    parser.add_argument('--increment_ratio', type=float, default=0.2, 
                        help='Increment ratio for progressive training (e.g., 0.2 for 20%)')
    
    args = parser.parse_args()
    
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    if device == 'cuda' and torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    else:
        print("Using CPU")
    
    if args.mode == 'explore':
        from Trainer import explore_data
        explore_data(args.data_dir)
        
    elif args.mode == 'test':
        if not args.checkpoint:
            print("Error: --checkpoint required for test mode")
            return
        from Trainer import test_model
        test_model(args.checkpoint, args.data_dir, device, args.participant_idx, args.num_slices)
        
    elif args.mode == 'progressive_test':
        if not args.checkpoint:
            print("Error: --checkpoint required for progressive_test mode")
            return
        test_model_progressive(args.checkpoint, args.data_dir, args.test_slices, 
                             device, args.participant_idx, args.num_slices)
        
    elif args.mode in ['train', 'progressive_train']:
        auto_config = get_model_config()
        
        config = {
            'device': device,
            'lr_g': args.lr_g,
            'lr_d': args.lr_d,
            'base_features': args.base_features if args.base_features > 0 else auto_config['base_features'],
            'target_slices': args.target_slices,
            'full_volume_slices': args.full_volume_slices,
            'target_size': args.target_size,
            'lambda_l1': 100.0,
            'lambda_gradient': 10.0
        }
        
        batch_size = 1
        
        print("Progressive Training Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        print(f"  batch_size: {batch_size}")
        print(f"  total_epochs: {args.epochs}")
        
        with open('progressive_training_config.json', 'w') as f:
            json.dump({**config, 'batch_size': batch_size, 'epochs': args.epochs}, f, indent=2)
        
        trainer = ProgressiveTrainer(config)
        
        if args.checkpoint and os.path.exists(args.checkpoint):
            trainer.load_model(args.checkpoint)
            print("Resumed from checkpoint")
        
        try:
            total_epochs = trainer.progressive_train(args.data_dir, args.epochs, batch_size,
                                                   args.start_slices, args.increment_ratio)
            
            print(f"\nTesting final progressive model with {args.full_volume_slices} slices...")
            test_model_progressive('final_model_progressive.pth', args.data_dir, 
                                 args.full_volume_slices, device)
            
        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving checkpoint...")
            trainer.save_model('interrupted_progressive_checkpoint')
        except Exception as e:
            print(f"Training failed: {e}")
            trainer.save_model('error_progressive_checkpoint')
            raise

if __name__ == "__main__":
    main()