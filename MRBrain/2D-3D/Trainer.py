#!/usr/bin/env python3
"""
Streamlined MRI Interslice GAN Trainer
Command-line interface for training MRI interpolation models
"""
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

# Import core classes from Main.py
from Main import (
    MRISliceDataset, Generator3D, Discriminator3D, GANLoss,
    clear_memory, get_memory_usage, get_model_config
)


class SimpleTrainer:
    """Simplified trainer for command-line usage"""
    
    def __init__(self, config):
        self.config = config
        self.device = config['device']
        
        print("Initializing Simple MRI GAN Trainer...")
        
        # Models
        self.generator = Generator3D(base_features=config['base_features']).to(self.device)
        self.discriminator = Discriminator3D(base_features=config['base_features']).to(self.device)
        
        # Optimizers
        self.g_optimizer = Adam(self.generator.parameters(), 
                               lr=config['lr_g'], betas=(0.5, 0.999))
        self.d_optimizer = Adam(self.discriminator.parameters(), 
                               lr=config['lr_d'], betas=(0.5, 0.999))
        
        # Loss
        self.criterion = GANLoss(
            lambda_l1=config.get('lambda_l1', 100.0),
            lambda_gradient=config.get('lambda_gradient', 10.0)
        )
        
        # Training history
        self.history = {'g_loss': [], 'd_loss': [], 'l1_loss': []}
        
        print(f"Generator params: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"Discriminator params: {sum(p.numel() for p in self.discriminator.parameters()):,}")
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        self.generator.train()
        self.discriminator.train()
        
        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_l1_loss = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}')
        
        for batch_idx, (sparse_input, real_target, _) in enumerate(pbar):
            batch_size = sparse_input.size(0)
            sparse_input = sparse_input.to(self.device)
            real_target = real_target.to(self.device)
            
            # Train Discriminator (every 3rd batch to prevent over-training)
            if batch_idx % 3 == 0:
                self.d_optimizer.zero_grad()
                
                # Real samples
                real_pred = self.discriminator(real_target)
                
                # Fake samples  
                with torch.no_grad():
                    fake_output = self.generator(sparse_input)
                fake_pred = self.discriminator(fake_output.detach())
                
                d_loss = self.criterion.discriminator_loss(real_pred, fake_pred)
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
                self.d_optimizer.step()
                
                epoch_d_loss += d_loss.item()
            
            # Train Generator
            self.g_optimizer.zero_grad()
            
            fake_output = self.generator(sparse_input)
            fake_pred = self.discriminator(fake_output)
            
            g_loss, adv_loss, l1_loss, grad_loss = self.criterion.generator_loss(
                fake_pred, fake_output, real_target)
            
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
            self.g_optimizer.step()
            
            epoch_g_loss += g_loss.item()
            epoch_l1_loss += l1_loss.item()
            
            # Update progress
            pbar.set_postfix({
                'G': f'{g_loss.item():.3f}',
                'D': f'{d_loss.item() if batch_idx % 3 == 0 else 0:.3f}',
                'L1': f'{l1_loss.item():.3f}'
            })
            
            if batch_idx % 10 == 0:
                clear_memory()
        
        # Calculate averages
        num_batches = len(dataloader)
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / max(num_batches // 3, 1)
        avg_l1_loss = epoch_l1_loss / num_batches
        
        return avg_g_loss, avg_d_loss, avg_l1_loss
    
    def train(self, data_dir, epochs, batch_size):
        """Main training loop"""
        # Create dataset
        dataset = MRISliceDataset(
            data_dir,
            target_slices=self.config['target_slices'],
            target_size=self.config['target_size'],
            cache_size=3
        )
        
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=True
        )
        
        print(f"Training on {len(dataset)} participants for {epochs} epochs")
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            g_loss, d_loss, l1_loss = self.train_epoch(dataloader, epoch)
            
            # Store history
            self.history['g_loss'].append(g_loss)
            self.history['d_loss'].append(d_loss)
            self.history['l1_loss'].append(l1_loss)
            
            # Print progress
            memory_info = get_memory_usage()
            print(f'Epoch {epoch+1}/{epochs}: G={g_loss:.4f}, D={d_loss:.4f}, L1={l1_loss:.4f}')
            print(f'GPU: {memory_info["gpu_allocated"]:.2f}GB')
            
            # Save best model
            if l1_loss < best_loss:
                best_loss = l1_loss
                self.save_model('best_model')
            
            # Regular checkpoints
            if (epoch + 1) % 10 == 0:
                self.save_model(f'checkpoint_epoch_{epoch+1}')
                self.save_training_plot(f'training_progress_epoch_{epoch+1}.png')
        
        # Final save
        self.save_model('final_model')
        self.save_training_plot('final_training_plot.png')
        print("Training completed!")
    
    def save_model(self, prefix):
        """Save model checkpoint"""
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'config': self.config,
            'history': self.history
        }, f'{prefix}.pth')
        print(f"Model saved: {prefix}.pth")
    
    def load_model(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        print(f"Model loaded from: {checkpoint_path}")
    
    def save_training_plot(self, filename):
        """Save training progress plot"""
        if not self.history['g_loss']:
            return
            
        epochs = range(1, len(self.history['g_loss']) + 1)
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.plot(epochs, self.history['g_loss'], 'b-', label='Generator')
        plt.plot(epochs, self.history['d_loss'], 'r-', label='Discriminator')
        plt.title('GAN Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.plot(epochs, self.history['l1_loss'], 'g-')
        plt.title('L1 Reconstruction Loss')
        plt.xlabel('Epoch')
        plt.ylabel('L1 Loss')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        if len(self.history['g_loss']) > 5:
            g_smooth = np.convolve(self.history['g_loss'], np.ones(5)/5, mode='valid')
            l1_smooth = np.convolve(self.history['l1_loss'], np.ones(5)/5, mode='valid')
            smooth_epochs = range(3, len(self.history['g_loss']) - 1)
            plt.plot(smooth_epochs, g_smooth, 'b-', label='G (smoothed)')
            plt.plot(smooth_epochs, l1_smooth, 'g-', label='L1 (smoothed)')
            plt.title('Smoothed Losses')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Training plot saved: {filename}")

def train_multistage(checkpoint_path, data_dir, device='cuda', epochs_per_stage=20):
    """
    Train model on different slice ranges sequentially
    
    Args:
        checkpoint_path: Path to initial model (e.g., 'final_model.pth')
        data_dir: Path to data directory
        device: Device to train on
        epochs_per_stage: Number of epochs for each stage
    """
    print("Starting multi-stage training for full volume coverage...")
    
    # Define slice ranges for 256 total slices, 64 slices each
    slice_ranges = [
        (0, 64),      # Stage 1: slices 0-63
        (64, 128),    # Stage 2: slices 64-127  
        (128, 192),   # Stage 3: slices 128-191
        (192, 256)    # Stage 4: slices 192-255
    ]
    
    # Load initial checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint {checkpoint_path} not found!")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config'].copy()
    
    print(f"Loaded base model from: {checkpoint_path}")
    print(f"Original config: target_slices={config['target_slices']}")
    
    # Train each stage
    for stage, (start_slice, end_slice) in enumerate(slice_ranges):
        print(f"\n{'='*50}")
        print(f"STAGE {stage + 1}: Training on slices {start_slice}-{end_slice-1}")
        print(f"{'='*50}")
        
        # Update config for this slice range
        stage_config = config.copy()
        stage_config['slice_start'] = start_slice
        stage_config['slice_end'] = end_slice
        stage_config['target_slices'] = 64  # Always 64 slices per stage
        
        # Create trainer for this stage
        trainer = SimpleTrainer(stage_config)
        
        # Load previous stage weights (or initial weights for stage 1)
        if stage == 0:
            # First stage: load original checkpoint
            trainer.load_model(checkpoint_path)
        else:
            # Later stages: load previous stage checkpoint
            prev_stage_path = f'stage_{stage}_final.pth'
            trainer.load_model(prev_stage_path)
        
        # Create dataset for this slice range
        dataset = MRISliceDatasetRanged(
            data_dir,
            slice_start=start_slice,
            slice_end=end_slice,
            target_slices=64,
            target_size=stage_config['target_size'],
            cache_size=3
        )
        
        dataloader = DataLoader(
            dataset, 
            batch_size=1,  # Reduced batch size for memory
            shuffle=True,
            num_workers=0, 
            pin_memory=True
        )
        
        print(f"Training on {len(dataset)} participants, slice range {start_slice}-{end_slice-1}")
        
        # Train this stage
        best_loss = float('inf')
        
        for epoch in range(epochs_per_stage):
            g_loss, d_loss, l1_loss = trainer.train_epoch(dataloader, epoch)
            
            # Store history
            trainer.history['g_loss'].append(g_loss)
            trainer.history['d_loss'].append(d_loss)
            trainer.history['l1_loss'].append(l1_loss)
            
            print(f'Stage {stage+1}, Epoch {epoch+1}/{epochs_per_stage}: G={g_loss:.4f}, D={d_loss:.4f}, L1={l1_loss:.4f}')
            
            # Save best model for this stage
            if l1_loss < best_loss:
                best_loss = l1_loss
                trainer.save_model(f'stage_{stage+1}_best')
        
        # Save final model for this stage
        trainer.save_model(f'stage_{stage+1}_final')
        trainer.save_training_plot(f'stage_{stage+1}_training.png')
        
        print(f"Stage {stage+1} completed. Best L1 loss: {best_loss:.6f}")
        
        # Clear memory
        clear_memory()
    
    print(f"\nMulti-stage training completed!")
    print("Saved models:")
    for stage in range(4):
        print(f"  stage_{stage+1}_best.pth")
        print(f"  stage_{stage+1}_final.pth")


class MRISliceDatasetRanged(MRISliceDataset):
    """Extended dataset that can load specific slice ranges"""
    
    def __init__(self, data_dir, slice_start=0, slice_end=256, **kwargs):
        self.slice_start = slice_start
        self.slice_end = slice_end
        self.slice_range_size = slice_end - slice_start
        
        super().__init__(data_dir, **kwargs)
        print(f"Dataset configured for slice range: {slice_start}-{slice_end-1} ({self.slice_range_size} slices)")
    
    def load_participant_data(self, participant_dir):
        """Load data for specific slice range"""
        try:
            # Load original volume
            volume_path = os.path.join(participant_dir, 'mri_volume.npy')
            if not os.path.exists(volume_path):
                return None, None
            
            volume = np.load(volume_path)  # Shape: (256, H, W)
            
            # Extract the specific slice range
            volume_range = volume[self.slice_start:self.slice_end]  # Shape: (64, H, W)
            
            # Resize if needed
            if volume_range.shape[1] != self.target_size or volume_range.shape[2] != self.target_size:
                resized_volume = np.zeros((self.slice_range_size, self.target_size, self.target_size))
                for i in range(self.slice_range_size):
                    resized_volume[i] = cv2.resize(
                        volume_range[i], 
                        (self.target_size, self.target_size), 
                        interpolation=cv2.INTER_LINEAR
                    )
                volume_range = resized_volume
            
            # Normalize
            volume_range = (volume_range - volume_range.min()) / (volume_range.max() - volume_range.min() + 1e-8)
            
            # Create sparse input (every ~5th slice from the range)
            sparse_indices = np.linspace(0, self.slice_range_size-1, 25, dtype=int)
            sparse_input = np.zeros_like(volume_range)
            for idx in sparse_indices:
                sparse_input[idx] = volume_range[idx]
            
            # Target is the full range
            target_volume = volume_range
            
            # Convert to torch tensors
            sparse_tensor = torch.FloatTensor(sparse_input).unsqueeze(0)  # Add channel dim
            target_tensor = torch.FloatTensor(target_volume).unsqueeze(0)
            
            return sparse_tensor, target_tensor
            
        except Exception as e:
            if self.debug:
                print(f"Error loading {participant_dir}: {e}")
            return None, None


def generate_slice_range(checkpoint_path, data_dir, slice_start=0, slice_end=64, device='cuda', participant_idx=0, fps=5):
    """
    Generate interpolations for a specific slice range using trained model
    
    Args:
        checkpoint_path: Path to trained model for this slice range
        data_dir: Path to data directory
        slice_start: Starting slice index (0-255)
        slice_end: Ending slice index (0-255)
        device: Device to run inference on
        participant_idx: Participant to test
        fps: Frame rate for GIFs
    """
    print(f"Generating interpolations for slice range {slice_start}-{slice_end-1}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Create model
    generator = Generator3D(base_features=config['base_features']).to(device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    
    # Create ranged dataset
    dataset = MRISliceDatasetRanged(
        data_dir,
        slice_start=slice_start,
        slice_end=slice_end,
        target_slices=slice_end - slice_start,
        target_size=config['target_size']
    )
    
    if participant_idx >= len(dataset):
        participant_idx = 0
        print(f"Invalid participant index, using 0")
    
    sparse_input, real_target, participant_id = dataset[participant_idx]
    
    # Generate output
    with torch.no_grad():
        sparse_input_gpu = sparse_input.unsqueeze(0).to(device)
        generated_output = generator(sparse_input_gpu)
        generated_output = generated_output.squeeze().cpu().numpy()
    
    sparse_np = sparse_input.squeeze().numpy()
    target_np = real_target.squeeze().numpy()
    
    # Calculate metrics
    mse = np.mean((generated_output - target_np) ** 2)
    psnr = 20 * np.log10(1.0 / np.sqrt(mse + 1e-8))
    
    print(f"Results for {participant_id}, slices {slice_start}-{slice_end-1}:")
    print(f"  MSE: {mse:.6f}")
    print(f"  PSNR: {psnr:.2f} dB")
    
    # Create GIFs for this slice range
    return create_range_gifs(
        sparse_np, generated_output, target_np, 
        participant_id, slice_start, slice_end, psnr, fps
    )


def create_range_gifs(sparse_np, generated_output, target_np, participant_id, slice_start, slice_end, psnr, fps=5):
    """Create GIFs for a specific slice range"""
    
    def normalize_for_display(img_slice):
        normalized = np.clip(img_slice, 0, 1)
        return (normalized * 255).astype(np.uint8)
    
    def add_text_overlay(img_array, text, position=(10, 10)):
        img_pil = Image.fromarray(img_array)
        draw = ImageDraw.Draw(img_pil)
        
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        draw.rectangle([position, (position[0] + len(text) * 10, position[1] + 20)], 
                      fill=(0, 0, 0))
        draw.text((position[0] + 2, position[1] + 2), text, fill=(255, 255, 255), font=font)
        
        return np.array(img_pil)
    
    # Find input slices
    input_frames = []
    generated_frames = []
    target_frames = []
    
    depth = sparse_np.shape[0]
    
    # Input GIF (only non-zero slices)
    for i in range(depth):
        if np.any(sparse_np[i] > 0.01):
            frame = normalize_for_display(sparse_np[i])
            frame_rgb = np.stack([frame, frame, frame], axis=-1)
            actual_slice_num = slice_start + i
            frame_with_text = add_text_overlay(
                frame_rgb, 
                f"Input Slice {actual_slice_num}"
            )
            input_frames.append(frame_with_text)
    
    # Generated and target GIFs (all slices in range)
    for i in range(depth):
        actual_slice_num = slice_start + i
        
        # Generated
        gen_frame = normalize_for_display(generated_output[i])
        gen_frame_rgb = np.stack([gen_frame, gen_frame, gen_frame], axis=-1)
        gen_frame_with_text = add_text_overlay(
            gen_frame_rgb,
            f"Generated Slice {actual_slice_num}"
        )
        generated_frames.append(gen_frame_with_text)
        
        # Target
        target_frame = normalize_for_display(target_np[i])
        target_frame_rgb = np.stack([target_frame, target_frame, target_frame], axis=-1)
        target_frame_with_text = add_text_overlay(
            target_frame_rgb,
            f"Target Slice {actual_slice_num}"
        )
        target_frames.append(target_frame_with_text)
    
    # Save GIFs
    duration = 1.0 / fps
    range_label = f"{slice_start}to{slice_end-1}"
    
    input_filename = f'input_range_{range_label}_{participant_id}.gif'
    generated_filename = f'generated_range_{range_label}_{participant_id}_PSNR{psnr:.1f}dB.gif'
    target_filename = f'target_range_{range_label}_{participant_id}.gif'
    
    imageio.mimsave(input_filename, input_frames, duration=duration, loop=0)
    imageio.mimsave(generated_filename, generated_frames, duration=duration, loop=0)
    imageio.mimsave(target_filename, target_frames, duration=duration, loop=0)
    
    print(f"Range GIFs saved:")
    print(f"  {input_filename} ({len(input_frames)} frames)")
    print(f"  {generated_filename} ({len(generated_frames)} frames)")
    print(f"  {target_filename} ({len(target_frames)} frames)")
    
    return {
        'input_gif': input_filename,
        'generated_gif': generated_filename, 
        'target_gif': target_filename,
        'psnr': psnr,
        'slice_range': (slice_start, slice_end)
    }

def test_model(checkpoint_path, data_dir, device='cuda', participant_idx=0, num_slices=3):
    """Test a trained model"""
    print(f"Testing model: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Create model
    generator = Generator3D(base_features=config['base_features']).to(device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    
    # Load test data
    dataset = MRISliceDataset(
        data_dir,
        target_slices=config['target_slices'],
        target_size=config['target_size']
    )
    
    if participant_idx >= len(dataset):
        participant_idx = 0
        print(f"Invalid participant index, using 0")
    
    sparse_input, real_target, participant_id = dataset[participant_idx]
    
    # Generate output
    with torch.no_grad():
        sparse_input_gpu = sparse_input.unsqueeze(0).to(device)
        generated_output = generator(sparse_input_gpu)
        generated_output = generated_output.squeeze().cpu().numpy()
    
    sparse_np = sparse_input.squeeze().numpy()
    target_np = real_target.squeeze().numpy()
    
    # Find original slice numbers for input data
    # These are the actual slice numbers from the original volume (0-255 range)
    original_slice_numbers = []
    depth = sparse_np.shape[0]
    
    for i in range(depth):
        if np.any(sparse_np[i] > 0.01):
            # Map back to original slice number (0-255 range)
            original_slice_num = int((i * 25) / (depth - 1))
            original_slice_numbers.append((i, original_slice_num))
    
    print(f"Found {len(original_slice_numbers)} input slices from original volume")
    if original_slice_numbers:
        print(f"Original slice range: {original_slice_numbers[0][1]} to {original_slice_numbers[-1][1]}")
    
    # Calculate metrics
    mse = np.mean((generated_output - target_np) ** 2)
    mae = np.mean(np.abs(generated_output - target_np))
    psnr = 20 * np.log10(1.0 / np.sqrt(mse + 1e-8))
    
    print(f"Test Results for {participant_id}:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  PSNR: {psnr:.2f} dB")
    
    # Create linearly spaced slice indices for visualization
    max_depth = depth - 1
    slice_indices = np.linspace(0, max_depth, num_slices, dtype=int)
    
    print(f"Visualizing {num_slices} slices at positions: {slice_indices}")
    
    # Create dynamic subplot layout
    fig, axes = plt.subplots(3, num_slices, figsize=(5*num_slices, 15))
    if num_slices == 1:
        axes = axes.reshape(3, 1)  # Handle single slice case
    
    fig.suptitle(f'Test Results - {participant_id}\nMSE: {mse:.6f}, PSNR: {psnr:.2f}dB\n'
                 f'Showing {num_slices} slices from 0 to {max_depth}', fontsize=14)
    
    for i, slice_idx in enumerate(slice_indices):
        # Map current slice to original volume slice number
        original_slice_for_display = int((slice_idx * 255) / max_depth)
        
        # Check if this slice has original input data
        has_input_data = np.any(sparse_np[slice_idx] > 0.01)
        actual_input_slice = None
        
        if has_input_data:
            # Find the closest original slice number for this position
            for vol_idx, orig_slice in original_slice_numbers:
                if vol_idx == slice_idx:
                    actual_input_slice = orig_slice
                    break
        
        # Input row
        axes[0, i].imshow(sparse_np[slice_idx], cmap='gray', vmin=0, vmax=1)
        if has_input_data and actual_input_slice is not None:
            axes[0, i].set_title(f'Input - Slice {slice_idx}\n(Original #{actual_input_slice})', fontsize=10)
        else:
            axes[0, i].set_title(f'Input - Slice {slice_idx}\n(Interpolated)', fontsize=10)
        axes[0, i].axis('off')
        
        # Generated row
        axes[1, i].imshow(generated_output[slice_idx], cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f'Generated - Slice {slice_idx}\n(Target #{original_slice_for_display})', fontsize=10)
        axes[1, i].axis('off')
        
        # Target row
        axes[2, i].imshow(target_np[slice_idx], cmap='gray', vmin=0, vmax=1)
        axes[2, i].set_title(f'Ground Truth - Slice {slice_idx}\n(Original #{original_slice_for_display})', fontsize=10)
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'test_results_{num_slices}slices.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Test visualization saved: test_results_{num_slices}slices.png")
    
    return {'mse': mse, 'mae': mae, 'psnr': psnr}

def create_mri_gifs(checkpoint_path, data_dir, device='cuda', participant_idx=0, fps=5):
    """
    Create GIFs showing all slices for input, generated, and ground truth data
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        data_dir: Path to data directory
        device: Device to run inference on
        participant_idx: Index of participant to visualize
        fps: Frames per second for the GIFs
    """
    print(f"Creating MRI GIFs from model: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Create model
    generator = Generator3D(base_features=config['base_features']).to(device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    
    # Load test data
    dataset = MRISliceDataset(
        data_dir,
        target_slices=config['target_slices'],
        target_size=config['target_size']
    )
    
    if participant_idx >= len(dataset):
        participant_idx = 0
        print(f"Invalid participant index, using 0")
    
    sparse_input, real_target, participant_id = dataset[participant_idx]
    
    # Generate output
    with torch.no_grad():
        sparse_input_gpu = sparse_input.unsqueeze(0).to(device)
        generated_output = generator(sparse_input_gpu)
        generated_output = generated_output.squeeze().cpu().numpy()
    
    sparse_np = sparse_input.squeeze().numpy()
    target_np = real_target.squeeze().numpy()
    
    print(f"Processing participant: {participant_id}")
    print(f"Data shapes - Input: {sparse_np.shape}, Generated: {generated_output.shape}, Target: {target_np.shape}")
    
    # Find which slices have actual input data (original 25 slices)
    input_slice_info = []
    depth = sparse_np.shape[0]
    
    for i in range(depth):
        if np.any(sparse_np[i] > 0.01):  # Non-zero slice
            original_slice_num = int((i * 255) / (depth - 1))  # Map to original slice number
            input_slice_info.append((i, original_slice_num))
    
    print(f"Found {len(input_slice_info)} input slices")
    
    # Helper function to normalize image for display
    def normalize_for_display(img_slice):
        """Normalize slice to 0-255 uint8 range"""
        normalized = np.clip(img_slice, 0, 1)  # Ensure 0-1 range
        return (normalized * 255).astype(np.uint8)
    
    # Helper function to add text overlay
    def add_text_overlay(img_array, text, position=(10, 10)):
        """Add text overlay to image array"""
        img_pil = Image.fromarray(img_array)
        draw = ImageDraw.Draw(img_pil)
        
        # Try to use a font, fallback to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
            except:
                font = ImageFont.load_default()
        
        # Add text with background for better visibility
        draw.rectangle([position, (position[0] + len(text) * 12, position[1] + 25)], 
                      fill=(0, 0, 0))
        draw.text((position[0] + 2, position[1] + 2), text, fill=(255, 255, 255), font=font)
        
        return np.array(img_pil)
    
    # Create frames for each type
    input_frames = []
    generated_frames = []
    target_frames = []
    
    print("Creating frames...")
    
    # Create input GIF (only non-zero slices)
    print(f"Processing {len(input_slice_info)} input frames...")
    for vol_idx, orig_slice_num in input_slice_info:
        frame = normalize_for_display(sparse_np[vol_idx])
        frame_rgb = np.stack([frame, frame, frame], axis=-1)  # Convert to RGB
        frame_with_text = add_text_overlay(
            frame_rgb, 
            f"Input Slice {vol_idx} (Orig #{orig_slice_num})"
        )
        input_frames.append(frame_with_text)
    
    # Create generated and target GIFs (all slices)
    print(f"Processing {depth} generated and target frames...")
    for i in range(depth):
        # Generated frame
        gen_frame = normalize_for_display(generated_output[i])
        gen_frame_rgb = np.stack([gen_frame, gen_frame, gen_frame], axis=-1)
        original_slice_for_display = int((i * 255) / (depth - 1))
        gen_frame_with_text = add_text_overlay(
            gen_frame_rgb,
            f"Generated Slice {i} (Target #{original_slice_for_display})"
        )
        generated_frames.append(gen_frame_with_text)
        
        # Target frame
        target_frame = normalize_for_display(target_np[i])
        target_frame_rgb = np.stack([target_frame, target_frame, target_frame], axis=-1)
        target_frame_with_text = add_text_overlay(
            target_frame_rgb,
            f"Ground Truth Slice {i} (Orig #{original_slice_for_display})"
        )
        target_frames.append(target_frame_with_text)
    
    # Calculate metrics for display in filenames
    mse = np.mean((generated_output - target_np) ** 2)
    psnr = 20 * np.log10(1.0 / np.sqrt(mse + 1e-8))
    
    # Save GIFs
    duration = 1.0 / fps
    
    # Input GIF (original 25 slices)
    input_filename = f'input_slices_{participant_id}_{len(input_frames)}frames.gif'
    imageio.mimsave(input_filename, input_frames, duration=duration, loop=0)
    print(f"Input GIF saved: {input_filename} ({len(input_frames)} frames)")
    
    # Generated GIF (all interpolated slices)
    generated_filename = f'generated_slices_{participant_id}_{depth}frames_PSNR{psnr:.1f}dB.gif'
    imageio.mimsave(generated_filename, generated_frames, duration=duration, loop=0)
    print(f"Generated GIF saved: {generated_filename} ({len(generated_frames)} frames)")
    
    # Ground truth GIF (all target slices)
    target_filename = f'target_slices_{participant_id}_{depth}frames.gif'
    imageio.mimsave(target_filename, target_frames, duration=duration, loop=0)
    print(f"Target GIF saved: {target_filename} ({len(target_frames)} frames)")
    
    # Create a side-by-side comparison GIF
    print("Creating comparison GIF...")
    comparison_frames = []
    
    for i in range(depth):
        # Create side-by-side layout: Generated | Target
        gen_frame = normalize_for_display(generated_output[i])
        target_frame = normalize_for_display(target_np[i])
        
        # Resize frames if needed and ensure same height
        height, width = gen_frame.shape
        
        # Create side-by-side image
        comparison = np.zeros((height, width * 2), dtype=np.uint8)
        comparison[:, :width] = gen_frame
        comparison[:, width:] = target_frame
        
        # Convert to RGB and add labels
        comparison_rgb = np.stack([comparison, comparison, comparison], axis=-1)
        
        # Add labels
        comparison_with_labels = add_text_overlay(comparison_rgb, "Generated", (10, 10))
        comparison_with_labels = add_text_overlay(comparison_with_labels, "Ground Truth", (width + 10, 10))
        
        # Add slice number at bottom
        slice_info = f"Slice {i}/{depth-1} - PSNR: {psnr:.1f}dB"
        comparison_with_labels = add_text_overlay(
            comparison_with_labels, 
            slice_info, 
            (10, height - 35)
        )
        
        comparison_frames.append(comparison_with_labels)
    
    # Save comparison GIF
    comparison_filename = f'comparison_{participant_id}_{depth}frames_PSNR{psnr:.1f}dB.gif'
    imageio.mimsave(comparison_filename, comparison_frames, duration=duration, loop=0)
    print(f"Comparison GIF saved: {comparison_filename} ({len(comparison_frames)} frames)")
    
    # Print summary
    print(f"\nGIF Creation Summary for {participant_id}:")
    print(f"  Input GIF: {len(input_frames)} frames from original {len(input_slice_info)} slices")
    print(f"  Generated GIF: {len(generated_frames)} frames (interpolated to {depth} slices)")
    print(f"  Target GIF: {len(target_frames)} frames (ground truth)")
    print(f"  Comparison GIF: {len(comparison_frames)} frames (side-by-side)")
    print(f"  Model Performance: MSE={mse:.6f}, PSNR={psnr:.2f}dB")
    print(f"  Frame rate: {fps} FPS")
    
    return {
        'input_gif': input_filename,
        'generated_gif': generated_filename,
        'target_gif': target_filename,
        'comparison_gif': comparison_filename,
        'metrics': {'mse': mse, 'psnr': psnr},
        'frame_counts': {
            'input': len(input_frames),
            'generated': len(generated_frames),
            'target': len(target_frames)
        }
    }

def explore_data(data_dir):
    """Explore dataset structure"""
    print("Exploring dataset...")
    
    dataset = MRISliceDataset(data_dir, target_slices=128, target_size=256, debug=True)
    
    print(f"Found {len(dataset)} participants")
    
    if len(dataset) > 0:
        sparse_input, real_target, participant_id = dataset[0]
        print(f"\nSample data (Participant: {participant_id}):")
        print(f"  Input shape: {sparse_input.shape}")
        print(f"  Target shape: {real_target.shape}")
        print(f"  Input range: [{sparse_input.min():.3f}, {sparse_input.max():.3f}]")
        print(f"  Target range: [{real_target.min():.3f}, {real_target.max():.3f}]")
        
        # Count non-zero slices
        sparse_np = sparse_input.squeeze().numpy()
        non_zero_slices = np.sum(np.any(sparse_np > 0.01, axis=(1, 2)))
        print(f"  Non-zero input slices: {non_zero_slices}/{sparse_np.shape[0]}")

def evaluate_all_participants(checkpoint_path, data_dir, device='cuda'):
    """
    Evaluate all participants to find the ones with highest and lowest PSNR
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        data_dir: Path to data directory
        device: Device to run inference on
    
    Returns:
        dict: Contains best and worst participant info with their metrics
    """
    print(f"Evaluating all participants with model: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Create model
    generator = Generator3D(base_features=config['base_features']).to(device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    
    # Load dataset
    dataset = MRISliceDataset(
        data_dir,
        target_slices=config['target_slices'],
        target_size=config['target_size']
    )
    
    print(f"Evaluating {len(dataset)} participants...")
    
    results = []
    
    # Evaluate each participant
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Evaluating participants"):
            try:
                sparse_input, real_target, participant_id = dataset[idx]
                
                # Generate output
                sparse_input_gpu = sparse_input.unsqueeze(0).to(device)
                generated_output = generator(sparse_input_gpu)
                generated_output_np = generated_output.squeeze().cpu().numpy()
                target_np = real_target.squeeze().numpy()
                
                # Calculate metrics
                mse = np.mean((generated_output_np - target_np) ** 2)
                mae = np.mean(np.abs(generated_output_np - target_np))
                psnr = 20 * np.log10(1.0 / np.sqrt(mse + 1e-8))
                
                # Store results
                results.append({
                    'participant_idx': idx,
                    'participant_id': participant_id,
                    'mse': mse,
                    'mae': mae,
                    'psnr': psnr
                })
                
                # Clear memory periodically
                if idx % 10 == 0:
                    clear_memory()
                    
            except Exception as e:
                print(f"Error processing participant {idx}: {e}")
                continue
    
    if not results:
        raise ValueError("No participants could be evaluated successfully")
    
    # Sort by PSNR
    results.sort(key=lambda x: x['psnr'])
    
    worst_participant = results[0]   # Lowest PSNR
    best_participant = results[-1]   # Highest PSNR
    
    # Calculate statistics
    psnr_values = [r['psnr'] for r in results]
    mean_psnr = np.mean(psnr_values)
    std_psnr = np.std(psnr_values)
    median_psnr = np.median(psnr_values)
    
    print(f"\nEvaluation Results ({len(results)} participants):")
    print(f"  Mean PSNR: {mean_psnr:.2f} ± {std_psnr:.2f} dB")
    print(f"  Median PSNR: {median_psnr:.2f} dB")
    print(f"  PSNR Range: {worst_participant['psnr']:.2f} - {best_participant['psnr']:.2f} dB")
    print(f"\nBest Performance:")
    print(f"  Participant: {best_participant['participant_id']} (index {best_participant['participant_idx']})")
    print(f"  PSNR: {best_participant['psnr']:.2f} dB")
    print(f"  MSE: {best_participant['mse']:.6f}")
    print(f"  MAE: {best_participant['mae']:.6f}")
    print(f"\nWorst Performance:")
    print(f"  Participant: {worst_participant['participant_id']} (index {worst_participant['participant_idx']})")
    print(f"  PSNR: {worst_participant['psnr']:.2f} dB")
    print(f"  MSE: {worst_participant['mse']:.6f}")
    print(f"  MAE: {worst_participant['mae']:.6f}")
    
    # Save detailed results
    
    return {
        'best': best_participant,
        'worst': worst_participant,
        'all_results': results,
        'summary': {
            'mean_psnr': mean_psnr,
            'std_psnr': std_psnr,
            'median_psnr': median_psnr
        }
    }


def create_best_worst_gifs(checkpoint_path, data_dir, device='cuda', fps=5):
    """
    Create GIFs for the participants with best and worst PSNR performance
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        data_dir: Path to data directory  
        device: Device to run inference on
        fps: Frames per second for GIFs
    """
    print("Finding best and worst performing participants...")
    
    # Evaluate all participants
    evaluation_results = evaluate_all_participants(checkpoint_path, data_dir, device)
    
    best_participant = evaluation_results['best']
    worst_participant = evaluation_results['worst']
    
    print(f"\nCreating GIFs for best and worst performers...")
    
    # Create GIFs for best performer
    print(f"\n=== Creating GIFs for BEST performer ===")
    print(f"Participant: {best_participant['participant_id']} (PSNR: {best_participant['psnr']:.2f} dB)")
    best_gifs = create_mri_gifs(
        checkpoint_path, 
        data_dir, 
        device, 
        best_participant['participant_idx'], 
        fps
    )
    
    # Rename files to include "best" prefix
    best_files = {}
    for gif_type, filename in best_gifs.items():
        if gif_type.endswith('_gif'):
            new_filename = f"BEST_{filename}"
            os.rename(filename, new_filename)
            best_files[gif_type] = new_filename
            print(f"  Renamed: {filename} -> {new_filename}")
    
    # Create GIFs for worst performer  
    print(f"\n=== Creating GIFs for WORST performer ===")
    print(f"Participant: {worst_participant['participant_id']} (PSNR: {worst_participant['psnr']:.2f} dB)")
    worst_gifs = create_mri_gifs(
        checkpoint_path, 
        data_dir, 
        device, 
        worst_participant['participant_idx'], 
        fps
    )
    
    # Rename files to include "worst" prefix
    worst_files = {}
    for gif_type, filename in worst_gifs.items():
        if gif_type.endswith('_gif'):
            new_filename = f"WORST_{filename}"
            os.rename(filename, new_filename)
            worst_files[gif_type] = new_filename
            print(f"  Renamed: {filename} -> {new_filename}")
    
    # Create summary comparison plot
    create_performance_comparison_plot(evaluation_results, best_participant, worst_participant)
    
    print(f"\n=== Summary ===")
    print(f"Best Performer: {best_participant['participant_id']} (PSNR: {best_participant['psnr']:.2f} dB)")
    print(f"  Files: BEST_input_*, BEST_generated_*, BEST_target_*, BEST_comparison_*")
    print(f"Worst Performer: {worst_participant['participant_id']} (PSNR: {worst_participant['psnr']:.2f} dB)")
    print(f"  Files: WORST_input_*, WORST_generated_*, WORST_target_*, WORST_comparison_*")
    print(f"Performance comparison plot: performance_comparison.png")
    
    return {
        'best': {
            'participant': best_participant,
            'gifs': best_files
        },
        'worst': {
            'participant': worst_participant, 
            'gifs': worst_files
        },
        'evaluation_summary': evaluation_results['summary']
    }


def create_performance_comparison_plot(evaluation_results, best_participant, worst_participant):
    """Create a plot showing performance distribution and highlighting best/worst"""
    
    all_results = evaluation_results['all_results']
    psnr_values = [r['psnr'] for r in all_results]
    
    plt.figure(figsize=(15, 5))
    
    # Histogram of PSNR values
    plt.subplot(1, 3, 1)
    plt.hist(psnr_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(best_participant['psnr'], color='green', linestyle='--', linewidth=2, 
                label=f'Best: {best_participant["psnr"]:.1f} dB')
    plt.axvline(worst_participant['psnr'], color='red', linestyle='--', linewidth=2,
                label=f'Worst: {worst_participant["psnr"]:.1f} dB')
    plt.axvline(evaluation_results['summary']['mean_psnr'], color='orange', linestyle='-', linewidth=2,
                label=f'Mean: {evaluation_results["summary"]["mean_psnr"]:.1f} dB')
    plt.xlabel('PSNR (dB)')
    plt.ylabel('Number of Participants')
    plt.title('PSNR Distribution Across All Participants')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Box plot
    plt.subplot(1, 3, 2)
    plt.boxplot(psnr_values, vert=True)
    plt.scatter([1], [best_participant['psnr']], color='green', s=100, marker='o', 
                label='Best', zorder=5)
    plt.scatter([1], [worst_participant['psnr']], color='red', s=100, marker='o',
                label='Worst', zorder=5)
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR Distribution Summary')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Performance ranking plot
    plt.subplot(1, 3, 3)
    sorted_psnr = sorted(psnr_values)
    participant_ranks = range(1, len(sorted_psnr) + 1)
    plt.plot(participant_ranks, sorted_psnr, 'b-', alpha=0.7, linewidth=2)
    
    # Highlight best and worst
    best_rank = sorted_psnr.index(best_participant['psnr']) + 1
    worst_rank = sorted_psnr.index(worst_participant['psnr']) + 1
    
    plt.scatter([best_rank], [best_participant['psnr']], color='green', s=100, 
                marker='o', label=f'Best (Rank {best_rank})', zorder=5)
    plt.scatter([worst_rank], [worst_participant['psnr']], color='red', s=100,
                marker='o', label=f'Worst (Rank {worst_rank})', zorder=5)
    
    plt.xlabel('Participant Rank')
    plt.ylabel('PSNR (dB)')
    plt.title('Performance Ranking')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Performance comparison plot saved: performance_comparison.png")

def main():
    parser = argparse.ArgumentParser(description='Simple MRI GAN Trainer')
    parser.add_argument('--data_dir', type=str, required=True, 
                       help='Path to data directory')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'explore', 'gif', 'best_worst', 'multistage', 'range_gen'], 
                       default='train', help='Mode: train, test, explore, gif, best_worst, multistage, or range_gen')
    parser.add_argument('--epochs', type=int, default=50, 
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=0, 
                       help='Batch size (0 for auto)')
    parser.add_argument('--lr_g', type=float, default=2e-4, 
                       help='Generator learning rate')
    parser.add_argument('--lr_d', type=float, default=1e-4, 
                       help='Discriminator learning rate')
    parser.add_argument('--base_features', type=int, default=0, 
                       help='Base features (0 for auto)')
    parser.add_argument('--target_slices', type=int, default=128, 
                       help='Target number of slices')
    parser.add_argument('--target_size', type=int, default=256, 
                       help='Target image size')
    parser.add_argument('--device', type=str, default='auto', 
                       help='Device (auto/cuda/cpu)')
    parser.add_argument('--checkpoint', type=str, default='', 
                       help='Checkpoint path for testing or GIF creation')
    parser.add_argument('--participant_idx', type=int, default=0, 
                       help='Participant index for testing/GIF creation')
    parser.add_argument('--num_slices', type=int, default=3, 
                        help='Number of slices to visualize in test mode')
    parser.add_argument('--fps', type=int, default=5, 
                        help='Frames per second for GIF creation')
    parser.add_argument('--slice_start', type=int, default=0, 
                        help='Starting slice index for range generation (0-255)')
    parser.add_argument('--slice_end', type=int, default=64, 
                        help='Ending slice index for range generation (1-256)')
    parser.add_argument('--epochs_per_stage', type=int, default=20, 
                        help='Epochs per stage for multistage training')
    
    args = parser.parse_args()
    
    # Device setup
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    if device == 'cuda' and torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    
    # Mode selection
    if args.mode == 'explore':
        explore_data(args.data_dir)
        
    elif args.mode == 'test':
        if not args.checkpoint:
            print("Error: --checkpoint required for test mode")
            return
        test_model(args.checkpoint, args.data_dir, device, args.participant_idx, args.num_slices)
        
    elif args.mode == 'gif':
        if not args.checkpoint:
            print("Error: --checkpoint required for GIF mode")
            return
        create_mri_gifs(args.checkpoint, args.data_dir, device, args.participant_idx, args.fps)
        
    elif args.mode == 'best_worst':
        if not args.checkpoint:
            print("Error: --checkpoint required for best_worst mode")
            return
        create_best_worst_gifs(args.checkpoint, args.data_dir, device, args.fps)
        
    elif args.mode == 'multistage':
        if not args.checkpoint:
            print("Error: --checkpoint required for multistage mode (base model)")
            return
        train_multistage(args.checkpoint, args.data_dir, device, args.epochs_per_stage)
        
    elif args.mode == 'range_gen':
        if not args.checkpoint:
            print("Error: --checkpoint required for range_gen mode")
            return
        if args.slice_end <= args.slice_start:
            print("Error: slice_end must be greater than slice_start")
            return
        generate_slice_range(args.checkpoint, args.data_dir, args.slice_start, args.slice_end, 
                           device, args.participant_idx, args.fps)
        
    elif args.mode == 'train':
        # Get optimal config
        auto_config = get_model_config()
        
        # Create training config
        config = {
            'device': device,
            'lr_g': args.lr_g,
            'lr_d': args.lr_d,
            'base_features': args.base_features if args.base_features > 0 else auto_config['base_features'],
            'target_slices': args.target_slices,
            'target_size': args.target_size,
            'lambda_l1': 100.0,
            'lambda_gradient': 10.0
        }
        
        batch_size = args.batch_size if args.batch_size > 0 else auto_config['batch_size']
        
        print("Training Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        print(f"  batch_size: {batch_size}")
        print(f"  epochs: {args.epochs}")
        
        # Save config
        with open('training_config.json', 'w') as f:
            json.dump({**config, 'batch_size': batch_size, 'epochs': args.epochs}, f, indent=2)
        
        # Create trainer
        trainer = SimpleTrainer(config)
        
        # Resume from checkpoint if provided
        if args.checkpoint and os.path.exists(args.checkpoint):
            trainer.load_model(args.checkpoint)
            print("Resumed from checkpoint")
        
        # Train
        try:
            trainer.train(args.data_dir, args.epochs, batch_size)
            
            # Test final model
            print("\nTesting final model...")
            test_model('final_model.pth', args.data_dir, device)
            
        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving checkpoint...")
            trainer.save_model('interrupted_checkpoint')
        except Exception as e:
            print(f"Training failed: {e}")
            trainer.save_model('error_checkpoint')
            raise
if __name__ == "__main__":
    main()