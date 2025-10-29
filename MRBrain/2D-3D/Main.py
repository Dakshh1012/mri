import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import glob
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict
import random
from scipy.ndimage import zoom
import gc
import psutil
import argparse
import json
from tqdm import tqdm
from torch.utils.checkpoint import checkpoint

# Memory management utilities
def clear_memory():
    """Clear GPU and CPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def get_memory_usage():
    """Get current memory usage"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3
        gpu_reserved = torch.cuda.memory_reserved() / 1024**3
    else:
        gpu_memory, gpu_reserved = 0, 0
    
    cpu_memory = psutil.virtual_memory().used / 1024**3
    cpu_percent = psutil.virtual_memory().percent
    
    return {
        'gpu_allocated': gpu_memory,
        'gpu_reserved': gpu_reserved,
        'cpu_used': cpu_memory,
        'cpu_percent': cpu_percent
    }

class MRISliceDataset(Dataset):
    """Clean MRI dataset for interslice interpolation"""
    
    def __init__(self, data_dir: str, target_slices: int = 128, target_size: int = 256, 
                 cache_size: int = 3, debug: bool = False):
        self.data_dir = data_dir
        self.target_slices = target_slices
        self.target_size = target_size
        self.debug = debug
        
        # Find all participants
        self.participants = self._get_participants()
        
        # Simple cache system
        self.cache = {}
        self.cache_order = []
        self.max_cache_size = cache_size
        
        print(f"Found {len(self.participants)} participants")
        print(f"Target volume: {target_slices} slices of {target_size}x{target_size}")
        
        # Debug: Check first participant
        if self.debug and len(self.participants) > 0:
            self._debug_participant(self.participants[0])
        
    def _get_participants(self):
        """Get unique participant IDs"""
        files = glob.glob(os.path.join(self.data_dir, "*.npy"))
        participants = set()
        
        for file in files:
            basename = os.path.basename(file)
            # Extract participant ID (everything before .nii_slice)
            participant_id = basename.split('.nii_slice')[0]
            participants.add(participant_id)
            
        return list(participants)
    
    def _debug_participant(self, participant_id: str):
        """Debug function to check data loading"""
        lr_files = glob.glob(os.path.join(self.data_dir, f"{participant_id}.nii_*_lr.npy"))
        hr_files = glob.glob(os.path.join(self.data_dir, f"{participant_id}.nii_*_hr.npy"))
        
        print(f"\n=== Debug Participant: {participant_id} ===")
        print(f"LR files found: {len(lr_files)}")
        print(f"HR files found: {len(hr_files)}")
        
        if lr_files:
            sample_lr = np.load(lr_files[0])
            print(f"LR slice shape: {sample_lr.shape}")
            print(f"LR slice range: [{sample_lr.min():.3f}, {sample_lr.max():.3f}]")
        
        if hr_files:
            sample_hr = np.load(hr_files[0])
            print(f"HR slice shape: {sample_hr.shape}")
            print(f"HR slice range: [{sample_hr.min():.3f}, {sample_hr.max():.3f}]")
    
    def _load_participant_data(self, participant_id: str):
        """Load all data for a participant"""
        # Check cache first
        if participant_id in self.cache:
            self.cache_order.remove(participant_id)
            self.cache_order.append(participant_id)
            return self.cache[participant_id]
        
        # Load LR slices (should be ~25-26 slices)
        lr_files = glob.glob(os.path.join(self.data_dir, f"{participant_id}.nii_*_lr.npy"))
        lr_slices = {}
        
        for file in lr_files:
            basename = os.path.basename(file)
            slice_num = int(basename.split('slice')[1].split('_')[0])
            lr_data = np.load(file).squeeze().astype(np.float32)
            lr_slices[slice_num] = lr_data
        
        # Load HR slices (should be ~255-256 slices)
        hr_files = glob.glob(os.path.join(self.data_dir, f"{participant_id}.nii_*_hr.npy"))
        hr_slices = {}
        
        for file in hr_files:
            basename = os.path.basename(file)
            slice_num = int(basename.split('slice')[1].split('_')[0])
            hr_data = np.load(file).squeeze().astype(np.float32)
            hr_slices[slice_num] = hr_data
        
        # Cache the data
        data = (lr_slices, hr_slices)
        self.cache[participant_id] = data
        self.cache_order.append(participant_id)
        
        # Manage cache size
        while len(self.cache) > self.max_cache_size:
            oldest = self.cache_order.pop(0)
            del self.cache[oldest]
            clear_memory()
        
        return data
    
    def _create_volume(self, slices_dict: dict, target_depth: int):
        """Create 3D volume from slice dictionary"""
        volume = np.zeros((target_depth, self.target_size, self.target_size), dtype=np.float32)
        
        if not slices_dict:
            return volume
        
        # Get sorted slice indices
        slice_indices = sorted(slices_dict.keys())
        max_slice_idx = max(slice_indices)
        
        for slice_idx in slice_indices:
            slice_data = slices_dict[slice_idx]
            
            # Map original slice index to target volume index
            mapped_idx = int((slice_idx * (target_depth - 1)) / max_slice_idx)
            mapped_idx = min(mapped_idx, target_depth - 1)
            
            # Resize slice if needed
            if slice_data.shape != (self.target_size, self.target_size):
                zoom_factors = (self.target_size / slice_data.shape[0], 
                              self.target_size / slice_data.shape[1])
                slice_data = zoom(slice_data, zoom_factors, order=1)
            
            volume[mapped_idx] = slice_data
        
        return volume
    
    def _create_sparse_input(self, lr_slices: dict, target_depth: int):
        """Create sparse input volume from LR slices with better interpolation"""
        sparse_volume = np.zeros((target_depth, self.target_size, self.target_size), dtype=np.float32)
        placed_indices = []
        
        if not lr_slices:
            return sparse_volume, placed_indices
        
        # Place LR slices in sparse volume
        slice_indices = sorted(lr_slices.keys())
        max_slice_idx = max(slice_indices)
        
        for slice_idx in slice_indices:
            slice_data = lr_slices[slice_idx]
            
            # Map to target volume
            mapped_idx = int((slice_idx * (target_depth - 1)) / max_slice_idx)
            mapped_idx = min(mapped_idx, target_depth - 1)
            
            # Resize if needed
            if slice_data.shape != (self.target_size, self.target_size):
                zoom_factors = (self.target_size / slice_data.shape[0], 
                              self.target_size / slice_data.shape[1])
                slice_data = zoom(slice_data, zoom_factors, order=1)
            
            sparse_volume[mapped_idx] = slice_data
            placed_indices.append(mapped_idx)
        
        # Add stronger linear interpolation between known slices
        self._add_linear_interpolation(sparse_volume, placed_indices)
        
        return sparse_volume, placed_indices
    
    def _add_linear_interpolation(self, volume: np.ndarray, known_indices: List[int]):
        """Add linear interpolation between known slices"""
        if len(known_indices) < 2:
            return
        
        known_indices = sorted(list(set(known_indices)))  # Remove duplicates and sort
        
        for i in range(len(known_indices) - 1):
            start_idx = known_indices[i]
            end_idx = known_indices[i + 1]
            
            if end_idx - start_idx > 1:
                start_slice = volume[start_idx]
                end_slice = volume[end_idx]
                
                for j in range(start_idx + 1, end_idx):
                    alpha = (j - start_idx) / (end_idx - start_idx)
                    interpolated = (1 - alpha) * start_slice + alpha * end_slice
                    volume[j] = interpolated * 0.5  # Stronger interpolation
    
    def _normalize_volume(self, volume):
        """Normalize volume to [0, 1]"""
        vol_min, vol_max = volume.min(), volume.max()
        if vol_max > vol_min:
            return (volume - vol_min) / (vol_max - vol_min)
        return np.zeros_like(volume)
    
    def __len__(self):
        return len(self.participants)
    
    def __getitem__(self, idx):
        participant_id = self.participants[idx]
        lr_slices, hr_slices = self._load_participant_data(participant_id)
        
        # Create HR target volume
        hr_volume = self._create_volume(hr_slices, self.target_slices)
        hr_volume = self._normalize_volume(hr_volume)
        
        # Create sparse input from LR slices
        sparse_volume, lr_indices = self._create_sparse_input(lr_slices, self.target_slices)
        sparse_volume = self._normalize_volume(sparse_volume)
        
        # Convert to tensors
        sparse_tensor = torch.from_numpy(sparse_volume).unsqueeze(0).float()  # [1, D, H, W]
        hr_tensor = torch.from_numpy(hr_volume).unsqueeze(0).float()  # [1, D, H, W]
        
        return sparse_tensor, hr_tensor, participant_id

class ResidualBlock3D(nn.Module):
    """3D Residual Block"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class Generator3D(nn.Module):
    """Enhanced 3D UNet Generator with proper skip connections"""
    
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_features: int = 32):
        super().__init__()
        
        # Encoder blocks
        self.enc1 = self._make_encoder_block(in_channels, base_features)
        self.enc2 = self._make_encoder_block(base_features, base_features * 2)
        self.enc3 = self._make_encoder_block(base_features * 2, base_features * 4)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(base_features * 4, base_features * 8, 3, padding=1),
            nn.BatchNorm3d(base_features * 8),
            nn.ReLU(inplace=True),
            ResidualBlock3D(base_features * 8),
            ResidualBlock3D(base_features * 8),
        )
        
        # Decoder blocks with proper input channel calculation
        # dec3: bottleneck (base*8) + enc2 skip (base*2) = base*10 -> base*4
        self.dec3 = self._make_decoder_block(base_features * 8 + base_features * 2, base_features * 4)
        
        # dec2: dec3 output (base*4) + enc1 skip (base*1) = base*5 -> base*2  
        self.dec2 = self._make_decoder_block(base_features * 4 + base_features, base_features * 2)
        
        # dec1: dec2 output (base*2) + input skip (1) = base*2+1 -> base*1
        self.dec1 = self._make_decoder_block(base_features * 2 + in_channels, base_features)
        
        # Final output
        self.final = nn.Sequential(
            nn.Conv3d(base_features, out_channels, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _make_encoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            ResidualBlock3D(out_ch),
            nn.MaxPool3d(2)
        )
    
    def _make_decoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            ResidualBlock3D(out_ch)
        )
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Encoder path - store feature maps for skip connections
        e1 = self.enc1(x)    # Shape: [B, base_features, D/2, H/2, W/2]
        e2 = self.enc2(e1)   # Shape: [B, base_features*2, D/4, H/4, W/4]  
        e3 = self.enc3(e2)   # Shape: [B, base_features*4, D/8, H/8, W/8]
        
        # Bottleneck
        b = self.bottleneck(e3)  # Shape: [B, base_features*8, D/8, H/8, W/8]
        
        # Decoder path with skip connections
        # Step 1: Upsample bottleneck to match e2 size and concatenate
        b_up = F.interpolate(b, size=e2.shape[2:], mode='trilinear', align_corners=False)
        d3_input = torch.cat([b_up, e2], dim=1)  # [B, base*8 + base*2, D/4, H/4, W/4]
        d3 = self.dec3(d3_input)  # [B, base*4, D/4, H/4, W/4]
        
        # Step 2: Upsample d3 to match e1 size and concatenate
        d3_up = F.interpolate(d3, size=e1.shape[2:], mode='trilinear', align_corners=False)
        d2_input = torch.cat([d3_up, e1], dim=1)  # [B, base*4 + base*1, D/2, H/2, W/2]
        d2 = self.dec2(d2_input)  # [B, base*2, D/2, H/2, W/2]
        
        # Step 3: Upsample d2 to match input size and concatenate
        d2_up = F.interpolate(d2, size=x.shape[2:], mode='trilinear', align_corners=False)
        d1_input = torch.cat([d2_up, x], dim=1)  # [B, base*2 + 1, D, H, W]
        d1 = self.dec1(d1_input)  # [B, base*1, D, H, W]
        
        # Final output with residual connection
        output = self.final(d1)  # [B, 1, D, H, W]
        
        # Add weak residual connection from input
        output = output + 0.1 * x
        
        return torch.clamp(output, 0, 1)

class Discriminator3D(nn.Module):
    """3D Discriminator with progressive difficulty"""
    
    def __init__(self, in_channels: int = 1, base_features: int = 32):
        super().__init__()
        
        # Progressive layers with dropout
        self.layers = nn.Sequential(
            # First layer - no batch norm
            nn.Conv3d(in_channels, base_features, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Gradually increase difficulty
            nn.Conv3d(base_features, base_features * 2, 4, stride=2, padding=1),
            nn.BatchNorm3d(base_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(0.3),
            
            nn.Conv3d(base_features * 2, base_features * 4, 4, stride=2, padding=1),
            nn.BatchNorm3d(base_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(0.4),
            
            nn.Conv3d(base_features * 4, base_features * 8, 4, stride=2, padding=1),
            nn.BatchNorm3d(base_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(0.5),
            
            # Final classification
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(base_features * 8, 1)
        )
        
        # Add noise for regularization
        self.noise_std = 0.05
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Add noise during training for regularization
        if self.training:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
            
        return self.layers(x)

class GANLoss(nn.Module):
    """Comprehensive GAN Loss"""
    
    def __init__(self, lambda_l1: float = 100.0, lambda_gradient: float = 10.0, lambda_perceptual: float = 1.0):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_gradient = lambda_gradient
        self.lambda_perceptual = lambda_perceptual
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
    def gradient_loss(self, pred, target):
        """Gradient loss along all dimensions"""
        # Depth gradient
        pred_grad_d = pred[:, :, 1:] - pred[:, :, :-1]
        target_grad_d = target[:, :, 1:] - target[:, :, :-1]
        grad_loss_d = self.l1_loss(pred_grad_d, target_grad_d)
        
        # Height gradient
        pred_grad_h = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        target_grad_h = target[:, :, :, 1:] - target[:, :, :, :-1]
        grad_loss_h = self.l1_loss(pred_grad_h, target_grad_h)
        
        # Width gradient
        pred_grad_w = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
        target_grad_w = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]
        grad_loss_w = self.l1_loss(pred_grad_w, target_grad_w)
        
        return (grad_loss_d + grad_loss_h + grad_loss_w) / 3.0
    
    def perceptual_loss(self, pred, target):
        """Simple perceptual loss using MSE on patches"""
        # Use MSE as a simple perceptual loss
        return self.mse_loss(pred, target)
    
    def generator_loss(self, fake_pred, fake_output, real_target):
        # Adversarial loss
        real_labels = torch.ones_like(fake_pred)
        adv_loss = self.bce_loss(fake_pred, real_labels)
        
        # Reconstruction losses
        l1_loss = self.l1_loss(fake_output, real_target)
        grad_loss = self.gradient_loss(fake_output, real_target)
        perceptual_loss = self.perceptual_loss(fake_output, real_target)
        
        total_loss = (adv_loss + 
                     self.lambda_l1 * l1_loss + 
                     self.lambda_gradient * grad_loss +
                     self.lambda_perceptual * perceptual_loss)
        
        return total_loss, adv_loss, l1_loss, grad_loss
    
    def discriminator_loss(self, real_pred, fake_pred):
        real_labels = torch.ones_like(real_pred)
        fake_labels = torch.zeros_like(fake_pred)
        
        real_loss = self.bce_loss(real_pred, real_labels)
        fake_loss = self.bce_loss(fake_pred, fake_labels)
        
        return (real_loss + fake_loss) * 0.5

class MRIIntersliceTrainer:
    """Main training class"""
    
    def __init__(self, data_dir: str, device: str = 'cuda', 
                 lr_g: float = 2e-4, lr_d: float = 1e-4,
                 base_features: int = 32, target_slices: int = 128, target_size: int = 256):
        
        self.device = device
        self.data_dir = data_dir
        
        print("Initializing MRI Interslice GAN...")
        
        # Models
        self.generator = Generator3D(base_features=base_features).to(device)
        self.discriminator = Discriminator3D(base_features=base_features).to(device)
        
        # Optimizers with different learning rates
        self.g_optimizer = Adam(self.generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
        self.d_optimizer = Adam(self.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
        
        # Schedulers
        self.g_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.g_optimizer, patience=5, factor=0.5, verbose=True)
        self.d_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.d_optimizer, patience=5, factor=0.5, verbose=True)
        
        # Loss function
        self.criterion = GANLoss()
        
        # Training parameters
        self.target_slices = target_slices
        self.target_size = target_size
        
        # History tracking
        self.history = {
            'g_loss': [], 'd_loss': [], 'adv_loss': [], 'l1_loss': [], 'grad_loss': []
        }
        
        print(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}")
        
        # Print model architecture for debugging
        self._print_model_info()
        
    def _print_model_info(self):
        """Print model architecture information for debugging"""
        print("\n=== Generator Architecture Debug ===")
        dummy_input = torch.randn(1, 1, 128, 256, 256).to(self.device)
        
        with torch.no_grad():
            x = dummy_input
            print(f"Input shape: {x.shape}")
            
            # Encoder
            e1 = self.generator.enc1(x)
            print(f"After enc1: {e1.shape}")
            
            e2 = self.generator.enc2(e1)
            print(f"After enc2: {e2.shape}")
            
            e3 = self.generator.enc3(e2)
            print(f"After enc3: {e3.shape}")
            
            # Bottleneck
            b = self.generator.bottleneck(e3)
            print(f"After bottleneck: {b.shape}")
            
            # Decoder
            b_up = F.interpolate(b, size=e2.shape[2:], mode='trilinear', align_corners=False)
            print(f"Bottleneck upsampled: {b_up.shape}")
            d3_input = torch.cat([b_up, e2], dim=1)
            print(f"d3_input (b_up + e2): {d3_input.shape}")
            
            d3 = self.generator.dec3(d3_input)
            print(f"After dec3: {d3.shape}")
            
            d3_up = F.interpolate(d3, size=e1.shape[2:], mode='trilinear', align_corners=False)
            print(f"d3 upsampled: {d3_up.shape}")
            d2_input = torch.cat([d3_up, e1], dim=1)
            print(f"d2_input (d3_up + e1): {d2_input.shape}")
            
            d2 = self.generator.dec2(d2_input)
            print(f"After dec2: {d2.shape}")
            
            d2_up = F.interpolate(d2, size=x.shape[2:], mode='trilinear', align_corners=False)
            print(f"d2 upsampled: {d2_up.shape}")
            d1_input = torch.cat([d2_up, x], dim=1)
            print(f"d1_input (d2_up + x): {d1_input.shape}")
            
            d1 = self.generator.dec1(d1_input)
            print(f"After dec1: {d1.shape}")
            
            output = self.generator.final(d1)
            print(f"Final output: {output.shape}")
        
        print("=== Architecture Debug Complete ===\n")
    
    def train_epoch(self, dataloader, epoch):
        self.generator.train()
        self.discriminator.train()
        
        epoch_losses = {'g_loss': 0, 'd_loss': 0, 'adv_loss': 0, 'l1_loss': 0, 'grad_loss': 0}
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}')
        
        for batch_idx, (sparse_input, real_target, participant_ids) in enumerate(pbar):
            batch_size = sparse_input.size(0)
            
            sparse_input = sparse_input.to(self.device)
            real_target = real_target.to(self.device)
            
            # Train Discriminator (less frequently to prevent convergence)
            if batch_idx % 3 == 0:  # Train discriminator every 3rd batch
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
                
                epoch_losses['d_loss'] += d_loss.item()
            
            # Train Generator
            self.g_optimizer.zero_grad()
            
            fake_output = self.generator(sparse_input)
            fake_pred = self.discriminator(fake_output)
            
            g_loss, adv_loss, l1_loss, grad_loss = self.criterion.generator_loss(
                fake_pred, fake_output, real_target)
            
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
            self.g_optimizer.step()
            
            # Update losses
            epoch_losses['g_loss'] += g_loss.item()
            epoch_losses['adv_loss'] += adv_loss.item()
            epoch_losses['l1_loss'] += l1_loss.item()
            epoch_losses['grad_loss'] += grad_loss.item()
            
            # Update progress bar
            d_batches = (batch_idx // 3) + (1 if batch_idx % 3 == 0 else 0)
            current_d_loss = epoch_losses['d_loss'] / max(d_batches, 1)
            current_g_loss = epoch_losses['g_loss'] / (batch_idx + 1)
            
            pbar.set_postfix({
                'G': f'{current_g_loss:.3f}',
                'D': f'{current_d_loss:.3f}',
                'L1': f'{epoch_losses["l1_loss"]/(batch_idx+1):.3f}'
            })
            
            # Memory cleanup
            if batch_idx % 5 == 0:
                clear_memory()
        
        # Calculate epoch averages
        num_batches = len(dataloader)
        avg_g_loss = epoch_losses['g_loss'] / num_batches
        d_batches = (num_batches // 3) + (1 if num_batches % 3 != 0 else 0)
        avg_d_loss = epoch_losses['d_loss'] / max(d_batches, 1)
        
        # Update schedulers
        self.g_scheduler.step(avg_g_loss)
        self.d_scheduler.step(avg_d_loss)
        
        # Store history
        self.history['g_loss'].append(avg_g_loss)
        self.history['d_loss'].append(avg_d_loss)
        self.history['adv_loss'].append(epoch_losses['adv_loss'] / num_batches)
        self.history['l1_loss'].append(epoch_losses['l1_loss'] / num_batches)
        self.history['grad_loss'].append(epoch_losses['grad_loss'] / num_batches)
        
        return avg_g_loss, avg_d_loss
    
    def train(self, epochs: int = 50, batch_size: int = 1, save_every: int = 10):
        # Create dataset
        dataset = MRISliceDataset(
            self.data_dir, 
            target_slices=self.target_slices, 
            target_size=self.target_size,
            debug=True  # Enable debug for first run
        )
        
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, 
            num_workers=0, pin_memory=True
        )
        
        print(f"Training on {len(dataset)} participants")
        
        for epoch in range(epochs):
            g_loss, d_loss = self.train_epoch(dataloader, epoch)
            
            memory_info = get_memory_usage()
            print(f'Epoch {epoch+1}/{epochs}: G_Loss: {g_loss:.4f}, D_Loss: {d_loss:.4f}')
            print(f'GPU: {memory_info["gpu_allocated"]:.2f}GB, CPU: {memory_info["cpu_percent"]:.1f}%')
            
            # Save test inference every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_test_inference(dataset, f'test_inference_epoch_{epoch+1}.png')
            
            if (epoch + 1) % save_every == 0:
                self.save_models(f'models_epoch_{epoch+1}')
        
        # Final save
        self.save_models('models_final')
        print("Training completed!")
    
    def save_test_inference(self, dataset, filename):
        """Save a test inference during training"""
        try:
            self.generator.eval()
            with torch.no_grad():
                sparse_input, real_target, participant_id = dataset[0]
                sparse_input = sparse_input.unsqueeze(0).to(self.device)
                fake_output = self.generator(sparse_input)
                
                # Save visualization
                self._visualize_inference(sparse_input.squeeze().cpu().numpy(),
                                        fake_output.squeeze().cpu().numpy(),
                                        real_target.squeeze().cpu().numpy(),
                                        participant_id, filename)
            self.generator.train()
        except Exception as e:
            print(f"Could not save test inference: {e}")
    
    def _visualize_inference(self, sparse_input, generated, target, participant_id, filename):
        """Visualize inference results"""
        depth = generated.shape[0]
        slice_indices = [depth//4, depth//2, 3*depth//4]
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        fig.suptitle(f'Training Progress - {participant_id}')
        
        for i, slice_idx in enumerate(slice_indices):
            # Sparse input
            axes[0, i].imshow(sparse_input[slice_idx], cmap='gray', vmin=0, vmax=1)
            axes[0, i].set_title(f'Input - Slice {slice_idx}')
            axes[0, i].axis('off')
            
            # Generated
            axes[1, i].imshow(generated[slice_idx], cmap='gray', vmin=0, vmax=1)
            axes[1, i].set_title(f'Generated - Slice {slice_idx}')
            axes[1, i].axis('off')
            
            # Target
            axes[2, i].imshow(target[slice_idx], cmap='gray', vmin=0, vmax=1)
            axes[2, i].set_title(f'Target - Slice {slice_idx}')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_models(self, prefix: str):
        """Save complete models (not just state dicts)"""
        torch.save(self.generator, f'{prefix}_generator.pth')
        torch.save(self.discriminator, f'{prefix}_discriminator.pth')
        
        # Also save training state
        torch.save({
            'g_optimizer': self.g_optimizer.state_dict(),
            'd_optimizer': self.d_optimizer.state_dict(),
            'g_scheduler': self.g_scheduler.state_dict(),
            'd_scheduler': self.d_scheduler.state_dict(),
            'history': self.history
        }, f'{prefix}_training_state.pth')
        
        print(f"Models saved with prefix: {prefix}")
        
    def load_models(self, prefix: str):
        """Load complete models"""
        self.generator = torch.load(f'{prefix}_generator.pth', map_location=self.device)
        self.discriminator = torch.load(f'{prefix}_discriminator.pth', map_location=self.device)
        
        # Load training state if exists
        try:
            state = torch.load(f'{prefix}_training_state.pth', map_location=self.device)
            self.g_optimizer.load_state_dict(state['g_optimizer'])
            self.d_optimizer.load_state_dict(state['d_optimizer'])
            self.g_scheduler.load_state_dict(state['g_scheduler'])
            if 'd_scheduler' in state:
                self.d_scheduler.load_state_dict(state['d_scheduler'])
            self.history = state['history']
            print(f"Training state restored from: {prefix}")
        except FileNotFoundError:
            print(f"Training state not found for: {prefix}")
            
        print(f"Models loaded from: {prefix}")
def explore_dataset(data_dir: str, max_samples: int = 3):
    """Explore the dataset structure"""
    dataset = MRISliceDataset(data_dir, debug=True)
    
    print(f"\n=== Dataset Overview ===")
    print(f"Total participants: {len(dataset)}")
    
    # Sample a few participants
    sample_indices = random.sample(range(len(dataset)), min(max_samples, len(dataset)))
    
    for idx in sample_indices:
        sparse_input, real_target, participant_id = dataset[idx]
        print(f"\nParticipant {participant_id}:")
        print(f"  Sparse input shape: {sparse_input.shape}")
        print(f"  Target shape: {real_target.shape}")
        print(f"  Sparse range: [{sparse_input.min():.3f}, {sparse_input.max():.3f}]")
        print(f"  Target range: [{real_target.min():.3f}, {real_target.max():.3f}]")
        
        # Check how many non-zero slices in sparse input
        sparse_np = sparse_input.squeeze().numpy()
        non_zero_slices = np.sum(np.any(sparse_np > 0.01, axis=(1, 2)))
        print(f"  Non-zero slices in sparse input: {non_zero_slices}")
        
        # Check slice distribution
        non_zero_indices = []
        for i in range(sparse_np.shape[0]):
            if np.any(sparse_np[i] > 0.01):
                non_zero_indices.append(i)
        
        if non_zero_indices:
            print(f"  Non-zero slice positions: {non_zero_indices[:5]}...{non_zero_indices[-5:] if len(non_zero_indices) > 10 else non_zero_indices[5:]}")
            print(f"  Slice distribution: {min(non_zero_indices)} to {max(non_zero_indices)}")

def get_model_config(gpu_memory_gb: float = None):
    """Get optimal model configuration based on GPU memory"""
    if gpu_memory_gb is None:
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        else:
            gpu_memory_gb = 0
    
    if gpu_memory_gb >= 24:  # High-end GPUs
        config = {
            'batch_size': 2,
            'base_features': 48,
            'target_slices': 128,
            'target_size': 256,
            'lr_g': 2e-4,
            'lr_d': 1e-4
        }
    elif gpu_memory_gb >= 16:  # Mid-range GPUs
        config = {
            'batch_size': 1,
            'base_features': 32,
            'target_slices': 128,
            'target_size': 256,
            'lr_g': 2e-4,
            'lr_d': 1e-4
        }
    elif gpu_memory_gb >= 8:  # Lower-end GPUs
        config = {
            'batch_size': 1,
            'base_features': 24,
            'target_slices': 96,
            'target_size': 192,
            'lr_g': 2e-4,
            'lr_d': 1e-4
        }
    else:  # CPU or very low memory
        config = {
            'batch_size': 1,
            'base_features': 16,
            'target_slices': 64,
            'target_size': 128,
            'lr_g': 2e-4,
            'lr_d': 1e-4
        }
    
    print(f"GPU Memory: {gpu_memory_gb:.1f}GB - Using config: {config}")
    return config

# Helper functions
def visualize_training_batch(dataset: MRISliceDataset, save_path: str = 'training_batch_debug.png'):
    """Visualize a training batch for debugging"""
    sparse_input, real_target, participant_id = dataset[0]
    
    sparse_np = sparse_input.squeeze().cpu().numpy()
    target_np = real_target.squeeze().cpu().numpy()
    
    # Select some slices to visualize - including slices with actual data
    depth = sparse_np.shape[0]
    
    # Find slices with actual data
    non_zero_slices = []
    for i in range(depth):
        if np.any(sparse_np[i] > 0.01):
            non_zero_slices.append(i)
    
    # Select visualization slices: some with data, some interpolated, some empty
    if len(non_zero_slices) >= 3:
        slice_indices = [non_zero_slices[0], non_zero_slices[len(non_zero_slices)//2], non_zero_slices[-1]]
    else:
        slice_indices = [depth//4, depth//2, 3*depth//4]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Training Data Debug - Participant: {participant_id}\nNon-zero slices: {len(non_zero_slices)}/{depth}')
    
    for i, slice_idx in enumerate(slice_indices):
        # Sparse input
        axes[0, i].imshow(sparse_np[slice_idx], cmap='gray', vmin=0, vmax=1)
        has_data = "✓" if np.any(sparse_np[slice_idx] > 0.01) else "✗"
        axes[0, i].set_title(f'Sparse Input - Slice {slice_idx} {has_data}')
        axes[0, i].axis('off')
        
        # Target
        axes[1, i].imshow(target_np[slice_idx], cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f'Target - Slice {slice_idx}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comprehensive evaluation saved to: {save_path}")
    

def benchmark_training_speed(data_dir: str, device: str = 'cuda', base_features: int = 32):
    """Benchmark training speed and memory usage"""
    print("=== Training Speed Benchmark ===")
    
    # Create a small trainer for benchmarking
    trainer = MRIIntersliceTrainer(
        data_dir=data_dir,
        device=device,
        base_features=base_features,
        target_slices=64,  # Smaller for benchmark
        target_size=128
    )
    
    # Create small dataset
    dataset = MRISliceDataset(data_dir, target_slices=64, target_size=128, cache_size=1)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Time a few iterations
    import time
    
    trainer.generator.train()
    trainer.discriminator.train()
    
    times = []
    memory_usage = []
    
    for i, (sparse_input, real_target, _) in enumerate(dataloader):
        if i >= 5:  # Just test 5 batches
            break
            
        start_time = time.time()
        
        sparse_input = sparse_input.to(device)
        real_target = real_target.to(device)
        
        # Generator forward pass
        trainer.g_optimizer.zero_grad()
        fake_output = trainer.generator(sparse_input)
        fake_pred = trainer.discriminator(fake_output)
        
        g_loss, _, _, _ = trainer.criterion.generator_loss(fake_pred, fake_output, real_target)
        g_loss.backward()
        trainer.g_optimizer.step()
        
        # Discriminator pass
        trainer.d_optimizer.zero_grad()
        real_pred = trainer.discriminator(real_target)
        fake_pred = trainer.discriminator(fake_output.detach())
        d_loss = trainer.criterion.discriminator_loss(real_pred, fake_pred)
        d_loss.backward()
        trainer.d_optimizer.step()
        
        end_time = time.time()
        times.append(end_time - start_time)
        
        mem_info = get_memory_usage()
        memory_usage.append(mem_info)
        
        clear_memory()
    
    avg_time = np.mean(times)
    avg_gpu_memory = np.mean([m['gpu_allocated'] for m in memory_usage])
    
    print(f"Average time per batch: {avg_time:.2f} seconds")
    print(f"Average GPU memory usage: {avg_gpu_memory:.2f} GB")
    print(f"Estimated time per epoch (56 participants): {avg_time * 56 / 60:.1f} minutes")
    
    return {
        'avg_time_per_batch': avg_time,
        'avg_gpu_memory': avg_gpu_memory,
        'estimated_epoch_time_minutes': avg_time * 56 / 60
    }

def create_comparison_visualization(generator_path: str, data_dir: str, device: str = 'cuda',
                                  participant_idx: int = 0, save_path: str = 'detailed_comparison.png'):
    """Create detailed comparison visualization"""
    # Load generator and data
    generator = torch.load(generator_path, map_location=device)
    generator.eval()
    
    dataset = MRISliceDataset(data_dir, target_slices=128, target_size=256)
    sparse_input, real_target, participant_id = dataset[participant_idx]
    
    # Generate output
    with torch.no_grad():
        sparse_input_gpu = sparse_input.unsqueeze(0).to(device)
        generated_output = generator(sparse_input_gpu)
        generated_output = generated_output.squeeze().cpu().numpy()
    
    sparse_np = sparse_input.squeeze().cpu().numpy()
    target_np = real_target.squeeze().cpu().numpy()
    
    # Find slices with actual data
    depth = generated_output.shape[0]
    non_zero_slices = []
    for i in range(depth):
        if np.any(sparse_np[i] > 0.01):
            non_zero_slices.append(i)
    
    # Select 5 representative slices
    if len(non_zero_slices) >= 5:
        step = len(non_zero_slices) // 5
        slice_indices = [non_zero_slices[i * step] for i in range(5)]
    else:
        slice_indices = [depth//6, depth//3, depth//2, 2*depth//3, 5*depth//6]
    
    # Create detailed visualization
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    fig.suptitle(f'Detailed Model Comparison - {participant_id}', fontsize=16)
    
    for i, slice_idx in enumerate(slice_indices):
        # Row 1: Sparse input
        axes[0, i].imshow(sparse_np[slice_idx], cmap='gray', vmin=0, vmax=1)
        has_data = "✓" if np.any(sparse_np[slice_idx] > 0.01) else "✗"
        axes[0, i].set_title(f'Input {slice_idx} {has_data}')
        axes[0, i].axis('off')
        
        # Row 2: Generated output
        axes[1, i].imshow(generated_output[slice_idx], cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f'Generated {slice_idx}')
        axes[1, i].axis('off')
        
        # Row 3: Ground truth
        axes[2, i].imshow(target_np[slice_idx], cmap='gray', vmin=0, vmax=1)
        axes[2, i].set_title(f'Target {slice_idx}')
        axes[2, i].axis('off')
        
        # Row 4: Difference map
        diff = np.abs(generated_output[slice_idx] - target_np[slice_idx])
        im = axes[3, i].imshow(diff, cmap='hot', vmin=0, vmax=0.5)
        axes[3, i].set_title(f'Error {slice_idx}')
        axes[3, i].axis('off')
        
        # Add colorbar for difference maps
        if i == 4:  # Only for last subplot
            plt.colorbar(im, ax=axes[3, i], fraction=0.046, pad=0.04)
    
    # Add row labels
    row_labels = ['Sparse Input', 'Generated', 'Ground Truth', 'Absolute Error']
    for i, label in enumerate(row_labels):
        axes[i, 0].set_ylabel(label, rotation=90, fontsize=12, labelpad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Detailed comparison saved to: {save_path}")
def plot_training_history(history: dict, save_path: str = 'training_history.png'):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Generator vs Discriminator Loss
    epochs = range(1, len(history['g_loss']) + 1)
    axes[0, 0].plot(epochs, history['g_loss'], 'b-', label='Generator', linewidth=2)
    axes[0, 0].plot(epochs, history['d_loss'], 'r-', label='Discriminator', linewidth=2)
    axes[0, 0].set_title('Generator vs Discriminator Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss ratio
    if len(history['g_loss']) > 0 and len(history['d_loss']) > 0:
        ratio = [g/(d+1e-8) for g, d in zip(history['g_loss'], history['d_loss'])]
        axes[0, 1].plot(epochs, ratio, 'g-', linewidth=2)
        axes[0, 1].set_title('G/D Loss Ratio')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Ratio')
        axes[0, 1].axhline(y=1, color='orange', linestyle='--', alpha=0.7, label='Balance Line')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # L1 Loss
    axes[1, 0].plot(epochs, history['l1_loss'], 'purple', linewidth=2)
    axes[1, 0].set_title('L1 Reconstruction Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('L1 Loss')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Adversarial Loss
    axes[1, 1].plot(epochs, history['adv_loss'], 'orange', linewidth=2)
    axes[1, 1].set_title('Adversarial Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Adversarial Loss')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to: {save_path}")

def save_volume_as_nifti(volume: np.ndarray, output_path: str, affine: np.ndarray = None):
    """Save volume as NIfTI file (requires nibabel)"""
    try:
        import nibabel as nib
        
        if affine is None:
            # Create simple identity affine
            affine = np.eye(4)
        
        # Create NIfTI image
        nifti_img = nib.Nifti1Image(volume, affine)
        
        # Save
        nib.save(nifti_img, output_path)
        print(f"Volume saved as NIfTI: {output_path}")
        
    except ImportError:
        print("nibabel not available - cannot save as NIfTI")
        # Save as numpy array instead
        np_path = output_path.replace('.nii', '.npy').replace('.gz', '')
        np.save(np_path, volume)
        print(f"Volume saved as numpy array: {np_path}")
def test_model_inference(generator_path: str, data_dir: str, device: str = 'cuda', 
                        participant_idx: int = 0, save_path: str = 'inference_test.png'):
    """Test model inference and visualize results"""
    # Load generator
    generator = torch.load(generator_path, map_location=device)
    generator.eval()
    
    # Load dataset
    dataset = MRISliceDataset(data_dir, target_slices=128, target_size=256)
    sparse_input, real_target, participant_id = dataset[participant_idx]
    
    print(f"Testing inference on participant: {participant_id}")
    
    # Generate output
    with torch.no_grad():
        sparse_input_gpu = sparse_input.unsqueeze(0).to(device)  # Add batch dimension
        generated_output = generator(sparse_input_gpu)
        generated_output = generated_output.squeeze().cpu().numpy()
    
    sparse_np = sparse_input.squeeze().cpu().numpy()
    target_np = real_target.squeeze().cpu().numpy()
    
    # Calculate metrics
    mse = np.mean((generated_output - target_np) ** 2)
    mae = np.mean(np.abs(generated_output - target_np))
    psnr = 20 * np.log10(1.0 / np.sqrt(mse + 1e-8))
    
    # Calculate structural similarity (SSIM) approximation
    ssim_approx = 1 - mse  # Simple approximation
    
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM (approx): {ssim_approx:.4f}")
    
    # Find slices with actual data for better visualization
    depth = generated_output.shape[0]
    non_zero_slices = []
    for i in range(depth):
        if np.any(sparse_np[i] > 0.01):
            non_zero_slices.append(i)
    
    if len(non_zero_slices) >= 3:
        slice_indices = [non_zero_slices[0], non_zero_slices[len(non_zero_slices)//2], non_zero_slices[-1]]
    else:
        slice_indices = [depth//4, depth//2, 3*depth//4]
    
    # Visualize results
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f'Model Inference Test - {participant_id}\nMSE: {mse:.6f}, PSNR: {psnr:.2f}dB, SSIM: {ssim_approx:.4f}')
    
    for i, slice_idx in enumerate(slice_indices):
        # Sparse input
        axes[0, i].imshow(sparse_np[slice_idx], cmap='gray', vmin=0, vmax=1)
        has_data = "✓" if np.any(sparse_np[slice_idx] > 0.01) else "✗"
        axes[0, i].set_title(f'Input - Slice {slice_idx} {has_data}')
        axes[0, i].axis('off')
        
        # Generated output
        axes[1, i].imshow(generated_output[slice_idx], cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f'Generated - Slice {slice_idx}')
        axes[1, i].axis('off')
        
        # Ground truth
        axes[2, i].imshow(target_np[slice_idx], cmap='gray', vmin=0, vmax=1)
        axes[2, i].set_title(f'Ground Truth - Slice {slice_idx}')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Inference test visualization saved to: {save_path}")
    
    return {
        'mse': mse,
        'mae': mae,
        'psnr': psnr,
        'ssim_approx': ssim_approx,
        'generated_output': generated_output
    }

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='MRI Interslice Interpolation GAN')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'explore', 'animate', 'evaluate', 'benchmark'], 
                       default='train', help='Mode: train, test, explore, animate, evaluate, or benchmark')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (0 for auto)')
    parser.add_argument('--base_features', type=int, default=0, help='Base features (0 for auto)')
    parser.add_argument('--target_slices', type=int, default=128, help='Target number of slices')
    parser.add_argument('--target_size', type=int, default=128, help='Target slice size')
    parser.add_argument('--lr_g', type=float, default=2e-4, help='Generator learning rate')
    parser.add_argument('--lr_d', type=float, default=1e-4, help='Discriminator learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--save_every', type=int, default=10, help='Save models every N epochs')
    parser.add_argument('--generator_path', type=str, default='', help='Path to generator for testing')
    parser.add_argument('--output_dir', type=str, default='./', help='Output directory')
    parser.add_argument('--participant_idx', type=int, default=0, help='Participant index for testing')
    parser.add_argument('--num_eval_samples', type=int, default=5, help='Number of samples for evaluation')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == 'explore':
        print("=== Exploring Dataset ===")
        explore_dataset(args.data_dir)
        
        # Create debug visualization
        dataset = MRISliceDataset(args.data_dir, target_slices=args.target_slices, 
                                 target_size=args.target_size)
        visualize_training_batch(dataset, os.path.join(args.output_dir, 'dataset_debug.png'))
        
    elif args.mode == 'benchmark':
        print("=== Benchmarking Training Speed ===")
        config = get_model_config()
        base_features = args.base_features if args.base_features > 0 else config['base_features']
        
        benchmark_results = benchmark_training_speed(
            args.data_dir, 
            device=args.device, 
            base_features=base_features
        )
        
        # Save benchmark results
        with open(os.path.join(args.output_dir, 'benchmark_results.json'), 'w') as f:
            json.dump(benchmark_results, f, indent=2)
    
    elif args.mode == 'evaluate':
        if not args.generator_path:
            print("Error: --generator_path required for evaluation mode")
            return
            
        print("=== Comprehensive Model Evaluation ===")
        evaluation_results = evaluate_model_comprehensive(
            args.generator_path,
            args.data_dir,
            device=args.device,
            num_samples=args.num_eval_samples,
            save_path=os.path.join(args.output_dir, 'comprehensive_evaluation.png')
        )
        
        # Save detailed comparison
        create_comparison_visualization(
            args.generator_path,
            args.data_dir,
            device=args.device,
            participant_idx=args.participant_idx,
            save_path=os.path.join(args.output_dir, 'detailed_comparison.png')
        )
        
        # Save evaluation results
        with open(os.path.join(args.output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
    elif args.mode == 'animate':
        if not args.generator_path:
            print("Error: --generator_path required for animation mode")
            return
            
        print("=== Creating Animation ===")
        create_animation(
            args.generator_path,
            args.data_dir,
            participant_idx=args.participant_idx,
            device=args.device,
            output_path=os.path.join(args.output_dir, 'mri_animation.gif')
        )
        
    elif args.mode == 'test':
        if not args.generator_path:
            print("Error: --generator_path required for testing mode")
            return
        
        print("=== Testing Model ===")
        results = test_model_inference(
            args.generator_path, 
            args.data_dir, 
            device=args.device,
            participant_idx=args.participant_idx,
            save_path=os.path.join(args.output_dir, 'test_results.png')
        )
        
        # Save results
        with open(os.path.join(args.output_dir, 'test_metrics.json'), 'w') as f:
            json.dump({k: v for k, v in results.items() if k != 'generated_output'}, f, indent=2)
        
        print("=== Test Results ===")
        for key, value in results.items():
            if key != 'generated_output':
                print(f"{key}: {value}")
        
    elif args.mode == 'train':
        print("=== Training Mode ===")
        
        # Get optimal configuration
        config = get_model_config()
        
        # Override with command line arguments
        if args.batch_size > 0:
            config['batch_size'] = args.batch_size
        if args.base_features > 0:
            config['base_features'] = args.base_features
            
        config.update({
            'target_slices': args.target_slices,
            'target_size': args.target_size,
            'lr_g': args.lr_g,
            'lr_d': args.lr_d
        })
        
        # Save configuration
        with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        print("=== Training Configuration ===")
        for key, value in config.items():
            print(f"{key}: {value}")
        
        # Initialize trainer
        trainer = MRIIntersliceTrainer(
            data_dir=args.data_dir,
            device=args.device,
            lr_g=config['lr_g'],
            lr_d=config['lr_d'],
            base_features=config['base_features'],
            target_slices=config['target_slices'],
            target_size=config['target_size']
        )
        
        # Create debug visualizations before training
        dataset = MRISliceDataset(args.data_dir, target_slices=config['target_slices'], 
                                 target_size=config['target_size'], debug=True)
        visualize_training_batch(dataset, os.path.join(args.output_dir, 'training_batch_debug.png'))
        
        # Train
        try:
            trainer.train(
                epochs=args.epochs,
                batch_size=config['batch_size'],
                save_every=args.save_every
            )
            
            # Save final training history
            plot_training_history(trainer.history, 
                                os.path.join(args.output_dir, 'training_history.png'))
            
            with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
                json.dump(trainer.history, f, indent=2)
            
            # Test the final model
            print("\n=== Testing Final Model ===")
            final_results = test_model_inference(
                'models_final_generator.pth',
                args.data_dir,
                device=args.device,
                save_path=os.path.join(args.output_dir, 'final_model_test.png')
            )
            
            # Save final results
            with open(os.path.join(args.output_dir, 'final_results.json'), 'w') as f:
                json.dump({k: v for k, v in final_results.items() if k != 'generated_output'}, f, indent=2)
            
            print("=== Final Results ===")
            for key, value in final_results.items():
                if key != 'generated_output':
                    print(f"{key}: {value}")
            
            # Comprehensive evaluation
            print("\n=== Comprehensive Evaluation ===")
            evaluation_results = evaluate_model_comprehensive(
                'models_final_generator.pth',
                args.data_dir,
                device=args.device,
                num_samples=5,
                save_path=os.path.join(args.output_dir, 'final_comprehensive_evaluation.png')
            )
            
            with open(os.path.join(args.output_dir, 'final_evaluation.json'), 'w') as f:
                json.dump(evaluation_results, f, indent=2)
            
        except Exception as e:
            print(f"Training failed: {e}")
            # Try to save emergency checkpoint
            try:
                trainer.save_models('emergency_checkpoint')
                print("Emergency checkpoint saved")
            except:
                print("Could not save emergency checkpoint")
            raise e
        
        finally:
            clear_memory()


def interactive_mode():
    """Interactive mode for easy usage"""
    print("=== MRI Interslice Interpolation GAN - Interactive Mode ===")
    print()
    
    # Get data directory
    while True:
        data_dir = input("Enter path to data directory: ").strip()
        if os.path.exists(data_dir):
            break
        print("Directory not found. Please try again.")
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
        device = 'cuda'
    else:
        print("No GPU detected - using CPU (training will be very slow)")
        device = 'cpu'
    
    # Get configuration
    config = get_model_config(gpu_memory if device == 'cuda' else 0)
    
    print("\nRecommended configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Ask what to do
    print("\nWhat would you like to do?")
    print("1. Explore dataset")
    print("2. Train new model")
    print("3. Test existing model")
    print("4. Create animation")
    print("5. Comprehensive evaluation")
    print("6. Benchmark training speed")
    
    choice = input("Enter choice (1-6): ").strip()
    
    if choice == '1':
        explore_dataset(data_dir)
        dataset = MRISliceDataset(data_dir, target_slices=config['target_slices'], 
                                 target_size=config['target_size'])
        visualize_training_batch(dataset, 'dataset_debug.png')
        
    elif choice == '2':
        epochs = int(input(f"Number of epochs (default {50}): ").strip() or 50)
        
        trainer = MRIIntersliceTrainer(
            data_dir=data_dir,
            device=device,
            lr_g=config['lr_g'],
            lr_d=config['lr_d'],
            base_features=config['base_features'],
            target_slices=config['target_slices'],
            target_size=config['target_size']
        )
        
        try:
            trainer.train(epochs=epochs, batch_size=config['batch_size'])
            plot_training_history(trainer.history, 'training_history.png')
            print("Training completed successfully!")
            
        except Exception as e:
            print(f"Training failed: {e}")
            trainer.save_models('emergency_checkpoint')
            
    elif choice == '3':
        model_path = input("Enter path to generator model: ").strip()
        if os.path.exists(model_path):
            participant_idx = int(input("Participant index to test (default 0): ").strip() or 0)
            results = test_model_inference(model_path, data_dir, device, participant_idx)
            print("Test completed!")
        else:
            print("Model file not found")
            
    elif choice == '4':
        model_path = input("Enter path to generator model: ").strip()
        if os.path.exists(model_path):
            participant_idx = int(input("Participant index for animation (default 0): ").strip() or 0)
            create_animation(model_path, data_dir, participant_idx, device)
            print("Animation created!")
        else:
            print("Model file not found")
            
    elif choice == '5':
        model_path = input("Enter path to generator model: ").strip()
        if os.path.exists(model_path):
            num_samples = int(input("Number of samples to evaluate (default 5): ").strip() or 5)
            evaluation_results = evaluate_model_comprehensive(model_path, data_dir, device, num_samples)
            print("Comprehensive evaluation completed!")
        else:
            print("Model file not found")
            
    elif choice == '6':
        benchmark_results = benchmark_training_speed(data_dir, device, config['base_features'])
        print("Benchmark completed!")
        
    else:
        print("Invalid choice")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        # Check if we want interactive mode
        try:
            interactive_mode()
        except KeyboardInterrupt:
            print("\nExiting...")
        except Exception as e:
            print(f"Error in interactive mode: {e}")
            print()
            print("MRI Interslice Interpolation GAN")
            print("=====================================")
            print("This tool trains a 3D GAN to interpolate between sparse MRI slices.")
            print("You have ~25 low-resolution slices and need to generate 128 complete slices.")
            print()
            print("Usage examples:")
            print("  # Explore dataset structure")
            print("  python Main.py --data_dir /path/to/data --mode explore")
            print()
            print("  # Train model")
            print("  python Main.py --data_dir /path/to/data --mode train --epochs 50")
            print()
            print("  # Test trained model")
            print("  python Main.py --data_dir /path/to/data --mode test --generator_path models_final_generator.pth")
            print()
            print("  # Create animation")
            print("  python Main.py --data_dir /path/to/data --mode animate --generator_path models_final_generator.pth")
            print()
            print("  # Comprehensive evaluation")
            print("  python Main.py --data_dir /path/to/data --mode evaluate --generator_path models_final_generator.pth")
            print()
            print("  # Benchmark training speed")
            print("  python Main.py --data_dir /path/to/data --mode benchmark")
            print()
            
            # Show current configuration
            config = get_model_config()
            print("Current auto-configuration:")
            for key, value in config.items():
                print(f"  {key}: {value}")
            
            # Quick dataset check if data_dir is provided as environment variable
            data_dir = os.environ.get('MRI_DATA_DIR')
            if data_dir and os.path.exists(data_dir):
                print(f"\nQuick dataset check for: {data_dir}")
                explore_dataset(data_dir, max_samples=1)
    plt.close()
    

def create_animation(generator_path: str, data_dir: str, participant_idx: int = 0, 
                    device: str = 'cuda', output_path: str = 'mri_animation.gif'):
    """Create animation showing slice-by-slice comparison"""
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
        
        # Load generator and data
        generator = torch.load(generator_path, map_location=device)
        generator.eval()
        
        dataset = MRISliceDataset(data_dir, target_slices=128, target_size=256)
        sparse_input, real_target, participant_id = dataset[participant_idx]
        
        # Generate output
        with torch.no_grad():
            sparse_input_gpu = sparse_input.unsqueeze(0).to(device)
            generated_output = generator(sparse_input_gpu)
            generated_output = generated_output.squeeze().cpu().numpy()
        
        sparse_np = sparse_input.squeeze().cpu().numpy()
        target_np = real_target.squeeze().cpu().numpy()
        
        # Create animation
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'MRI Slice Animation - {participant_id}')
        
        def animate(frame):
            axes[0].clear()
            axes[1].clear()
            axes[2].clear()
            
            axes[0].imshow(sparse_np[frame], cmap='gray', vmin=0, vmax=1)
            has_data = "✓" if np.any(sparse_np[frame] > 0.01) else "✗"
            axes[0].set_title(f'Input - Slice {frame} {has_data}')
            axes[0].axis('off')
            
            axes[1].imshow(generated_output[frame], cmap='gray', vmin=0, vmax=1)
            axes[1].set_title(f'Generated - Slice {frame}')
            axes[1].axis('off')
            
            axes[2].imshow(target_np[frame], cmap='gray', vmin=0, vmax=1)
            axes[2].set_title(f'Target - Slice {frame}')
            axes[2].axis('off')
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=generated_output.shape[0], 
                           interval=200, repeat=True)
        
        # Save animation
        writer = PillowWriter(fps=5)
        anim.save(output_path, writer=writer)
        plt.close()
        
        print(f"Animation saved to: {output_path}")
        
    except ImportError:
        print("Could not create animation - matplotlib animation not available")
    except Exception as e:
        print(f"Could not create animation: {e}")

def evaluate_model_comprehensive(generator_path: str, data_dir: str, device: str = 'cuda', 
                                num_samples: int = 5, save_path: str = 'comprehensive_evaluation.png'):
    """Comprehensive model evaluation with multiple metrics"""
    # Load generator
    generator = torch.load(generator_path, map_location=device)
    generator.eval()
    
    # Load dataset
    dataset = MRISliceDataset(data_dir, target_slices=128, target_size=256)
    
    print(f"Evaluating model on {min(num_samples, len(dataset))} participants...")
    
    all_metrics = []
    
    for i in range(min(num_samples, len(dataset))):
        sparse_input, real_target, participant_id = dataset[i]
        
        # Generate output
        with torch.no_grad():
            sparse_input_gpu = sparse_input.unsqueeze(0).to(device)
            generated_output = generator(sparse_input_gpu)
            generated_output = generated_output.squeeze().cpu().numpy()
        
        target_np = real_target.squeeze().cpu().numpy()
        
        # Calculate metrics
        mse = np.mean((generated_output - target_np) ** 2)
        mae = np.mean(np.abs(generated_output - target_np))
        psnr = 20 * np.log10(1.0 / np.sqrt(mse + 1e-8))
        
        # Slice-wise metrics for better understanding
        slice_mse = np.mean((generated_output - target_np) ** 2, axis=(1, 2))
        slice_mae = np.mean(np.abs(generated_output - target_np), axis=(1, 2))
        
        metrics = {
            'participant_id': participant_id,
            'mse': mse,
            'mae': mae,
            'psnr': psnr,
            'slice_mse_mean': np.mean(slice_mse),
            'slice_mse_std': np.std(slice_mse),
            'slice_mae_mean': np.mean(slice_mae),
            'slice_mae_std': np.std(slice_mae)
        }
        all_metrics.append(metrics)
        
        print(f"Participant {participant_id}: MSE={mse:.6f}, PSNR={psnr:.2f}dB")
    
    # Calculate overall statistics
    overall_mse = np.mean([m['mse'] for m in all_metrics])
    overall_mae = np.mean([m['mae'] for m in all_metrics])
    overall_psnr = np.mean([m['psnr'] for m in all_metrics])
    
    print(f"\n=== Overall Performance ===")
    print(f"Average MSE: {overall_mse:.6f}")
    print(f"Average MAE: {overall_mae:.6f}")
    print(f"Average PSNR: {overall_psnr:.2f} dB")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # MSE distribution
    mse_values = [m['mse'] for m in all_metrics]
    axes[0, 0].hist(mse_values, bins=10, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('MSE Distribution')
    axes[0, 0].set_xlabel('MSE')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].axvline(overall_mse, color='red', linestyle='--', label=f'Mean: {overall_mse:.6f}')
    axes[0, 0].legend()
    
    # PSNR distribution
    psnr_values = [m['psnr'] for m in all_metrics]
    axes[0, 1].hist(psnr_values, bins=10, alpha=0.7, edgecolor='black', color='green')
    axes[0, 1].set_title('PSNR Distribution')
    axes[0, 1].set_xlabel('PSNR (dB)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].axvline(overall_psnr, color='red', linestyle='--', label=f'Mean: {overall_psnr:.2f}dB')
    axes[0, 1].legend()
    
    # Slice-wise error analysis
    slice_mse_means = [m['slice_mse_mean'] for m in all_metrics]
    slice_mse_stds = [m['slice_mse_std'] for m in all_metrics]
    participant_labels = [m['participant_id'][:8] for m in all_metrics]  # Truncate for readability
    
    x_pos = np.arange(len(participant_labels))
    axes[1, 0].bar(x_pos, slice_mse_means, yerr=slice_mse_stds, alpha=0.7, capsize=5)
    axes[1, 0].set_title('Slice-wise MSE per Participant')
    axes[1, 0].set_xlabel('Participant')
    axes[1, 0].set_ylabel('Mean Slice MSE')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(participant_labels, rotation=45)
    
    # MAE vs MSE correlation
    mae_values = [m['mae'] for m in all_metrics]
    axes[1, 1].scatter(mse_values, mae_values, alpha=0.7)
    axes[1, 1].set_title('MSE vs MAE Correlation')
    axes[1, 1].set_xlabel('MSE')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add correlation coefficient
    correlation = np.corrcoef(mse_values, mae_values)[0, 1]
    axes[1, 1].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                   transform=axes[1, 1].transAxes, fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()