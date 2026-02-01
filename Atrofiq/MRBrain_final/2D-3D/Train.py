#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import json
from scipy.ndimage import zoom as scipy_zoom

# SynthSeg Label Mapping
LABEL_MAP = {
    'background': 0, '3rd ventricle': 14, '4th ventricle': 15, 'brain-stem': 16, 'csf': 24, 
    'left cerebral white matter': 2, 'left cerebral cortex': 3, 'left lateral ventricle': 4, 
    'left inferior lateral ventricle': 5, 'left cerebellum white matter': 7, 'left cerebellum cortex': 8, 
    'left thalamus': 10, 'left caudate': 11, 'left putamen': 12, 'left pallidum': 13, 
    'left hippocampus': 17, 'left amygdala': 18, 'left accumbens area': 26, 'left ventral DC': 28, 
    'right cerebral white matter': 41, 'right cerebral cortex': 42, 'right lateral ventricle': 43, 
    'right inferior lateral ventricle': 44, 'right cerebellum white matter': 46, 'right cerebellum cortex': 47, 
    'right thalamus': 49, 'right caudate': 50, 'right putamen': 51, 'right pallidum': 52, 
    'right hippocampus': 53, 'right amygdala': 54, 'right accumbens area': 58, 'right ventral DC': 60
}

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
        
        # Final refinement to ensure rich features
        self.final = nn.Sequential(
            nn.Conv3d(base_ch*8, base_ch*8, 3, padding=1),
            nn.GroupNorm(32, base_ch*8),
            nn.GELU()
        )
    
    def forward(self, x):
        # x: [B, 1, D, H, W]
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        return self.final(x)


class AnatomyDecoder(nn.Module):
    def __init__(self, in_ch=512):
        super().__init__()
        
        # Initial processing of the 2D slice
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
        """
        Sample a 2D slice from the 3D feature volume at normalized depth t.
        volume_features: [B, C, D, H, W]
        t: [B] or scalar, normalized depth in [0, 1]
        """
        B, C, D, H, W = volume_features.shape
        
        # Create sampling grid
        # We want to sample a plane at z = 2*t - 1 (converting [0,1] to [-1,1])
        # The grid should cover the full X-Y extent [-1, 1]
        
        # If t is a scalar, expand it
        if not torch.is_tensor(t):
            t = torch.tensor([t], device=volume_features.device).repeat(B)
        elif t.dim() == 0:
            t = t.unsqueeze(0).repeat(B)
            
        z = 2 * t - 1  # Map [0, 1] -> [-1, 1]
        z = z.view(B, 1, 1, 1)
        
        # Create meshgrid for H, W
        y = torch.linspace(-1, 1, H, device=volume_features.device)
        x = torch.linspace(-1, 1, W, device=volume_features.device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        
        # Expand to batch size and add depth dim
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1) # [B, 1, H, W]
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1) # [B, 1, H, W]
        z = z.expand(-1, 1, H, W) # [B, 1, H, W]
        
        # Combine to create the sampling grid [B, 1, H, W, 3]
        # Grid coordinates are (x, y, z)
        grid = torch.stack([grid_x, grid_y, z], dim=-1)
        
        # Sample
        # align_corners=True matches the geometric interpretation of pixels
        sampled = F.grid_sample(volume_features, grid, align_corners=True, mode='bilinear', padding_mode='border')
        
        # sampled is [B, C, 1, H, W] -> squeeze to [B, C, H, W]
        return sampled.squeeze(2)

    def forward(self, sparse_volume, t):
        # sparse_volume: [B, 1, D_in, H_in, W_in]
        features = self.encoder(sparse_volume)
        
        # Sample the slice features
        slice_features = self.sample_slice(features, t)
        
        # Decode
        return self.decoder(slice_features)


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.scales = nn.ModuleList([
            self._make_disc(64),
            self._make_disc(48),
            self._make_disc(32)
        ])
        
    def _make_disc(self, base_ch):
        return nn.Sequential(
            nn.Conv2d(1, base_ch, 4, 2, 1),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1),
            nn.GroupNorm(16, base_ch * 2),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1),
            nn.GroupNorm(32, base_ch * 4),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(base_ch * 4, base_ch * 8, 4, 2, 1),
            nn.GroupNorm(32, base_ch * 8),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(base_ch * 8, 1, 4, 1, 1)
        )
    
    def forward(self, x):
        outputs = []
        for scale in self.scales:
            outputs.append(scale(x))
            x = F.avg_pool2d(x, 2)
        return outputs


class AnatomicalConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                     dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                     dtype=torch.float32).view(1, 1, 3, 3)
        
    def forward(self, pred, target, device, weight_map=None):
        if not hasattr(self, '_sobel_x_cuda'):
            self._sobel_x_cuda = self.sobel_x.to(device)
            self._sobel_y_cuda = self.sobel_y.to(device)
        
        pred_edges_x = F.conv2d(pred, self._sobel_x_cuda, padding=1)
        pred_edges_y = F.conv2d(pred, self._sobel_y_cuda, padding=1)
        target_edges_x = F.conv2d(target, self._sobel_x_cuda, padding=1)
        target_edges_y = F.conv2d(target, self._sobel_y_cuda, padding=1)
        
        diff_x = torch.abs(pred_edges_x - target_edges_x)
        diff_y = torch.abs(pred_edges_y - target_edges_y)
        
        if weight_map is not None:
            diff_x = diff_x * weight_map
            diff_y = diff_y * weight_map
            
        edge_loss = diff_x.mean() + diff_y.mean()
        
        return edge_loss


class StructuralSimilarityLoss(nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        
    def forward(self, pred, target):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu_pred = F.avg_pool2d(pred, self.window_size, stride=1, padding=self.window_size // 2)
        mu_target = F.avg_pool2d(target, self.window_size, stride=1, padding=self.window_size // 2)
        
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target
        
        sigma_pred = F.avg_pool2d(pred ** 2, self.window_size, stride=1, padding=self.window_size // 2) - mu_pred_sq
        sigma_target = F.avg_pool2d(target ** 2, self.window_size, stride=1, padding=self.window_size // 2) - mu_target_sq
        sigma_pred_target = F.avg_pool2d(pred * target, self.window_size, stride=1, padding=self.window_size // 2) - mu_pred_target
        
        ssim = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred + sigma_target + C2))
        
        return 1 - ssim.mean()


class FrequencyLoss(nn.Module):
    def forward(self, pred, target):
        pred_fft = torch.fft.fft2(pred)
        target_fft = torch.fft.fft2(target)
        
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        return F.l1_loss(pred_mag, target_mag)


class MRIDataset(Dataset):
    def __init__(self, data_dir, samples_per_volume=64, augment=True):
        self.data_dir = Path(data_dir)
        self.samples_per_volume = samples_per_volume
        self.augment = augment
        
        self.subjects = []
        for subject_dir in sorted(self.data_dir.iterdir()):
            if not subject_dir.is_dir():
                continue
            
            sparse_path = subject_dir / 'sparse_volume.npy'
            dense_path = subject_dir / 'dense_volume.npy'
            label_path = subject_dir / 'labels.npy'
            
            if sparse_path.exists() and dense_path.exists():
                self.subjects.append({
                    'sparse': sparse_path,
                    'dense': dense_path,
                    'labels': label_path if label_path.exists() else None,
                    'id': subject_dir.name
                })
        
        print(f"Loaded {len(self.subjects)} subjects")
    
    def __len__(self):
        return len(self.subjects) * self.samples_per_volume
    
    def __getitem__(self, idx):
        subject_idx = idx // self.samples_per_volume
        subj = self.subjects[subject_idx]
        
        sparse = np.load(subj['sparse']).astype(np.float32)
        dense = np.load(subj['dense']).astype(np.float32)
        labels = np.load(subj['labels']).astype(np.int32) if subj['labels'] else None
        
        # Multi-view augmentation: randomly select axis (view)
        # 0 = Axial (original), 1 = Coronal, 2 = Sagittal
        # Multi-view augmentation: randomly select axis (view)
        # 0 = Axial (original), 1 = Coronal, 2 = Sagittal
        # CHANGED: Force equal distribution (1/3 each) to improve 3D consistency
        if self.augment:
            view_axis = np.random.choice([0, 1, 2], p=[0.34, 0.33, 0.33])
        else:
            view_axis = 0  # Default axial view
        
        # Transpose to put the selected axis first
        if view_axis == 1:  # Coronal
            sparse = np.transpose(sparse, (1, 0, 2))
            dense = np.transpose(dense, (1, 0, 2))
            if labels is not None: labels = np.transpose(labels, (1, 0, 2))
        elif view_axis == 2:  # Sagittal
            sparse = np.transpose(sparse, (2, 0, 1))
            dense = np.transpose(dense, (2, 0, 1))
            if labels is not None: labels = np.transpose(labels, (2, 0, 1))
        # view_axis == 0 keeps original (D, H, W) = Axial
        
        # Ensure sparse volume is always (25, 256, 256) by resizing if needed
        target_sparse_shape = (25, 256, 256)
        if sparse.shape != target_sparse_shape:
            zoom_factors = (target_sparse_shape[0] / sparse.shape[0],
                          target_sparse_shape[1] / sparse.shape[1],
                          target_sparse_shape[2] / sparse.shape[2])
            sparse = scipy_zoom(sparse, zoom_factors, order=1).astype(np.float32)
        
        valid_start = 0
        valid_end = dense.shape[0] - 1
        
        slice_idx = np.random.randint(valid_start, valid_end)
        
        # Ensure slice_idx is within bounds for next slice
        slice_idx_next = min(slice_idx + 1, dense.shape[0] - 1)
        
        t = slice_idx / (dense.shape[0] - 1)
        # Calculate t for the next slice
        t_next = slice_idx_next / (dense.shape[0] - 1)
        
        target_slice = dense[slice_idx]
        target_slice_next = dense[slice_idx_next]
        
        label_slice = labels[slice_idx] if labels is not None else np.zeros_like(target_slice, dtype=np.int32)
        
        # Ensure target_slice is always 256x256 by resizing if needed
        if target_slice.shape != (256, 256):
            zoom_factors = (256 / target_slice.shape[0], 256 / target_slice.shape[1])
            target_slice = scipy_zoom(target_slice, zoom_factors, order=1).astype(np.float32)
            target_slice_next = scipy_zoom(target_slice_next, zoom_factors, order=1).astype(np.float32)
            if labels is not None:
                 label_slice = scipy_zoom(label_slice, zoom_factors, order=0).astype(np.int32)
        
        if self.augment:
            # Horizontal flip
            if np.random.rand() > 0.5:
                sparse = np.flip(sparse, 2).copy()
                target_slice = np.flip(target_slice, 1).copy()
                target_slice_next = np.flip(target_slice_next, 1).copy()
                label_slice = np.flip(label_slice, 1).copy()
            
            # Vertical flip
            if np.random.rand() > 0.5:
                sparse = np.flip(sparse, 1).copy()
                target_slice = np.flip(target_slice, 0).copy()
                target_slice_next = np.flip(target_slice_next, 0).copy()
                label_slice = np.flip(label_slice, 0).copy()
            
            # Intensity scaling (only affects images, not labels)
            if np.random.rand() > 0.7:
                scale = np.random.uniform(0.9, 1.1)
                sparse = np.clip(sparse * scale, 0, 1)
                target_slice = np.clip(target_slice * scale, 0, 1)
                target_slice_next = np.clip(target_slice_next * scale, 0, 1)
            
            # Gamma adjustment
            if np.random.rand() > 0.8:
                gamma = np.random.uniform(0.85, 1.15)
                sparse = np.power(sparse, gamma)
                target_slice = np.power(target_slice, gamma)
                target_slice_next = np.power(target_slice_next, gamma)
        
        return (
            torch.from_numpy(sparse).unsqueeze(0),
            torch.tensor(t, dtype=torch.float32),
            torch.tensor(t_next, dtype=torch.float32),
            torch.from_numpy(target_slice).unsqueeze(0),
            torch.from_numpy(target_slice_next).unsqueeze(0),
            torch.from_numpy(label_slice).long()
        )
        
        return (
            torch.from_numpy(sparse).unsqueeze(0),
            torch.tensor(t, dtype=torch.float32),
            torch.from_numpy(target_slice).unsqueeze(0),
            torch.from_numpy(label_slice).long()
        )


class Trainer:
    def __init__(self, device='cuda', lr_g=5e-5, lr_d=1e-5):
        self.device = torch.device(device)
        
        self.G = Generator().to(self.device)
        self.D = MultiScaleDiscriminator().to(self.device)
        
        # Enable DataParallel if multiple GPUs are available
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for training!")
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)
            # opt parameters are already registered from the base model, so this is fine
            # providing we don't re-initialize optimizer after wrapping. 
            # Actually, typically we optimize parameters of the model. 
            # self.G.parameters() returns the same params even if wrapped?
            # Yes, DataParallel.parameters() yields module.parameters().
        
        # Re-initialize optimizers to be safe with pointers (though usually fine)
        self.opt_G = torch.optim.AdamW(self.G.parameters(), lr=lr_g, betas=(0.5, 0.999), weight_decay=1e-5)
        self.opt_D = torch.optim.AdamW(self.D.parameters(), lr=lr_d, betas=(0.5, 0.999), weight_decay=1e-5)
        
        self.anatomy_loss = AnatomicalConsistencyLoss()
        self.ssim_loss = StructuralSimilarityLoss()
        self.freq_loss = FrequencyLoss()
        
        self.history = {'g_loss': [], 'd_loss': [], 'edge_loss': [], 'psnr': []}
        self.best_psnr = 0
        self.start_epoch = 1
        
        # Redirect outputs to Region_Focused_Results
        self.weights_dir = Path('Region_Focused_Results/weights')
        self.plots_dir = Path('Region_Focused_Results/Plots') # Use 'Plots' to match refined script
        
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Load region weights if provided via environment variable
        self.region_weights = {}
        weights_path = os.getenv('REGION_WEIGHTS_PATH')
        
        # Fallback: Look for region_weights.json in the script directory
        if not weights_path:
            possible_path = Path(__file__).parent / "region_weights.json"
            if possible_path.exists():
                weights_path = str(possible_path)
                print(f"Found region weights at default location: {weights_path}")
        
        if weights_path and os.path.isfile(weights_path):
            try:
                with open(weights_path, 'r') as f:
                    self.region_weights = json.load(f)
                print(f"Loaded region weights: {self.region_weights}")
            except Exception as e:
                print(f"Failed to load region weights from {weights_path}: {e}")
        else:
            print("No region weights provided; using default weighting.")
        
    def compute_psnr(self, pred, target):
        mse = F.mse_loss(pred, target)
        return 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))
    
    def get_weight_map(self, labels):
        """
        Create a pixel-wise weight map based on labels and self.region_weights.
        labels: [B, H, W]
        """
        B, H, W = labels.shape
        weight_map = torch.ones((B, 1, H, W), device=self.device)
        
        if not self.region_weights:
            return weight_map
            
        for region_name, weight in self.region_weights.items():
            if region_name in LABEL_MAP:
                label_id = LABEL_MAP[region_name]
                # Create mask for this label
                mask = (labels == label_id).unsqueeze(1).float() # [B, 1, H, W]
                # Apply weight: where mask is 1, value becomes weight. 
                # Note: Overlaps shouldn't happen with segmentation labels.
                # We want base 1.0, and if label is present, use weight.
                # If weight > 1, we increase importance.
                weight_map = torch.where(mask > 0.5, torch.tensor(weight, device=self.device), weight_map)
                
        return weight_map

    def train_epoch(self, loader, epoch):
        self.G.train()
        self.D.train()
        
        total_g = 0
        total_d = 0
        total_edge = 0
        total_psnr = 0
        
        pbar = tqdm(loader, desc=f'Epoch {epoch}')
        
        for batch_idx, (sparse, t, t_next, target, target_next, labels) in enumerate(pbar):
            sparse = sparse.to(self.device)
            t = t.to(self.device)
            t_next = t_next.to(self.device)
            target = target.to(self.device)
            target_next = target_next.to(self.device)
            labels = labels.to(self.device)
            
            # Create weight map for this batch
            weight_map = self.get_weight_map(labels)
            
            if batch_idx % 2 == 0:
                self.opt_D.zero_grad()
                
                fake = self.G(sparse, t).detach()
                
                real_scores = self.D(target)
                fake_scores = self.D(fake)
                
                d_loss = 0
                for real_score, fake_score in zip(real_scores, fake_scores):
                    d_loss += F.relu(1 - real_score).mean() + F.relu(1 + fake_score).mean()
                d_loss = d_loss / len(real_scores)
                
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.D.parameters(), 1.0)
                self.opt_D.step()
                
                total_d += d_loss.item()
            
            self.opt_G.zero_grad()
            
            fake = self.G(sparse, t)
            fake_scores = self.D(fake)
            
            adv_loss = 0
            for fake_score in fake_scores:
                adv_loss += -fake_score.mean()
            adv_loss = adv_loss / len(fake_scores)
            
            # Weighted L1 Loss
            l1_diff = torch.abs(fake - target)
            l1_loss = (l1_diff * weight_map).mean()
            
            mse_loss = F.mse_loss(fake, target)
            
            # Weighted Edge Loss
            edge_loss = self.anatomy_loss(fake, target, self.device, weight_map)
            
            ssim_loss = self.ssim_loss(fake, target)
            freq_loss = self.freq_loss(fake, target)
            
             # Z-Consistency Loss
            fake_next = self.G(sparse, t_next)
            delta_fake = fake_next - fake
            delta_target = target_next - target
            consistency_loss = F.l1_loss(delta_fake, delta_target)
            
            g_loss = (5.0 * adv_loss + 
                     10.0 * mse_loss + 
                     10.0 * l1_loss + 
                     20.0 * edge_loss + 
                     10.0 * ssim_loss + 
                     10.0 * consistency_loss + # New term 
                     5.0 * freq_loss)
            
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.G.parameters(), 1.0)
            self.opt_G.step()
            
            psnr = self.compute_psnr(fake, target).item()
            
            total_g += g_loss.item()
            total_edge += edge_loss.item()
            total_psnr += psnr
            
            pbar.set_postfix({
                'G': f'{g_loss.item():.3f}',
                'D': f'{d_loss.item():.3f}' if batch_idx % 2 == 0 else '-',
                'Edge': f'{edge_loss.item():.3f}',
                'Z-Con': f'{consistency_loss.item():.3f}',
                'PSNR': f'{psnr:.1f}'
            })
        
        avg_g = total_g / len(loader)
        avg_d = total_d / (len(loader) // 2)
        avg_edge = total_edge / len(loader)
        avg_psnr = total_psnr / len(loader)
        
        print(f"\nEpoch {epoch}: G={avg_g:.3f}, D={avg_d:.3f}, Edge={avg_edge:.3f}, PSNR={avg_psnr:.2f}dB")
        
        self.history['g_loss'].append(avg_g)
        self.history['d_loss'].append(avg_d)
        self.history['edge_loss'].append(avg_edge)
        self.history['psnr'].append(avg_psnr)
        
        # Log to CSV
        log_path = 'loss_history.csv'
        file_exists = os.path.isfile(log_path)
        with open(log_path, 'a') as f:
            if not file_exists:
                f.write('epoch,g_loss,d_loss,edge_loss,psnr\n')
            f.write(f'{epoch},{avg_g:.6f},{avg_d:.6f},{avg_edge:.6f},{avg_psnr:.6f}\n')
        
        if avg_psnr > self.best_psnr:
            self.best_psnr = avg_psnr
            self.save(self.weights_dir / 'best_model.pth', epoch)
            print(f"★ NEW BEST: {self.best_psnr:.2f}dB")
            
        # ALWAYS save the latest model so we can resume exactly from here
        self.save(self.weights_dir / 'latest_model.pth', epoch)
        
        return avg_g, avg_d, avg_psnr
    
    def train(self, dataset, epochs=500, batch_size=4):
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True
        )
        
        print(f"\nTraining from epoch {self.start_epoch} to {epochs}\n")
        
        for epoch in range(self.start_epoch, epochs + 1):
            self.train_epoch(loader, epoch)
            
            if epoch % 1 == 0:
                self.visualize(dataset, epoch)
                self.plot_curves()
            
            if epoch % 50 == 0:
                self.save(self.weights_dir / f'checkpoint_epoch_{epoch}.pth', epoch)
        
        print(f"\n✓ Complete! Best PSNR: {self.best_psnr:.2f}dB")
    
    def visualize(self, dataset, epoch):
        self.G.eval()
        
        subj = dataset.subjects[0]
        sparse = np.load(subj['sparse'])
        dense = np.load(subj['dense'])
        
        sparse_t = torch.from_numpy(sparse).unsqueeze(0).unsqueeze(0).to(self.device)
        
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))
        
        positions = [0.2, 0.35, 0.5, 0.65, 0.8]
        
        for i, pos in enumerate(positions):
            idx = int(pos * (dense.shape[0] - 1))
            t = torch.tensor([pos], device=self.device)
            
            with torch.no_grad():
                pred_slice = self.G(sparse_t, t).squeeze().cpu().numpy()
            
            target_slice = dense[idx]
            
            axes[0, i].imshow(target_slice, cmap='gray', vmin=0, vmax=1)
            axes[0, i].set_title(f'GT {idx}')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(pred_slice, cmap='gray', vmin=0, vmax=1)
            psnr = 20 * np.log10(1.0 / np.sqrt(np.mean((pred_slice - target_slice)**2) + 1e-8))
            axes[1, i].set_title(f'Pred ({psnr:.1f}dB)')
            axes[1, i].axis('off')
            
            diff = np.abs(pred_slice - target_slice)
            axes[2, i].imshow(diff, cmap='hot', vmin=0, vmax=0.3)
            axes[2, i].set_title(f'Diff')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'epoch_{epoch:03d}.png', dpi=150)
        plt.close()
    
    def plot_curves(self):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        epochs = range(1, len(self.history['psnr']) + 1)
        
        axes[0, 0].plot(epochs, self.history['psnr'], 'b-', linewidth=2)
        axes[0, 0].axhline(self.best_psnr, color='g', linestyle='--')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('PSNR (dB)')
        axes[0, 0].set_title('PSNR')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(epochs, self.history['edge_loss'], 'purple', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Edge Loss')
        axes[0, 1].set_title('Anatomical Edge Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(epochs, self.history['g_loss'], 'r-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Generator Loss')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(epochs, self.history['d_loss'], 'orange', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Discriminator Loss')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'training_curves.png', dpi=100)
        plt.close()
    
    def save(self, path, epoch):
        # Unwrap models if DataParallel
        G = self.G.module if isinstance(self.G, nn.DataParallel) else self.G
        D = self.D.module if isinstance(self.D, nn.DataParallel) else self.D
        
        torch.save({
            'epoch': epoch,
            'generator': G.state_dict(),
            'discriminator': D.state_dict(),
            'opt_G': self.opt_G.state_dict(),
            'opt_D': self.opt_D.state_dict(),
            'best_psnr': self.best_psnr,
            'history': self.history
        }, path)
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        
        # Helper to strip 'module.' if it somehow got in
        def clean_state_dict(sd):
            new_sd = {}
            for k, v in sd.items():
                if k.startswith('module.'):
                    new_sd[k[7:]] = v
                else:
                    new_sd[k] = v
            return new_sd

        # Load into underlying models
        G = self.G.module if isinstance(self.G, nn.DataParallel) else self.G
        D = self.D.module if isinstance(self.D, nn.DataParallel) else self.D
        
        G.load_state_dict(clean_state_dict(checkpoint['generator']))
        D.load_state_dict(clean_state_dict(checkpoint['discriminator']))
        
        self.opt_G.load_state_dict(checkpoint['opt_G'])
        self.opt_D.load_state_dict(checkpoint['opt_D'])
        
        self.best_psnr = checkpoint.get('best_psnr', 0)
        self.history = checkpoint.get('history', {'g_loss': [], 'd_loss': [], 'edge_loss': [], 'psnr': []})
        self.start_epoch = checkpoint.get('epoch', 1) + 1
        print(f"Loaded from epoch {self.start_epoch - 1}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr_g', type=float, default=5e-5)
    parser.add_argument('--lr_d', type=float, default=1e-5)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    
    dataset = MRIDataset(args.data_dir, samples_per_volume=64, augment=True)
    
    trainer = Trainer(device=args.device, lr_g=args.lr_g, lr_d=args.lr_d)
    
    if args.checkpoint and Path(args.checkpoint).exists():
        trainer.load_checkpoint(args.checkpoint)
    
    trainer.train(dataset, epochs=args.epochs, batch_size=args.batch_size)


if __name__ == '__main__':
    main()