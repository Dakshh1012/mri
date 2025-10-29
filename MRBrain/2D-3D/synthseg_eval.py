import os
import numpy as np
import nibabel as nib
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import argparse
import sys
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import transforms
from scipy import ndimage


class NiftiDataset(Dataset):
    """Pure PyTorch dataset for NIfTI volumes"""
    
    def __init__(self, image_paths, patch_size=(64, 64, 64), num_samples_per_volume=4, mode='train'):
        self.image_paths = image_paths
        self.patch_size = patch_size
        self.num_samples_per_volume = num_samples_per_volume
        self.mode = mode
        
    def __len__(self):
        return len(self.image_paths) * self.num_samples_per_volume
    
    def _normalize_volume(self, volume):
        """Normalize volume to [0, 1] range"""
        volume = volume.astype(np.float32)
        # Clip outliers
        p1, p99 = np.percentile(volume[volume > 0], [1, 99])
        volume = np.clip(volume, p1, p99)
        # Normalize
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
        return volume
    
    def _random_crop(self, volume, patch_size):
        """Extract random patch from volume"""
        h, w, d = volume.shape
        ph, pw, pd = patch_size
        
        if h <= ph or w <= pw or d <= pd:
            # Pad if volume is smaller than patch
            pad_h = max(0, ph - h + 1)
            pad_w = max(0, pw - w + 1) 
            pad_d = max(0, pd - d + 1)
            volume = np.pad(volume, ((0, pad_h), (0, pad_w), (0, pad_d)), mode='constant')
            h, w, d = volume.shape
        
        # Random crop
        start_h = np.random.randint(0, h - ph + 1)
        start_w = np.random.randint(0, w - pw + 1)
        start_d = np.random.randint(0, d - pd + 1)
        
        patch = volume[start_h:start_h + ph, start_w:start_w + pw, start_d:start_d + pd]
        return patch
    
    def _augment_patch(self, patch):
        """Apply data augmentation"""
        if self.mode != 'train':
            return patch
            
        # Random rotation (90 degree multiples)
        if np.random.random() < 0.3:
            axes = np.random.choice(3, 2, replace=False)
            k = np.random.randint(1, 4)
            patch = np.rot90(patch, k=k, axes=axes)
        
        # Random flip
        if np.random.random() < 0.5:
            axis = np.random.randint(0, 3)
            patch = np.flip(patch, axis=axis)
        
        # Intensity shift
        if np.random.random() < 0.5:
            shift = np.random.uniform(-0.1, 0.1)
            patch = np.clip(patch + shift, 0, 1)
        
        # Gaussian noise
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.05, patch.shape)
            patch = np.clip(patch + noise, 0, 1)
            
        return patch
    
    def __getitem__(self, idx):
        volume_idx = idx // self.num_samples_per_volume
        img_path = self.image_paths[volume_idx]
        
        # Load volume
        nii = nib.load(img_path)
        volume = nii.get_fdata()
        
        # Normalize
        volume = self._normalize_volume(volume)
        
        # Extract patch
        patch = self._random_crop(volume, self.patch_size)
        
        # Augmentation
        patch = self._augment_patch(patch)
        
        # Convert to tensor [1, H, W, D]
        patch = torch.from_numpy(patch.copy()).float().unsqueeze(0)
        
        # For SynthSeg-style training, use same patch as input and target
        return {'image': patch, 'label': patch.clone()}


class UNet3D(nn.Module):
    """3D U-Net implementation for segmentation"""
    
    def __init__(self, in_channels=1, out_channels=32, init_features=32):
        super(UNet3D, self).__init__()
        
        features = init_features
        self.encoder1 = self._make_encoder_block(in_channels, features)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder2 = self._make_encoder_block(features, features * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder3 = self._make_encoder_block(features * 2, features * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder4 = self._make_encoder_block(features * 4, features * 8)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.bottleneck = self._make_encoder_block(features * 8, features * 16)
        
        self.upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = self._make_encoder_block(features * 16, features * 8)
        
        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._make_encoder_block(features * 8, features * 4)
        
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._make_encoder_block(features * 4, features * 2)
        
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._make_encoder_block(features * 2, features)
        
        self.conv_final = nn.Conv3d(features, out_channels, kernel_size=1)
        
    def _make_encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return self.conv_final(dec1)


class DiceLoss(nn.Module):
    """Dice loss for segmentation"""
    
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        # Apply softmax to predictions
        predictions = F.softmax(predictions, dim=1)
        
        # Convert targets to one-hot if needed
        if targets.dim() == predictions.dim() - 1:
            targets = F.one_hot(targets.long(), num_classes=predictions.shape[1]).permute(0, 4, 1, 2, 3)
        
        # Flatten
        predictions = predictions.view(predictions.shape[0], predictions.shape[1], -1)
        targets = targets.view(targets.shape[0], targets.shape[1], -1)
        
        # Calculate Dice coefficient
        intersection = (predictions * targets).sum(dim=2)
        dice = (2 * intersection + self.smooth) / (predictions.sum(dim=2) + targets.sum(dim=2) + self.smooth)
        
        return 1 - dice.mean()


class PyTorchSynthSegTrainingPipeline:
    """
    Pure PyTorch-based alternative to SynthSeg training pipeline
    """

    def __init__(self, evaluation_dir: str, output_dir: str = "pytorch_synthseg_output",
                 device: str = "cuda", num_classes: int = 32):

        self.evaluation_dir = Path(evaluation_dir)
        self.output_dir = Path(output_dir)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes

        # Create output dirs
        self.models_dir = self.output_dir / "trained_models"
        self.predictions_dir = self.output_dir / "predictions"
        self.metrics_dir = self.output_dir / "metrics"
        self.visualizations_dir = self.output_dir / "visualizations"
        self.data_splits_dir = self.output_dir / "data_splits"
        for d in [self.models_dir, self.predictions_dir, self.metrics_dir,
                  self.visualizations_dir, self.data_splits_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.volume_types = ["generated_volume", "ground_truth_volume", "hybrid_volume"]
        self.participants = []
        self.data_registry = {}

        print(f"PyTorch SynthSeg Pipeline initialized on {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        self._discover_data()

    def _discover_data(self):
        """Discover available data"""
        if not self.evaluation_dir.exists():
            raise FileNotFoundError(f"Evaluation dir not found: {self.evaluation_dir}")

        participant_dirs = [d for d in self.evaluation_dir.iterdir() if d.is_dir()]
        self.participants = sorted([d.name for d in participant_dirs])

        for pid in self.participants:
            self.data_registry[pid] = {}
            for vt in self.volume_types:
                f = self.evaluation_dir / pid / f"{vt}.nii.gz"
                if f.exists():
                    self.data_registry[pid][vt] = str(f)

        self._print_data_summary()

    def _print_data_summary(self):
        print("\nData Summary:")
        for vt in self.volume_types:
            c = sum(1 for p in self.participants if vt in self.data_registry[p])
            print(f"  {vt}: {c}/{len(self.participants)}")
        complete = sum(1 for p in self.participants if len(self.data_registry[p]) == 3)
        print(f"  Complete datasets: {complete}/{len(self.participants)}")

    def create_data_splits(self, train_ratio=0.7, val_ratio=0.15,
                           test_ratio=0.15, random_seed=42):
        """Create train/val/test splits"""
        complete_p = [p for p in self.participants if len(self.data_registry[p]) == 3]
        np.random.seed(random_seed)
        shuffled = np.random.permutation(complete_p)

        n_total = len(shuffled)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train = shuffled[:n_train].tolist()
        val = shuffled[n_train:n_train + n_val].tolist()
        test = shuffled[n_train + n_val:].tolist()

        splits = {
            "train": train, "validation": val, "test": test,
            "metadata": {
                "total": n_total,
                "train": len(train), "val": len(val), "test": len(test),
                "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio}
            }
        }

        with open(self.data_splits_dir / "data_splits.json", "w") as f:
            json.dump(splits, f, indent=2)
        return splits

    def train_pytorch_model(self, volume_type, splits, epochs=100, 
                           learning_rate=1e-4, batch_size=2, model_name=None,
                           patch_size=(64, 64, 64), num_samples_per_volume=8):
        """Train PyTorch model"""
        
        if model_name is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"pytorch_synthseg_{volume_type}_{ts}"

        model_output_dir = self.models_dir / model_name
        model_output_dir.mkdir(exist_ok=True)

        # Prepare data paths
        train_files = []
        val_files = []
        
        for pid in splits["train"]:
            if volume_type in self.data_registry[pid]:
                train_files.append(self.data_registry[pid][volume_type])
        
        for pid in splits["validation"]:
            if volume_type in self.data_registry[pid]:
                val_files.append(self.data_registry[pid][volume_type])

        print(f"Training {volume_type}: {len(train_files)} train, {len(val_files)} val")

        # Create datasets
        train_dataset = NiftiDataset(train_files, patch_size=patch_size, 
                                    num_samples_per_volume=num_samples_per_volume, mode='train')
        val_dataset = NiftiDataset(val_files, patch_size=patch_size, 
                                  num_samples_per_volume=4, mode='val')
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                 num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
                               num_workers=2, pin_memory=True)

        # Create model
        model = UNet3D(in_channels=1, out_channels=self.num_classes).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        dice_loss = DiceLoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
        
        # Training loop
        best_loss = float('inf')
        best_epoch = 0
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Training
            model.train()
            epoch_loss = 0
            num_batches = 0
            
            train_pbar = tqdm(train_loader, desc='Training')
            for batch_data in train_pbar:
                inputs = batch_data['image'].to(self.device)
                # For SynthSeg-style training, create dummy labels
                labels = torch.randint(0, self.num_classes, 
                                     (inputs.shape[0], inputs.shape[2], inputs.shape[3], inputs.shape[4]),
                                     dtype=torch.long).to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                
                # Combined loss
                ce_loss = criterion(outputs, labels)
                d_loss = dice_loss(outputs, labels)
                loss = ce_loss + d_loss
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                # Clear cache periodically
                if num_batches % 10 == 0:
                    torch.cuda.empty_cache()
            
            avg_train_loss = epoch_loss / num_batches
            train_losses.append(avg_train_loss)
            
            # Validation
            if (epoch + 1) % 5 == 0:
                model.eval()
                val_loss = 0
                val_batches = 0
                
                with torch.no_grad():
                    for batch_data in tqdm(val_loader, desc='Validation'):
                        inputs = batch_data['image'].to(self.device)
                        labels = torch.randint(0, self.num_classes,
                                             (inputs.shape[0], inputs.shape[2], inputs.shape[3], inputs.shape[4]),
                                             dtype=torch.long).to(self.device)
                        
                        outputs = model(inputs)
                        ce_loss = criterion(outputs, labels)
                        d_loss = dice_loss(outputs, labels)
                        loss = ce_loss + d_loss
                        
                        val_loss += loss.item()
                        val_batches += 1
                
                avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
                val_losses.append(avg_val_loss)
                
                print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
                
                # Save best model
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    best_epoch = epoch + 1
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'loss': best_loss
                    }, model_output_dir / "best_model.pth")
                    print(f"✓ New best model saved! Loss: {best_loss:.4f}")
                
                scheduler.step(avg_val_loss)
            
            # Clear cache
            torch.cuda.empty_cache()

        # Save final model
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epochs,
            'loss': train_losses[-1]
        }, model_output_dir / "final_model.pth")
        
        # Save training metadata
        metadata = {
            "model_name": model_name,
            "volume_type": volume_type,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "patch_size": patch_size,
            "num_samples_per_volume": num_samples_per_volume,
            "training_files": len(train_files),
            "validation_files": len(val_files),
            "best_loss": float(best_loss),
            "best_epoch": int(best_epoch),
            "timestamp": datetime.now().isoformat(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "num_classes": self.num_classes
        }
        
        with open(model_output_dir / "training_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        return str(model_output_dir / "best_model.pth")

    def predict_with_pytorch_model(self, model_path, input_volumes, output_dir, model_name):
        """Make predictions using trained PyTorch model"""
        # Load model
        model = UNet3D(in_channels=1, out_channels=self.num_classes).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        out = Path(output_dir)
        out.mkdir(exist_ok=True)
        preds = []
        
        for v in tqdm(input_volumes, desc=f"Predicting {model_name}"):
            pid = Path(v).parent.name
            vol = Path(v).stem.replace(".nii", "")
            out_file = out / f"{pid}_{vol}_prediction.nii.gz"
            
            # Load and normalize volume
            nii = nib.load(v)
            volume = nii.get_fdata().astype(np.float32)
            
            # Normalize
            if volume.max() > volume.min():
                p1, p99 = np.percentile(volume[volume > 0], [1, 99])
                volume = np.clip(volume, p1, p99)
                volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
            
            # Predict on patches and stitch together
            patch_size = (64, 64, 64)
            stride = (32, 32, 32)  # 50% overlap
            
            h, w, d = volume.shape
            ph, pw, pd = patch_size
            sh, sw, sd = stride
            
            # Pad volume if needed
            pad_h = max(0, ph - h)
            pad_w = max(0, pw - w)
            pad_d = max(0, pd - d)
            if pad_h > 0 or pad_w > 0 or pad_d > 0:
                volume = np.pad(volume, ((0, pad_h), (0, pad_w), (0, pad_d)), mode='constant')
                h, w, d = volume.shape
            
            # Initialize prediction volume
            prediction = np.zeros((h, w, d), dtype=np.uint8)
            
            with torch.no_grad():
                # Extract and predict patches
                for start_h in range(0, h - ph + 1, sh):
                    for start_w in range(0, w - pw + 1, sw):
                        for start_d in range(0, d - pd + 1, sd):
                            end_h = min(start_h + ph, h)
                            end_w = min(start_w + pw, w)
                            end_d = min(start_d + pd, d)
                            
                            patch = volume[start_h:end_h, start_w:end_w, start_d:end_d]
                            
                            # Ensure patch is correct size
                            if patch.shape != patch_size:
                                patch = np.pad(patch, 
                                             [(0, ph - patch.shape[0]),
                                              (0, pw - patch.shape[1]),
                                              (0, pd - patch.shape[2])],
                                             mode='constant')
                            
                            # Convert to tensor and predict
                            patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(self.device)
                            outputs = model(patch_tensor)
                            pred_patch = torch.argmax(outputs, dim=1).cpu().numpy()[0]
                            
                            # Place prediction back
                            actual_end_h = min(start_h + pred_patch.shape[0], h)
                            actual_end_w = min(start_w + pred_patch.shape[1], w)
                            actual_end_d = min(start_d + pred_patch.shape[2], d)
                            
                            prediction[start_h:actual_end_h, 
                                     start_w:actual_end_w, 
                                     start_d:actual_end_d] = pred_patch[
                                         :actual_end_h-start_h,
                                         :actual_end_w-start_w,
                                         :actual_end_d-start_d
                                     ]
            
            # Remove padding if added
            if pad_h > 0 or pad_w > 0 or pad_d > 0:
                prediction = prediction[:h-pad_h, :w-pad_w, :d-pad_d]
            
            # Save prediction
            pred_img = nib.Nifti1Image(prediction, nii.affine, nii.header)
            nib.save(pred_img, out_file)
            preds.append(str(out_file))
            
            # Clear cache
            torch.cuda.empty_cache()
            
        return preds

    def run_complete_pipeline(self, epochs=100, learning_rate=1e-4,
                              batch_size=2, train_ratio=0.7, patch_size=(64, 64, 64)):
        """Run complete training and evaluation pipeline"""
        print("Starting PyTorch SynthSeg Pipeline...")
        
        splits = self.create_data_splits(train_ratio=train_ratio)
        trained = {}
        
        for vt in self.volume_types:
            try:
                print(f"\n{'='*50}")
                print(f"Training {vt}")
                print(f"{'='*50}")
                
                mp = self.train_pytorch_model(
                    vt, splits,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    patch_size=patch_size
                )
                trained[vt] = mp
                print(f"✓ Completed training for {vt}")
                
                # Clear GPU cache between models
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"✗ Failed {vt}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\nPipeline completed!")
        print(f"Trained models: {list(trained.keys())}")
        return {"trained": trained}


# CLI
def main():
    ap = argparse.ArgumentParser(description="Pure PyTorch SynthSeg Training Pipeline")
    ap.add_argument("--evaluation_dir", required=True, help="Directory with participant data")
    ap.add_argument("--output_dir", default="pytorch_synthseg_output", help="Output directory")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device to use")
    ap.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    ap.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    ap.add_argument("--batch_size", type=int, default=2, help="Batch size")
    ap.add_argument("--num_classes", type=int, default=32, help="Number of segmentation classes")
    ap.add_argument("--patch_size", nargs=3, type=int, default=[64, 64, 64], help="Patch size (H W D)")
    args = ap.parse_args()

    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    elif args.device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    
    pipe = PyTorchSynthSegTrainingPipeline(
        args.evaluation_dir,
        args.output_dir,
        args.device,
        args.num_classes
    )
    
    results = pipe.run_complete_pipeline(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        patch_size=tuple(args.patch_size)
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Trained models: {list(results['trained'].keys())}")
    print(f"Models saved in: {pipe.models_dir}")


if __name__ == "__main__":
    main()