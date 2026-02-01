import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, activation='elu'):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        if activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Identity()

    def forward(self, x):
        return self.activation(self.conv(x))

class Encoder(nn.Module):
    def __init__(self, in_channels, features, levels, conv_per_level=2, feat_mult=2):
        super(Encoder, self).__init__()
        self.levels = levels
        self.down_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        self.max_pool = nn.MaxPool3d(2, stride=2)
        
        current_in = in_channels
        for level in range(levels):
            level_blocks = nn.ModuleList()
            out_channels = int(features * (feat_mult ** level))
            
            for i in range(conv_per_level):
                level_blocks.append(ConvBlock(current_in, out_channels))
                current_in = out_channels
            
            self.down_blocks.append(nn.Sequential(*level_blocks))
            self.bn_blocks.append(nn.BatchNorm3d(out_channels))

    def forward(self, x):
        skips = []
        for i in range(self.levels):
            x = self.down_blocks[i](x)
            x = self.bn_blocks[i](x)
            if i < self.levels - 1:
                skips.append(x)
                x = self.max_pool(x)
        return x, skips

class Decoder(nn.Module):
    def __init__(self, in_channels, out_labels, features, levels, conv_per_level=2, feat_mult=2):
        super(Decoder, self).__init__()
        self.levels = levels
        self.up_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # Calculate channels for the bottom level (which is the input to decoder)
        # The encoder output has features * (feat_mult ** (levels - 1)) channels
        current_in = int(features * (feat_mult ** (levels - 1)))
        
        for level in range(levels - 1):
            # Going up from bottom
            # Level index in decoder (0 is bottom-most upsampling)
            # Corresponding encoder level index: levels - 2 - level
            
            encoder_level = levels - 2 - level
            skip_channels = int(features * (feat_mult ** encoder_level))
            out_channels = skip_channels # We want to match the skip connection channels
            
            # Input to this level is (upsampled previous) + skip
            level_in = current_in + skip_channels
            
            level_blocks = nn.ModuleList()
            
            # First conv reduces channel count
            level_blocks.append(ConvBlock(level_in, out_channels))
            current_in = out_channels
            
            for i in range(1, conv_per_level):
                level_blocks.append(ConvBlock(current_in, out_channels))
            
            self.up_blocks.append(nn.Sequential(*level_blocks))
            self.bn_blocks.append(nn.BatchNorm3d(out_channels))
            
        self.final_conv = nn.Conv3d(current_in, out_labels, kernel_size=1)

    def forward(self, x, skips):
        for i in range(self.levels - 1):
            x = self.upsample(x)
            # Skip connection from corresponding encoder level
            # Skips are stored from top to bottom (0 to levels-2)
            # We need them in reverse order
            skip = skips[-(i+1)]
            x = torch.cat([skip, x], dim=1)
            x = self.up_blocks[i](x)
            x = self.bn_blocks[i](x)
        
        return self.final_conv(x)

class SynthSegParc(nn.Module):
    def __init__(self, n_labels_seg, n_labels_parc, features=24, levels=5, feat_mult=2):
        super(SynthSegParc, self).__init__()
        self.unet_seg = UNet3D(in_channels=1, out_labels=n_labels_seg, features=features, levels=levels, feat_mult=feat_mult)
        
        # Parcellation UNet takes input image (1 channel) + one-hot encoded segmentation (2 channels: background/cortex)
        # Wait, let's check mri_pipeline_clean.py line 598-599
        # last_tensor = KL.Lambda(lambda x: tf.one_hot(tf.cast(x, 'int32'), depth=2, axis=-1))(last_tensor)
        # last_tensor = KL.Lambda(lambda x: tf.cast(tf.concat(x, axis=-1), 'float32'))([input_image, last_tensor])
        # So input is 1 + 2 = 3 channels.
        self.unet_parc = UNet3D(in_channels=3, out_labels=n_labels_parc, features=features, levels=levels, feat_mult=feat_mult)
        
        self.gaussian_blur = GaussianBlur(sigma=0.5)
        
        self.n_labels_seg = n_labels_seg

    def forward(self, x, flip_indices=None):
        # Segmentation
        seg_out = self.unet_seg(x) # (B, n_labels_seg, D, H, W)
        
        if flip_indices is not None:
            # Flip input along D (axis 2) - assuming RAS/Sagittal is first spatial dim
            x_flipped = torch.flip(x, dims=[2])
            seg_flipped = self.unet_seg(x_flipped)
            # Flip output back
            seg_flipped = torch.flip(seg_flipped, dims=[2])
            # Reorder channels
            seg_flipped = seg_flipped[:, flip_indices, ...]
            
            seg_out = 0.5 * (seg_out + seg_flipped)
        
        # Softmax output of segmentation
        seg_probs = seg_out # Already softmaxed in UNet3D.
        # Wait, UNet3D applies softmax at the end.
        # If we average probabilities, it's fine.
        # But if UNet3D returns logits, we should softmax after averaging?
        # mri_pipeline_clean.py averages the *predictions*.
        # "last_tensor = KL.Lambda(lambda x: 0.5 * (x[0] + x[1]))([seg, last_tensor])"
        # And "net" output was "pred_tensor" which is Softmax (if final_pred_activation='softmax').
        # So it averages probabilities.
        
        # Use hard argmax to match TensorFlow implementation
        # TensorFlow implementation:
        # last_tensor = KL.Lambda(lambda x: tf.cast(tf.argmax(x, axis=-1), 'int32'))(last_tensor)
        # ...
        # last_tensor = ConvertLabels(labels_segmentation, parcellation_masking_values)(last_tensor)
        # ...
        
        # Get hard segmentation indices
        seg_argmax = torch.argmax(seg_out, dim=1) # (B, D, H, W)
        
        # Create mask for cortex (labels 3 and 42, which are at indices 2 and 20)
        # We use indices because the model outputs channels corresponding to sorted unique labels.
        mask = (seg_argmax == 2) | (seg_argmax == 20)
        mask = mask.float().unsqueeze(1) # (B, 1, D, H, W)
        
        cortex_prob = mask
        background_prob = 1.0 - mask
        
        # Concatenate: Image, Background, Cortex
        # Image is (B, 1, H, W, D) -> Wait, input x is (B, 1, D, H, W)
        parc_input = torch.cat([x, background_prob, cortex_prob], dim=1)
        
        parc_out = self.unet_parc(parc_input)
        
        # Apply Gaussian Blur to outputs to match TensorFlow implementation
        if hasattr(self, 'gaussian_blur'):
            seg_out = self.gaussian_blur(seg_out)
            parc_out = self.gaussian_blur(parc_out)
            
        return seg_out, parc_out

class GaussianBlur(nn.Module):
    def __init__(self, sigma=0.5, channels=1):
        super(GaussianBlur, self).__init__()
        self.sigma = sigma
        self.channels = channels
        
        # Create kernel
        # Default truncate=4.0 like scipy
        size = int(2 * 4 * sigma + 1)
        if size % 2 == 0: size += 1
        
        # Create 3D kernel
        coords = torch.arange(size).float() - (size - 1) / 2
        grid = torch.stack(torch.meshgrid([coords, coords, coords]), dim=-1)
        dist_sq = (grid**2).sum(dim=-1)
        kernel = torch.exp(-dist_sq / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        
        # Reshape for conv3d: (out_channels, in_channels/groups, k, k, k)
        # We use groups=channels, so weight shape is (channels, 1, k, k, k)
        kernel = kernel.view(1, 1, size, size, size)
        self.register_buffer('kernel', kernel)
        
    def forward(self, x):
        # x: (B, C, D, H, W)
        b, c, d, h, w = x.shape
        # Replicate kernel for each channel
        kernel = self.kernel.repeat(c, 1, 1, 1, 1)
        # Apply depthwise convolution (groups=c)
        return F.conv3d(x, kernel, padding=self.kernel.shape[2]//2, groups=c)


class SynthSeg(nn.Module):
    def __init__(self, in_channels=1, out_labels=33, features=24, levels=5, feat_mult=2, sigma=0.5):
        super(SynthSeg, self).__init__()
        self.unet = UNet3D(in_channels=in_channels, out_labels=out_labels, features=features, levels=levels, feat_mult=feat_mult)
        self.gaussian_blur = GaussianBlur(sigma=sigma)
        
    def forward(self, x, flip_indices=None):
        seg_out = self.unet(x)
        
        if flip_indices is not None:
            # Flip input along D (axis 2)
            x_flipped = torch.flip(x, dims=[2])
            seg_flipped = self.unet(x_flipped)
            # Flip output back
            seg_flipped = torch.flip(seg_flipped, dims=[2])
            # Reorder channels
            seg_flipped = seg_flipped[:, flip_indices, ...]
            
            seg_out = 0.5 * (seg_out + seg_flipped)
            
        seg_out = self.gaussian_blur(seg_out)
        return seg_out

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_labels=33, features=24, levels=5, feat_mult=2):
        super(UNet3D, self).__init__()
        self.encoder = Encoder(in_channels, features, levels, feat_mult=feat_mult)
        self.decoder = Decoder(features, out_labels, features, levels, feat_mult=feat_mult)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.decoder(x, skips)
        return self.softmax(x)

# Helper to load weights from Keras model (placeholder for now)
def load_keras_weights(pytorch_model, keras_h5_path):
    # This function will need to map Keras layer names/weights to PyTorch
    pass
