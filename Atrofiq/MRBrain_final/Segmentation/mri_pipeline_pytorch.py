#!/usr/bin/env python
import os
import sys
import csv
import glob
import time
import argparse
import traceback
import numpy as np
import nibabel as nib
import torch
from datetime import timedelta
from itertools import combinations
from scipy.ndimage import label as scipy_label
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter, distance_transform_edt, binary_fill_holes

# Import PyTorch models from local file
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from synthseg_pytorch import SynthSegParc, UNet3D

def main():
    parser = argparse.ArgumentParser(description="SynthSeg PyTorch", epilog='\n')
    parser.add_argument("--i", help="Image(s) to segment. Can be a path to an image or to a folder.")
    parser.add_argument("--o", help="Segmentation output(s). Must be a folder if --i designates a folder.")
    parser.add_argument("--parc", action="store_true", help="(optional) Whether to perform cortex parcellation.")
    parser.add_argument("--robust", action="store_true", help="(optional) Whether to use robust predictions (slower). (Not fully supported in PyTorch version yet)")
    parser.add_argument("--fast", action="store_true", help="(optional) Bypass some processing for faster prediction.")
    parser.add_argument("--ct", action="store_true", help="(optional) Clip CT scans in Hounsfield scale to [0, 80]")
    parser.add_argument("--vol", help="(optional) Output CSV file with volumes for all structures and subjects.")
    parser.add_argument("--qc", help="(optional) Output CSV file with qc scores for all subjects. (Not supported in PyTorch version yet)")
    parser.add_argument("--post", help="(optional) Posteriors output(s). Must be a folder if --i designates a folder.")
    parser.add_argument("--resample", help="(optional) Resampled image(s). Must be a folder if --i is a folder.")
    parser.add_argument("--crop", nargs='+', type=int, help="(optional) Only analyse an image patch of the given size.")
    parser.add_argument("--threads", type=int, default=1, help="(optional) Number of cores to be used. Default is 1.")
    parser.add_argument("--cpu", action="store_true", help="(optional) Enforce running with CPU rather than GPU.")
    parser.add_argument("--v1", action="store_true", help="(optional) Use SynthSeg 1.0 (updated 25/06/22). (Not supported in PyTorch version yet)")
    
    parser.add_argument("--device", help="Device to use (e.g. cuda:0, cuda:1, cpu)", default="cuda:1")
    
    args = parser.parse_args()

    # Paths
    synthseg_home = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(synthseg_home, 'models')
    
    labels_segmentation = os.path.join(models_dir, 'synthseg_segmentation_labels_2.0.npy')
    labels_denoiser = os.path.join(models_dir, 'synthseg_denoiser_labels_2.0.npy')
    labels_parcellation = os.path.join(models_dir, 'synthseg_parcellation_labels.npy')
    names_segmentation_labels = os.path.join(models_dir, 'synthseg_segmentation_names_2.0.npy')
    names_parcellation_labels = os.path.join(models_dir, 'synthseg_parcellation_names.npy')
    topology_classes = os.path.join(models_dir, 'synthseg_topological_classes_2.0.npy')
    n_neutral_labels = 19
    
    # Model paths
    path_model_segmentation = os.path.join(models_dir, 'synthseg_2.0_pytorch.pth')
    path_model_parcellation = os.path.join(models_dir, 'synthseg_parc_2.0_pytorch.pth')
    
    if args.robust:
        print("Warning: Robust mode not fully implemented in PyTorch version. Using standard model.")
    if args.v1:
        print("Warning: v1 mode not implemented in PyTorch version. Using standard 2.0 model.")
        
    # Device
    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    if args.threads > 1:
        torch.set_num_threads(args.threads)
        print(f"PyTorch using {torch.get_num_threads()} threads")

    # Run prediction
    predict(
        path_images=args.i,
        path_segmentations=args.o,
        path_model_segmentation=path_model_segmentation,
        labels_segmentation=labels_segmentation,
        robust=args.robust,
        fast=args.fast,
        v1=False, # Force v2
        do_parcellation=args.parc,
        n_neutral_labels=n_neutral_labels,
        names_segmentation=names_segmentation_labels,
        labels_denoiser=labels_denoiser,
        path_posteriors=args.post,
        path_resampled=args.resample,
        path_volumes=args.vol,
        path_model_parcellation=path_model_parcellation,
        labels_parcellation=labels_parcellation,
        names_parcellation=names_parcellation_labels,
        cropping=args.crop,
        topology_classes=topology_classes,
        ct=args.ct,
        device=device
    )

def predict(path_images,
            path_segmentations,
            path_model_segmentation,
            labels_segmentation,
            robust,
            fast,
            v1,
            n_neutral_labels,
            names_segmentation,
            labels_denoiser,
            path_posteriors,
            path_resampled,
            path_volumes,
            do_parcellation,
            path_model_parcellation,
            labels_parcellation,
            names_parcellation,
            cropping,
            topology_classes,
            ct,
            device):
            
    # Prepare output files
    # Note: We are skipping QC for now as requested/implied by missing model
    outputs = prepare_output_files(path_images, path_segmentations, path_posteriors, path_resampled, path_volumes)
    path_images = outputs[0]
    path_segmentations = outputs[1]
    path_posteriors = outputs[2]
    path_resampled = outputs[3]
    path_volumes = outputs[4]
    unique_vol_file = outputs[5]

    # Load labels
    labels_segmentation, _ = get_list_labels(label_list=labels_segmentation)
    if (n_neutral_labels is not None) & (not fast) & (not robust):
        labels_segmentation, flip_indices, unique_idx = get_flip_indices(labels_segmentation, n_neutral_labels)
    else:
        labels_segmentation, unique_idx = np.unique(labels_segmentation, return_index=True)
        flip_indices = None
        
    names_segmentation = load_array_if_path(names_segmentation)[unique_idx]
    topology_classes = load_array_if_path(topology_classes, load_as_numpy=True)[unique_idx]
    labels_denoiser = np.unique(get_list_labels(labels_denoiser)[0])
    
    if do_parcellation:
        labels_parcellation, unique_i_parc = np.unique(get_list_labels(labels_parcellation)[0], return_index=True)
        labels_volumes = np.concatenate([labels_segmentation, labels_parcellation[1:]])
        names_parcellation = load_array_if_path(names_parcellation)[unique_i_parc][1:]
        names_volumes = np.concatenate([names_segmentation, names_parcellation])
    else:
        labels_volumes = labels_segmentation
        names_volumes = names_segmentation
        
    if not v1:
        labels_volumes = np.concatenate([labels_volumes, np.array([np.max(labels_volumes + 1)])])
        names_volumes = np.concatenate([names_volumes, np.array(['total intracranial'])])

    # Cropping
    if cropping is not None:
        cropping = reformat_to_list(cropping, length=3, dtype='int')
        min_pad = cropping
    else:
        min_pad = 128

    # Initialize CSV
    if unique_vol_file & (path_volumes[0] is not None):
        write_csv(path_volumes[0], None, True, labels_volumes, names_volumes, last_first=(not v1))

    # Build/Load Model
    model = build_model_pytorch(
        path_model_segmentation, 
        path_model_parcellation, 
        do_parcellation, 
        device
    )

    # Loop over images
    if len(path_images) <= 10:
        loop_info = LoopInfo(len(path_images), 1, 'predicting', True)
    else:
        loop_info = LoopInfo(len(path_images), 10, 'predicting', True)
        
    list_errors = list()
    
    # Prepare flip_indices tensor if needed
    flip_indices_tensor = None
    if flip_indices is not None:
        flip_indices_tensor = torch.from_numpy(flip_indices).long().to(device)
    
    for i in range(len(path_images)):
        loop_info.update(i)
        try:
            # Preprocess
            image, aff, h, im_res, shape, pad_idx, crop_idx = preprocess(
                path_image=path_images[i],
                ct=ct,
                crop=cropping,
                min_pad=min_pad,
                path_resample=path_resampled[i]
            )
            
            # Prepare for PyTorch: (D, H, W, 1) -> (1, 1, D, H, W)
            # image is currently (D, H, W, 1) from preprocess
            image_tensor = torch.from_numpy(image).float()
            image_tensor = image_tensor.permute(3, 0, 1, 2).unsqueeze(0) # (1, C, D, H, W)
            image_tensor = image_tensor.to(device)
            
            # Inference
            with torch.no_grad():
                if do_parcellation:
                    # Model returns (seg_out, parc_out)
                    post_patch_segmentation, post_patch_parcellation = model(image_tensor, flip_indices=flip_indices_tensor)
                else:
                    # Model returns seg_out
                    post_patch_segmentation = model(image_tensor, flip_indices=flip_indices_tensor)
                    post_patch_parcellation = None
            
            # Convert back to numpy: (1, C, D, H, W) -> (D, H, W, C)
            post_patch_segmentation = post_patch_segmentation.cpu().numpy()[0] # (C, D, H, W)
            post_patch_segmentation = np.transpose(post_patch_segmentation, (1, 2, 3, 0)) # (D, H, W, C)
            
            if post_patch_parcellation is not None:
                post_patch_parcellation = post_patch_parcellation.cpu().numpy()[0]
                post_patch_parcellation = np.transpose(post_patch_parcellation, (1, 2, 3, 0))

            # Postprocess
            seg, posteriors, volumes = postprocess(
                post_patch_seg=post_patch_segmentation,
                post_patch_parc=post_patch_parcellation,
                shape=shape,
                pad_idx=pad_idx,
                crop_idx=crop_idx,
                labels_segmentation=labels_segmentation,
                labels_parcellation=labels_parcellation,
                aff=aff,
                im_res=im_res,
                fast=fast,
                topology_classes=topology_classes,
                v1=v1
            )
            
            # Save
            save_volume(seg, aff, h, path_segmentations[i], dtype='int32')
            if path_posteriors[i] is not None:
                save_volume(posteriors, aff, h, path_posteriors[i], dtype='float32')
            if path_volumes[i] is not None:
                row = [os.path.basename(path_images[i]).replace('.nii.gz', '')] + [str(vol) for vol in volumes]
                write_csv(path_volumes[i], row, unique_vol_file, labels_volumes, names_volumes, last_first=(not v1))
                
        except Exception as e:
            list_errors.append(path_images[i])
            print('\nthe following problem occured with image %s :' % path_images[i])
            print(traceback.format_exc())
            print('resuming program execution\n')
            continue

    # Summary
    if len(list_errors) > 0:
        print('\nERROR: some problems occured for the following inputs:')
        for path_error_image in list_errors:
            print(path_error_image)
        sys.exit(1)
    else:
        print('\nDone!')

def build_model_pytorch(path_seg, path_parc, do_parcellation, device):
    """
    Build and load the SynthSeg PyTorch model.
    
    Note on checkpoint structure:
    - path_seg (synthseg_2.0_pytorch.pth): Contains UNet3D weights only (keys: encoder.*, decoder.*)
    - path_parc (synthseg_parc_2.0_pytorch.pth): Contains full SynthSegParc weights (keys: unet_seg.*, unet_parc.*)
    """
    n_labels_seg = 33  # Standard for SynthSeg 2.0
    
    # Import SynthSeg class
    from synthseg_pytorch import SynthSeg
    
    if do_parcellation:
        n_labels_parc = 69  # Standard for SynthSeg Parcellation
        model = SynthSegParc(n_labels_seg=n_labels_seg, n_labels_parc=n_labels_parc, features=24, levels=5, feat_mult=2)
        
        # Load from the parcellation checkpoint which has the full model
        if os.path.exists(path_parc):
            print(f"Loading full SynthSegParc model from {path_parc}...")
            state_dict = torch.load(path_parc, map_location=device)
            model.load_state_dict(state_dict, strict=False)
        else:
            raise FileNotFoundError(f"Parcellation model not found at {path_parc}")
            
        model.to(device)
        model.eval()
        return model
        
    else:
        # Load segmentation-only model
        model_seg = SynthSeg(in_channels=1, out_labels=n_labels_seg, features=24, levels=5, feat_mult=2)
        
        if os.path.exists(path_seg):
            print(f"Loading segmentation model from {path_seg}...")
            state_dict = torch.load(path_seg, map_location=device)
            # Load into model_seg.unet
            model_seg.unet.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"Segmentation model not found at {path_seg}")
             
        model_seg.to(device)
        model_seg.eval()
        return model_seg

# -----------------------------------------------------------------------------
# Helper Functions (Copied/Adapted from mri_pipeline_clean.py)
# -----------------------------------------------------------------------------

def prepare_output_files(path_images, out_seg, out_posteriors, out_resampled, out_volumes):
    if path_images is None:
        sys.exit('please specify an input file/folder (--i)')
    if out_seg is None:
        sys.exit('please specify an output file/folder (--o)')
        
    path_images = os.path.abspath(path_images)
    basename = os.path.basename(path_images)
    out_seg = os.path.abspath(out_seg)
    out_posteriors = os.path.abspath(out_posteriors) if (out_posteriors is not None) else out_posteriors
    out_resampled = os.path.abspath(out_resampled) if (out_resampled is not None) else out_resampled
    out_volumes = os.path.abspath(out_volumes) if (out_volumes is not None) else out_volumes
    
    # Handle directory input
    if os.path.isdir(path_images):
        path_images = list_images_in_folder(path_images)
        def helper_dir(path, name, file_type, suffix):
            unique_file = False
            if path is not None:
                if file_type == 'csv':
                    if path[-4:] != '.csv':
                        path += '.csv'
                    path = [path] * len(path_images)
                    recompute_files = [True] * len(path_images)
                    unique_file = True
                else:
                    path = [os.path.join(path, os.path.basename(p)) for p in path_images]
                    path = [p.replace('.nii', '_%s.nii' % suffix) for p in path]
                    path = [p.replace('.mgz', '_%s.mgz' % suffix) for p in path]
                    path = [p.replace('.npz', '_%s.npz' % suffix) for p in path]
                    recompute_files = [not os.path.isfile(p) for p in path]
                mkdir(os.path.dirname(path[0]))
            else:
                path = [None] * len(path_images)
                recompute_files = [False] * len(path_images)
            return path, recompute_files, unique_file
            
        out_seg, _, _ = helper_dir(out_seg, 'path_segmentations', '', 'synthseg')
        out_posteriors, _, _ = helper_dir(out_posteriors, 'path_posteriors', '', 'posteriors')
        out_resampled, _, _ = helper_dir(out_resampled, 'path_resampled', '', 'resampled')
        out_volumes, _, unique_volume_file = helper_dir(out_volumes, 'path_volumes', 'csv', '')
        
    else: # Single file
        path_images = [path_images]
        def helper_im(path, name, file_type, suffix):
            unique_file = False
            if path is not None:
                if file_type == 'csv':
                    if path[-4:] != '.csv':
                        path += '.csv'
                    recompute_files = [True]
                    unique_file = True
                else:
                    if ('.nii.gz' not in path) & ('.nii' not in path) & ('.mgz' not in path) & ('.npz' not in path):
                        file_name = os.path.basename(path_images[0]).replace('.nii', '_%s.nii' % suffix)
                        file_name = file_name.replace('.mgz', '_%s.mgz' % suffix)
                        file_name = file_name.replace('.npz', '_%s.npz' % suffix)
                        path = os.path.join(path, file_name)
                    recompute_files = [not os.path.isfile(path)]
                mkdir(os.path.dirname(path))
            else:
                recompute_files = [False]
            path = [path]
            return path, recompute_files, unique_file
            
        out_seg, _, _ = helper_im(out_seg, 'path_segmentations', '', 'synthseg')
        out_posteriors, _, _ = helper_im(out_posteriors, 'path_posteriors', '', 'posteriors')
        out_resampled, _, _ = helper_im(out_resampled, 'path_resampled', '', 'resampled')
        out_volumes, _, unique_volume_file = helper_im(out_volumes, 'path_volumes', 'csv', '')

    return path_images, out_seg, out_posteriors, out_resampled, out_volumes, unique_volume_file

def preprocess(path_image, ct, n_levels=5, crop=None, min_pad=None, path_resample=None):
    im, _, aff, n_dims, n_channels, h, im_res = get_volume_info(path_image, True)
    if n_dims < 3:
        sys.exit('input should have 3 dimensions, had %s' % n_dims)
    elif n_dims == 4 and n_channels == 1:
        n_dims = 3
        im = im[..., 0]
    elif n_dims > 3:
        sys.exit('input should have 3 dimensions, had %s' % n_dims)
    elif n_channels > 1:
        print('Detected more than 1 channel, we keep the average.')
        im = np.mean(im, axis=-1)
        
    if np.any((im_res > 1.05) | (im_res < 0.95)):
        im_res = np.array([1.]*3)
        im, aff = resample_volume(im, aff, im_res)
        if path_resample is not None:
            save_volume(im, aff, h, path_resample)
            
    im = align_volume_to_ref(im, aff, aff_ref=np.eye(4), n_dims=n_dims, return_copy=False)
    shape = list(im.shape[:n_dims])
    
    if crop is not None:
        crop = reformat_to_list(crop, length=n_dims, dtype='int')
        crop_shape = [find_closest_number_divisible_by_m(s, 2 ** n_levels, 'higher') for s in crop]
        im, crop_idx = crop_volume(im, cropping_shape=crop_shape, return_crop_idx=True)
    else:
        crop_idx = None
        
    if ct:
        im = np.clip(im, 0, 80)
        
    im = rescale_volume(im, new_min=0, new_max=1, min_percentile=0.5, max_percentile=99.5)
    input_shape = im.shape[:n_dims]
    pad_shape = [find_closest_number_divisible_by_m(s, 2 ** n_levels, 'higher') for s in input_shape]
    min_pad = reformat_to_list(min_pad, length=n_dims, dtype='int')
    min_pad = [find_closest_number_divisible_by_m(s, 2 ** n_levels, 'higher') for s in min_pad]
    pad_shape = np.maximum(pad_shape, min_pad)
    im, pad_idx = pad_volume(im, padding_shape=pad_shape, return_pad_idx=True)
    im = add_axis(im, axis=[0, -1]) # (1, D, H, W, 1) - wait, add_axis adds at 0 and -1.
    # Original add_axis(im, axis=[0, -1]) on 3D vol (D,H,W) -> (1, D, H, W, 1)
    # But here we want to return the image for pytorch conversion.
    # Let's keep it consistent with original for now, but strip batch dim.
    im = np.squeeze(im, axis=0) # (D, H, W, 1)
    
    return im, aff, h, im_res, shape, pad_idx, crop_idx

def postprocess(post_patch_seg, post_patch_parc, shape, pad_idx, crop_idx,
                labels_segmentation, labels_parcellation, aff, im_res, fast, topology_classes, v1):
    # post_patch_seg is (D, H, W, C)
    
    if fast:
        post_patch_seg = crop_volume_with_idx(post_patch_seg, pad_idx, n_dims=3, return_copy=False)
        
    tmp_post_patch_seg = post_patch_seg[..., 1:]
    post_patch_seg_mask = np.sum(tmp_post_patch_seg, axis=-1) > 0.25
    post_patch_seg_mask = get_largest_connected_component(post_patch_seg_mask)
    post_patch_seg_mask = np.stack([post_patch_seg_mask]*tmp_post_patch_seg.shape[-1], axis=-1)
    tmp_post_patch_seg = mask_volume(tmp_post_patch_seg, mask=post_patch_seg_mask, return_copy=False)
    post_patch_seg[..., 1:] = tmp_post_patch_seg
    
    if not fast:
        post_patch_seg_mask = post_patch_seg > 0.25
        for topology_class in np.unique(topology_classes)[1:]:
            tmp_topology_indices = np.where(topology_classes == topology_class)[0]
            tmp_mask = np.any(post_patch_seg_mask[..., tmp_topology_indices], axis=-1)
            tmp_mask = get_largest_connected_component(tmp_mask)
            for idx in tmp_topology_indices:
                post_patch_seg[..., idx] *= tmp_mask
        post_patch_seg = crop_volume_with_idx(post_patch_seg, pad_idx, n_dims=3, return_copy=False)
    else:
        post_patch_seg_mask = post_patch_seg > 0.2
        post_patch_seg[..., 1:] *= post_patch_seg_mask[..., 1:]
        
    post_patch_seg /= np.sum(post_patch_seg, axis=-1)[..., np.newaxis]
    seg_patch = labels_segmentation[post_patch_seg.argmax(-1).astype('int32')].astype('int32')
    
    if post_patch_parc is not None:
        post_patch_parc = crop_volume_with_idx(post_patch_parc, pad_idx, n_dims=3, return_copy=False)
        mask = (seg_patch == 3) | (seg_patch == 42)
        post_patch_parc[..., 0] = np.ones_like(post_patch_parc[..., 0])
        post_patch_parc[..., 0] = mask_volume(post_patch_parc[..., 0], mask=mask < 0.1, return_copy=False)
        post_patch_parc /= np.sum(post_patch_parc, axis=-1)[..., np.newaxis]
        parc_patch = labels_parcellation[post_patch_parc.argmax(-1).astype('int32')].astype('int32')
        seg_patch[mask] = parc_patch[mask]
        
    if crop_idx is not None:
        seg = np.zeros(shape=shape, dtype='int32')
        posteriors = np.zeros(shape=[*shape, labels_segmentation.shape[0]])
        posteriors[..., 0] = np.ones(shape)
        seg[crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5]] = seg_patch
        posteriors[crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5], :] = post_patch_seg
    else:
        seg = seg_patch
        posteriors = post_patch_seg
        
    seg = align_volume_to_ref(seg, aff=np.eye(4), aff_ref=aff, n_dims=3, return_copy=False)
    posteriors = align_volume_to_ref(posteriors, np.eye(4), aff_ref=aff, n_dims=3, return_copy=False)
    
    volumes = np.sum(posteriors[..., 1:], axis=tuple(range(0, len(posteriors.shape) - 1)))
    if not v1:
        volumes = np.concatenate([np.array([np.sum(volumes)]), volumes])
        
    if post_patch_parc is not None:
        volumes_parc = np.sum(post_patch_parc[..., 1:], axis=tuple(range(0, len(posteriors.shape) - 1)))
        total_volume_cortex = np.sum(volumes[np.where((labels_segmentation == 3) | (labels_segmentation == 42))[0] - 1])
        volumes_parc = volumes_parc / np.sum(volumes_parc) * total_volume_cortex
        volumes = np.concatenate([volumes, volumes_parc])
        
    volumes = np.around(volumes * np.prod(im_res), 3)
    return seg, posteriors, volumes

def load_volume(path_volume, im_only=True, squeeze=True, dtype=None, aff_ref=None):
    if path_volume.endswith(('.nii', '.nii.gz', '.mgz')):
        x = nib.load(path_volume)
        if squeeze:
            volume = np.squeeze(x.get_fdata())
        else:
            volume = x.get_fdata()
        aff = x.affine
        header = x.header
    else:
        volume = np.load(path_volume)['vol_data']
        if squeeze:
            volume = np.squeeze(volume)
        aff = np.eye(4)
        header = nib.Nifti1Header()
    if dtype is not None:
        if 'int' in dtype:
            volume = np.round(volume)
        volume = volume.astype(dtype=dtype)
    if aff_ref is not None:
        n_dims, _ = get_dims(list(volume.shape), max_channels=10)
        volume, aff = align_volume_to_ref(volume, aff, aff_ref=aff_ref, return_aff=True, n_dims=n_dims)
    if im_only:
        return volume
    else:
        return volume, aff, header

def save_volume(volume, aff, header, path, res=None, dtype=None, n_dims=3):
    mkdir(os.path.dirname(path))
    if '.npz' in path:
        np.savez_compressed(path, vol_data=volume)
    else:
        if header is None:
            header = nib.Nifti1Header()
        if isinstance(aff, str):
            if aff == 'FS':
                aff = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        elif aff is None:
            aff = np.eye(4)
        nifty = nib.Nifti1Image(volume, aff, header)
        if dtype is not None:
            if 'int' in dtype:
                volume = np.round(volume)
            volume = volume.astype(dtype=dtype)
            nifty.set_data_dtype(dtype)
        if res is not None:
            if n_dims is None:
                n_dims, _ = get_dims(volume.shape)
            res = reformat_to_list(res, length=n_dims, dtype=None)
            nifty.header.set_zooms(res)
        nib.save(nifty, path)

def get_volume_info(path_volume, return_volume=False, aff_ref=None, max_channels=10):
    im, aff, header = load_volume(path_volume, im_only=False)
    im_shape = list(im.shape)
    n_dims, n_channels = get_dims(im_shape, max_channels=max_channels)
    im_shape = im_shape[:n_dims]
    if '.nii' in path_volume:
        data_res = np.array(header['pixdim'][1:n_dims + 1])
    elif '.mgz' in path_volume:
        data_res = np.array(header['delta'])
    else:
        data_res = np.array([1.0] * n_dims)
    if aff_ref is not None:
        ras_axes = get_ras_axes(aff, n_dims=n_dims)
        ras_axes_ref = get_ras_axes(aff_ref, n_dims=n_dims)
        im = align_volume_to_ref(im, aff, aff_ref=aff_ref, n_dims=n_dims)
        im_shape = np.array(im_shape)
        data_res = np.array(data_res)
        im_shape[ras_axes_ref] = im_shape[ras_axes]
        data_res[ras_axes_ref] = data_res[ras_axes]
        im_shape = im_shape.tolist()
    if return_volume:
        return im, im_shape, aff, n_dims, n_channels, header, data_res
    else:
        return im_shape, aff, n_dims, n_channels, header, data_res

def get_dims(shape, max_channels=10):
    if shape[-1] <= max_channels:
        n_dims = len(shape) - 1
        n_channels = shape[-1]
    else:
        n_dims = len(shape)
        n_channels = 1
    return n_dims, n_channels

def add_axis(x, axis=0):
    axis = reformat_to_list(axis)
    for ax in axis:
        x = np.expand_dims(x, axis=ax)
    return x

def get_list_labels(label_list=None, labels_dir=None, save_label_list=None, FS_sort=False):
    if label_list is not None:
        label_list = np.array(reformat_to_list(label_list, load_as_numpy=True, dtype='int'))
    elif labels_dir is not None:
        print('Compiling list of unique labels')
        labels_paths = list_images_in_folder(labels_dir)
        label_list = np.empty(0)
        loop_info = LoopInfo(len(labels_paths), 10, 'processing', print_time=True)
        for lab_idx, path in enumerate(labels_paths):
            loop_info.update(lab_idx)
            y = load_volume(path, dtype='int32')
            y_unique = np.unique(y)
            label_list = np.unique(np.concatenate((label_list, y_unique))).astype('int')
    else:
        raise Exception('either label_list, path_label_list or labels_dir should be provided')
    n_neutral_labels = 0
    if FS_sort:
        # Simplified FS sort logic or copy full if needed. 
        # For now, just returning list as is if FS_sort is False, which is default for most calls except one in main
        pass 
    if save_label_list is not None:
        np.save(save_label_list, np.int32(label_list))
    return np.int32(label_list), None

def load_array_if_path(var, load_as_numpy=True):
    if (isinstance(var, str)) & load_as_numpy:
        assert os.path.isfile(var), 'No such path: %s' % var
        var = np.load(var)
    return var

def reformat_to_list(var, length=None, load_as_numpy=False, dtype=None):
    if var is None:
        return None
    var = load_array_if_path(var, load_as_numpy=load_as_numpy)
    if isinstance(var, (int, float, np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64)):
        var = [var]
    elif isinstance(var, tuple):
        var = list(var)
    elif isinstance(var, np.ndarray):
        if var.shape == (1,):
            var = [var[0]]
        else:
            var = np.squeeze(var).tolist()
    elif isinstance(var, str):
        var = [var]
    elif isinstance(var, bool):
        var = [var]
    if isinstance(var, list):
        if length is not None:
            if len(var) == 1:
                var = var * length
            elif len(var) != length:
                raise ValueError('if var is a list/tuple/numpy array, it should be of length 1 or {0}, '
                                 'had {1}'.format(length, var))
    else:
        raise TypeError('var should be an int, float, tuple, list, numpy array, or path to numpy array')
    if dtype is not None:
        if dtype == 'int':
            var = [int(v) for v in var]
        elif dtype == 'float':
            var = [float(v) for v in var]
        elif dtype == 'bool':
            var = [bool(v) for v in var]
        elif dtype == 'str':
            var = [str(v) for v in var]
    return var

def list_images_in_folder(path_dir, include_single_image=True, check_if_empty=True):
    basename = os.path.basename(path_dir)
    if include_single_image & \
            (('.nii.gz' in basename) | ('.nii' in basename) | ('.mgz' in basename) | ('.npz' in basename)):
        assert os.path.isfile(path_dir), 'file %s does not exist' % path_dir
        list_images = [path_dir]
    else:
        if os.path.isdir(path_dir):
            list_images = sorted(glob.glob(os.path.join(path_dir, '*nii.gz')) +
                                 glob.glob(os.path.join(path_dir, '*nii')) +
                                 glob.glob(os.path.join(path_dir, '*.mgz')) +
                                 glob.glob(os.path.join(path_dir, '*.npz')))
        else:
            raise Exception('Folder does not exist: %s' % path_dir)
        if check_if_empty:
            assert len(list_images) > 0, 'no .nii, .nii.gz, .mgz or .npz image could be found in %s' % path_dir
    return list_images

def mkdir(path_dir):
    if len(path_dir)>0:
        if path_dir[-1] == '/':
            path_dir = path_dir[:-1]
        if not os.path.isdir(path_dir):
            os.makedirs(path_dir, exist_ok=True)

class LoopInfo:
    def __init__(self, n_iterations, spacing=10, text='processing', print_time=False):
        self.n_iterations = n_iterations
        self.spacing = spacing
        self.text = text
        self.print_time = print_time
        self.print_previous_time = False
        self.align = len(str(self.n_iterations)) * 2 + 1 + 3
        self.iteration_durations = np.zeros((n_iterations,))
        self.start = time.time()
        self.previous = time.time()
    def update(self, idx):
        now = time.time()
        self.iteration_durations[idx] = now - self.previous
        self.previous = now
        if idx == 0:
            print(self.text + ' 1/{}'.format(self.n_iterations))
        elif idx % self.spacing == self.spacing - 1:
            iteration = str(idx + 1) + '/' + str(self.n_iterations)
            if self.print_time:
                max_duration = np.max(self.iteration_durations)
                average_duration = np.mean(self.iteration_durations[self.iteration_durations > .01 * max_duration])
                remaining_time = int(average_duration * (self.n_iterations - idx))
                if (remaining_time > 1) | self.print_previous_time:
                    eta = str(timedelta(seconds=remaining_time))
                    print(self.text + ' {:<{x}} remaining time: {}'.format(iteration, eta, x=self.align))
                    self.print_previous_time = True
                else:
                    print(self.text + ' {}'.format(iteration))
            else:
                print(self.text + ' {}'.format(iteration))

def find_closest_number_divisible_by_m(n, m, answer_type='lower'):
    if n % m == 0:
        return n
    else:
        q = int(n / m)
        lower = q * m
        higher = (q + 1) * m
        if answer_type == 'lower':
            return lower
        elif answer_type == 'higher':
            return higher
        elif answer_type == 'closer':
            return lower if (n - lower) < (higher - n) else higher

def build_binary_structure(connectivity, n_dims, shape=None):
    if shape is None:
        shape = [connectivity * 2 + 1] * n_dims
    else:
        shape = reformat_to_list(shape, length=n_dims)
    dist = np.ones(shape)
    center = tuple([tuple([int(s / 2)]) for s in shape])
    dist[center] = 0
    dist = distance_transform_edt(dist)
    struct = (dist <= connectivity) * 1
    return struct

def mask_volume(volume, mask=None, threshold=0.1, dilate=0, erode=0, fill_holes=False, masking_value=0,
                return_mask=False, return_copy=True):
    new_volume = volume.copy() if return_copy else volume
    vol_shape = list(new_volume.shape)
    n_dims, n_channels = get_dims(vol_shape)
    if mask is None:
        mask = new_volume >= threshold
    else:
        mask = mask > 0
    if dilate > 0:
        dilate_struct = build_binary_structure(dilate, n_dims)
        mask_to_apply = binary_dilation(mask, dilate_struct)
    else:
        mask_to_apply = mask
    if erode > 0:
        erode_struct = build_binary_structure(erode, n_dims)
        mask_to_apply = binary_erosion(mask_to_apply, erode_struct)
    if fill_holes:
        mask_to_apply = binary_fill_holes(mask_to_apply)
    if mask_to_apply.shape == new_volume.shape:
        new_volume[np.logical_not(mask_to_apply)] = masking_value
    else:
        new_volume[np.stack([np.logical_not(mask_to_apply)] * n_channels, axis=-1)] = masking_value
    if return_mask:
        return new_volume, mask_to_apply
    else:
        return new_volume

def rescale_volume(volume, new_min=0, new_max=255, min_percentile=2., max_percentile=98., use_positive_only=False):
    new_volume = volume.copy()
    intensities = new_volume[new_volume > 0] if use_positive_only else new_volume.flatten()
    robust_min = np.min(intensities) if min_percentile == 0 else np.percentile(intensities, min_percentile)
    robust_max = np.max(intensities) if max_percentile == 100 else np.percentile(intensities, max_percentile)
    new_volume = np.clip(new_volume, robust_min, robust_max)
    if robust_min != robust_max:
        return new_min + (new_volume - robust_min) / (robust_max - robust_min) * (new_max - new_min)
    else:
        return np.zeros_like(new_volume)

def crop_volume(volume, cropping_margin=None, cropping_shape=None, aff=None, return_crop_idx=False, mode='center'):
    new_volume = volume.copy()
    vol_shape = new_volume.shape
    n_dims, _ = get_dims(vol_shape)
    if cropping_margin is not None:
        cropping_margin = reformat_to_list(cropping_margin, length=n_dims)
        do_cropping = np.array(vol_shape[:n_dims]) > 2 * np.array(cropping_margin)
        min_crop_idx = [cropping_margin[i] if do_cropping[i] else 0 for i in range(n_dims)]
        max_crop_idx = [vol_shape[i] - cropping_margin[i] if do_cropping[i] else vol_shape[i] for i in range(n_dims)]
    else:
        cropping_shape = reformat_to_list(cropping_shape, length=n_dims)
        if mode == 'center':
            min_crop_idx = np.maximum([int((vol_shape[i] - cropping_shape[i]) / 2) for i in range(n_dims)], 0)
            max_crop_idx = np.minimum([min_crop_idx[i] + cropping_shape[i] for i in range(n_dims)],
                                      np.array(vol_shape)[:n_dims])
    crop_idx = np.concatenate([np.array(min_crop_idx), np.array(max_crop_idx)])
    if n_dims == 2:
        new_volume = new_volume[crop_idx[0]: crop_idx[2], crop_idx[1]: crop_idx[3], ...]
    elif n_dims == 3:
        new_volume = new_volume[crop_idx[0]: crop_idx[3], crop_idx[1]: crop_idx[4], crop_idx[2]: crop_idx[5], ...]
    output = [new_volume]
    if aff is not None:
        aff[0:3, -1] = aff[0:3, -1] + aff[:3, :3] @ np.array(min_crop_idx)
        output.append(aff)
    if return_crop_idx:
        output.append(crop_idx)
    return output[0] if len(output) == 1 else tuple(output)

def crop_volume_with_idx(volume, crop_idx, aff=None, n_dims=None, return_copy=True):
    new_volume = volume.copy() if return_copy else volume
    n_dims = int(np.array(crop_idx).shape[0] / 2) if n_dims is None else n_dims
    if n_dims == 2:
        new_volume = new_volume[crop_idx[0]:crop_idx[2], crop_idx[1]:crop_idx[3], ...]
    elif n_dims == 3:
        new_volume = new_volume[crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5], ...]
    if aff is not None:
        aff[0:3, -1] = aff[0:3, -1] + aff[:3, :3] @ crop_idx[:3]
        return new_volume, aff
    else:
        return new_volume

def pad_volume(volume, padding_shape, padding_value=0, aff=None, return_pad_idx=False):
    new_volume = volume.copy()
    vol_shape = new_volume.shape
    n_dims, n_channels = get_dims(vol_shape)
    padding_shape = reformat_to_list(padding_shape, length=n_dims, dtype='int')
    if np.any(np.array(padding_shape, dtype='int32') > np.array(vol_shape[:n_dims], dtype='int32')):
        min_margins = np.maximum(np.int32(np.floor((np.array(padding_shape) - np.array(vol_shape)[:n_dims]) / 2)), 0)
        max_margins = np.maximum(np.int32(np.ceil((np.array(padding_shape) - np.array(vol_shape)[:n_dims]) / 2)), 0)
        pad_idx = np.concatenate([min_margins, min_margins + np.array(vol_shape[:n_dims])])
        pad_margins = tuple([(min_margins[i], max_margins[i]) for i in range(n_dims)])
        if n_channels > 1:
            pad_margins = tuple(list(pad_margins) + [(0, 0)])
        new_volume = np.pad(new_volume, pad_margins, mode='constant', constant_values=padding_value)
        if aff is not None:
            if n_dims == 2:
                min_margins = np.append(min_margins, 0)
            aff[:-1, -1] = aff[:-1, -1] - aff[:-1, :-1] @ min_margins
    else:
        pad_idx = np.concatenate([np.array([0] * n_dims), np.array(vol_shape[:n_dims])])
    output = [new_volume]
    if aff is not None:
        output.append(aff)
    if return_pad_idx:
        output.append(pad_idx)
    return output[0] if len(output) == 1 else tuple(output)

def resample_volume(volume, aff, new_vox_size, interpolation='linear'):
    pixdim = np.sqrt(np.sum(aff * aff, axis=0))[:-1]
    new_vox_size = np.array(new_vox_size)
    factor = pixdim / new_vox_size
    sigmas = 0.25 / factor
    sigmas[factor > 1] = 0
    volume_filt = gaussian_filter(volume, sigmas)
    x = np.arange(0, volume_filt.shape[0])
    y = np.arange(0, volume_filt.shape[1])
    z = np.arange(0, volume_filt.shape[2])
    my_interpolating_function = RegularGridInterpolator((x, y, z), volume_filt, method=interpolation)
    start = - (factor - 1) / (2 * factor)
    step = 1.0 / factor
    stop = start + step * np.ceil(volume_filt.shape * factor)
    xi = np.arange(start=start[0], stop=stop[0], step=step[0])
    yi = np.arange(start=start[1], stop=stop[1], step=step[1])
    zi = np.arange(start=start[2], stop=stop[2], step=step[2])
    xi[xi < 0] = 0
    yi[yi < 0] = 0
    zi[zi < 0] = 0
    xi[xi > (volume_filt.shape[0] - 1)] = volume_filt.shape[0] - 1
    yi[yi > (volume_filt.shape[1] - 1)] = volume_filt.shape[1] - 1
    zi[zi > (volume_filt.shape[2] - 1)] = volume_filt.shape[2] - 1
    xig, yig, zig = np.meshgrid(xi, yi, zi, indexing='ij', sparse=True)
    volume2 = my_interpolating_function((xig, yig, zig))
    aff2 = aff.copy()
    for c in range(3):
        aff2[:-1, c] = aff2[:-1, c] / factor[c]
    aff2[:-1, -1] = aff2[:-1, -1] - np.matmul(aff2[:-1, :-1], 0.5 * (factor - 1))
    return volume2, aff2

def get_ras_axes(aff, n_dims=3):
    aff_inverted = np.linalg.inv(aff)
    img_ras_axes = np.argmax(np.absolute(aff_inverted[0:n_dims, 0:n_dims]), axis=0)
    for i in range(n_dims):
        if i not in img_ras_axes:
            unique, counts = np.unique(img_ras_axes, return_counts=True)
            incorrect_value = unique[np.argmax(counts)]
            img_ras_axes[np.where(img_ras_axes == incorrect_value)[0][-1]] = i
    return img_ras_axes

def align_volume_to_ref(volume, aff, aff_ref=None, return_aff=False, n_dims=None, return_copy=True):
    new_volume = volume.copy() if return_copy else volume
    aff_flo = aff.copy()
    if aff_ref is None:
        aff_ref = np.eye(4)
    if n_dims is None:
        n_dims, _ = get_dims(new_volume.shape)
    ras_axes_ref = get_ras_axes(aff_ref, n_dims=n_dims)
    ras_axes_flo = get_ras_axes(aff_flo, n_dims=n_dims)
    aff_flo[:, ras_axes_ref] = aff_flo[:, ras_axes_flo]
    for i in range(n_dims):
        if ras_axes_flo[i] != ras_axes_ref[i]:
            new_volume = np.swapaxes(new_volume, ras_axes_flo[i], ras_axes_ref[i])
            swapped_axis_idx = np.where(ras_axes_flo == ras_axes_ref[i])
            ras_axes_flo[swapped_axis_idx], ras_axes_flo[i] = ras_axes_flo[i], ras_axes_flo[swapped_axis_idx]
    dot_products = np.sum(aff_flo[:3, :3] * aff_ref[:3, :3], axis=0)
    for i in range(n_dims):
        if dot_products[i] < 0:
            new_volume = np.flip(new_volume, axis=i)
            aff_flo[:, i] = - aff_flo[:, i]
            aff_flo[:3, 3] = aff_flo[:3, 3] - aff_flo[:3, i] * (new_volume.shape[i] - 1)
    if return_aff:
        return new_volume, aff_flo
    else:
        return new_volume

def get_largest_connected_component(mask, structure=None):
    components, n_components = scipy_label(mask, structure)
    return components == np.argmax(np.bincount(components.flat)[1:]) + 1 if n_components > 0 else mask.copy()

def get_flip_indices(labels_segmentation, n_neutral_labels):
    # Simplified version or full if needed. 
    # The original script uses this for robust prediction which we aren't fully supporting yet.
    # But it's called in predict() so we need it.
    n_sided_labels = int((len(labels_segmentation) - n_neutral_labels) / 2)
    neutral_labels = labels_segmentation[:n_neutral_labels]
    left = labels_segmentation[n_neutral_labels:n_neutral_labels + n_sided_labels]
    lr_corresp = np.stack([labels_segmentation[n_neutral_labels:n_neutral_labels + n_sided_labels],
                           labels_segmentation[n_neutral_labels + n_sided_labels:]])
    lr_corresp_unique, lr_corresp_indices = np.unique(lr_corresp[0, :], return_index=True)
    lr_corresp_unique = np.stack([lr_corresp_unique, lr_corresp[1, lr_corresp_indices]])
    lr_corresp_unique = lr_corresp_unique[:, 1:] if not np.all(lr_corresp_unique[:, 0]) else lr_corresp_unique
    labels_segmentation, unique_idx = np.unique(labels_segmentation, return_index=True)
    lr_indices = np.zeros_like(lr_corresp_unique)
    for i in range(lr_corresp_unique.shape[0]):
        for j, lab in enumerate(lr_corresp_unique[i]):
            lr_indices[i, j] = np.where(labels_segmentation == lab)[0]
    flip_indices = np.zeros_like(labels_segmentation)
    for i in range(len(flip_indices)):
        if labels_segmentation[i] in neutral_labels:
            flip_indices[i] = i
        elif labels_segmentation[i] in left:
            flip_indices[i] = lr_indices[1, np.where(lr_corresp_unique[0, :] == labels_segmentation[i])]
        else:
            flip_indices[i] = lr_indices[0, np.where(lr_corresp_unique[1, :] == labels_segmentation[i])]
    return labels_segmentation, flip_indices, unique_idx

def write_csv(path_csv, data, unique_file, labels, names, skip_first=True, last_first=False):
    print(f"DEBUG: Inside write_csv. Path: {path_csv}, Unique: {unique_file}", flush=True)
    mkdir(os.path.dirname(path_csv))
    labels, unique_idx = np.unique(labels, return_index=True)
    if skip_first:
        labels = labels[1:]
    if names is not None:
        names = names[unique_idx].tolist()
        if skip_first:
            names = names[1:]
        header = names
    else:
        header = [str(lab) for lab in labels]
    if last_first:
        header = [header[-1]] + header[:-1]
    if (not unique_file) & (data is None):
        raise ValueError('data can only be None when initialising a unique volume file')
    if unique_file:
        if data is None:
            type_open = 'w'
            data = ['subject'] + header
        else:
            type_open = 'a'
        data = [data]
    else:
        type_open = 'w'
        header = [''] + header
        data = [header, data]
    with open(path_csv, type_open) as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(data)

def get_mapping_lut(source, dest=None):
    source = np.array(reformat_to_list(source), dtype='int32')
    n_labels = source.shape[0]
    if dest is None:
        dest = np.arange(n_labels, dtype='int32')
    else:
        assert len(source) == len(dest), 'label_list and new_label_list should have the same length'
        dest = np.array(reformat_to_list(dest, dtype='int'))
    lut = np.zeros(np.max(source) + 1, dtype='int32')
    for source, dest in zip(source, dest):
        lut[source] = dest
    return lut

if __name__ == "__main__":
    main()
