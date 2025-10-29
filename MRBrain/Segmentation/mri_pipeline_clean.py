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
from datetime import timedelta
from itertools import combinations
from scipy.ndimage import label as scipy_label
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter, distance_transform_edt, binary_fill_holes
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow import keras
import tensorflow.keras.layers as KL
from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')
def _n_features_of(tensor):
    try:
        s = tensor.shape.as_list()
        if s is not None:
            return int(s[-1])
    except Exception:
        pass
    try:
        s = tf.keras.backend.int_shape(tensor)
        if s is not None:
            return int(s[-1])
    except Exception:
        pass
    shape_attr = getattr(tensor, "_shape", None) or getattr(tensor, "shape", None)
    if shape_attr is None:
        raise RuntimeError("Unable to determine number of features for tensor: no shape attr found.")
    try:
        return int(list(shape_attr)[-1])
    except Exception:
        return int(tuple(shape_attr)[-1])
def main():
    parser = argparse.ArgumentParser(description="SynthSeg", epilog='\n')
    parser.add_argument("--i", help="Image(s) to segment. Can be a path to an image or to a folder.")
    parser.add_argument("--o", help="Segmentation output(s). Must be a folder if --i designates a folder.")
    parser.add_argument("--parc", action="store_true", help="(optional) Whether to perform cortex parcellation.")
    parser.add_argument("--robust", action="store_true", help="(optional) Whether to use robust predictions (slower).")
    parser.add_argument("--fast", action="store_true", help="(optional) Bypass some processing for faster prediction.")
    parser.add_argument("--ct", action="store_true", help="(optional) Clip CT scans in Hounsfield scale to [0, 80]")
    parser.add_argument("--vol", help="(optional) Output CSV file with volumes for all structures and subjects.")
    parser.add_argument("--qc", help="(optional) Output CSV file with qc scores for all subjects.")
    parser.add_argument("--post", help="(optional) Posteriors output(s). Must be a folder if --i designates a folder.")
    parser.add_argument("--resample", help="(optional) Resampled image(s). Must be a folder if --i is a folder.")
    parser.add_argument("--crop", nargs='+', type=int, help="(optional) Only analyse an image patch of the given size.")
    parser.add_argument("--threads", type=int, default=1, help="(optional) Number of cores to be used. Default is 1.")
    parser.add_argument("--cpu", action="store_true", help="(optional) Enforce running with CPU rather than GPU.")
    parser.add_argument("--v1", action="store_true", help="(optional) Use SynthSeg 1.0 (updated 25/06/22).")
    parser.add_argument("--photo", help="(optional) Photo-SynthSeg: segment 3D reconstructed stack of coronal dissection photos of the cerebrum; must be left, right, or both")
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
    synthseg_home = os.path.dirname(os.path.abspath(__file__))
    args = parser.parse_args()
    labels_segmentation = os.path.join(synthseg_home, 'models', 'synthseg_segmentation_labels_2.0.npy')
    labels_denoiser = os.path.join(synthseg_home, 'models', 'synthseg_denoiser_labels_2.0.npy')
    labels_parcellation = os.path.join(synthseg_home, 'models', 'synthseg_parcellation_labels.npy')
    labels_qc = os.path.join(synthseg_home, 'models', 'synthseg_qc_labels_2.0.npy')
    names_segmentation_labels = os.path.join(synthseg_home, 'models', 'synthseg_segmentation_names_2.0.npy')
    names_parcellation_labels = os.path.join(synthseg_home, 'models', 'synthseg_parcellation_names.npy')
    names_qc_labels = os.path.join(synthseg_home, 'models', 'synthseg_qc_names_2.0.npy')
    topology_classes = os.path.join(synthseg_home, 'models', 'synthseg_topological_classes_2.0.npy')
    n_neutral_labels = 19
    path_model_parcellation = os.path.join(synthseg_home, 'models', 'synthseg_parc_2.0.h5')
    path_model_qc = os.path.join(synthseg_home, 'models', 'synthseg_qc_2.0.h5')
    if args.photo is not None:
        if args.robust:
            sys.exit('The flag --robust is not compatible with Photo-SynthSeg (--photo).')
        if args.v1:
            sys.exit('The flag --v1 is not compatible with Photo-SynthSeg (--photo).')
        if args.ct:
            sys.exit('The flag --ct is not compatible with Photo-SynthSeg (--photo).')
        if args.qc is not None:
            sys.exit('The flag --qc is not compatible with Photo-SynthSeg (--photo).')
        version = 'Photo-SynthSeg 1.0'
        if args.photo=='both':
            path_model_segmentation = os.path.join(synthseg_home, 'models', 'synthseg_photo_both_1.0.h5')
        elif (args.photo=='left' or args.photo=='right'):
            path_model_segmentation = os.path.join(synthseg_home, 'models', 'synthseg_photo_single_1.0.h5')
            if args.parc:
                disp('Warning: the flag --parc is not well tested with single hemispheres.')
        else:
            sys.exit('The argument of --photo must be left, right, or both.')
    elif args.robust:
        args.fast = True
        if args.v1:
            sys.exit('The flag --v1 cannot be used with --robust since SynthSeg-robust only came out with 2.0.')
        version = 'SynthSeg-robust 2.0'
        path_model_segmentation = os.path.join(synthseg_home, 'models', 'synthseg_robust_2.0.h5')
    else:
        if args.v1:
            path_model_segmentation = os.path.join(synthseg_home, 'models', 'synthseg_1.0.h5')
            version = 'SynthSeg 1.0'
        else:
            path_model_segmentation = os.path.join(synthseg_home, 'models', 'synthseg_2.0.h5')
            version = 'SynthSeg 2.0'
        if args.fast:
            version += ' (fast)'
    print(version)
    if args.v1:
        labels_segmentation = labels_segmentation.replace('_2.0.npy', '.npy')
        labels_qc = labels_qc.replace('_2.0.npy', '.npy')
        names_segmentation_labels = names_segmentation_labels.replace('_2.0.npy', '.npy')
        names_qc_labels = names_qc_labels.replace('_2.0.npy', '.npy')
        topology_classes = topology_classes.replace('_2.0.npy', '.npy')
        n_neutral_labels = 18
    if args.photo=='both':
        labels_segmentation = labels_segmentation.replace('_2.0.npy', '_photo_both_1.0.npy')
        names_segmentation_labels = names_segmentation_labels.replace('_2.0.npy', '_photo_both_1.0.npy')
        topology_classes = topology_classes.replace('_2.0.npy', '_photo_both_1.0.npy')
        n_neutral_labels = 5
    elif (args.photo=='left' or args.photo=='right'):
        labels_segmentation = labels_segmentation.replace('_2.0.npy', '_photo_single_1.0.npy')
        names_segmentation_labels = names_segmentation_labels.replace('_2.0.npy', '_photo_single_1.0.npy')
        topology_classes = topology_classes.replace('_2.0.npy', '_photo_single_1.0.npy')
        n_neutral_labels = 5
    if args.cpu:
        print('using CPU, hiding all CUDA_VISIBLE_DEVICES')
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    if args.threads == 1:
        print('using 1 thread')
    else:
        print('using %s threads' % args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    predict(
        path_images=args.i,
        path_segmentations=args.o,
        path_model_segmentation=path_model_segmentation,
        labels_segmentation=labels_segmentation,
        robust=args.robust,
        fast=args.fast,
        v1=args.v1,
        photo=args.photo,
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
        path_qc_scores=args.qc,
        path_model_qc=path_model_qc,
        labels_qc=labels_qc,
        names_qc=names_qc_labels,
        cropping=args.crop,
        topology_classes=topology_classes,
        ct=args.ct,
    )
def predict(path_images,
            path_segmentations,
            path_model_segmentation,
            labels_segmentation,
            robust,
            fast,
            v1,
            photo,
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
            path_qc_scores,
            path_model_qc,
            labels_qc,
            names_qc,
            cropping,
            topology_classes,
            ct):
    outputs = prepare_output_files(path_images, path_segmentations, path_posteriors, path_resampled,
                                   path_volumes, path_qc_scores)
    path_images = outputs[0]
    path_segmentations = outputs[1]
    path_posteriors = outputs[2]
    path_resampled = outputs[3]
    path_volumes = outputs[4]
    unique_vol_file = outputs[5]
    path_qc_scores = outputs[6]
    unique_qc_file = outputs[7]
    labels_segmentation, _ = get_list_labels(label_list=labels_segmentation)
    if (n_neutral_labels is not None) & (not fast) & (not robust) & (photo!='left') & (photo!='right'):
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
    if (not v1) & (photo is None):
        labels_volumes = np.concatenate([labels_volumes, np.array([np.max(labels_volumes + 1)])])
        names_volumes = np.concatenate([names_volumes, np.array(['total intracranial'])])
    do_qc = True if path_qc_scores[0] is not None else False
    if do_qc:
        labels_qc = get_list_labels(labels_qc)[0][unique_idx]
        names_qc = load_array_if_path(names_qc)[unique_idx]
    if cropping is not None:
        cropping = reformat_to_list(cropping, length=3, dtype='int')
        min_pad = cropping
    else:
        min_pad = 128
    if unique_vol_file & (path_volumes[0] is not None):
        write_csv(path_volumes[0], None, True, labels_volumes, names_volumes, last_first=((not v1) & (photo is None)))
    if unique_qc_file & do_qc:
        write_csv(path_qc_scores[0], None, True, labels_qc, names_qc)
    net = build_model(model_file_segmentation=path_model_segmentation,
                      model_file_parcellation=path_model_parcellation,
                      model_file_qc=path_model_qc,
                      labels_segmentation=labels_segmentation,
                      labels_denoiser=labels_denoiser,
                      labels_parcellation=labels_parcellation,
                      labels_qc=labels_qc,
                      flip_indices=flip_indices,
                      robust=robust,
                      do_parcellation=do_parcellation,
                      do_qc=do_qc)
    if len(path_images) <= 10:
        loop_info = LoopInfo(len(path_images), 1, 'predicting', True)
    else:
        loop_info = LoopInfo(len(path_images), 10, 'predicting', True)
    list_errors = list()
    for i in range(len(path_images)):
        loop_info.update(i)
        try:
            image, aff, h, im_res, shape, pad_idx, crop_idx = preprocess(path_image=path_images[i],
                                                                         ct=ct,
                                                                         crop=cropping,
                                                                         min_pad=min_pad,
                                                                         path_resample=path_resampled[i])
            shape_input = add_axis(np.array(image.shape[1:-1]))
            if photo=='right':
                image = np.flip(image, axis=1)
            if do_parcellation & do_qc:
                post_patch_segmentation, post_patch_parcellation, qc_score = net.predict([image, shape_input])
            elif do_parcellation & (not do_qc):
                post_patch_segmentation, post_patch_parcellation = net.predict(image)
                qc_score = None
            elif (not do_parcellation) & do_qc:
                post_patch_segmentation, qc_score = net.predict([image, shape_input])
                post_patch_parcellation = None
            else:
                post_patch_segmentation = net.predict(image)
                post_patch_parcellation = qc_score = None
            if photo=='right':
                image = np.flip(image, axis=0)
                post_patch_segmentation = np.flip(post_patch_segmentation, axis=1)
                if post_patch_parcellation is not None:
                    post_patch_parcellation = np.flip(post_patch_parcellation, axis=1)
            seg, posteriors, volumes = postprocess(post_patch_seg=post_patch_segmentation,
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
                                                   v1=(v1 or (photo is not None)))
            save_volume(seg, aff, h, path_segmentations[i], dtype='int32')
            if path_posteriors[i] is not None:
                save_volume(posteriors, aff, h, path_posteriors[i], dtype='float32')
            if path_volumes[i] is not None:
                row = [os.path.basename(path_images[i]).replace('.nii.gz', '')] + [str(vol) for vol in volumes]
                write_csv(path_volumes[i], row, unique_vol_file, labels_volumes, names_volumes, last_first=((not v1) & (photo is None)))
            if path_qc_scores[i] is not None:
                qc_score = np.around(np.clip(np.squeeze(qc_score)[1:], 0, 1), 4)
                row = [os.path.basename(path_images[i]).replace('.nii.gz', '')] + ['%.4f' % q for q in qc_score]
                write_csv(path_qc_scores[i], row, unique_qc_file, labels_qc, names_qc)
        except Exception as e:
            list_errors.append(path_images[i])
            print('\nthe following problem occured with image %s :' % path_images[i])
            print(traceback.format_exc())
            print('resuming program execution\n')
            continue
    if (len(path_segmentations) == 1) & (len(list_errors) == 0):
        print('\nsegmentation  saved in:    ' + path_segmentations[0])
        if path_posteriors[0] is not None:
            print('posteriors saved in:       ' + path_posteriors[0])
        if path_resampled[0] is not None:
            print('resampled image saved in:  ' + path_resampled[0])
        if path_volumes[0] is not None:
            print('volumes saved in:          ' + path_volumes[0])
        if path_qc_scores[0] is not None:
            print('QC scores saved in:        ' + path_qc_scores[0])
    elif (len(path_segmentations) > 1) & (len(list_errors) < len(path_segmentations)):
        if len(set([os.path.dirname(path_segmentations[i]) for i in range(len(path_segmentations))])) <= 1:
            print('\nsegmentations saved in:    ' + os.path.dirname(path_segmentations[0]))
            if path_posteriors[0] is not None:
                print('posteriors saved in:       ' + os.path.dirname(path_posteriors[0]))
            if path_resampled[0] is not None:
                print('resampled images saved in: ' + os.path.dirname(path_resampled[0]))
            if path_volumes[0] is not None:
                print('volumes saved in:          ' + path_volumes[0])
            if path_qc_scores[0] is not None:
                print('QC scores saved in:        ' + path_qc_scores[0])
    if len(list_errors) > 0:
        print('\nERROR: some problems occured for the following inputs (see corresponding errors above):')
        for path_error_image in list_errors:
            print(path_error_image)
        sys.exit(1)
    else:
        print('\nIf you use this tool in a publication, please cite:')
        print('SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining')
        print('B. Billot, D.N. Greve, O. Puonti, A. Thielscher, K. Van Leemput, B. Fischl, A.V. Dalca, J.E. Iglesias')
        print('Medical Image Analysis, accepted for publication.')
        if robust:
            print('\nRobust machine learning segmentation for large-scale analysis of heterogeneous clinical brain MRI datasets')
            print('B. Billot, C. Magdamo, Y. Cheng, S.E. Arnold, S. Das, J.E. Iglesias')
            print('PNAS, accepted for publication.')
        if photo is not None:
            print('\n3D analysis of dissection photographs with surface scanning and machine learning for quantitative neuropathology')
            print('H. Gazula, H. Tregidgo, B. Billot, Y. Balbastre, ..., & J.E. Iglesias')
            print('Under revision')
def prepare_output_files(path_images, out_seg, out_posteriors, out_resampled, out_volumes, out_qc):
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
    out_qc = os.path.abspath(out_qc) if (out_qc is not None) else out_qc
    if basename[-4:] == '.txt':
        if not os.path.isfile(path_images):
            raise Exception('provided text file containing paths of input images does not exist' % path_images)
        with open(path_images, 'r') as f:
            path_images = [line.replace('\n', '') for line in f.readlines() if line != '\n']
        def text_helper(path, name):
            if path is not None:
                if path[-4:] != '.txt':
                    sys.exit('if path_images given as text file, so must be %s' % name)
                with open(path, 'r') as ff:
                    path = [line.replace('\n', '') for line in ff.readlines() if line != '\n']
                recompute_files = [not os.path.isfile(p) for p in path]
            else:
                path = [None] * len(path_images)
                recompute_files = [False] * len(path_images)
            unique_file = False
            return path, recompute_files, unique_file
        out_seg, recompute_seg, _ = text_helper(out_seg, 'path_segmentations')
        out_posteriors, recompute_post, _ = text_helper(out_posteriors, 'path_posteriors')
        out_resampled, recompute_resampled, _ = text_helper(out_resampled, 'path_resampled')
        out_volumes, recompute_volume, unique_volume_file = text_helper(out_volumes, 'path_volume')
        out_qc, recompute_qc, unique_qc_file = text_helper(out_qc, 'path_qc_scores')
    elif ('.nii.gz' not in basename) & ('.nii' not in basename) & ('.mgz' not in basename) & ('.npz' not in basename):
        if os.path.isfile(path_images):
            sys.exit('Extension not supported for %s, only use: .nii.gz, .nii, .mgz, or .npz' % path_images)
        path_images = list_images_in_folder(path_images)
        def helper_dir(path, name, file_type, suffix):
            unique_file = False
            if path is not None:
                if path[-4:] == '.txt':
                    sys.exit('%s can only be given as text file when path_images is.' % name)
                if file_type == 'csv':
                    if path[-4:] != '.csv':
                        print('%s provided without csv extension. Adding csv extension.' % name)
                        path += '.csv'
                    path = [path] * len(path_images)
                    recompute_files = [True] * len(path_images)
                    unique_file = True
                else:
                    if (path[-7:] == '.nii.gz') | (path[-4:] == '.nii') | (path[-4:] == '.mgz') | (path[-4:] == '.npz'):
                        raise Exception('Output FOLDER had a FILE extension' % path)
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
        out_seg, recompute_seg, _ = helper_dir(out_seg, 'path_segmentations', '', 'synthseg')
        out_posteriors, recompute_post, _ = helper_dir(out_posteriors, 'path_posteriors', '', 'posteriors')
        out_resampled, recompute_resampled, _ = helper_dir(out_resampled, 'path_resampled', '', 'resampled')
        out_volumes, recompute_volume, unique_volume_file = helper_dir(out_volumes, 'path_volumes', 'csv', '')
        out_qc, recompute_qc, unique_qc_file = helper_dir(out_qc, 'path_qc_scores', 'csv', '')
    else:
        if not os.path.isfile(path_images):
            sys.exit('file does not exist: %s \n'
                            'please make sure the path and the extension are correct' % path_images)
        path_images = [path_images]
        def helper_im(path, name, file_type, suffix):
            unique_file = False
            if path is not None:
                if path[-4:] == '.txt':
                    sys.exit('%s can only be given as text file when path_images is.' % name)
                if file_type == 'csv':
                    if path[-4:] != '.csv':
                        print('%s provided without csv extension. Adding csv extension.' % name)
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
        out_seg, recompute_seg, _ = helper_im(out_seg, 'path_segmentations', '', 'synthseg')
        out_posteriors, recompute_post, _ = helper_im(out_posteriors, 'path_posteriors', '', 'posteriors')
        out_resampled, recompute_resampled, _ = helper_im(out_resampled, 'path_resampled', '', 'resampled')
        out_volumes, recompute_volume, unique_volume_file = helper_im(out_volumes, 'path_volumes', 'csv', '')
        out_qc, recompute_qc, unique_qc_file = helper_im(out_qc, 'path_qc_scores', 'csv', '')
    return path_images, out_seg, out_posteriors, out_resampled, out_volumes, unique_volume_file, \
           out_qc, unique_qc_file
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
    im = add_axis(im, axis=[0, -1])
    return im, aff, h, im_res, shape, pad_idx, crop_idx
def build_model(model_file_segmentation,
                model_file_parcellation,
                model_file_qc,
                labels_segmentation,
                labels_denoiser,
                labels_parcellation,
                labels_qc,
                flip_indices,
                robust,
                do_parcellation,
                do_qc):
    if not os.path.isfile(model_file_segmentation):
        sys.exit("The provided model path does not exist.")
    n_labels_seg = len(labels_segmentation)
    if robust:
        n_groups = len(labels_denoiser)
        net = unet(nb_features=24,
                   input_shape=[None, None, None, 1],
                   nb_levels=5,
                   conv_size=3,
                   nb_labels=n_groups,
                   feat_mult=2,
                   activation='elu',
                   nb_conv_per_level=2,
                   batch_norm=-1,
                   name='unet')
        last_tensor = net.output
        import tensorflow as tf
        last_tensor = KL.Lambda(lambda x: tf.argmax(x, axis=-1))(last_tensor)
        last_tensor = KL.Lambda(lambda x: tf.one_hot(tf.cast(x, 'int32'), depth=n_groups, axis=-1))(last_tensor)
        net = keras.Model(inputs=net.inputs, outputs=last_tensor)
        net = unet(nb_features=16,
                   input_shape=[None, None, None, 1],
                   nb_levels=5,
                   conv_size=5,
                   nb_labels=n_groups,
                   feat_mult=2,
                   nb_conv_per_level=2,
                   batch_norm=-1,
                   activation='elu',
                   skip_n_concatenations=2,
                   input_model=net,
                   name='l2l')
        input_image = net.inputs[0]
        last_tensor = net.output
        last_tensor = KL.Lambda(lambda x: tf.argmax(x, axis=-1))(last_tensor)
        last_tensor = KL.Lambda(lambda x: tf.one_hot(tf.cast(x, 'int32'), depth=n_groups, axis=-1))(last_tensor)
        last_tensor = KL.Lambda(lambda x: tf.cast(tf.concat(x, axis=-1), 'float32'))([input_image, last_tensor])
        net = keras.Model(inputs=net.inputs, outputs=last_tensor)
        net = unet(nb_features=24,
                   input_shape=[None, None, None, 2],
                   nb_levels=5,
                   conv_size=3,
                   nb_labels=n_labels_seg,
                   feat_mult=2,
                   activation='elu',
                   nb_conv_per_level=2,
                   batch_norm=-1,
                   input_model=net,
                   name='unet2')
        net.load_weights(model_file_segmentation, by_name=True)
        name_segm_prediction_layer = 'unet2_prediction'
    else:
        net = unet(nb_features=24,
                   input_shape=[None, None, None, 1],
                   nb_levels=5,
                   conv_size=3,
                   nb_labels=n_labels_seg,
                   feat_mult=2,
                   activation='elu',
                   nb_conv_per_level=2,
                   batch_norm=-1,
                   name='unet')
        net.load_weights(model_file_segmentation, by_name=True)
        input_image = net.inputs[0]
        name_segm_prediction_layer = 'unet_prediction'
        last_tensor = net.output
        try:
            shape_list = last_tensor.shape.as_list()
        except Exception:
            try:
                import tensorflow as tf
                shape_list = tf.keras.backend.int_shape(last_tensor)
                if shape_list is None:
                    raise ValueError("tf.keras.backend.int_shape returned None")
            except Exception:
                shape_attr = getattr(last_tensor, "_shape", None) or getattr(last_tensor, "shape", None)
                if shape_attr is None:
                    raise RuntimeError("Unable to determine the shape of `last_tensor` (no shape/.get_shape/_shape available).")
                try:
                    shape_list = list(shape_attr)
                except Exception:
                    shape_list = list(tuple(shape_attr))
        if shape_list is None:
            raise RuntimeError("Resolved shape is None; cannot set _keras_shape.")
        shape_list = [None if s is None else int(s) for s in shape_list]
        last_tensor._keras_shape = tuple(shape_list)
        last_tensor = GaussianBlur(sigma=0.5)(last_tensor)
        net = keras.Model(inputs=net.inputs, outputs=last_tensor)
        if flip_indices is not None:
            seg = net.output
            image_flipped = RandomFlip(flip_axis=0, prob=1)(input_image)
            last_tensor = net(image_flipped)
            last_tensor = RandomFlip(flip_axis=0, prob=1)(last_tensor)
            import tensorflow as tf
            last_tensor = KL.Lambda(lambda x: tf.split(x, [1] * n_labels_seg, axis=-1), name='split')(last_tensor)
            reordered_channels = [last_tensor[flip_indices[i]] for i in range(n_labels_seg)]
            last_tensor = KL.Lambda(lambda x: tf.concat(x, -1)[0, ...], name='concat')(reordered_channels)
            name_segm_prediction_layer = 'average_lr'
            last_tensor = KL.Lambda(lambda x: 0.5 * (x[0] + x[1]), name=name_segm_prediction_layer)([seg, last_tensor])
            net = keras.Model(inputs=net.inputs, outputs=last_tensor)
    if do_parcellation:
        n_labels_parcellation = len(labels_parcellation)
        last_tensor = net.output
        last_tensor = KL.Lambda(lambda x: tf.cast(tf.argmax(x, axis=-1), 'int32'))(last_tensor)
        last_tensor = ConvertLabels(np.arange(n_labels_seg), labels_segmentation)(last_tensor)
        parcellation_masking_values = np.array([1 if ((ll == 3) | (ll == 42)) else 0 for ll in labels_segmentation])
        last_tensor = ConvertLabels(labels_segmentation, parcellation_masking_values)(last_tensor)
        last_tensor = KL.Lambda(lambda x: tf.one_hot(tf.cast(x, 'int32'), depth=2, axis=-1))(last_tensor)
        last_tensor = KL.Lambda(lambda x: tf.cast(tf.concat(x, axis=-1), 'float32'))([input_image, last_tensor])
        net = keras.Model(inputs=net.inputs, outputs=last_tensor)
        net = unet(nb_features=24,
                   input_shape=[None, None, None, 3],
                   nb_levels=5,
                   conv_size=3,
                   nb_labels=n_labels_parcellation,
                   feat_mult=2,
                   activation='elu',
                   nb_conv_per_level=2,
                   batch_norm=-1,
                   name='unet_parc',
                   input_model=net)
        net.load_weights(model_file_parcellation, by_name=True)
        last_tensor = net.output
        try:
            shape_list = last_tensor.shape.as_list()
        except Exception:
            try:
                import tensorflow as tf
                shape_list = tf.keras.backend.int_shape(last_tensor)
                if shape_list is None:
                    raise ValueError("tf.keras.backend.int_shape returned None")
            except Exception:
                shape_attr = getattr(last_tensor, "_shape", None) or getattr(last_tensor, "shape", None)
                if shape_attr is None:
                    raise RuntimeError("Unable to determine the shape of `last_tensor` (no shape/.get_shape/_shape available).")
                try:
                    shape_list = list(shape_attr)
                except Exception:
                    shape_list = list(tuple(shape_attr))
        if shape_list is None:
            raise RuntimeError("Resolved shape is None; cannot set _keras_shape.")
        shape_list = [None if s is None else int(s) for s in shape_list]
        last_tensor._keras_shape = tuple(shape_list)
        last_tensor = GaussianBlur(sigma=0.5)(last_tensor)
        net = keras.Model(inputs=net.inputs, outputs=[net.get_layer(name_segm_prediction_layer).output, last_tensor])
    if do_qc:
        n_labels_qc = len(np.unique(labels_qc))
        shape_prediction = KL.Input([3], dtype='int32')
        if do_parcellation:
            last_tensor = KL.Lambda(lambda x: tf.cast(tf.argmax(x[0], axis=-1), 'int32'))(net.outputs)
        else:
            last_tensor = KL.Lambda(lambda x: tf.cast(tf.argmax(x, axis=-1), 'int32'))(net.output)
        last_tensor = MakeShape(224)([last_tensor, shape_prediction])
        last_tensor = ConvertLabels(np.arange(n_labels_seg), labels_segmentation)(last_tensor)
        last_tensor = ConvertLabels(labels_segmentation, labels_qc)(last_tensor)
        last_tensor = KL.Lambda(lambda x: tf.one_hot(tf.cast(x, 'int32'), depth=n_labels_qc, axis=-1))(last_tensor)
        net = keras.Model(inputs=[*net.inputs, shape_prediction], outputs=last_tensor)
        net = conv_enc(nb_features=24,
                       input_shape=[None, None, None, n_labels_qc],
                       nb_levels=4,
                       conv_size=5,
                       name='qc',
                       feat_mult=2,
                       activation='relu',
                       use_residuals=True,
                       nb_conv_per_level=2,
                       batch_norm=-1,
                       input_model=net)
        last_tensor = net.outputs[0]
        conv_kwargs = {'padding': 'same', 'activation': 'relu', 'data_format': 'channels_last'}
        last_tensor = KL.MaxPool3D(pool_size=(2, 2, 2), name='qc_maxpool_3', padding='same')(last_tensor)
        last_tensor = KL.Conv3D(n_labels_qc, kernel_size=5, **conv_kwargs, name='qc_final_conv_0')(last_tensor)
        last_tensor = KL.Conv3D(n_labels_qc, kernel_size=5, **conv_kwargs, name='qc_final_conv_1')(last_tensor)
        last_tensor = KL.Lambda(lambda x: tf.reduce_mean(x, axis=[1, 2, 3]), name='qc_final_pred')(last_tensor)
        if do_parcellation:
            outputs = [net.get_layer(name_segm_prediction_layer).output,
                       net.get_layer('unet_parc_prediction').output,
                       last_tensor]
        else:
            outputs = [net.get_layer(name_segm_prediction_layer).output, last_tensor]
        net = keras.Model(inputs=net.inputs, outputs=outputs)
        net.load_weights(model_file_qc, by_name=True)
    return net
def postprocess(post_patch_seg, post_patch_parc, shape, pad_idx, crop_idx,
                labels_segmentation, labels_parcellation, aff, im_res, fast, topology_classes, v1):
    post_patch_seg = np.squeeze(post_patch_seg)
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
        post_patch_parc = np.squeeze(post_patch_parc)
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
class MakeShape(KL.Layer):
    def __init__(self, target_shape, **kwargs):
        self.n_dims = None
        self.target_shape = target_shape
        self.cropping_shape = None
        super(MakeShape, self).__init__(**kwargs)
    def get_config(self):
        config = super().get_config()
        config["target_shape"] = self.target_shape
        return config
    def build(self, input_shape):
        self.n_dims = input_shape[1][1]
        self.cropping_shape = np.array(reformat_to_list(self.target_shape, length=self.n_dims))
        self.built = True
        super(MakeShape, self).build(input_shape)
    def call(self, inputs, **kwargs):
        elems = (tf.cast(inputs[0], dtype='int32'), tf.cast(inputs[1], dtype='int32'))
        return tf.map_fn(self._single_process, elems, dtype=tf.int32)
    def _single_process(self, inputs):
        x = inputs[0]
        shape = inputs[1]
        mask = tf.logical_and(tf.not_equal(x, 0), tf.not_equal(x, 24))
        indices = tf.cast(tf.where(mask), 'int32')
        min_idx = K.switch(tf.equal(tf.shape(indices)[0], 0),
                           tf.zeros(self.n_dims, dtype='int32'),
                           tf.maximum(tf.reduce_min(indices, axis=0), 0))
        max_idx = K.switch(tf.equal(tf.shape(indices)[0], 0),
                           tf.minimum(shape, self.cropping_shape),
                           tf.minimum(tf.reduce_max(indices, axis=0) + 1, shape))
        intermediate_vol_shape = max_idx - min_idx
        min_idx = min_idx - tf.cast(tf.math.ceil((self.cropping_shape - intermediate_vol_shape) / 2), 'int32')
        max_idx = max_idx + tf.cast(tf.math.floor((self.cropping_shape - intermediate_vol_shape) / 2), 'int32')
        tmp_min_idx = tf.maximum(min_idx, 0)
        tmp_max_idx = tf.minimum(max_idx, shape)
        x = tf.slice(x, begin=tmp_min_idx, size=tf.minimum(tmp_max_idx - tmp_min_idx, shape))
        min_padding = tf.abs(tf.minimum(min_idx, 0))
        max_padding = tf.maximum(max_idx - shape, 0)
        x = K.switch(tf.reduce_any(tf.logical_or(tf.greater(min_padding, 0), tf.greater(max_padding, 0))),
                     tf.pad(x, tf.stack([min_padding, max_padding], axis=1)),
                     x)
        return x
def get_flip_indices(labels_segmentation, n_neutral_labels):
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
def unet(nb_features,
         input_shape,
         nb_levels,
         conv_size,
         nb_labels,
         name='unet',
         prefix=None,
         feat_mult=1,
         pool_size=2,
         padding='same',
         dilation_rate_mult=1,
         activation='elu',
         skip_n_concatenations=0,
         use_residuals=False,
         final_pred_activation='softmax',
         nb_conv_per_level=1,
         layer_nb_feats=None,
         conv_dropout=0,
         batch_norm=None,
         input_model=None):
    model_name = name
    if prefix is None:
        prefix = model_name
    ndims = len(input_shape) - 1
    if isinstance(pool_size, int):
        pool_size = (pool_size,) * ndims
    enc_model = conv_enc(nb_features,
                         input_shape,
                         nb_levels,
                         conv_size,
                         name=model_name,
                         prefix=prefix,
                         feat_mult=feat_mult,
                         pool_size=pool_size,
                         padding=padding,
                         dilation_rate_mult=dilation_rate_mult,
                         activation=activation,
                         use_residuals=use_residuals,
                         nb_conv_per_level=nb_conv_per_level,
                         layer_nb_feats=layer_nb_feats,
                         conv_dropout=conv_dropout,
                         batch_norm=batch_norm,
                         input_model=input_model)
    lnf = layer_nb_feats[(nb_levels * nb_conv_per_level):] if layer_nb_feats is not None else None
    dec_model = conv_dec(nb_features,
                         None,
                         nb_levels,
                         conv_size,
                         nb_labels,
                         name=model_name,
                         prefix=prefix,
                         feat_mult=feat_mult,
                         pool_size=pool_size,
                         use_skip_connections=True,
                         skip_n_concatenations=skip_n_concatenations,
                         padding=padding,
                         dilation_rate_mult=dilation_rate_mult,
                         activation=activation,
                         use_residuals=use_residuals,
                         final_pred_activation=final_pred_activation,
                         nb_conv_per_level=nb_conv_per_level,
                         batch_norm=batch_norm,
                         layer_nb_feats=lnf,
                         conv_dropout=conv_dropout,
                         input_model=enc_model)
    final_model = dec_model
    return final_model
def conv_enc(nb_features,
             input_shape,
             nb_levels,
             conv_size,
             name=None,
             prefix=None,
             feat_mult=1,
             pool_size=2,
             dilation_rate_mult=1,
             padding='same',
             activation='elu',
             layer_nb_feats=None,
             use_residuals=False,
             nb_conv_per_level=2,
             conv_dropout=0,
             batch_norm=None,
             input_model=None):
    model_name = name
    if prefix is None:
        prefix = model_name
    name = '%s_input' % prefix
    if input_model is None:
        input_tensor = keras.layers.Input(shape=input_shape, name=name)
        last_tensor = input_tensor
    else:
        input_tensor = input_model.inputs
        last_tensor = input_model.outputs
        if isinstance(last_tensor, list):
            last_tensor = last_tensor[0]
    ndims = len(input_shape) - 1
    if isinstance(pool_size, int):
        pool_size = (pool_size,) * ndims
    convL = getattr(keras.layers, 'Conv%dD' % ndims)
    conv_kwargs = {'padding': padding, 'activation': activation, 'data_format': 'channels_last'}
    maxpool = getattr(keras.layers, 'MaxPooling%dD' % ndims)
    lfidx = 0
    for level in range(nb_levels):
        lvl_first_tensor = last_tensor
        nb_lvl_feats = np.round(nb_features * feat_mult ** level).astype(int)
        conv_kwargs['dilation_rate'] = dilation_rate_mult ** level
        for conv in range(nb_conv_per_level):
            if layer_nb_feats is not None:
                nb_lvl_feats = layer_nb_feats[lfidx]
                lfidx += 1
            name = '%s_conv_downarm_%d_%d' % (prefix, level, conv)
            if conv < (nb_conv_per_level - 1) or (not use_residuals):
                last_tensor = convL(nb_lvl_feats, conv_size, **conv_kwargs, name=name)(last_tensor)
            else:
                last_tensor = convL(nb_lvl_feats, conv_size, padding=padding, name=name)(last_tensor)
            if conv_dropout > 0:
                name = '%s_dropout_downarm_%d_%d' % (prefix, level, conv)
                noise_shape = [None, *[1] * ndims, nb_lvl_feats]
                last_tensor = keras.layers.Dropout(conv_dropout, noise_shape=noise_shape, name=name)(last_tensor)
        if use_residuals:
            convarm_layer = last_tensor
            nb_feats_in = _n_features_of(lvl_first_tensor)
            nb_feats_out = _n_features_of(convarm_layer)
            add_layer = lvl_first_tensor
            if nb_feats_in > 1 and nb_feats_out > 1 and (nb_feats_in != nb_feats_out):
                name = '%s_expand_down_merge_%d' % (prefix, level)
                last_tensor = convL(nb_lvl_feats, conv_size, **conv_kwargs, name=name)(lvl_first_tensor)
                add_layer = last_tensor
                if conv_dropout > 0:
                    name = '%s_dropout_down_merge_%d_%d' % (prefix, level, conv)
                    noise_shape = [None, *[1] * ndims, nb_lvl_feats]
            name = '%s_res_down_merge_%d' % (prefix, level)
            last_tensor = keras.layers.add([add_layer, convarm_layer], name=name)
            name = '%s_res_down_merge_act_%d' % (prefix, level)
            last_tensor = keras.layers.Activation(activation, name=name)(last_tensor)
        if batch_norm is not None:
            name = '%s_bn_down_%d' % (prefix, level)
            last_tensor = keras.layers.BatchNormalization(axis=batch_norm, name=name)(last_tensor)
        if level < (nb_levels - 1):
            name = '%s_maxpool_%d' % (prefix, level)
            last_tensor = maxpool(pool_size=pool_size, name=name, padding=padding)(last_tensor)
    model = keras.Model(inputs=input_tensor, outputs=[last_tensor], name=model_name)
    return model
def conv_dec(nb_features,
             input_shape,
             nb_levels,
             conv_size,
             nb_labels,
             name=None,
             prefix=None,
             feat_mult=1,
             pool_size=2,
             use_skip_connections=False,
             skip_n_concatenations=0,
             padding='same',
             dilation_rate_mult=1,
             activation='elu',
             use_residuals=False,
             final_pred_activation='softmax',
             nb_conv_per_level=2,
             layer_nb_feats=None,
             batch_norm=None,
             conv_dropout=0,
             input_model=None):
    model_name = name
    if prefix is None:
        prefix = model_name
    if use_skip_connections:
        assert input_model is not None, "is using skip connections, tensors dictionary is required"
    input_name = '%s_input' % prefix
    if input_model is None:
        input_tensor = keras.layers.Input(shape=input_shape, name=input_name)
        last_tensor = input_tensor
    else:
        input_tensor = input_model.input
        last_tensor = input_model.output
        if isinstance(last_tensor, (list, tuple)):
            last_tensor = last_tensor[0]
        try:
            input_shape = last_tensor.shape.as_list()[1:]
        except Exception:
            try:
                input_shape = last_tensor.get_shape().as_list()[1:]
            except Exception:
                shape_attr = getattr(last_tensor, "shape", None)
                if shape_attr is None:
                    raise RuntimeError("Unable to determine shape of the last tensor/output.")
                input_shape = list(shape_attr)[1:]
    ndims = len(input_shape) - 1
    if isinstance(pool_size, int):
        if ndims > 1:
            pool_size = (pool_size,) * ndims
    convL = getattr(keras.layers, 'Conv%dD' % ndims)
    conv_kwargs = {'padding': padding, 'activation': activation}
    upsample = getattr(keras.layers, 'UpSampling%dD' % ndims)
    lfidx = 0
    for level in range(nb_levels - 1):
        nb_lvl_feats = np.round(nb_features * feat_mult ** (nb_levels - 2 - level)).astype(int)
        conv_kwargs['dilation_rate'] = dilation_rate_mult ** (nb_levels - 2 - level)
        name = '%s_up_%d' % (prefix, nb_levels + level)
        last_tensor = upsample(size=pool_size, name=name)(last_tensor)
        up_tensor = last_tensor
        if use_skip_connections & (level < (nb_levels - skip_n_concatenations - 1)):
            conv_name = '%s_conv_downarm_%d_%d' % (prefix, nb_levels - 2 - level, nb_conv_per_level - 1)
            cat_tensor = input_model.get_layer(conv_name).output
            name = '%s_merge_%d' % (prefix, nb_levels + level)
            last_tensor = keras.layers.concatenate([cat_tensor, last_tensor], axis=ndims + 1, name=name)
        for conv in range(nb_conv_per_level):
            if layer_nb_feats is not None:
                nb_lvl_feats = layer_nb_feats[lfidx]
                lfidx += 1
            name = '%s_conv_uparm_%d_%d' % (prefix, nb_levels + level, conv)
            if conv < (nb_conv_per_level - 1) or (not use_residuals):
                last_tensor = convL(nb_lvl_feats, conv_size, **conv_kwargs, name=name)(last_tensor)
            else:
                last_tensor = convL(nb_lvl_feats, conv_size, padding=padding, name=name)(last_tensor)
            if conv_dropout > 0:
                name = '%s_dropout_uparm_%d_%d' % (prefix, level, conv)
                noise_shape = [None, *[1] * ndims, nb_lvl_feats]
                last_tensor = keras.layers.Dropout(conv_dropout, noise_shape=noise_shape, name=name)(last_tensor)
        if use_residuals:
            add_layer = up_tensor
            nb_feats_in = _n_features_of(add_layer)
            nb_feats_out = _n_features_of(last_tensor)
            if nb_feats_in > 1 and nb_feats_out > 1 and (nb_feats_in != nb_feats_out):
                name = '%s_expand_up_merge_%d' % (prefix, level)
                add_layer = convL(nb_lvl_feats, conv_size, **conv_kwargs, name=name)(add_layer)
                if conv_dropout > 0:
                    name = '%s_dropout_up_merge_%d_%d' % (prefix, level, conv)
                    noise_shape = [None, *[1] * ndims, nb_lvl_feats]
                    last_tensor = keras.layers.Dropout(conv_dropout, noise_shape=noise_shape, name=name)(last_tensor)
            name = '%s_res_up_merge_%d' % (prefix, level)
            last_tensor = keras.layers.add([last_tensor, add_layer], name=name)
            name = '%s_res_up_merge_act_%d' % (prefix, level)
            last_tensor = keras.layers.Activation(activation, name=name)(last_tensor)
        if batch_norm is not None:
            name = '%s_bn_up_%d' % (prefix, level)
            last_tensor = keras.layers.BatchNormalization(axis=batch_norm, name=name)(last_tensor)
    name = '%s_likelihood' % prefix
    last_tensor = convL(nb_labels, 1, activation=None, name=name)(last_tensor)
    like_tensor = last_tensor
    if final_pred_activation == 'softmax':
        name = '%s_prediction' % prefix
        softmax_lambda_fcn = lambda x: keras.activations.softmax(x, axis=ndims + 1)
        pred_tensor = keras.layers.Lambda(softmax_lambda_fcn, name=name)(last_tensor)
    else:
        name = '%s_prediction' % prefix
        pred_tensor = keras.layers.Activation('linear', name=name)(like_tensor)
    model = keras.Model(inputs=input_tensor, outputs=pred_tensor, name=model_name)
    return model
def load_volume(path_volume, im_only=True, squeeze=True, dtype=None, aff_ref=None):
    assert path_volume.endswith(('.nii', '.nii.gz', '.mgz', '.npz')), 'Unknown data file: %s' % path_volume
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
        neutral_FS_labels = [0, 14, 15, 16, 21, 22, 23, 24, 72, 77, 80, 85, 100, 101, 102, 103, 104, 105, 106, 107, 108,
                             109, 165, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210,
                             251, 252, 253, 254, 255, 258, 259, 260, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340,
                             502, 506, 507, 508, 509, 511, 512, 514, 515, 516, 517, 530,
                             531, 532, 533, 534, 535, 536, 537]
        neutral = list()
        left = list()
        right = list()
        for la in label_list:
            if la in neutral_FS_labels:
                if la not in neutral:
                    neutral.append(la)
            elif (0 < la < 14) | (16 < la < 21) | (24 < la < 40) | (135 < la < 139) | (1000 <= la <= 1035) | \
                    (la == 865) | (20100 < la < 20110):
                if la not in left:
                    left.append(la)
            elif (39 < la < 72) | (162 < la < 165) | (2000 <= la <= 2035) | (20000 < la < 20010) | (la == 139) | \
                    (la == 866):
                if la not in right:
                    right.append(la)
            else:
                raise Exception('label {} not in our current FS classification, '
                                'please update get_list_labels in utils.py'.format(la))
        label_list = np.concatenate([sorted(neutral), sorted(left), sorted(right)])
        if ((len(left) > 0) & (len(right) > 0)) | ((len(left) == 0) & (len(right) == 0)):
            n_neutral_labels = len(neutral)
        else:
            n_neutral_labels = len(label_list)
    if save_label_list is not None:
        np.save(save_label_list, np.int32(label_list))
    if FS_sort:
        return np.int32(label_list), n_neutral_labels
    else:
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
        else:
            raise ValueError("dtype should be 'str', 'float', 'int', or 'bool'; had {}".format(dtype))
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
            list_dir_to_create = [path_dir]
            while not os.path.isdir(os.path.dirname(list_dir_to_create[-1])):
                list_dir_to_create.append(os.path.dirname(list_dir_to_create[-1]))
            for dir_to_create in reversed(list_dir_to_create):
                os.mkdir(dir_to_create)
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
def infer(x):
    try:
        x = float(x)
    except ValueError:
        if x == 'False':
            x = False
        elif x == 'True':
            x = True
        elif not isinstance(x, str):
            raise TypeError('input should be an int/float/boolean/str, had {}'.format(type(x)))
    return x
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
        else:
            raise Exception('answer_type should be lower, higher, or closer, had : %s' % answer_type)
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
        assert list(mask.shape[:n_dims]) == vol_shape[:n_dims], 'mask should have shape {0}, or {1}, had {2}'.format(
            vol_shape[:n_dims], vol_shape[:n_dims] + [n_channels], list(mask.shape))
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
    assert (cropping_margin is not None) | (cropping_shape is not None), \
        'cropping_margin or cropping_shape should be provided'
    assert not ((cropping_margin is not None) & (cropping_shape is not None)), \
        'only one of cropping_margin or cropping_shape should be provided'
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
        elif mode == 'random':
            crop_max_val = np.maximum(np.array([vol_shape[i] - cropping_shape[i] for i in range(n_dims)]), 0)
            min_crop_idx = np.random.randint(0, high=crop_max_val + 1)
            max_crop_idx = np.minimum(min_crop_idx + np.array(cropping_shape), np.array(vol_shape)[:n_dims])
        else:
            raise ValueError('mode should be either "center" or "random", had %s' % mode)
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
    else:
        raise Exception('cannot crop volumes with more than 3 dimensions')
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
def flip_volume(volume, axis=None, direction=None, aff=None, return_copy=True):
    new_volume = volume.copy() if return_copy else volume
    assert (axis is not None) | ((aff is not None) & (direction is not None)), \
        'please provide either axis, or an affine matrix with a direction'
    if (axis is None) & (aff is not None):
        volume_axes = get_ras_axes(aff)
        if direction == 'rl':
            axis = volume_axes[0]
        elif direction == 'ap':
            axis = volume_axes[1]
        elif direction == 'si':
            axis = volume_axes[2]
        else:
            raise ValueError("direction should be 'rl', 'ap', or 'si', had %s" % direction)
    return np.flip(new_volume, axis=axis)
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
class RandomFlip(keras.layers.Layer):
    def __init__(self, flip_axis=None, swap_labels=False, label_list=None, n_neutral_labels=None, prob=0.5, **kwargs):
        self.several_inputs = True
        self.n_dims = None
        self.list_n_channels = None
        self.flip_axis = reformat_to_list(flip_axis)
        self.swap_labels = reformat_to_list(swap_labels)
        self.label_list = label_list
        self.n_neutral_labels = n_neutral_labels
        self.swap_lut = None
        self.prob = prob
        super(RandomFlip, self).__init__(**kwargs)
    def get_config(self):
        config = super().get_config()
        config["flip_axis"] = self.flip_axis
        config["swap_labels"] = self.swap_labels
        config["label_list"] = self.label_list
        config["n_neutral_labels"] = self.n_neutral_labels
        config["prob"] = self.prob
        return config
    def build(self, input_shape):
        if not isinstance(input_shape, list):
            self.several_inputs = False
            inputshape = [input_shape]
        else:
            inputshape = input_shape
        self.n_dims = len(inputshape[0][1:-1])
        self.list_n_channels = [i[-1] for i in inputshape]
        self.swap_labels = reformat_to_list(self.swap_labels, length=len(inputshape))
        if any(self.swap_labels):
            assert (self.label_list is not None) & (self.n_neutral_labels is not None), \
                'please provide a label_list, and n_neutral_labels when swapping the values of at least one input'
            n_labels = len(self.label_list)
            if self.n_neutral_labels == n_labels:
                self.swap_labels = [False] * len(self.swap_labels)
            else:
                rl_split = np.split(self.label_list, [self.n_neutral_labels,
                                                      self.n_neutral_labels + int((n_labels-self.n_neutral_labels)/2)])
                label_list_swap = np.concatenate((rl_split[0], rl_split[2], rl_split[1]))
                swap_lut = get_mapping_lut(self.label_list, label_list_swap)
                self.swap_lut = tf.convert_to_tensor(swap_lut, dtype='int32')
        self.built = True
        super(RandomFlip, self).build(input_shape)
    def call(self, inputs, **kwargs):
        if not self.several_inputs:
            inputs = [inputs]
        types = [v.dtype for v in inputs]
        batchsize = tf.split(tf.shape(inputs[0]), [1, self.n_dims + 1])[0]
        rand_flip = K.less(tf.random.uniform(tf.concat([batchsize, tf.ones(1, dtype='int32')], axis=0), 0, 1),
                           self.prob)
        swapped_inputs = list()
        for i in range(len(inputs)):
            if self.swap_labels[i]:
                swapped_inputs.append(tf.map_fn(self._single_swap, [inputs[i], rand_flip], dtype=types[i]))
            else:
                swapped_inputs.append(inputs[i])
        inputs = tf.concat([tf.cast(v, 'float32') for v in swapped_inputs], axis=-1)
        inputs = tf.map_fn(self._single_flip, [inputs, rand_flip], dtype=tf.float32)
        inputs = tf.split(inputs, self.list_n_channels, axis=-1)
        return [tf.cast(v, t) for (t, v) in zip(types, inputs)]
    def _single_swap(self, inputs):
        return K.switch(inputs[1], tf.gather(self.swap_lut, inputs[0]), inputs[0])
    def _single_flip(self, inputs):
        if self.flip_axis is None:
            flip_axis = tf.random.uniform([1], 0, self.n_dims, dtype='int32')
        else:
            idx = tf.squeeze(tf.random.uniform([1], 0, len(self.flip_axis), dtype='int32'))
            flip_axis = tf.expand_dims(tf.convert_to_tensor(self.flip_axis, dtype='int32')[idx], axis=0)
        return K.switch(inputs[1], K.reverse(inputs[0], axes=flip_axis), inputs[0])
class GaussianBlur(keras.layers.Layer):
    def __init__(self, sigma, random_blur_range=None, use_mask=False, **kwargs):
        self.sigma = reformat_to_list(sigma)
        assert np.all(np.array(self.sigma) >= 0), 'sigma should be superior or equal to 0'
        self.use_mask = use_mask
        self.n_dims = None
        self.n_channels = None
        self.blur_range = random_blur_range
        self.stride = None
        self.separable = None
        self.kernels = None
        self.convnd = None
        super(GaussianBlur, self).__init__(**kwargs)
    def get_config(self):
        config = super().get_config()
        config["sigma"] = self.sigma
        config["random_blur_range"] = self.blur_range
        config["use_mask"] = self.use_mask
        return config
    def build(self, input_shape):
        if self.use_mask:
            assert len(input_shape) == 2, 'please provide a mask as second layer input when use_mask=True'
            self.n_dims = len(input_shape[0]) - 2
            self.n_channels = input_shape[0][-1]
        else:
            self.n_dims = len(input_shape) - 2
            self.n_channels = input_shape[-1]
        self.stride = [1]*(self.n_dims+2)
        self.sigma = reformat_to_list(self.sigma, length=self.n_dims)
        self.separable = np.linalg.norm(np.array(self.sigma)) > 5
        if self.blur_range is None:
            self.kernels = gaussian_kernel(self.sigma, separable=self.separable)
        else:
            self.kernels = None
        self.convnd = getattr(tf.nn, 'conv%dd' % self.n_dims)
        self.built = True
        super(GaussianBlur, self).build(input_shape)
    def call(self, inputs, **kwargs):
        if self.use_mask:
            image = inputs[0]
            mask = tf.cast(inputs[1], 'bool')
        else:
            image = inputs
            mask = None
        if self.blur_range is not None:
            self.kernels = gaussian_kernel(self.sigma, blur_range=self.blur_range, separable=self.separable)
        if self.separable:
            for k in self.kernels:
                if k is not None:
                    image = tf.concat([self.convnd(tf.expand_dims(image[..., n], -1), k, self.stride, 'SAME')
                                       for n in range(self.n_channels)], -1)
                    if self.use_mask:
                        maskb = tf.cast(mask, 'float32')
                        maskb = tf.concat([self.convnd(tf.expand_dims(maskb[..., n], -1), k, self.stride, 'SAME')
                                           for n in range(self.n_channels)], -1)
                        image = image / (maskb + keras.backend.epsilon())
                        image = tf.where(mask, image, tf.zeros_like(image))
        else:
            if any(self.sigma):
                image = tf.concat([self.convnd(tf.expand_dims(image[..., n], -1), self.kernels, self.stride, 'SAME')
                                   for n in range(self.n_channels)], -1)
                if self.use_mask:
                    maskb = tf.cast(mask, 'float32')
                    maskb = tf.concat([self.convnd(tf.expand_dims(maskb[..., n], -1), self.kernels, self.stride, 'SAME')
                                       for n in range(self.n_channels)], -1)
                    image = image / (maskb + keras.backend.epsilon())
                    image = tf.where(mask, image, tf.zeros_like(image))
        return image
class ConvertLabels(keras.layers.Layer):
    def __init__(self, source_values, dest_values=None, **kwargs):
        self.source_values = source_values
        self.dest_values = dest_values
        self.lut = None
        super(ConvertLabels, self).__init__(**kwargs)
    def get_config(self):
        config = super().get_config()
        config["source_values"] = self.source_values
        config["dest_values"] = self.dest_values
        return config
    def build(self, input_shape):
        self.lut = tf.convert_to_tensor(get_mapping_lut(self.source_values, dest=self.dest_values), dtype='int32')
        self.built = True
        super(ConvertLabels, self).build(input_shape)
    def call(self, inputs, **kwargs):
        return tf.gather(self.lut, tf.cast(inputs, dtype='int32'))
def gaussian_kernel(sigma, max_sigma=None, blur_range=None, separable=True):
    if not tf.is_tensor(sigma):
        sigma_tens = tf.convert_to_tensor(reformat_to_list(sigma), dtype='float32')
    else:
        assert max_sigma is not None, 'max_sigma must be provided when sigma is given as a tensor'
        sigma_tens = sigma
    shape = sigma_tens.get_shape().as_list()
    if shape[0] is not None:
        n_dims = shape[0]
        batchsize = None
    else:
        n_dims = shape[1]
        batchsize = tf.split(tf.shape(sigma_tens), [1, -1])[0]
    if max_sigma is not None:
        max_sigma = np.array(reformat_to_list(max_sigma, length=n_dims))
    else:
        max_sigma = np.array(reformat_to_list(sigma, length=n_dims))
    if blur_range is not None:
        if blur_range != 1:
            sigma_tens = sigma_tens * tf.random.uniform(tf.shape(sigma_tens), minval=1 / blur_range, maxval=blur_range)
    windowsize = np.int32(np.ceil(2.5 * max_sigma) / 2) * 2 + 1
    if separable:
        split_sigma = tf.split(sigma_tens, [1] * n_dims, axis=-1)
        kernels = list()
        comb = np.array(list(combinations(list(range(n_dims)), n_dims - 1))[::-1])
        for (i, wsize) in enumerate(windowsize):
            if wsize > 1:
                locations = tf.cast(tf.range(0, wsize), 'float32') - (wsize - 1) / 2
                if batchsize is not None:
                    locations = tf.tile(tf.expand_dims(locations, axis=0),
                                        tf.concat([batchsize, tf.ones(tf.shape(tf.shape(locations)), dtype='int32')],
                                                  axis=0))
                    comb[i] += 1
                exp_term = -K.square(locations) / (2 * split_sigma[i] ** 2)
                g = tf.exp(exp_term - tf.math.log(np.sqrt(2 * np.pi) * split_sigma[i]))
                g = g / tf.reduce_sum(g)
                for axis in comb[i]:
                    g = tf.expand_dims(g, axis=axis)
                kernels.append(tf.expand_dims(tf.expand_dims(g, -1), -1))
            else:
                kernels.append(None)
    else:
        mesh = [tf.cast(f, 'float32') for f in volshape_to_meshgrid(windowsize, indexing='ij')]
        diff = tf.stack([mesh[f] - (windowsize[f] - 1) / 2 for f in range(len(windowsize))], axis=-1)
        if batchsize is not None:
            diff = tf.tile(tf.expand_dims(diff, axis=0),
                           tf.concat([batchsize, tf.ones(tf.shape(tf.shape(diff)), dtype='int32')], axis=0))
            for i in range(n_dims):
                sigma_tens = tf.expand_dims(sigma_tens, axis=1)
        else:
            for i in range(n_dims):
                sigma_tens = tf.expand_dims(sigma_tens, axis=0)
        sigma_is_0 = tf.equal(sigma_tens, 0)
        exp_term = -K.square(diff) / (2 * tf.where(sigma_is_0, tf.ones_like(sigma_tens), sigma_tens)**2)
        norms = exp_term - tf.math.log(tf.where(sigma_is_0, tf.ones_like(sigma_tens), np.sqrt(2 * np.pi) * sigma_tens))
        kernels = K.sum(norms, -1)
        kernels = tf.exp(kernels)
        kernels /= tf.reduce_sum(kernels)
        kernels = tf.expand_dims(tf.expand_dims(kernels, -1), -1)
    return kernels
def volshape_to_meshgrid(volshape, **kwargs):
    isint = [float(d).is_integer() for d in volshape]
    if not all(isint):
        raise ValueError("volshape needs to be a list of integers")
    linvec = [tf.range(0, d) for d in volshape]
    return meshgrid(*linvec, **kwargs)
def meshgrid(*args, **kwargs):
    indexing = kwargs.pop("indexing", "xy")
    if kwargs:
        key = list(kwargs.keys())[0]
        raise TypeError("'{}' is an invalid keyword argument "
                        "for this function".format(key))
    if indexing not in ("xy", "ij"):
        raise ValueError("indexing parameter must be either 'xy' or 'ij'")
    ndim = len(args)
    s0 = (1,) * ndim
    output = []
    for i, x in enumerate(args):
        output.append(tf.reshape(tf.stack(x), (s0[:i] + (-1,) + s0[i + 1::])))
    shapes = [tf.size(x) for x in args]
    sz = [x.get_shape().as_list()[0] for x in args]
    if indexing == "xy" and ndim > 1:
        output[0] = tf.reshape(output[0], (1, -1) + (1,) * (ndim - 2))
        output[1] = tf.reshape(output[1], (-1, 1) + (1,) * (ndim - 2))
        shapes[0], shapes[1] = shapes[1], shapes[0]
        sz[0], sz[1] = sz[1], sz[0]
    for i in range(len(output)):
        stack_sz = [*sz[:i], 1, *sz[(i + 1):]]
        if indexing == 'xy' and ndim > 1 and i < 2:
            stack_sz[0], stack_sz[1] = stack_sz[1], stack_sz[0]
        output[i] = tf.tile(output[i], tf.stack(stack_sz))
    return output
if __name__ == '__main__':
    main()