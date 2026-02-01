#!/usr/bin/env python3
"""
2D-3D API Integration for MRBrain_final
Converts DICOM/NIfTI files to high-resolution 3D volumes
"""

import os
import sys
import json
import tempfile
import logging
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import nibabel as nib
from datetime import datetime

# Add the 2D-3D directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "2D-3D"))

try:
    from Inference import run_inference, MRIInference
    INFERENCE_AVAILABLE = True
except Exception as e:
    print(f"Warning: 2D-3D inference not available: {e}")
    INFERENCE_AVAILABLE = False

logger = logging.getLogger(__name__)

class Volume2D3DProcessor:
    """
    Processor for converting 2D MRI to 3D high-resolution volumes
    """
    
    def __init__(self, checkpoint_path=None):
        self.checkpoint_path = checkpoint_path or self._find_checkpoint()
        self.device = 'cuda' if os.environ.get('CUDA_AVAILABLE', 'false').lower() == 'true' else 'cpu'
        self.model_loaded = False
        self.inferencer = None
        
        if self.checkpoint_path and Path(self.checkpoint_path).exists():
            logger.info(f"2D-3D checkpoint found: {self.checkpoint_path}")
        else:
            logger.warning("No 2D-3D checkpoint found - using fallback mode")
    
    def _find_checkpoint(self):
        """Find the 2D-3D model checkpoint"""
        possible_paths = [
            Path(__file__).parent / "2D-3D" / "checkpoints" / "best_model.pth",
            Path(__file__).parent / "2D-3D" / "models" / "generator.pth",
            Path(__file__).parent / "2D-3D" / "saved_models" / "latest.pth",
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path)
        
        return None
    
    def _load_model_lazy(self):
        """Load the model only when needed"""
        if not self.model_loaded and self.checkpoint_path and INFERENCE_AVAILABLE:
            try:
                logger.info("Loading 2D-3D model...")
                self.inferencer = MRIInference(
                    checkpoint_path=self.checkpoint_path,
                    device=self.device,
                    target_slices_output=256
                )
                self.model_loaded = True
                logger.info("2D-3D model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load 2D-3D model: {e}")
                self.model_loaded = False
    
    def convert_dicom_to_3d(self, dicom_path: str, output_dir: str = None) -> Dict:
        """
        Convert DICOM series to high-resolution 3D NIfTI volume
        
        Args:
            dicom_path: Path to DICOM file OR directory containing DICOM slices
            output_dir: Directory to save outputs (optional)
            
        Returns:
            Dict with conversion results including output_nifti_path
        """
        try:
            # Create output directory
            if not output_dir:
                output_dir = tempfile.mkdtemp(prefix="2d3d_output_")
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"2D-3D conversion started. Input: {dicom_path}, Output: {output_dir}")
            
            # Convert DICOM to NIfTI (this is the main goal)
            nifti_path = self._dicom_to_nifti(dicom_path, output_path)
            
            if not nifti_path:
                return {
                    'success': False,
                    'error': 'Failed to convert DICOM to NIfTI'
                }
            
            logger.info(f"Successfully converted DICOM to NIfTI: {nifti_path}")
            
            # Load NIfTI to get shape information
            try:
                import nibabel as nib
                nifti_img = nib.load(nifti_path)
                input_shape = list(nifti_img.shape)
                logger.info(f"Input NIfTI shape: {input_shape}")
            except Exception as e:
                logger.warning(f"Could not load NIfTI to get shape: {e}")
                input_shape = [256, 256, 30]  # default
            
            # Optional: Apply GAN enhancement if model is available
            enhanced_nifti_path = nifti_path
            output_shape = input_shape
            
            if self.model_loaded and INFERENCE_AVAILABLE:
                try:
                    logger.info("Applying GAN enhancement...")
                    enhanced_nifti_path = self._enhance_with_gan(nifti_path, output_path)
                    
                    if enhanced_nifti_path:
                        # Get enhanced shape
                        enhanced_img = nib.load(enhanced_nifti_path)
                        output_shape = list(enhanced_img.shape)
                        logger.info(f"Enhanced NIfTI shape: {output_shape}")
                    else:
                        enhanced_nifti_path = nifti_path
                except Exception as e:
                    logger.warning(f"GAN enhancement failed, using basic NIfTI: {e}")
                    enhanced_nifti_path = nifti_path
            
            # Return success with NIfTI path
            result = {
                'success': True,
                'output_nifti_path': str(enhanced_nifti_path),  # This is the key field!
                'output_3d_file': os.path.basename(enhanced_nifti_path),
                'input_shape': input_shape,
                'output_shape': output_shape,
                'conversion_method': 'DICOM → NIfTI' + (' → GAN enhanced' if enhanced_nifti_path != nifti_path else ''),
                'processing_time': datetime.now().isoformat()
            }
            
            logger.info(f"2D-3D conversion completed successfully: {result}")
            return result
        
        except Exception as e:
            logger.error(f"2D-3D conversion failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }
    
    def _dicom_to_nifti(self, dicom_path: str, output_dir: Path) -> Optional[str]:
        """
        Convert DICOM series to NIfTI format
        
        Args:
            dicom_path: Path to DICOM file or directory
            output_dir: Output directory for NIfTI
            
        Returns:
            Path to generated NIfTI file
        """
        try:
            import pydicom
            import nibabel as nib
            import numpy as np
            
            dicom_path = Path(dicom_path)
            
            # Check if it's a directory or single file
            if dicom_path.is_dir():
                # Directory of DICOM slices - need to load and stack them
                logger.info(f"Loading DICOM series from directory: {dicom_path}")
                
                # Get all DICOM files
                dicom_files = sorted([f for f in dicom_path.glob('*') 
                                     if f.suffix.lower() in ['.dcm', '.dicom'] or f.suffix == ''])
                
                if not dicom_files:
                    raise Exception("No DICOM files found in directory")
                
                logger.info(f"Found {len(dicom_files)} DICOM files")
                
                # Read first slice to get dimensions
                first_slice = pydicom.dcmread(str(dicom_files[0]))
                rows = first_slice.Rows
                cols = first_slice.Columns
                
                # Initialize volume array
                volume_data = []
                
                # Load all slices
                for dicom_file in dicom_files:
                    try:
                        ds = pydicom.dcmread(str(dicom_file))
                        slice_data = ds.pixel_array.astype(np.float32)
                        volume_data.append(slice_data)
                    except Exception as e:
                        logger.warning(f"Failed to read {dicom_file}: {e}")
                        continue
                
                # Stack into 3D volume
                volume_3d = np.stack(volume_data, axis=-1)
                logger.info(f"Created 3D volume with shape: {volume_3d.shape}")
                
            else:
                # Single DICOM file
                logger.info(f"Loading single DICOM file: {dicom_path}")
                ds = pydicom.dcmread(str(dicom_path))
                volume_3d = ds.pixel_array.astype(np.float32)
                
                if len(volume_3d.shape) == 2:
                    # Single slice, add dimension
                    volume_3d = volume_3d[:, :, np.newaxis]
            
            # Create NIfTI image
            nifti_img = nib.Nifti1Image(volume_3d, affine=np.eye(4))
            
            # Save NIfTI
            output_nifti = output_dir / "converted_volume.nii.gz"
            nib.save(nifti_img, str(output_nifti))
            
            logger.info(f"Successfully saved NIfTI: {output_nifti}")
            return str(output_nifti)
            
        except ImportError as e:
            logger.error(f"Missing required library for DICOM conversion: {e}")
            logger.error("Please install: pip install pydicom nibabel")
            return None
        except Exception as e:
            logger.error(f"DICOM to NIfTI conversion failed: {e}", exc_info=True)
            return None
    
    def _enhance_with_gan(self, nifti_path: str, output_dir: Path) -> Optional[str]:
        """
        Apply GAN enhancement to NIfTI volume (optional)
        
        Args:
            nifti_path: Path to input NIfTI
            output_dir: Output directory
            
        Returns:
            Path to enhanced NIfTI or None
        """
        if not INFERENCE_AVAILABLE or not self.inferencer:
            return None
        
        try:
            logger.info(f"Running GAN inference on {nifti_path}")
            
            metadata = run_inference(
                input_path=nifti_path,
                output_dir=str(output_dir),
                checkpoint_path=self.checkpoint_path,
                device=self.device,
                target_slices_lr=25,
                target_slices_hr=256,
                target_size=256,
                batch_size=8,
                save_visualization=True
            )
            
            if metadata.get('status') == 'success':
                enhanced_path = metadata.get('output_file')
                if enhanced_path and Path(enhanced_path).exists():
                    logger.info(f"GAN enhancement successful: {enhanced_path}")
                    return enhanced_path
            
            return None
            
        except Exception as e:
            logger.error(f"GAN enhancement failed: {e}")
            return None
    
    def get_status(self) -> Dict:
        """Get processor status"""
        return {
            'available': INFERENCE_AVAILABLE,
            'model_loaded': self.model_loaded,
            'checkpoint_path': self.checkpoint_path,
            'checkpoint_exists': self.checkpoint_path and Path(self.checkpoint_path).exists(),
            'device': self.device
        }

# Global processor instance
processor_2d3d = Volume2D3DProcessor()

def process_dicom_to_3d(dicom_path: str, output_dir: str = None) -> Dict:
    """
    Main function to process DICOM to 3D
    This can be called from the main API
    """
    return processor_2d3d.convert_dicom_to_3d(dicom_path, output_dir)