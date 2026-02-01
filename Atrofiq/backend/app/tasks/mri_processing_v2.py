"""
Updated MRI Processing Tasks for MRBrain_final Integration
Handles both NIfTI file processing and feature data processing
"""

import os
import json
import logging
import requests
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
from urllib.parse import urljoin

from celery import current_task
from minio import Minio
from minio.error import S3Error

from ..celery_app import celery_app, get_celery_db_session
from .. import db as dbmod

# Configure logging
logger = logging.getLogger(__name__)

# Configuration - Get from environment or use defaults
MRBRAIN_API_BASE = os.getenv('MRBRAIN_API_URL', 'http://localhost:8000')  # Default to localhost for local dev
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")  # Default to localhost for local dev
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin") 
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "atrofiq")

# Paths to MRBrain_final
MRBRAIN_FINAL_PATH = os.getenv("MRBRAIN_FINAL_PATH", "/app/MRBrain_final")

def get_minio_client() -> Minio:
    """Get MinIO client instance."""
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE,
    )

def update_task_progress(task_id: str, progress: int, status: str = None):
    """Update task progress in database."""
    try:
        with get_celery_db_session() as session:
            if session is not None:
                task_record = session.query(dbmod.ProcessingTask).filter_by(task_id=task_id).first()
                if task_record:
                    task_record.progress = progress
                    if status:
                        task_record.status = status
                    if status in ["SUCCESS", "FAILURE"]:
                        task_record.completed_at = datetime.now(timezone.utc)
                    elif status == "STARTED" and not task_record.started_at:
                        task_record.started_at = datetime.now(timezone.utc)
                    session.commit()
                    logger.info(f"Updated task {task_id}: progress={progress}, status={status}")
    except Exception as e:
        logger.warning(f"Failed to update task progress: {e}")

def download_file_from_minio(minio_client: Minio, bucket: str, object_key: str, suffix: str = None) -> str:
    """Download file from MinIO to temporary directory."""
    try:
        suffix = suffix or Path(object_key).suffix
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.close()
        
        minio_client.fget_object(bucket, object_key, temp_file.name)
        logger.info(f"Downloaded {object_key} to {temp_file.name}")
        return temp_file.name
    except Exception as e:
        logger.error(f"Failed to download file {object_key}: {e}")
        raise

def call_2d3d_conversion(dicom_dir_path: str, folder_name: str = None) -> Dict[str, Any]:
    """
    Call MRBrain API for 2D-3D volume conversion from DICOM slices directory
    
    Args:
        dicom_dir_path: Path to directory containing DICOM slice files
        folder_name: Optional folder name for participant ID
        
    Returns:
        Dictionary with conversion results including generated NIfTI path
    """
    try:
        url = urljoin(MRBRAIN_API_BASE, '/convert-2d-to-3d')
        logger.info(f"Calling 2D-3D conversion API: {url}")
        logger.info(f"DICOM directory: {dicom_dir_path}")
        
        # Get list of DICOM files in directory
        dicom_files = []
        for filename in os.listdir(dicom_dir_path):
            if filename.lower().endswith(('.dcm', '.dicom')):
                dicom_files.append(os.path.join(dicom_dir_path, filename))
        
        if not dicom_files:
            raise Exception(f"No DICOM files found in directory: {dicom_dir_path}")
        
        logger.info(f"Found {len(dicom_files)} DICOM files for conversion")
        
        # Prepare files for upload - send all DICOM slices
        files = []
        for dicom_file in dicom_files:
            with open(dicom_file, 'rb') as f:
                file_content = f.read()
                files.append(('dicom_files', (os.path.basename(dicom_file), file_content, 'application/dicom')))
        
        data = {
            'participant_id': folder_name or Path(dicom_dir_path).name,
            'num_slices': len(dicom_files)
        }
        
        # Make request with all files
        response = requests.post(url, files=files, data=data, timeout=1800)  # 30 minutes timeout
        
        logger.info(f"2D-3D API Response Status: {response.status_code}")
        response.raise_for_status()
        result = response.json()
        logger.info(f"2D-3D API Response: {json.dumps(result, indent=2)[:500]}...")
        return result
    
    except requests.exceptions.Timeout:
        logger.error(f"Timeout calling 2D-3D API: {url}")
        raise Exception("2D-3D API request timed out")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling 2D-3D API {url}: {e}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            logger.error(f"Response text: {e.response.text}")
        raise Exception(f"2D-3D API error: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON response from 2D-3D API: {e}")
        raise Exception("Invalid response from 2D-3D API")
    except Exception as e:
        logger.error(f"Unexpected error in 2D-3D conversion: {e}")
        raise

def call_mrbrain_api(endpoint: str, files: Dict = None, data: Dict = None, json_data: Dict = None) -> Dict[str, Any]:
    """Call MRBrain API endpoint with improved error handling."""
    try:
        url = urljoin(MRBRAIN_API_BASE, endpoint)
        logger.info(f"Calling MRBrain API: {url}")
        
        if files:
            # For file uploads
            response = requests.post(url, files=files, data=data, timeout=1800)
        elif json_data:
            # For JSON payload
            response = requests.post(url, json=json_data, timeout=1800)
        else:
            # For form data
            response = requests.post(url, data=data, timeout=1800)
        
        logger.info(f"API Response Status: {response.status_code}")
        response.raise_for_status()
        result = response.json()
        logger.info(f"API Response: {json.dumps(result, indent=2)[:500]}...")
        return result
    
    except requests.exceptions.Timeout:
        logger.error(f"Timeout calling MRBrain API: {url}")
        raise Exception("MRBrain API request timed out")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling MRBrain API {url}: {e}")
        if hasattr(e.response, 'text'):
            logger.error(f"Response text: {e.response.text}")
        raise Exception(f"MRBrain API error: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON response from MRBrain API: {e}")
        raise Exception("Invalid response from MRBrain API")

def create_mock_feature_data(participant_id: str, age: float, gender: str) -> pd.DataFrame:
    """
    Create dummy feature data with 71+ features to match brain age model requirements.
    THIS IS TEMPORARY - Replace with real SynthSeg feature extraction.
    """
    np.random.seed(hash(participant_id) % (2**32))  # Deterministic per participant
    
    # Base features
    mock_features = {
        'participant_id': [participant_id],
        'Age': [age],
        'Sex': [gender.upper()],
    }
    
    # Age-related decline factor
    age_factor = 1.0 - (max(0, age - 25) * 0.003)
    gender_factor = 1.1 if gender.upper() in ['M', 'MALE'] else 1.0
    
    # Generate 71+ brain region features (matching SynthSeg output)
    brain_regions = [
        # Cortical regions
        'left_cerebral_white_matter', 'right_cerebral_white_matter',
        'left_cerebral_cortex', 'right_cerebral_cortex',
        'left_frontal_lobe', 'right_frontal_lobe',
        'left_parietal_lobe', 'right_parietal_lobe',
        'left_temporal_lobe', 'right_temporal_lobe',
        'left_occipital_lobe', 'right_occipital_lobe',
        
        # Subcortical structures
        'left_lateral_ventricle', 'right_lateral_ventricle',
        'left_thalamus', 'right_thalamus',
        'left_caudate', 'right_caudate',
        'left_putamen', 'right_putamen',
        'left_pallidum', 'right_pallidum',
        'left_hippocampus', 'right_hippocampus',
        'left_amygdala', 'right_amygdala',
        'left_accumbens', 'right_accumbens',
        
        # Cerebellum
        'left_cerebellum_white_matter', 'right_cerebellum_white_matter',
        'left_cerebellum_cortex', 'right_cerebellum_cortex',
        
        # Ventricles
        'third_ventricle', 'fourth_ventricle',
        
        # Brainstem
        'brainstem', 'midbrain', 'pons', 'medulla',
        
        # Additional subcortical
        'left_inf_lat_vent', 'right_inf_lat_vent',
        'left_ventraldc', 'right_ventraldc',
        'left_vessel', 'right_vessel',
        'left_choroid_plexus', 'right_choroid_plexus',
        
        # Corpus callosum regions
        'cc_posterior', 'cc_mid_posterior', 'cc_central', 'cc_mid_anterior', 'cc_anterior',
        
        # Additional cortical parcellations
        'left_superior_frontal', 'right_superior_frontal',
        'left_middle_frontal', 'right_middle_frontal',
        'left_inferior_frontal', 'right_inferior_frontal',
        'left_precentral', 'right_precentral',
        'left_postcentral', 'right_postcentral',
        'left_superior_parietal', 'right_superior_parietal',
        'left_inferior_parietal', 'right_inferior_parietal',
        'left_superior_temporal', 'right_superior_temporal',
        'left_middle_temporal', 'right_middle_temporal',
        'left_inferior_temporal', 'right_inferior_temporal',
        'left_insula', 'right_insula',
        'left_cingulate', 'right_cingulate',
        'left_precuneus', 'right_precuneus',
        'left_cuneus', 'right_cuneus',
    ]
    
    # Base volumes for different structure types (in mm³)
    base_volumes = {
        'cerebral_white_matter': 250000,
        'cerebral_cortex': 230000,
        'lateral_ventricle': 12000,
        'thalamus': 8500,
        'caudate': 3800,
        'putamen': 5200,
        'pallidum': 1800,
        'hippocampus': 4200,
        'amygdala': 1600,
        'accumbens': 600,
        'cerebellum_white_matter': 15000,
        'cerebellum_cortex': 50000,
        'ventricle': 2000,
        'brainstem': 20000,
        'lobe': 50000,
        'frontal': 15000,
        'temporal': 12000,
        'parietal': 10000,
        'occipital': 8000,
        'cc': 1500,
        'small': 500,
    }
    
    # Generate volumes for each region
    for region in brain_regions:
        # Determine base volume based on region name
        if 'cerebral_white_matter' in region:
            base = base_volumes['cerebral_white_matter']
        elif 'cerebral_cortex' in region:
            base = base_volumes['cerebral_cortex']
        elif 'lateral_ventricle' in region:
            base = base_volumes['lateral_ventricle']
        elif 'thalamus' in region:
            base = base_volumes['thalamus']
        elif 'caudate' in region:
            base = base_volumes['caudate']
        elif 'putamen' in region:
            base = base_volumes['putamen']
        elif 'pallidum' in region:
            base = base_volumes['pallidum']
        elif 'hippocampus' in region:
            base = base_volumes['hippocampus']
        elif 'amygdala' in region:
            base = base_volumes['amygdala']
        elif 'accumbens' in region:
            base = base_volumes['accumbens']
        elif 'cerebellum_white_matter' in region:
            base = base_volumes['cerebellum_white_matter']
        elif 'cerebellum_cortex' in region:
            base = base_volumes['cerebellum_cortex']
        elif 'ventricle' in region:
            base = base_volumes['ventricle']
        elif 'brainstem' in region or 'midbrain' in region or 'pons' in region or 'medulla' in region:
            base = base_volumes['brainstem'] / 4
        elif '_lobe' in region:
            base = base_volumes['lobe']
        elif 'frontal' in region:
            base = base_volumes['frontal']
        elif 'temporal' in region:
            base = base_volumes['temporal']
        elif 'parietal' in region:
            base = base_volumes['parietal']
        elif 'occipital' in region or 'cuneus' in region:
            base = base_volumes['occipital']
        elif 'cc_' in region:
            base = base_volumes['cc']
        else:
            base = base_volumes['small']
        
        # Apply age and gender factors
        random_factor = np.random.normal(1.0, 0.1)
        volume = base * age_factor * gender_factor * random_factor
        mock_features[region] = [max(0, volume)]
    
    return pd.DataFrame(mock_features)

@celery_app.task(bind=True, name='mri_processing.run_inference_v2')
def run_mri_inference_v2(self, study_folder: str, age: str, gender: str, username: str = None) -> Dict[str, Any]:
    """
    Enhanced MRI inference pipeline that works with MRBrain_final models.
    This version handles feature data processing for the new models.
    """
    task_id = self.request.id
    logger.info(f"Starting MRI inference v2 task {task_id} for folder {study_folder}")
    
    try:
        # Validate inputs - allow "unknown" for DICOM-only processing
        if age == "unknown":
            age_float = 50.0  # Default age for DICOM processing
            logger.info("Using default age (50) for DICOM processing without metadata")
        else:
            age_float = float(age)
        
        gender = gender.strip().upper()
        if gender == "UNKNOWN":
            gender_normalized = 'M'  # Default gender for DICOM processing
            logger.info("Using default gender (M) for DICOM processing without metadata")
        elif gender not in ['M', 'F', 'MALE', 'FEMALE']:
            raise ValueError(f"Invalid gender: {gender}")
        else:
            # Normalize gender for API
            gender_normalized = 'M' if gender in ['M', 'MALE'] else 'F'
        
        # Update task status
        update_task_progress(task_id, 0, "STARTED")
        
        # Get study from database
        with get_celery_db_session() as session:
            if session is not None:
                study = session.query(dbmod.Study).filter_by(folder=study_folder).first()
                if not study:
                    raise Exception(f"Study not found: {study_folder}")
                
                # Update study status
                study.status = "Processing"
                study.processing_by = username
                study.current_task_id = task_id
                session.commit()
                
                participant_id = study.folder or study_folder
        
        # Get MinIO client
        minio_client = get_minio_client()
        update_task_progress(task_id, 10)
        
        # Look for files in the study
        feature_data_key = None
        nifti_object_key = None
        dicom_object_key = None
        prefix = f"{study_folder}/"
        
        for obj in minio_client.list_objects(MINIO_BUCKET, prefix=prefix, recursive=True):
            name = obj.object_name.lower()
            if name.endswith('.csv') or name.endswith('.xlsx'):
                feature_data_key = obj.object_name
                logger.info(f"Found feature data file: {feature_data_key}")
            elif (name.endswith('.nii.gz') or name.endswith('.nii')) and not nifti_object_key:
                nifti_object_key = obj.object_name
                logger.info(f"Found NIfTI file: {nifti_object_key}")
            elif (name.endswith('.dcm') or name.endswith('.dicom')) and not dicom_object_key:
                dicom_object_key = obj.object_name
                logger.info(f"Found DICOM file: {dicom_object_key}")
        
        update_task_progress(task_id, 20)
        
        # Initialize results structure
        processing_results = {
            'brainage_result': None,
            'normative_result': None,
            'volume_2d3d_result': None
        }
        
        # Step 1: Try 2D-3D conversion if DICOM file is available
        if dicom_object_key:
            try:
                logger.info("Running 2D-3D volume conversion...")
                update_task_progress(task_id, 30, "Processing 2D-3D conversion")
                
                # Download DICOM file
                local_dicom_path = download_file_from_minio(minio_client, MINIO_BUCKET, dicom_object_key, '.dcm')
                
                # Run 2D-3D conversion
                volume_2d3d_result = call_2d3d_conversion(local_dicom_path)
                processing_results['volume_2d3d_result'] = volume_2d3d_result
                
                logger.info(f"2D-3D conversion completed: {volume_2d3d_result.get('success', False)}")
                
                # Clean up temp DICOM file
                try:
                    os.unlink(local_dicom_path)
                except:
                    pass
                    
            except Exception as e:
                logger.warning(f"2D-3D conversion failed: {e}")
                processing_results['volume_2d3d_result'] = {
                    'success': False,
                    'error': str(e),
                    'note': '2D-3D conversion failed, continuing with standard processing'
                }
        
        update_task_progress(task_id, 40)
        
        # Step 2: Prepare feature data for brain age analysis
        if feature_data_key:
            # Download existing feature data
            logger.info("Using existing feature data file")
            local_feature_path = download_file_from_minio(minio_client, MINIO_BUCKET, feature_data_key, '.csv')
        elif nifti_object_key:
            # NIfTI file exists but no features - use dummy data temporarily
            logger.warning(f"NIfTI file found ({nifti_object_key}) but no feature extraction implemented")
            logger.warning("Using DUMMY features with 71+ regions - predictions may not be accurate")
            logger.warning("TODO: Implement real SynthSeg segmentation → volume extraction")
            
            feature_df = create_mock_feature_data(participant_id, age_float, gender_normalized)
            temp_feature_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='w')
            feature_df.to_csv(temp_feature_file.name, index=False)
            temp_feature_file.close()
            local_feature_path = temp_feature_file.name
        else:
            # No NIfTI and no features - use dummy data
            logger.warning("No NIfTI or feature file found - using DUMMY feature data")
            logger.warning("Predictions will be approximate - need real brain imaging pipeline:")
            logger.warning("  1. Upload DICOM files")
            logger.warning("  2. Convert DICOM → NIfTI (2D-3D model)")
            logger.warning("  3. Segment NIfTI → Brain regions (SynthSeg)")
            logger.warning("  4. Extract volumes → CSV with 71+ features")
            
            feature_df = create_mock_feature_data(participant_id, age_float, gender_normalized)
            temp_feature_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='w')
            feature_df.to_csv(temp_feature_file.name, index=False)
            temp_feature_file.close()
            local_feature_path = temp_feature_file.name
        
        update_task_progress(task_id, 50)
        
        try:
            # Step 3: Run Brain Age Prediction
            logger.info("Running brain age prediction with MRBrain_final...")
            
            # Skip brain age if using default values (unknown metadata)
            if age == "unknown" or gender == "unknown":
                logger.info("Skipping brain age prediction due to unknown metadata")
                processing_results['brainage_result'] = {
                    'status': 'skipped',
                    'reason': 'Unknown metadata - brain age requires valid age and gender',
                    'predicted_age': None
                }
            else:
                with open(local_feature_path, 'rb') as feature_file:
                    files = {'feature_data': feature_file}
                    data = {
                        'age': str(age_float),
                        'gender': gender_normalized,
                        'participant_id': participant_id
                    }
                    
                    brainage_result = call_mrbrain_api('/brain-age', files=files, data=data)
                    processing_results['brainage_result'] = brainage_result
            
            logger.info("Brain age prediction completed")
            update_task_progress(task_id, 70)
            
            # Step 4: Run Normative Modeling (if applicable)
            try:
                # Skip normative modeling if using default values (unknown metadata)
                if age == "unknown" or gender == "unknown":
                    logger.info("Skipping normative modeling due to unknown metadata")
                    processing_results['normative_result'] = {
                        'status': 'skipped',
                        'reason': 'Unknown metadata - normative modeling requires valid demographic data',
                        'regions': []
                    }
                else:
                    with open(local_feature_path, 'rb') as feature_file:
                        files = {'feature_data': feature_file}
                        data = {'participant_id': participant_id}
                        
                        normative_result = call_mrbrain_api('/normative', files=files, data=data)
                        
                        # Flatten the response structure for frontend compatibility
                        # MRBrain API returns {status, results: {percentile_scores, ...}}
                        # Frontend expects {status, percentile_scores, ...}
                        if normative_result and 'results' in normative_result:
                            flattened_result = {
                                'status': normative_result.get('status', 'success'),
                                'job_id': normative_result.get('job_id'),
                                'participant_id': normative_result.get('participant_id'),
                                **normative_result['results']  # Spread results into top level
                            }
                            processing_results['normative_result'] = flattened_result
                        else:
                            processing_results['normative_result'] = normative_result
                
                logger.info("Normative modeling completed")
            except Exception as e:
                logger.warning(f"Normative modeling failed: {e}")
                processing_results['normative_result'] = {
                    'status': 'failed',
                    'error': str(e),
                    'note': 'Normative modeling failed, using brain age only'
                }
            
            update_task_progress(task_id, 90)
            
            # Step 5: Combine all results
            combined_results = {
                'brainAge': processing_results['brainage_result'],
                'normative': processing_results['normative_result'],
                'volume_2d3d': processing_results['volume_2d3d_result'],
                'metadata': {
                    'participant_id': participant_id,
                    'age': age_float,
                    'gender': gender_normalized,
                    'study_folder': study_folder,
                    'feature_data_used': feature_data_key or 'mock_data',
                    'nifti_file': nifti_object_key,
                    'dicom_file': dicom_object_key,
                    'processed_at': datetime.now(timezone.utc).isoformat(),
                    'task_id': task_id,
                    'model_version': 'MRBrain_final_v2.0',
                    'has_2d3d_conversion': dicom_object_key is not None and processing_results['volume_2d3d_result'] is not None
                }
            }
            
            # Update study with results
            with get_celery_db_session() as session:
                if session is not None:
                    study = session.query(dbmod.Study).filter_by(folder=study_folder).first()
                    if study:
                        study.status = "Completed"
                        study.completed_by = username
                        study.current_task_id = None
                        study.normative_results = processing_results['normative_result']
                        study.brainage_results = processing_results['brainage_result']
                        study.nifti_object = nifti_object_key
                        session.commit()
            
            # Update task completion
            update_task_progress(task_id, 100, "SUCCESS")
            
            logger.info(f"MRI inference v2 task {task_id} completed successfully")
            return combined_results
            
        finally:
            # Clean up temporary files
            try:
                os.unlink(local_feature_path)
                logger.info(f"Cleaned up temporary feature file: {local_feature_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {e}")
            # Clean up temporary files
            try:
                os.unlink(local_feature_path)
                logger.info(f"Cleaned up temporary feature file: {local_feature_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {e}")
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"MRI inference v2 task {task_id} failed: {error_msg}")
        
        # Update task as failed
        try:
            with get_celery_db_session() as session:
                if session is not None:
                    # Update task record
                    task_record = session.query(dbmod.ProcessingTask).filter_by(task_id=task_id).first()
                    if task_record:
                        task_record.status = "FAILURE"
                        task_record.error_info = error_msg
                        task_record.completed_at = datetime.now(timezone.utc)
                    
                    # Update study status
                    study = session.query(dbmod.Study).filter_by(folder=study_folder).first()
                    if study:
                        study.status = "Failed"
                        study.current_task_id = None
                    
                    session.commit()
        except Exception as db_e:
            logger.error(f"Failed to update failure status in DB: {db_e}")
        
        # Re-raise the exception for Celery
        raise

# Keep the original task for backward compatibility
@celery_app.task(bind=True, name='mri_processing.run_inference')
def run_mri_inference(self, study_folder: str, age: str, gender: str, username: str = None) -> Dict[str, Any]:
    """
    Legacy MRI inference task - delegates to new version
    """
    logger.info(f"Legacy task called, executing v2 logic for {study_folder}")
    # Call the task with the current task context by manually updating the request
    original_task = run_mri_inference_v2
    return original_task.__wrapped__(self, study_folder, age, gender, username)