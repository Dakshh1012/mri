import os
import json
import logging
import requests
import tempfile
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

# Configuration
MRBRAIN_API_BASE = os.getenv('MRBRAIN_API_URL', 'http://localhost:8000')
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "atrofiq")


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


def download_nifti_file(minio_client: Minio, bucket: str, object_key: str) -> str:
    """Download NIfTI file from MinIO to temporary directory."""
    try:
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz')
        temp_file.close()
        
        # Download file
        minio_client.fget_object(bucket, object_key, temp_file.name)
        logger.info(f"Downloaded {object_key} to {temp_file.name}")
        return temp_file.name
    except Exception as e:
        logger.error(f"Failed to download NIfTI file {object_key}: {e}")
        raise


def call_mrbrain_api(endpoint: str, files: Dict = None, data: Dict = None) -> Dict[str, Any]:
    """Call MRBrain API endpoint."""
    try:
        url = urljoin(MRBRAIN_API_BASE, endpoint)
        logger.info(f"Calling MRBrain API: {url}")
        
        if files:
            # For file uploads
            response = requests.post(url, files=files, data=data, timeout=1800)  # 30 min timeout
        else:
            # For JSON data
            response = requests.post(url, json=data, timeout=1800)
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.Timeout:
        logger.error(f"Timeout calling MRBrain API: {url}")
        raise Exception("MRBrain API request timed out")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling MRBrain API {url}: {e}")
        raise Exception(f"MRBrain API error: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON response from MRBrain API: {e}")
        raise Exception("Invalid response from MRBrain API")


@celery_app.task(bind=True, name='mri_processing.run_inference')
def run_mri_inference(self, study_folder: str, age: str, gender: str, username: str = None) -> Dict[str, Any]:
    """
    Run complete MRI inference pipeline for a study.
    
    Args:
        study_folder: Study folder name in MinIO
        age: Patient age
        gender: Patient gender
        username: User who initiated processing
        
    Returns:
        Dict containing results from normative modeling and brain age prediction
    """
    task_id = self.request.id
    logger.info(f"Starting MRI inference task {task_id} for folder {study_folder}")
    
    try:
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
        
        # Get MinIO client
        minio_client = get_minio_client()
        
        # Find NIfTI file
        nifti_object_key = None
        prefix = f"{study_folder}/"
        
        for obj in minio_client.list_objects(MINIO_BUCKET, prefix=prefix, recursive=True):
            name = obj.object_name.lower()
            if name.endswith('.nii.gz') or name.endswith('.nii'):
                nifti_object_key = obj.object_name
                break
        
        if not nifti_object_key:
            raise Exception(f"No NIfTI file found in study folder {study_folder}")
        
        logger.info(f"Found NIfTI file: {nifti_object_key}")
        update_task_progress(task_id, 10)
        
        # Download NIfTI file
        local_nifti_path = download_nifti_file(minio_client, MINIO_BUCKET, nifti_object_key)
        update_task_progress(task_id, 20)
        
        try:
            # Step 1: Run normative modeling
            logger.info("Running normative modeling...")
            with open(local_nifti_path, 'rb') as nifti_file:
                files = {'nifti_file': nifti_file}
                data = {'age': float(age), 'gender': gender}
                
                normative_result = call_mrbrain_api('/normative', files=files, data=data)
            
            logger.info("Normative modeling completed")
            update_task_progress(task_id, 60)
            
            # Step 2: Run brain age prediction
            logger.info("Running brain age prediction...")
            with open(local_nifti_path, 'rb') as nifti_file:
                files = {'nifti_file': nifti_file}
                data = {'age': float(age), 'gender': gender}
                
                brainage_result = call_mrbrain_api('/brain-age', files=files, data=data)
            
            logger.info("Brain age prediction completed")
            update_task_progress(task_id, 90)
            
            # Combine results
            combined_results = {
                'normative': normative_result,
                'brainAge': brainage_result,
                'metadata': {
                    'age': age,
                    'gender': gender,
                    'nifti_object': nifti_object_key,
                    'processed_at': datetime.now(timezone.utc).isoformat(),
                    'task_id': task_id
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
                        study.normative_results = normative_result
                        study.brainage_results = brainage_result
                        study.nifti_object = nifti_object_key
                        session.commit()
            
            # Update task completion
            update_task_progress(task_id, 100, "SUCCESS")
            
            logger.info(f"MRI inference task {task_id} completed successfully")
            return combined_results
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(local_nifti_path)
                logger.info(f"Cleaned up temporary file: {local_nifti_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {e}")
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"MRI inference task {task_id} failed: {error_msg}")
        
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


