import io
import json
import os
import shutil
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from minio import Minio
from minio.error import S3Error
import logging
import subprocess
from contextlib import contextmanager

# Image processing
try:
    from PIL import Image
    import numpy as np
    IMAGE_PROCESSING_AVAILABLE = True
except ImportError:
    IMAGE_PROCESSING_AVAILABLE = False

# Celery imports
from .celery_app import celery_app
from .tasks.mri_processing_v2 import run_mri_inference_v2 as run_mri_inference

# DICOM integration
from .dicom_utils import get_orthanc_client, is_dicom_file

# Local DB utilities
try:
    from . import db as dbmod  # type: ignore
except Exception:
    dbmod = None

# Configure logging
logger = logging.getLogger(__name__)

@contextmanager
def db_session():
    if not dbmod:
        yield None
        return
    session = dbmod.SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

# Load environment from a .env file if present (backend root or project root)
try:
    from dotenv import load_dotenv  # type: ignore

    # Try backend root .env (../../backend/.env from this file) then current working dir
    backend_root = Path(__file__).resolve().parent.parent
    env_file = backend_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)  # do not override existing environment
    else:
        load_dotenv()
except Exception:
    # dotenv is optional; proceed if not installed
    pass


def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.lower() in {"1", "true", "yes", "on"}


MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "10.198.63.20:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = env_bool("MINIO_SECURE", False)
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "atrofiq")

# Frontend origin; allow all for dev by default
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")


def minio_client() -> Minio:
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE,
    )


app = FastAPI(title="Atrofiq API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS.split(",")] if ALLOWED_ORIGINS != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def ensure_bucket():
    """Ensure target bucket exists; don't crash app if MinIO is unavailable."""
    try:
        client = minio_client()
        found = client.bucket_exists(MINIO_BUCKET)
        if not found:
            client.make_bucket(MINIO_BUCKET)
    except Exception as e:
        logging.getLogger("atrofiq").warning(
            "MinIO check skipped at startup (endpoint=%s): %s",
            MINIO_ENDPOINT,
            e,
        )
    # Initialize database tables if available
    try:
        if dbmod:
            dbmod.init_db()
    except Exception as e:
        logging.getLogger("atrofiq").warning("DB init failed/skipped: %s", e)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def put_text_object(client: Minio, bucket: str, object_name: str, text: str) -> None:
    data = text.encode("utf-8")
    client.put_object(
        bucket,
        object_name,
        data=io.BytesIO(data),
        length=len(data),
        content_type="application/json",
    )


def get_text_object(client: Minio, bucket: str, object_name: str) -> Optional[str]:
    try:
        resp = client.get_object(bucket, object_name)
        try:
            return resp.read().decode("utf-8")
        finally:
            resp.close()
            resp.release_conn()
    except S3Error as e:
        if e.code in {"NoSuchKey", "NoSuchObject"}:
            return None
        raise


class StartProcessingRequest(BaseModel):
    username: str


class ProcessRequest(BaseModel):
    username: str
    age: str 
    gender: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: int
    result: Optional[dict] = None
    error_info: Optional[str] = None


@app.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    age: Optional[str] = Form(None),
    gender: Optional[str] = Form(None),
    username: Optional[str] = Form(None),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    client = minio_client()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    folder = f"study-{ts}"

    # Ensure bucket exists and is reachable; return clear error if not
    try:
        if not client.bucket_exists(MINIO_BUCKET):
            client.make_bucket(MINIO_BUCKET)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Storage unavailable: cannot access bucket '{MINIO_BUCKET}' at {MINIO_ENDPOINT}: {e}")

    # Save files under a common prefix (folder/filename) and process DICOM files
    count = 0
    object_keys = []
    orthanc_client = get_orthanc_client()
    dicom_processed = 0
    dicom_files = []  # Track DICOM files for 2D-3D conversion
    
    for f in files:
        # Read file data first
        file_data = await f.read()
        await f.seek(0)  # Reset for MinIO upload
        
        try:
            key = f"{folder}/{f.filename}"
            # Upload to MinIO
            client.put_object(
                MINIO_BUCKET,
                key,
                data=io.BytesIO(file_data),
                length=len(file_data),
                content_type=f.content_type or "application/octet-stream",
            )
            count += 1
            object_keys.append(key)
            
            # Check if it's a DICOM file (returns tuple: (is_dicom, info))
            is_dicom, dicom_info = is_dicom_file(file_data, f.filename)
            if is_dicom:
                dicom_files.append(key)
                dicom_processed += 1
                logger.info(f"Detected DICOM file: {f.filename} (size: {len(file_data)} bytes)")
                
                # Upload to Orthanc if available
                if orthanc_client and orthanc_client.is_available():
                    try:
                        # Pass the file_data bytes directly (already in memory)
                        instance_id = orthanc_client.upload_dicom(
                            file_data,  # file_data is bytes object, safe to reuse
                            metadata={
                                "folder_id": folder,
                                "uploaded_by": username,
                                "study_id": folder
                            }
                        )
                        if instance_id:
                            logger.info(f"DICOM file {f.filename} uploaded to Orthanc with instance ID: {instance_id}")
                        else:
                            logger.error(f"Failed to get instance ID for {f.filename}")
                    except Exception as e:
                        logger.warning(f"Failed to upload DICOM file {f.filename} to Orthanc: {e}")
            
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Storage unavailable during upload: {e}")
        finally:
            await f.close()

    # Write metadata file for the folder
    meta = {
        "age": age,
        "gender": gender,
        "uploaded_by": username,
        "status": "Available",
        "created_at": now_iso(),
        "last_updated": now_iso(),
        "processing_by": None,
        "completed_by": None,
    }
    try:
        put_text_object(client, MINIO_BUCKET, f"{folder}/_meta.json", json.dumps(meta))
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Storage unavailable when writing metadata: {e}")

    # Persist to DB if configured
    try:
        if dbmod:
            with db_session() as s:
                if s is not None:
                    record = dbmod.Study(
                        folder=folder,
                        age=age,
                        gender=gender,
                        uploaded_by=username,
                        status="Available",
                        bucket=MINIO_BUCKET,
                        object_keys=object_keys,
                    )
                    s.add(record)
                    s.commit()
    except Exception as e:
        logging.getLogger("atrofiq").warning("DB insert failed for folder %s: %s", folder, e)

    # If DICOM files detected, convert to NIfTI first
    task_id = None
    nifti_file_generated = None
    
    if dicom_files and age and gender:
        logger.info(f"Found {len(dicom_files)} DICOM files, converting to NIfTI...")
        try:
            # Call 2D-3D conversion directly (synchronous)
            from .tasks.mri_processing_v2 import call_2d3d_conversion
            import tempfile
            
            # Create temp directory and download DICOM files
            temp_dir = tempfile.mkdtemp(prefix="dicom_upload_")
            try:
                # Download all DICOM files
                for dicom_key in dicom_files:
                    filename = os.path.basename(dicom_key)
                    local_path = os.path.join(temp_dir, filename)
                    client.fget_object(MINIO_BUCKET, dicom_key, local_path)
                
                # Convert DICOM to NIfTI
                logger.info(f"Converting {len(dicom_files)} DICOM slices to NIfTI...")
                conversion_result = call_2d3d_conversion(temp_dir, folder_name=folder)
                
                if conversion_result.get('success'):
                    # Upload generated NIfTI to MinIO
                    output_nifti_path = conversion_result.get('output_3d_file')
                    if output_nifti_path and os.path.exists(output_nifti_path):
                        nifti_key = f"{folder}/converted_from_dicom.nii.gz"
                        client.fput_object(MINIO_BUCKET, nifti_key, output_nifti_path)
                        logger.info(f"Uploaded converted NIfTI to MinIO: {nifti_key}")
                        object_keys.append(nifti_key)
                        nifti_file_generated = nifti_key
                        
                        # Update metadata
                        meta["has_dicom_conversion"] = True
                        meta["nifti_file"] = nifti_key
                        meta["num_dicom_slices"] = len(dicom_files)
                        put_text_object(client, MINIO_BUCKET, f"{folder}/_meta.json", json.dumps(meta))
                    else:
                        logger.error("2D-3D conversion succeeded but no output file found")
                else:
                    error_msg = conversion_result.get('error', 'Unknown error')
                    logger.error(f"2D-3D conversion failed: {error_msg}")
                    
            finally:
                # Clean up temp directory
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    
        except Exception as e:
            logger.error(f"Failed to convert DICOM to NIfTI: {e}", exc_info=True)
    
    # Auto-start processing if age and gender are provided
    # For DICOM: process the generated NIfTI file
    # For direct NIfTI upload: process normally
    if age and gender:
        try:
            task = run_mri_inference.delay(
                study_folder=folder,
                age=age,
                gender=gender,
                username=username or "system"
            )
            task_id = task.id
            
            # Update study with task ID and processing status
            if dbmod:
                with db_session() as s:
                    if s is not None:
                        study = s.query(dbmod.Study).filter_by(folder=folder).first()
                        if study:
                            study.current_task_id = task.id
                            study.status = "Processing"
                            study.processing_by = username or "system"
                            
                            # Create task record
                            task_record = dbmod.ProcessingTask(
                                task_id=task.id,
                                task_name="mri_inference",
                                study_id=study.id,
                                input_params={
                                    "age": age,
                                    "gender": gender,
                                    "username": username or "system",
                                    "from_dicom": len(dicom_files) > 0
                                }
                            )
                            s.add(task_record)
                            s.commit()
            
            logger.info(f"Auto-started processing for {folder} with task {task.id}")
            
        except Exception as e:
            logger.error(f"Failed to auto-start processing for {folder}: {e}")

    return {
        "ok": True, 
        "folder": folder, 
        "files_count": count, 
        "dicom_files_processed": dicom_processed,
        "nifti_generated": nifti_file_generated is not None,
        "nifti_file": nifti_file_generated,
        "task_id": task_id, 
        "auto_processing_started": task_id is not None
    }


@app.get("/folders/")
def list_folders():
    client = minio_client()
    # Build map of folder -> latest timestamp
    folders = {}
    for obj in client.list_objects(MINIO_BUCKET, recursive=True):
        # Expect keys like 'folder/file.dcm' or 'folder/_meta.json'
        key = obj.object_name
        if "/" not in key:
            # Skip root-level objects; we treat only prefixed ones as folders
            continue
        folder, _ = key.split("/", 1)
        info = folders.setdefault(
            folder,
            {
                "name": folder,
                "patient_name": None,
                "patient_id": None,
                "accession": None,
                "description": None,
                "study_instance_uid": None,
                "modality": None,
                "status": "Available",
                "processing_by": None,
                "completed_by": None,
                "last_updated": None,
            },
        )
        # track latest timestamp
        ts = obj.last_modified
        if ts is not None:
            cur = info["last_updated"]
            if cur is None or ts > datetime.fromisoformat(cur):
                info["last_updated"] = ts.replace(tzinfo=timezone.utc).isoformat()

    # Try to enrich with metadata if present
    result = []
    for folder, info in sorted(folders.items(), key=lambda kv: kv[0]):
        meta_text = get_text_object(client, MINIO_BUCKET, f"{folder}/_meta.json")
        if meta_text:
            try:
                meta = json.loads(meta_text)
                info["status"] = meta.get("status") or info["status"]
                info["processing_by"] = meta.get("processing_by")
                info["completed_by"] = meta.get("completed_by")
                info["last_updated"] = meta.get("last_updated") or info["last_updated"]
            except Exception:
                pass
        result.append(info)
    return {"folders": result}


@app.get("/studies")
def list_studies():
    """List studies from PostgreSQL if available, else fall back to MinIO listing.

    Response mirrors `/folders/` for frontend compatibility.
    """
    # Prefer DB
    if dbmod:
        try:
            with db_session() as s:
                if s is not None:
                    rows = s.query(dbmod.Study).order_by(dbmod.Study.last_updated.desc()).all()
                    return {"folders": [dbmod.to_worklist_dict(r) for r in rows]}
        except Exception as e:
            logging.getLogger("atrofiq").warning("DB list failed, falling back: %s", e)
    # Fallback
    return list_folders()


@app.post("/start_processing/{folder}")
def start_processing(folder: str, payload: StartProcessingRequest):
    client = minio_client()
    # Load existing meta
    meta_text = get_text_object(client, MINIO_BUCKET, f"{folder}/_meta.json")
    meta = {}
    if meta_text:
        try:
            meta = json.loads(meta_text)
        except Exception:
            meta = {}
    # Update and save
    meta.update(
        {
            "status": "Processing",
            "processing_by": payload.username,
            "last_updated": now_iso(),
        }
    )
    put_text_object(client, MINIO_BUCKET, f"{folder}/_meta.json", json.dumps(meta))
    # Update DB
    if dbmod:
        try:
            with db_session() as s:
                if s is not None:
                    r = s.query(dbmod.Study).filter_by(folder=folder).one_or_none()
                    if r:
                        r.status = "Processing"
                        r.processing_by = payload.username
        except Exception as e:
            logging.getLogger("atrofiq").warning("DB update failed for %s: %s", folder, e)
    return {"ok": True}


@app.post("/process/{folder}")
def start_processing_async(folder: str, payload: ProcessRequest):
    """Start asynchronous MRI processing using Celery."""
    import uuid
    try:
        # Validate study exists
        if dbmod:
            with db_session() as s:
                if s is not None:
                    study = s.query(dbmod.Study).filter_by(folder=folder).first()
                    if not study:
                        raise HTTPException(status_code=404, detail=f"Study not found: {folder}")
                    
                    if study.current_task_id:
                        raise HTTPException(status_code=409, detail="Study is already being processed")
        
        # Start Celery task first to get actual task ID
        task = run_mri_inference.delay(
            study_folder=folder,
            age=payload.age,
            gender=payload.gender,
            username=payload.username
        )
        
        # Create task record with actual task ID
        if dbmod:
            with db_session() as s:
                if s is not None:
                    study = s.query(dbmod.Study).filter_by(folder=folder).first()
                    if study:
                        task_record = dbmod.ProcessingTask(
                            task_id=task.id,
                            task_name="mri_inference",
                            study_id=study.id,
                            input_params={
                                "age": payload.age,
                                "gender": payload.gender,
                                "username": payload.username
                            }
                        )
                        s.add(task_record)
                        study.current_task_id = task.id
                        s.commit()
        
        return {
            "ok": True,
            "task_id": task.id,
            "message": f"Processing started for study {folder}",
            "check_status_url": f"/task-status/{task.id}"
        }
        
    except Exception as e:
        logger.error(f"Failed to start processing for {folder}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/task-status/{task_id}")
def get_task_status(task_id: str) -> TaskStatusResponse:
    """Get status of a Celery task."""
    try:
        # Get task result from Celery
        task_result = celery_app.AsyncResult(task_id)
        
        # Get detailed info from database
        task_info = None
        if dbmod:
            with db_session() as s:
                if s is not None:
                    task_record = s.query(dbmod.ProcessingTask).filter_by(task_id=task_id).first()
                    if task_record:
                        task_info = {
                            "progress": task_record.progress,
                            "error_info": task_record.error_info,
                            "started_at": task_record.started_at.isoformat() if task_record.started_at else None,
                            "completed_at": task_record.completed_at.isoformat() if task_record.completed_at else None,
                        }
        
        return TaskStatusResponse(
            task_id=task_id,
            status=task_result.status,
            progress=task_info.get("progress", 0) if task_info else 0,
            result=task_result.result if task_result.successful() else None,
            error_info=task_info.get("error_info") if task_info else str(task_result.info) if task_result.failed() else None
        )
        
    except Exception as e:
        logger.error(f"Failed to get task status for {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/study/{folder}/results")
def get_study_results(folder: str):
    """Get analysis results for a completed study."""
    try:
        if not dbmod:
            raise HTTPException(status_code=503, detail="Database not available")
        
        with db_session() as s:
            if s is not None:
                study = s.query(dbmod.Study).filter_by(folder=folder).first()
                if not study:
                    raise HTTPException(status_code=404, detail=f"Study not found: {folder}")
                
                if study.status != "Completed":
                    return {
                        "status": study.status,
                        "message": f"Study is not completed yet. Current status: {study.status}",
                        "current_task_id": study.current_task_id
                    }
                
                return {
                    "status": "completed",
                    "normative_results": study.normative_results,
                    "brainage_results": study.brainage_results,
                    "metadata": {
                        "age": study.age,
                        "gender": study.gender,
                        "completed_by": study.completed_by,
                        "last_updated": study.last_updated.isoformat() if study.last_updated else None
                    }
                }
            else:
                raise HTTPException(status_code=503, detail="Database session unavailable")
                
    except Exception as e:
        logger.error(f"Failed to get results for study {folder}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Basic health check
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/test-2d3d")
def test_2d3d_endpoint():
    """Test endpoint to verify 2D-3D route registration"""
    return {"message": "2D-3D endpoint is registered", "available": True}


@app.delete("/study/{folder}")
async def delete_study(folder: str):
    """Delete a study folder and all its contents from ALL storage systems"""
    deletion_summary = {
        "folder": folder,
        "minio_objects_deleted": 0,
        "database_records_deleted": 0,
        "redis_keys_cleared": 0,
        "errors": []
    }
    
    try:
        # 1. MINIO CLEANUP - Delete all files in the study folder
        logger.info(f"Starting complete deletion for study folder: {folder}")
        client = minio_client()
        
        objects_to_delete = []
        try:
            for obj in client.list_objects(MINIO_BUCKET, prefix=f"{folder}/", recursive=True):
                objects_to_delete.append(obj.object_name)
        except Exception as e:
            logger.error(f"Error listing MinIO objects for deletion in folder {folder}: {e}")
            deletion_summary["errors"].append(f"MinIO list error: {str(e)}")
        
        # Delete all objects in the folder one by one
        if objects_to_delete:
            try:
                for obj_name in objects_to_delete:
                    try:
                        client.remove_object(MINIO_BUCKET, obj_name)
                        deletion_summary["minio_objects_deleted"] += 1
                        logger.debug(f"Deleted MinIO object: {obj_name}")
                    except Exception as obj_error:
                        logger.error(f"Error deleting MinIO object {obj_name}: {obj_error}")
                        deletion_summary["errors"].append(f"MinIO object {obj_name}: {str(obj_error)}")
                logger.info(f"Deleted {deletion_summary['minio_objects_deleted']} objects from MinIO folder {folder}")
            except Exception as e:
                logger.error(f"Error deleting MinIO objects from folder {folder}: {e}")
                deletion_summary["errors"].append(f"MinIO deletion error: {str(e)}")
        
        # 2. REDIS CLEANUP - Clear any cached task data or results
        try:
            import redis
            redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            
            # Get all keys related to this study/folder
            study_keys = []
            all_keys = redis_client.keys('*')
            
            for key in all_keys:
                # Look for keys that might contain the folder name or study info
                if folder in str(key):
                    study_keys.append(key)
                # Also check for Celery task result keys
                elif 'celery-task-meta-' in key:
                    try:
                        # Check if the task result contains our folder
                        result = redis_client.get(key)
                        if result and folder in str(result):
                            study_keys.append(key)
                    except:
                        pass  # Skip if we can't read the key
            
            # Delete found keys
            if study_keys:
                deleted_keys = redis_client.delete(*study_keys)
                deletion_summary["redis_keys_cleared"] = deleted_keys
                logger.info(f"Deleted {deleted_keys} Redis keys for study {folder}: {study_keys}")
            else:
                logger.info(f"No Redis keys found for study {folder}")
                
        except Exception as e:
            logger.error(f"Error cleaning Redis for study {folder}: {e}")
            deletion_summary["errors"].append(f"Redis cleanup error: {str(e)}")
        
        # 3. POSTGRESQL CLEANUP - Delete study and all related processing tasks
        if dbmod:
            try:
                with db_session() as db:
                    # Find the study record (using correct column name 'folder')
                    study = db.query(dbmod.Study).filter_by(folder=folder).first()
                    
                    if study:
                        # Get current task ID before deletion for Redis cleanup
                        current_task_id = study.current_task_id
                        
                        # Delete all processing tasks for this study (CASCADE will handle this automatically)
                        tasks_deleted = db.query(dbmod.ProcessingTask).filter_by(study_id=study.id).count()
                        
                        # Delete the study record (this will cascade delete processing tasks)
                        db.delete(study)
                        db.commit()
                        
                        deletion_summary["database_records_deleted"] = 1 + tasks_deleted  # Study + Tasks
                        logger.info(f"Deleted study record and {tasks_deleted} processing tasks for folder {folder}")
                        
                        # Additional Redis cleanup for current task if it exists
                        if current_task_id:
                            try:
                                import redis
                                redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
                                task_result_key = f"celery-task-meta-{current_task_id}"
                                if redis_client.exists(task_result_key):
                                    redis_client.delete(task_result_key)
                                    deletion_summary["redis_keys_cleared"] += 1
                                    logger.info(f"Deleted Redis task result for {current_task_id}")
                            except:
                                pass  # Non-critical if we can't clean this up
                    else:
                        logger.warning(f"Study record not found in database for folder {folder}")
                        deletion_summary["errors"].append("Study not found in database")
                        
            except Exception as e:
                logger.error(f"Error deleting study from database: {e}")
                deletion_summary["errors"].append(f"Database deletion error: {str(e)}")
        
        # 4. SUMMARY AND RESPONSE
        total_deleted = (deletion_summary["minio_objects_deleted"] + 
                        deletion_summary["database_records_deleted"] + 
                        deletion_summary["redis_keys_cleared"])
        
        if deletion_summary["errors"]:
            logger.warning(f"Study {folder} deletion completed with {len(deletion_summary['errors'])} errors")
            return {
                "message": f"Study {folder} deletion completed with some errors",
                "summary": deletion_summary,
                "total_items_deleted": total_deleted,
                "status": "partial_success"
            }
        else:
            logger.info(f"Study {folder} completely deleted from all storage systems")
            return {
                "message": f"Study {folder} completely deleted from all storage systems",
                "summary": deletion_summary,
                "total_items_deleted": total_deleted,
                "status": "success"
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during complete deletion of study {folder}: {e}")
        deletion_summary["errors"].append(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail={
            "message": f"Failed to delete study {folder}",
            "summary": deletion_summary,
            "error": str(e)
        })


@app.get("/folders/{folder}/nifti-url")
def presign_nifti_url(
    folder: str,
    expires_seconds: int = 3600,
    object: Optional[str] = None,
    pattern: Optional[str] = None,
):
    """Return a presigned URL to a NIfTI file within a given folder and include stored metadata.

    - Scans objects with prefix ``{folder}/`` for a ``.nii`` or ``.nii.gz`` file (case-insensitive).
    - Returns HTTP 404 if no NIfTI object is found.
    - Expires defaults to 3600 seconds.

    Response example:
    {
      "folder": "study-20240101-120000",
      "nifti_object": "study-20240101-120000/scan.nii.gz",
      "nifti_url": "https://minio/...",
      "url": "https://minio/...",               # alias for compatibility
      "presigned_url": "https://minio/...",      # alias for compatibility
      "expires_in": 3600,
      "meta": { ... }                             # contents of _meta.json if present
    }
    """
    client = minio_client()

    # Direct object key provided by client
    if object:
        target_key = object
    else:
        # Find a NIfTI object within the folder (prefer .nii.gz then .nii)
        nifti_candidates = []
        prefix = f"{folder}/"
        for obj in client.list_objects(MINIO_BUCKET, prefix=prefix, recursive=True):
            name = obj.object_name
            low = name.lower()
            if (low.endswith(".nii.gz") or low.endswith(".nii")) and (not pattern or pattern.lower() in low):
                nifti_candidates.append(name)

        # If not found under direct prefix, try a broader search across bucket
        if not nifti_candidates:
            for obj in client.list_objects(MINIO_BUCKET, recursive=True):
                name = obj.object_name
                low = name.lower()
                if (low.endswith(".nii.gz") or low.endswith(".nii")) and (f"/{folder.lower()}/" in low or low.startswith(prefix.lower())):
                    if not pattern or pattern.lower() in low:
                        nifti_candidates.append(name)

        # Prefer .nii.gz first
        nifti_key = None
        for cand in nifti_candidates:
            if cand.lower().endswith(".nii.gz"):
                nifti_key = cand
                break
        if nifti_key is None and nifti_candidates:
            nifti_key = nifti_candidates[0]
        target_key = nifti_key

    if not target_key:
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"No NIfTI file found for folder '{folder}'",
                "searched_prefix": f"{folder}/",
                "hint": "Provide ?object=<exact/key.nii.gz> or ?pattern=subdir to narrow search",
            },
        )

    # Generate a presigned GET URL
    try:
        url = client.presigned_get_object(
            MINIO_BUCKET,
            target_key,
            expires=timedelta(seconds=max(1, min(expires_seconds, 7 * 24 * 3600))),  # cap to 7 days
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to generate presigned URL: {e}")

    # Load metadata if present
    meta_text = get_text_object(client, MINIO_BUCKET, f"{folder}/_meta.json")
    meta = None
    if meta_text:
        try:
            meta = json.loads(meta_text)
        except Exception:
            meta = {"raw": meta_text}

    resp = {
        "folder": folder,
        "nifti_object": target_key,
        "nifti_url": url,
        "url": url,
        "presigned_url": url,
        "expires_in": expires_seconds,
        "meta": meta,
    }

    # Save chosen nifti key to DB for this folder
    if dbmod and target_key:
        try:
            with db_session() as s:
                if s is not None:
                    r = s.query(dbmod.Study).filter_by(folder=folder).one_or_none()
                    if r:
                        r.nifti_object = target_key
        except Exception as e:
            logging.getLogger("atrofiq").warning("DB nifti update failed for %s: %s", folder, e)

    return resp


@app.get("/nifti-url")
def presign_nifti_url_query(
    folder: str,
    expires_seconds: int = 3600,
    object: Optional[str] = None,
    pattern: Optional[str] = None,
):
    """Alias for clients calling `/nifti-url?folder=...` instead of the nested path.

    This delegates to the folder-based endpoint for consistent behavior.
    """
    return presign_nifti_url(folder=folder, expires_seconds=expires_seconds, object=object, pattern=pattern)

@app.post("/open-visualizer")
def open_visualizer():
    subprocess.Popen(['python3', 'visualizer.py'])
    return 'Visualizer launched', 200

@app.post("/convert-2d-3d/{folder}")
def trigger_2d3d_conversion(folder: str):
    """
    Trigger 2D-3D conversion for DICOM slices in a folder.
    Collects all DICOM slices, converts to NIfTI, then processes through pipeline.
    """
    logger.info(f"2D-3D conversion triggered for folder: {folder}")
    
    try:
        from .tasks.mri_processing_v2 import call_2d3d_conversion
        import tempfile
        import os
        import shutil
        
        client = minio_client()
        
        # Collect ALL DICOM files from the folder
        objects = client.list_objects(MINIO_BUCKET, prefix=folder, recursive=True)
        dicom_files = []
        
        for obj in objects:
            obj_lower = obj.object_name.lower()
            if obj_lower.endswith(('.dcm', '.dicom')):
                dicom_files.append(obj.object_name)
        
        if not dicom_files:
            error_msg = f"No DICOM files found in folder: {folder}"
            logger.error(error_msg)
            raise HTTPException(status_code=404, detail=error_msg)
        
        logger.info(f"Found {len(dicom_files)} DICOM slices in {folder}")
        
        # Create temporary directory for DICOM slices
        temp_dir = tempfile.mkdtemp(prefix="dicom_slices_")
        
        try:
            # Download all DICOM slices to temp directory
            logger.info(f"Downloading {len(dicom_files)} DICOM slices to {temp_dir}")
            for dicom_file in dicom_files:
                filename = os.path.basename(dicom_file)
                local_path = os.path.join(temp_dir, filename)
                client.fget_object(MINIO_BUCKET, dicom_file, local_path)
                logger.info(f"Downloaded: {filename}")
            
            # Call 2D-3D conversion with the directory containing all slices
            logger.info(f"Starting 2D-3D conversion for {len(dicom_files)} slices...")
            result = call_2d3d_conversion(temp_dir, folder_name=folder)
            
            if not result.get('success'):
                error_msg = result.get('error', 'Unknown error during 2D-3D conversion')
                logger.error(f"2D-3D conversion failed: {error_msg}")
                raise HTTPException(status_code=500, detail=error_msg)
            
            logger.info(f"2D-3D conversion successful: {result.get('output_nifti_path', 'N/A')}")
            
            # Upload generated NIfTI back to MinIO
            output_nifti_path = result.get('output_nifti_path')
            if output_nifti_path and os.path.exists(output_nifti_path):
                nifti_key = f"{folder}/converted_3d.nii.gz"
                logger.info(f"Uploading generated NIfTI to MinIO: {nifti_key}")
                client.fput_object(MINIO_BUCKET, nifti_key, output_nifti_path)
                result['minio_path'] = nifti_key
            
            # Update metadata
            meta_text = get_text_object(client, MINIO_BUCKET, f"{folder}/_meta.json")
            meta = {}
            if meta_text:
                try:
                    meta = json.loads(meta_text)
                except Exception:
                    pass
            
            meta.update({
                "volume_2d3d": result,
                "last_updated": now_iso(),
                "has_2d3d_conversion": True,
                "nifti_file": nifti_key if 'minio_path' in result else None,
                "num_input_slices": len(dicom_files)
            })
            
            put_text_object(client, MINIO_BUCKET, f"{folder}/_meta.json", json.dumps(meta))
            
            # Trigger normative modeling and brain age prediction on the generated NIfTI
            if result.get('minio_path'):
                logger.info("Triggering analysis pipeline for generated NIfTI...")
                try:
                    from .tasks.mri_processing_v2 import process_nifti_with_mrbrain_final
                    task = process_nifti_with_mrbrain_final.delay(folder, result['minio_path'])
                    result['analysis_task_id'] = task.id
                    logger.info(f"Started analysis task: {task.id}")
                except Exception as e:
                    logger.error(f"Failed to start analysis task: {e}")
            
            return result
            
        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temp directory: {temp_dir}")
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"2D-3D conversion failed for {folder}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"2D-3D conversion error: {str(e)}")

@app.get("/volumes/slice/{slice_id}")
def get_volume_slice(slice_id: int):
    """Serve volume slice images (mock endpoint for development)."""
    if not IMAGE_PROCESSING_AVAILABLE:
        raise HTTPException(status_code=501, detail="Image processing not available - PIL/numpy not installed")
    
    from fastapi.responses import Response
    
    # Create a mock brain slice image
    size = 256
    image_data = np.zeros((size, size), dtype=np.uint8)
    
    # Create a simple brain-like pattern
    center = size // 2
    for y in range(size):
        for x in range(size):
            dx = x - center
            dy = y - center
            r = np.sqrt(dx*dx + dy*dy)
            
            # Create brain-like structure with some variation per slice
            if r < size * 0.4:
                noise = np.random.random() * 0.3
                slice_variation = np.sin(slice_id / 30.0) * 0.2
                intensity = max(0, min(255, 180 + slice_variation * 75 + noise * 50))
                image_data[y, x] = intensity
    
    # Convert to PIL Image
    img = Image.fromarray(image_data, mode='L')
    
    # Save to bytes
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    return Response(content=img_buffer.getvalue(), media_type="image/png")

@app.get("/verify_storage")
def verify_storage():
    """Verify connection to MinIO and report status without raising errors.

    Returns a JSON payload like:
    {
        "ok": true/false,
        "endpoint": "host:port",
        "secure": false,
        "bucket": "atrofiq",
        "can_connect": true/false,
        "bucket_exists": true/false,
        "errors": { "connect": "...", "bucket": "..." }
    }
    """
    info = {
        "endpoint": MINIO_ENDPOINT,
        "secure": MINIO_SECURE,
        "bucket": MINIO_BUCKET,
        "can_connect": False,
        "bucket_exists": False,
        "errors": {},
    }

    try:
        client = minio_client()
    except Exception as e:
        info["errors"]["client"] = str(e)
        info["ok"] = False
        return info

    # Check general connectivity/credentials by listing buckets
    try:
        client.list_buckets()
        info["can_connect"] = True
    except Exception as e:
        info["errors"]["connect"] = str(e)

    # Check if target bucket exists
    try:
        info["bucket_exists"] = bool(client.bucket_exists(MINIO_BUCKET))
    except Exception as e:
        info["errors"]["bucket"] = str(e)

    info["ok"] = bool(info["can_connect"])  # connectivity is the primary signal
    return info


if __name__ == "__main__":
    import uvicorn

    # When running as a script (e.g., `python app/main.py`), `__package__` is None
    # and importing "app.main" will fail because the project root isn't on sys.path.
    # Use a module path that matches the execution context so reload works.
    target = "app.main:app" if __package__ else "main:app"
    uvicorn.run(target, host="0.0.0.0", port=7000, reload=True)
