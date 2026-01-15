import io
import json
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Optional
import zipfile
import tempfile

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from minio import Minio
from minio.error import S3Error
import logging
import subprocess
from contextlib import contextmanager

# Celery imports
from .celery_app import celery_app
from .tasks.mri_processing import run_mri_inference

# DICOM utilities
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

# Orthanc DICOM Server Configuration
ORTHANC_ENDPOINT = os.getenv("ORTHANC_ENDPOINT")
ORTHANC_USERNAME = os.getenv("ORTHANC_USERNAME", "orthanc")
ORTHANC_PASSWORD = os.getenv("ORTHANC_PASSWORD", "orthanc")
ORTHANC_ENABLED = env_bool("ORTHANC_ENABLED", True)

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


def extract_files_from_zip(zip_content: bytes, original_filename: str) -> List[tuple]:
    """
    Extract files from a zip archive.
    Returns a list of tuples: (filename, file_content)
    """
    extracted_files = []
    try:
        with zipfile.ZipFile(io.BytesIO(zip_content), 'r') as zip_ref:
            for file_info in zip_ref.filelist:
                # Skip directories and hidden files
                if file_info.is_dir() or file_info.filename.startswith('.'):
                    continue
                
                # Extract file content
                with zip_ref.open(file_info.filename) as file:
                    file_content = file.read()
                    # Use just the filename without path
                    clean_filename = os.path.basename(file_info.filename)
                    if clean_filename:  # Skip empty filenames
                        extracted_files.append((clean_filename, file_content))
                        
        logger.info(f"Extracted {len(extracted_files)} files from zip archive: {original_filename}")
        return extracted_files
        
    except zipfile.BadZipFile:
        logger.error(f"Invalid zip file: {original_filename}")
        raise HTTPException(status_code=400, detail=f"Invalid zip file: {original_filename}")
    except Exception as e:
        logger.error(f"Error extracting zip file {original_filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error extracting zip file: {str(e)}")


def is_zip_file(filename: str, content: bytes) -> bool:
    """Check if a file is a zip archive by extension and magic bytes."""
    if not filename:
        return False
    
    # Check by extension
    is_zip_ext = filename.lower().endswith('.zip')
    
    # Check by magic bytes
    is_zip_magic = content.startswith(b'PK\x03\x04') or content.startswith(b'PK\x05\x06') or content.startswith(b'PK\x07\x08')
    
    return is_zip_ext or is_zip_magic


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
    orthanc_client = get_orthanc_client() if ORTHANC_ENABLED else None
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    folder = f"study-{ts}"

    # Ensure bucket exists and is reachable; return clear error if not
    try:
        if not client.bucket_exists(MINIO_BUCKET):
            client.make_bucket(MINIO_BUCKET)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Storage unavailable: cannot access bucket '{MINIO_BUCKET}' at {MINIO_ENDPOINT}: {e}")

    # Check Orthanc availability if enabled
    orthanc_available = False
    if orthanc_client and ORTHANC_ENABLED:
        orthanc_available = orthanc_client.is_available()
        if not orthanc_available:
            logger.warning("Orthanc server is not available, DICOM files will be stored in MinIO only")

    # Process all files (including extraction from zip files)
    all_files_to_process = []
    
    for f in files:
        # Read file data once
        file_content = await f.read()
        
        try:
            # Check if this is a zip file
            if is_zip_file(f.filename, file_content):
                logger.info(f"Processing zip file: {f.filename}")
                # Extract files from zip
                extracted_files = extract_files_from_zip(file_content, f.filename)
                
                for extracted_filename, extracted_content in extracted_files:
                    all_files_to_process.append({
                        'filename': extracted_filename,
                        'content': extracted_content,
                        'original_source': f"zip:{f.filename}",
                        'content_type': 'application/octet-stream'  # Will be determined later based on content
                    })
            else:
                # Regular file
                all_files_to_process.append({
                    'filename': f.filename,
                    'content': file_content,
                    'original_source': 'direct_upload',
                    'content_type': f.content_type or 'application/octet-stream'
                })
        finally:
            await f.close()

    if not all_files_to_process:
        raise HTTPException(status_code=400, detail="No valid files found to process")

    # Now process all files (extracted and direct)
    count = 0
    object_keys = []
    dicom_instances = []
    zip_extractions = []
    
    for file_info in all_files_to_process:
        filename = file_info['filename']
        file_content = file_info['content']
        original_source = file_info['original_source']
        content_type = file_info['content_type']
        
        try:
            # Check if this is a DICOM file
            is_dicom, dicom_info = is_dicom_file(file_content, filename)
            
            if is_dicom and orthanc_available:
                # Upload to Orthanc if it's a DICOM file and Orthanc is available
                metadata = {
                    'study_folder': folder,
                    'uploaded_by': username or 'system',
                    'original_source': original_source
                }
                
                instance_id = orthanc_client.upload_dicom(file_content, metadata)
                if instance_id:
                    dicom_instances.append({
                        'filename': filename,
                        'instance_id': instance_id,
                        'dicom_info': dicom_info,
                        'original_source': original_source
                    })
                    logger.info(f"DICOM file {filename} uploaded to Orthanc with instance ID: {instance_id}")
                else:
                    logger.warning(f"Failed to upload DICOM file {filename} to Orthanc, storing in MinIO")
            
            # Always store in MinIO as well for consistency with existing workflow
            key = f"{folder}/{filename}"
            # Create a new BytesIO object from the content
            file_stream = io.BytesIO(file_content)
            
            # Determine content type for DICOM files
            if is_dicom:
                content_type = "application/dicom"
            
            client.put_object(
                MINIO_BUCKET,
                key,
                data=file_stream,
                length=len(file_content),
                content_type=content_type,
            )
            count += 1
            object_keys.append(key)
            
            # Track zip extractions for metadata
            if original_source.startswith('zip:'):
                zip_source = original_source.replace('zip:', '')
                zip_extractions.append({
                    'zip_file': zip_source,
                    'extracted_file': filename,
                    'is_dicom': is_dicom
                })
            
        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}")
            raise HTTPException(status_code=503, detail=f"Storage unavailable during upload: {e}")

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
        "dicom_instances": dicom_instances,  # Include DICOM information
        "orthanc_available": orthanc_available,
        "total_files": count,
        "dicom_files": len(dicom_instances),
        "zip_extractions": zip_extractions,  # Track which files came from zip archives
        "processing_notes": f"Processed {len(all_files_to_process)} files from {len(files)} uploaded files"
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

    # Auto-start processing if age and gender are provided AND no DICOM files are present
    # DICOM files require different processing pipeline (not NIfTI-based MRI inference)
    task_id = None
    skip_auto_processing = len(dicom_instances) > 0
    
    if age and gender and not skip_auto_processing:
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
                                    "username": username or "system"
                                }
                            )
                            s.add(task_record)
                            s.commit()
            
            logger.info(f"Auto-started processing for {folder} with task {task.id}")
            
        except Exception as e:
            logger.error(f"Failed to auto-start processing for {folder}: {e}")
    elif skip_auto_processing:
        logger.info(f"Skipping auto-processing for {folder} - contains DICOM files which require different processing pipeline")

    return {
        "ok": True, 
        "folder": folder, 
        "files_count": count, 
        "task_id": task_id, 
        "auto_processing_started": task_id is not None,
        "auto_processing_skipped": skip_auto_processing,
        "auto_processing_skip_reason": "DICOM files detected - requires different processing pipeline" if skip_auto_processing else None,
        "dicom_files": len(dicom_instances),
        "dicom_instances": dicom_instances,
        "orthanc_available": orthanc_available,
        "zip_extractions": zip_extractions,
        "total_uploaded_files": len(files),
        "total_processed_files": len(all_files_to_process)
    }


@app.get("/orthanc/status")
def orthanc_status():
    """Get Orthanc server status and configuration."""
    if not ORTHANC_ENABLED:
        return {"enabled": False, "message": "Orthanc integration disabled"}
    
    orthanc_client = get_orthanc_client()
    if not orthanc_client:
        return {"enabled": True, "available": False, "message": "Orthanc not configured"}
    
    available = orthanc_client.is_available()
    return {
        "enabled": True,
        "available": available,
        "endpoint": ORTHANC_ENDPOINT,
        "message": "Orthanc server is reachable" if available else "Orthanc server is not reachable"
    }


@app.get("/orthanc/studies")
def list_orthanc_studies():
    """List all studies stored in Orthanc."""
    if not ORTHANC_ENABLED:
        raise HTTPException(status_code=404, detail="Orthanc integration disabled")
    
    orthanc_client = get_orthanc_client()
    if not orthanc_client or not orthanc_client.is_available():
        raise HTTPException(status_code=503, detail="Orthanc server not available")
    
    try:
        response = orthanc_client.session.get(f"{orthanc_client.base_url}/studies")
        if response.status_code == 200:
            studies = response.json()
            study_details = []
            
            for study_id in studies:
                study_response = orthanc_client.session.get(f"{orthanc_client.base_url}/studies/{study_id}")
                if study_response.status_code == 200:
                    study_info = study_response.json()
                    study_details.append({
                        'id': study_id,
                        'patient_id': study_info.get('PatientMainDicomTags', {}).get('PatientID'),
                        'patient_name': study_info.get('PatientMainDicomTags', {}).get('PatientName'),
                        'study_date': study_info.get('MainDicomTags', {}).get('StudyDate'),
                        'study_description': study_info.get('MainDicomTags', {}).get('StudyDescription'),
                        'study_instance_uid': study_info.get('MainDicomTags', {}).get('StudyInstanceUID'),
                        'series_count': len(study_info.get('Series', [])),
                        'instances_count': sum(len(series.get('Instances', [])) for series in study_info.get('Series', []))
                    })
            
            return {"studies": study_details}
        else:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch studies from Orthanc")
            
    except Exception as e:
        logger.error(f"Error fetching studies from Orthanc: {e}")
        raise HTTPException(status_code=503, detail=f"Error communicating with Orthanc: {str(e)}")


@app.get("/orthanc/studies/{study_id}")
def get_orthanc_study_details(study_id: str):
    """Get detailed information about a specific DICOM study including series and instances."""
    if not ORTHANC_ENABLED:
        raise HTTPException(status_code=404, detail="Orthanc integration disabled")
    
    orthanc_client = get_orthanc_client()
    if not orthanc_client or not orthanc_client.is_available():
        raise HTTPException(status_code=503, detail="Orthanc server not available")
    
    try:
        study_response = orthanc_client.session.get(f"{orthanc_client.base_url}/studies/{study_id}")
        if study_response.status_code == 200:
            study_info = study_response.json()
            
            # Get detailed series information
            series_details = []
            for series_id in study_info.get('Series', []):
                series_response = orthanc_client.session.get(f"{orthanc_client.base_url}/series/{series_id}")
                if series_response.status_code == 200:
                    series_info = series_response.json()
                    series_details.append({
                        'id': series_id,
                        'series_description': series_info.get('MainDicomTags', {}).get('SeriesDescription'),
                        'modality': series_info.get('MainDicomTags', {}).get('Modality'),
                        'series_number': series_info.get('MainDicomTags', {}).get('SeriesNumber'),
                        'instances': series_info.get('Instances', []),
                        'instances_count': len(series_info.get('Instances', []))
                    })
            
            return {
                'id': study_id,
                'patient_id': study_info.get('PatientMainDicomTags', {}).get('PatientID'),
                'patient_name': study_info.get('PatientMainDicomTags', {}).get('PatientName'),
                'study_date': study_info.get('MainDicomTags', {}).get('StudyDate'),
                'study_time': study_info.get('MainDicomTags', {}).get('StudyTime'),
                'study_description': study_info.get('MainDicomTags', {}).get('StudyDescription'),
                'study_instance_uid': study_info.get('MainDicomTags', {}).get('StudyInstanceUID'),
                'accession_number': study_info.get('MainDicomTags', {}).get('AccessionNumber'),
                'series': series_details,
                'series_count': len(series_details),
                'instances_count': sum(len(series.get('Instances', [])) for series in series_details)
            }
        elif study_response.status_code == 404:
            raise HTTPException(status_code=404, detail="Study not found")
        else:
            raise HTTPException(status_code=study_response.status_code, detail="Failed to fetch study from Orthanc")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching study {study_id} from Orthanc: {e}")
        raise HTTPException(status_code=503, detail=f"Error communicating with Orthanc: {str(e)}")


@app.get("/orthanc/instances/{instance_id}/file")
def download_dicom_instance(instance_id: str):
    """Download a DICOM instance file."""
    if not ORTHANC_ENABLED:
        raise HTTPException(status_code=404, detail="Orthanc integration disabled")
    
    orthanc_client = get_orthanc_client()
    if not orthanc_client or not orthanc_client.is_available():
        raise HTTPException(status_code=503, detail="Orthanc server not available")
    
    try:
        # Get instance info first to get filename
        instance_response = orthanc_client.session.get(f"{orthanc_client.base_url}/instances/{instance_id}")
        if instance_response.status_code != 200:
            raise HTTPException(status_code=404, detail="Instance not found")
            
        instance_info = instance_response.json()
        sop_instance_uid = instance_info.get('MainDicomTags', {}).get('SOPInstanceUID', instance_id)
        
        # Download the DICOM file
        file_response = orthanc_client.session.get(f"{orthanc_client.base_url}/instances/{instance_id}/file")
        if file_response.status_code == 200:
            from fastapi.responses import Response
            return Response(
                content=file_response.content,
                media_type="application/dicom",
                headers={
                    "Content-Disposition": f"attachment; filename={sop_instance_uid}.dcm"
                }
            )
        elif file_response.status_code == 404:
            raise HTTPException(status_code=404, detail="Instance file not found")
        else:
            raise HTTPException(status_code=file_response.status_code, detail="Failed to download instance from Orthanc")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading instance {instance_id} from Orthanc: {e}")
        raise HTTPException(status_code=503, detail=f"Error communicating with Orthanc: {str(e)}")


@app.get("/orthanc/instances/{instance_id}/preview")
def get_dicom_instance_preview(instance_id: str):
    """Get a preview image (PNG) of a DICOM instance."""
    if not ORTHANC_ENABLED:
        raise HTTPException(status_code=404, detail="Orthanc integration disabled")
    
    orthanc_client = get_orthanc_client()
    if not orthanc_client or not orthanc_client.is_available():
        raise HTTPException(status_code=503, detail="Orthanc server not available")
    
    try:
        # Try to get the preview image
        preview_response = orthanc_client.session.get(f"{orthanc_client.base_url}/instances/{instance_id}/preview")
        if preview_response.status_code == 200:
            from fastapi.responses import Response
            return Response(
                content=preview_response.content,
                media_type="image/png"
            )
        elif preview_response.status_code == 404:
            raise HTTPException(status_code=404, detail="Instance or preview not found")
        else:
            raise HTTPException(status_code=preview_response.status_code, detail="Failed to get preview from Orthanc")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting preview for instance {instance_id} from Orthanc: {e}")
        raise HTTPException(status_code=503, detail=f"Error communicating with Orthanc: {str(e)}")


@app.get("/folders/{folder}/dicom-studies")
def get_folder_dicom_studies(folder: str):
    """Get DICOM studies associated with a specific folder."""
    client = minio_client()
    
    # Get folder metadata
    meta_text = get_text_object(client, MINIO_BUCKET, f"{folder}/_meta.json")
    if not meta_text:
        raise HTTPException(status_code=404, detail="Folder not found")
    
    try:
        meta = json.loads(meta_text)
        dicom_instances = meta.get('dicom_instances', [])
        
        if not dicom_instances:
            return {"folder": folder, "dicom_instances": [], "message": "No DICOM files in this folder"}
        
        # If Orthanc is available, fetch current information
        if ORTHANC_ENABLED:
            orthanc_client = get_orthanc_client()
            if orthanc_client and orthanc_client.is_available():
                # Update DICOM instance information with current Orthanc data
                for instance in dicom_instances:
                    instance_id = instance.get('instance_id')
                    if instance_id:
                        try:
                            instance_response = orthanc_client.session.get(f"{orthanc_client.base_url}/instances/{instance_id}")
                            if instance_response.status_code == 200:
                                instance_info = instance_response.json()
                                instance['current_status'] = 'available'
                                instance['tags'] = instance_info.get('MainDicomTags', {})
                            else:
                                instance['current_status'] = 'not_found'
                        except Exception:
                            instance['current_status'] = 'error'
                    else:
                        instance['current_status'] = 'no_instance_id'
        
        return {
            "folder": folder,
            "dicom_instances": dicom_instances,
            "orthanc_available": ORTHANC_ENABLED and orthanc_client and orthanc_client.is_available() if ORTHANC_ENABLED else False
        }
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid folder metadata")
    except Exception as e:
        logger.error(f"Error getting DICOM studies for folder {folder}: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving DICOM studies: {str(e)}")


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
