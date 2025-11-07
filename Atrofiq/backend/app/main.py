import io
import json
import os
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

# Local DB utilities
try:
    from . import db as dbmod  # type: ignore
except Exception:
    dbmod = None

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

    # Save files under a common prefix (folder/filename)
    count = 0
    object_keys = []
    for f in files:
        # Read stream and upload using multipart (unknown length)
        try:
            key = f"{folder}/{f.filename}"
            client.put_object(
                MINIO_BUCKET,
                key,
                data=f.file,
                length=-1,
                part_size=10 * 1024 * 1024,  # 10MB parts
                content_type=f.content_type or "application/octet-stream",
            )
            count += 1
            object_keys.append(key)
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
    except Exception as e:
        logging.getLogger("atrofiq").warning("DB insert failed for folder %s: %s", folder, e)

    return {"ok": True, "folder": folder, "files_count": count}


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


# Basic health check
@app.get("/health")
def health():
    return {"status": "ok"}


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
