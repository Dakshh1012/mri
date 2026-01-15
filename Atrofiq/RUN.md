# AtrofIQ - Quick Run Guide

## Prerequisites
- Docker and Docker Compose
- Python 3.9+
- Node.js 18+
- Git

## Quick Start (Recommended)

### All Services with Docker Compose
```bash
docker-compose up --build -d
```

Access URLs:
- Frontend: http://localhost:3000
- Backend API: http://localhost:7000
- MRBrain API: http://localhost:8000
- MinIO Console: http://localhost:9001
- Keycloak: http://localhost:8080
- Orthanc DICOM Server: http://localhost:8042

## Manual Setup

### 1. Start Infrastructure Services

#### PostgreSQL Database
```bash
docker run -d --name atrofiq_postgres \
  -e POSTGRES_DB=brain_mri_db \
  -e POSTGRES_USER=brainuser \
  -e POSTGRES_PASSWORD=securepassword123 \
  -p 5432:5432 \
  postgres:15
```

#### Redis (Message Broker)
```bash
docker run -d --name atrofiq_redis \
  -p 6379:6379 \
  redis:7-alpine
```

#### MinIO (Object Storage)
```bash
docker run -d --name atrofiq_minio \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  -p 9000:9000 \
  -p 9001:9001 \
  minio/minio server /data --console-address ":9001"
```

#### Orthanc DICOM Server
```bash
docker run -d --name atrofiq_orthanc \
  -e ORTHANC_JSON_FILE=/etc/orthanc/orthanc.json \
  -p 8042:8042 \
  -p 4242:4242 \
  -v orthanc_data:/var/lib/orthanc/db \
  -v $(pwd)/orthanc-config:/etc/orthanc \
  orthancteam/orthanc:latest
```

#### Keycloak (Authentication)
```bash
docker run -d --name atrofiq_keycloak \
  -e KEYCLOAK_ADMIN=admin \
  -e KEYCLOAK_ADMIN_PASSWORD=admin123 \
  -p 8080:8080 \
  quay.io/keycloak/keycloak:23.0 start-dev
```

### 2. Start Backend Services

#### AtrofIQ Backend API
```bash
cd backend
pip install -r requirements.txt
export REDIS_URL=redis://localhost:6379/0
export DB_HOST=localhost
export DB_NAME=brain_mri_db
export DB_USER=brainuser
export DB_PASSWORD=securepassword123
export ORTHANC_ENDPOINT=localhost:8042
export ORTHANC_USERNAME=orthanc
export ORTHANC_PASSWORD=orthanc
uvicorn app.main:app --host 0.0.0.0 --port 7000 --reload
```

#### Celery Worker
```bash
cd backend
celery -A app.celery_app worker --loglevel=info
```

#### MRBrain API
```bash
cd MRBrain
pip install -r requirements.txt
uvicorn main_api:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Start Frontend
```bash
cd frontend
npm install
npm start
```

## Windows Commands

### All Services with Docker Compose
```cmd
docker-compose up --build -d
```

### Manual Setup - Windows

#### PostgreSQL Database
```cmd
docker run -d --name atrofiq_postgres -e POSTGRES_DB=brain_mri_db -e POSTGRES_USER=brainuser -e POSTGRES_PASSWORD=securepassword123 -p 5432:5432 postgres:15
```

#### Redis
```cmd
docker run -d --name atrofiq_redis -p 6379:6379 redis:7-alpine
```

#### MinIO
```cmd
docker run -d --name atrofiq_minio -e MINIO_ROOT_USER=minioadmin -e MINIO_ROOT_PASSWORD=minioadmin -p 9000:9000 -p 9001:9001 minio/minio server /data --console-address ":9001"
```

#### Orthanc DICOM Server
```cmd
docker run -d --name atrofiq_orthanc -e ORTHANC_JSON_FILE=/etc/orthanc/orthanc.json -p 8042:8042 -p 4242:4242 -v orthanc_data:/var/lib/orthanc/db -v %cd%/orthanc-config:/etc/orthanc orthancteam/orthanc:latest
```

#### Keycloak
```cmd
docker run -d --name atrofiq_keycloak -e KEYCLOAK_ADMIN=admin -e KEYCLOAK_ADMIN_PASSWORD=admin123 -p 8080:8080 quay.io/keycloak/keycloak:23.0 start-dev
```

#### Backend API (Windows CMD)
```cmd
cd backend
pip install -r requirements.txt
set REDIS_URL=redis://localhost:6379/0
set DB_HOST=localhost
set DB_NAME=brain_mri_db
set DB_USER=brainuser
set DB_PASSWORD=securepassword123
set ORTHANC_ENDPOINT=localhost:8042
set ORTHANC_USERNAME=orthanc
set ORTHANC_PASSWORD=orthanc
uvicorn app.main:app --host 0.0.0.0 --port 7000 --reload
```

#### Backend API (Windows PowerShell)
```powershell
cd backend
pip install -r requirements.txt
$env:REDIS_URL="redis://localhost:6379/0"
$env:DB_HOST="localhost"
$env:DB_NAME="brain_mri_db"
$env:DB_USER="brainuser"
$env:DB_PASSWORD="securepassword123"
$env:ORTHANC_ENDPOINT="localhost:8042"
$env:ORTHANC_USERNAME="orthanc"
$env:ORTHANC_PASSWORD="orthanc"
uvicorn app.main:app --host 0.0.0.0 --port 7000 --reload
```

#### Celery Worker
```cmd
cd backend
celery -A app.celery_app worker --loglevel=info
```

#### Flower (Optional - Testing Only)
```cmd
REM Only needed for debugging/monitoring Celery tasks
cd backend
celery -A app.celery_app flower --host=0.0.0.0 --port=5555
```

#### MRBrain API
```cmd
cd MRBrain
pip install -r requirements.txt
uvicorn main_api:app --host 0.0.0.0 --port 8000 --reload
```

#### Frontend
```cmd
cd frontend
npm install
npm start
```

## Stopping Services

### Stop Docker Compose
```bash
docker-compose down
```

### Stop Individual Containers
```bash
docker stop atrofiq_postgres atrofiq_redis atrofiq_minio atrofiq_keycloak atrofiq_orthanc
docker rm atrofiq_postgres atrofiq_redis atrofiq_minio atrofiq_keycloak atrofiq_orthanc
```

## Health Check Commands

### Check Container Status
```bash
docker ps
```

### Check Service Health
```bash
curl http://localhost:7000/health
curl http://localhost:8000/status
curl http://localhost:8042/system  # Orthanc system info
```

### View Logs
```bash
docker-compose logs -f [service-name]
```

## Environment Variables

### Required Environment Variables
```bash
REDIS_URL=redis://localhost:6379/0
DB_HOST=localhost
DB_NAME=brain_mri_db
DB_USER=brainuser
DB_PASSWORD=securepassword123
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
ORTHANC_ENDPOINT=localhost:8042
ORTHANC_USERNAME=orthanc
ORTHANC_PASSWORD=orthanc
ORTHANC_ENABLED=true
```

## Port Usage
- 3000: React Frontend
- 4242: Orthanc DICOM Protocol
- 5432: PostgreSQL Database
- 5555: Flower (Optional - Testing Only)
- 6379: Redis
- 7000: AtrofIQ Backend API
- 8000: MRBrain Inference API
- 8042: Orthanc DICOM Server Web UI/REST API
- 8080: Keycloak
- 9000: MinIO API
- 9001: MinIO Console

## DICOM Support

### Orthanc DICOM Server
AtrofIQ now includes integrated DICOM support through Orthanc:

- **Automatic DICOM Detection**: When uploading files, DICOM files are automatically detected and stored in Orthanc
- **Dual Storage**: DICOM files are stored both in Orthanc (for DICOM-specific operations) and MinIO (for existing workflow compatibility)
- **DICOM Web UI**: Access the Orthanc web interface at http://localhost:8042 (username: orthanc, password: orthanc)
- **DICOM Protocol**: Receive DICOM files via DICOM C-STORE protocol on port 4242
- **REST API**: Query and retrieve DICOM data via Orthanc's REST API

### DICOM File Support
- Supports standard DICOM file formats (.dcm, .dicom)
- Detects DICOM files by content (magic bytes) regardless of file extension
- Extracts and stores DICOM metadata (Patient ID, Study Date, Modality, etc.)
- Maintains backward compatibility with existing .nii and .nii.gz workflows

### API Endpoints for DICOM
- `GET /orthanc/status` - Check Orthanc server status
- `GET /orthanc/studies` - List all DICOM studies in Orthanc
- Upload endpoint automatically handles DICOM files when detected

### Configuration
Orthanc integration can be controlled via environment variables:
- `ORTHANC_ENABLED=true/false` - Enable/disable Orthanc integration
- `ORTHANC_ENDPOINT` - Orthanc server endpoint
- `ORTHANC_USERNAME` - Authentication username  
- `ORTHANC_PASSWORD` - Authentication password

Note: If Orthanc is unavailable, DICOM files will still be stored in MinIO to maintain system functionality.