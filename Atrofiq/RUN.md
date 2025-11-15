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
docker stop atrofiq_postgres atrofiq_redis atrofiq_minio atrofiq_keycloak
docker rm atrofiq_postgres atrofiq_redis atrofiq_minio atrofiq_keycloak
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
```

## Port Usage
- 3000: React Frontend
- 5432: PostgreSQL Database
- 5555: Flower (Optional - Testing Only)
- 6379: Redis
- 7000: AtrofIQ Backend API
- 8000: MRBrain Inference API
- 8080: Keycloak
- 9000: MinIO API
- 9001: MinIO Console