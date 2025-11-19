# AtrofIQ Storage Architecture Guide

## 📊 Overview
Your AtrofIQ system uses three different storage systems, each optimized for specific purposes:

---

## 1. 📦 PostgreSQL Database (Port 5432)

### Purpose
Stores **structured metadata** and **application state**

### Tables

#### `studies` Table
| Field | Type | Purpose |
|-------|------|---------|
| `id` | Integer | Primary key |
| `folder` | String | Study folder name (unique) |
| `age` | String | Patient age |
| `gender` | String | Patient gender |
| `status` | String | Current status (Available, Processing, Completed, Failed) |
| `uploaded_by` | String | Username who uploaded |
| `processing_by` | String | Username processing the study |
| `nifti_object` | Text | Path to NIfTI file in MinIO |
| `object_keys` | JSON | List of all uploaded files |
| `current_task_id` | String | Active Celery task ID |
| `normative_results` | JSON | Normative modeling results |
| `brainage_results` | JSON | Brain age prediction results |
| `created_at` | DateTime | Upload timestamp |
| `last_updated` | DateTime | Last modification timestamp |

**Current Data (3 Studies):**
```
✓ study-20251115-080045 → Status: Completed, Age: 33, Gender: Male
✓ study-20251115-081835 → Status: Completed, Age: 33, Gender: Male
✓ study-20251115-110909 → Status: Failed, Age: 34, Gender: Male
```

#### `processing_tasks` Table
| Field | Type | Purpose |
|-------|------|---------|
| `id` | Integer | Primary key |
| `task_id` | String | Unique Celery task ID |
| `task_name` | String | Task type (e.g., mri_inference) |
| `study_id` | Integer | FK to studies table |
| `status` | String | PENDING, STARTED, SUCCESS, FAILURE, RETRY |
| `progress` | Integer | Progress percentage (0-100) |
| `input_params` | JSON | Task parameters |
| `result` | JSON | Task result data |
| `error_info` | Text | Error message if failed |
| `started_at` | DateTime | When task started |
| `completed_at` | DateTime | When task completed |
| `created_at` | DateTime | Task creation time |



### How to Query


## 2. ⚡ Redis (Port 6379)

### Purpose
**Message broker** and **task queue** for Celery workers

### What's Stored
- **Celery task queue**: Pending tasks waiting to be processed
- **Task status**: Current execution status
- **Worker heartbeats**: Worker alive signals
- **Result backend**: Temporary task results (TTL-based)

### Current Keys (3 Keys)
```
1. _kombu.binding.celery
   ├─ Type: Set
   └─ Purpose: Celery message routing configuration

2. _kombu.binding.celery.pidbox
   ├─ Type: Set
   └─ Purpose: Worker process tracking

3. _kombu.binding.celeryev
   ├─ Type: Set
   ├─ Members: 2 worker events
   └─ Purpose: Worker event broadcasting
```

### Important Notes
- ⚠️ **No task queue items** - All tasks have been processed
- ⚠️ **No task results** - Results are stored in PostgreSQL for persistence
- ✓ **Worker is healthy** - Event bindings are active
- **TTL**: Keys have no expiration (permanent configuration)



## 3. 📁 MinIO Object Storage (Port 9000 / Console: 9001)

### Purpose
Stores **large binary files** (NIfTI images) and **analysis results**

### Bucket Structure
```
Bucket: brain-mri-data
├── study-20251115-080045/
│   ├── MRB_0135.nii.gz        (NIfTI scan file)
│   └── analysis_results/
├── study-20251115-081835/
│   ├── scan.nii.gz
│   └── analysis_results/
└── study-20251115-110909/
    ├── MRB_0135.nii.gz
    └── (no results - processing failed)
```

### Current Storage
```
✓ Bucket Status: EXISTS (auto-created if missing)
✓ Total Objects: 0
⚠️ Storage Empty: NIfTI files not yet uploaded via API
```

### File Organization Pattern
```
{study_folder}/
├── {scan_name}.nii or {scan_name}.nii.gz  ← Primary NIfTI file
├── _meta.json                             ← Metadata file
└── analysis_results/
    ├── normative_modeling_results.json
    └── brainage_prediction_results.json
```



### MinIO Console
- **URL**: http://localhost:9001
- **Username**: minioadmin
- **Password**: minioadmin
- Browse files visually through web interface

---

## 📊 Data Flow Architecture

```
User Upload (Frontend)
    ↓
FastAPI /upload endpoint
    ↓
PostgreSQL: Create Study record
    ↓
MinIO: Store NIfTI file
    ↓
Celery Task Created
    ↓
Redis: Queue task → mri_processing queue
    ↓
Celery Worker: Process task
    ├── Call MRBrain /normative endpoint
    ├── Call MRBrain /brain-age endpoint
    └── Store results
        ↓
PostgreSQL: Update Study.normative_results
PostgreSQL: Update Study.brainage_results
PostgreSQL: Create ProcessingTask record
    ↓
Frontend: Poll /task-status/{task_id}
    ├── Redis: Check current status
    └── PostgreSQL: Fetch full results
        ↓
Display Results to User
```

---

## 🔄 Typical Workflow Data Storage

### 1️⃣ **Upload Phase**
- **PostgreSQL**: Store `Study(folder=study-xxx, age=33, gender=Male, status=Available)`
- **MinIO**: Store `study-xxx/scan.nii.gz`
- **Redis**: (Nothing yet)

### 2️⃣ **Processing Phase**
- **PostgreSQL**: Update `Study(status=Processing, current_task_id=xxx-xxx)`
- **PostgreSQL**: Create `ProcessingTask(task_id=xxx-xxx, status=STARTED, progress=0)`
- **Redis**: Store task status (temporary)

### 3️⃣ **Inference Running**
- **PostgreSQL**: Update `ProcessingTask(progress=20, status=STARTED)`
- **PostgreSQL**: Update `ProcessingTask(progress=60, status=STARTED)`
- **Redis**: (Worker heartbeat signals)

### 4️⃣ **Completion Phase**
- **PostgreSQL**: Store `Study(normative_results={...}, brainage_results={...}, status=Completed)`
- **PostgreSQL**: Create `ProcessingTask(status=SUCCESS, progress=100, completed_at=2025-11-15 ...)`
- **Redis**: (Task cleaned up, bindings remain)

---

## 🎯 Quick Reference

| Need | Storage | Command |
|------|---------|---------|
| Check study metadata | PostgreSQL | `SELECT * FROM studies WHERE folder='study-xxx'` |
| Check processing status | PostgreSQL | `SELECT * FROM processing_tasks WHERE task_id='xxx-xxx'` |
| Check stored results | PostgreSQL | `SELECT normative_results FROM studies WHERE id=1` |
| Monitor workers | Redis | `redis-cli KEYS '*'` |
| Browse files | MinIO Console | http://localhost:9001 |
| Download NIfTI file | MinIO API | `client.fget_object(...)` |

---

## 📈 Database Credentials
```
PostgreSQL:
- Host: localhost
- Port: 5432
- Database: brain_mri_db
- User: brainuser
- Password: securepassword123

Redis:
- Host: localhost
- Port: 6379
- Database: 0
- Password: (none)

MinIO:
- Host: localhost
- Port: 9000
- Access Key: minioadmin
- Secret Key: minioadmin
- Bucket: brain-mri-data
```
