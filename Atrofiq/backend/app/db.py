import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Integer,
    String,
    Text,
    create_engine,
    func,
    ForeignKey,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v is not None else default


# Database configuration (defaults to the provided Docker run command)
DB_HOST = _env("DB_HOST", "127.0.0.1")
DB_PORT = _env("DB_PORT", "5432")
DB_NAME = _env("DB_NAME", "brain_mri_db")
DB_USER = _env("DB_USER", "brainuser")
DB_PASSWORD = _env("DB_PASSWORD", "securepassword123")

DATABASE_URL = _env(
    "DATABASE_URL",
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
)


engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

Base = declarative_base()


class Study(Base):
    __tablename__ = "studies"

    id = Column(Integer, primary_key=True, index=True)
    folder = Column(String(255), unique=True, nullable=False, index=True)

    # Meta-tags
    age = Column(String(32), nullable=True)
    gender = Column(String(16), nullable=True)
    uploaded_by = Column(String(128), nullable=True)

    status = Column(String(64), nullable=False, default="Available")
    processing_by = Column(String(128), nullable=True)
    completed_by = Column(String(128), nullable=True)

    # MinIO
    bucket = Column(String(255), nullable=True)
    nifti_object = Column(Text, nullable=True)  # primary nifti object key if selected
    object_keys = Column(JSON, nullable=True)   # list of uploaded object keys (folder/filename)

    # Task tracking
    current_task_id = Column(String(255), nullable=True, index=True)  # Current Celery task ID
    
    # Analysis results
    normative_results = Column(JSON, nullable=True)  # Store normative modeling results
    brainage_results = Column(JSON, nullable=True)   # Store brain age prediction results

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_updated = Column(DateTime(timezone=True), onupdate=func.now(), default=func.now(), nullable=False)
    
    # Relationship to tasks
    tasks = relationship("ProcessingTask", back_populates="study", cascade="all, delete-orphan")


class ProcessingTask(Base):
    __tablename__ = "processing_tasks"
    
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String(255), unique=True, nullable=False, index=True)  # Celery task ID
    task_name = Column(String(128), nullable=False)  # e.g., "mri_inference", "normative_modeling"
    
    # Foreign key to study
    study_id = Column(Integer, ForeignKey("studies.id"), nullable=False, index=True)
    
    # Task metadata
    status = Column(String(32), nullable=False, default="PENDING")  # PENDING, STARTED, SUCCESS, FAILURE, RETRY
    progress = Column(Integer, default=0)  # Progress percentage (0-100)
    
    # Task parameters and results
    input_params = Column(JSON, nullable=True)  # Input parameters for the task
    result = Column(JSON, nullable=True)        # Task result data
    error_info = Column(Text, nullable=True)    # Error information if failed
    
    # Timing
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationship back to study
    study = relationship("Study", back_populates="tasks")


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def to_worklist_dict(s: Study) -> Dict[str, Any]:
    return {
        "name": s.folder,
        "status": s.status,
        "processing_by": s.processing_by,
        "completed_by": s.completed_by,
        "last_updated": (s.last_updated or s.created_at).isoformat() if (s.last_updated or s.created_at) else None,
        "age": s.age,
        "gender": s.gender,
        "uploaded_by": s.uploaded_by,
        "bucket": s.bucket,
        "nifti_object": s.nifti_object,
        "object_keys": s.object_keys or [],
        "current_task_id": s.current_task_id,
        "normative_results": s.normative_results,
        "brainage_results": s.brainage_results,
        "task_count": len(s.tasks) if s.tasks else 0,
    }

