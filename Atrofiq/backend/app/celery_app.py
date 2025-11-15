import os
from celery import Celery
from celery.signals import worker_process_init
from contextlib import contextmanager

# Load environment variables
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', REDIS_URL)
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', REDIS_URL)

# Create Celery instance
celery_app = Celery(
    'atrofiq',
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=[
        'app.tasks.mri_processing',  # Our main processing tasks
    ]
)

# Celery configuration
celery_app.conf.update(
    # Task routing
    task_routes={
        'app.tasks.mri_processing.*': {'queue': 'mri_processing'},
    },
    
    # Task serialization
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Task execution settings
    task_always_eager=False,  # Set to True for testing without worker
    task_eager_propagates=True,
    
    # Result settings
    result_expires=3600 * 24,  # 24 hours
    result_persistent=True,
    
    # Worker settings
    worker_prefetch_multiplier=1,  # Process one task at a time per worker
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    
    # Task time limits (in seconds)
    task_soft_time_limit=1800,  # 30 minutes soft limit
    task_time_limit=2400,       # 40 minutes hard limit
    
    # Windows compatibility
    worker_pool='solo',  # Use solo pool for Windows
    
    # Beat schedule (if using celery beat for periodic tasks)
    beat_schedule={},
)

# Optional: Database connection per worker process
@worker_process_init.connect
def init_worker(**kwargs):
    """Initialize worker process with database connections."""
    print("Initializing Celery worker process...")
    # Any worker-specific initialization can go here

@contextmanager
def get_celery_db_session():
    """Get database session for Celery tasks."""
    try:
        from .db import SessionLocal
        session = SessionLocal()
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()