"""
Main MRI Processing API - Entry Point
This file imports and mounts routes from the inference files
All routes are defined in their respective inference files
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directory paths
BASE_DIR = Path(__file__).parent

# Create main FastAPI app
app = FastAPI(
    title="MRBrain Processing API",
    description="Unified API for MRI Brain Analysis with routes from inference modules",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Function to safely import and include routes from common inference
def include_brainage_routes():
    """Import and include BrainAge routes from common inference"""
    try:
        # Import the FastAPI app from common_inference.py
        from common_inference import brainage_app
        logger.info("Successfully imported BrainAge routes from common inference")
        
        # Include all routes from brainage_app
        for route in brainage_app.routes:
            app.router.routes.append(route)
        
        return True
    except Exception as e:
        logger.error(f"Failed to import BrainAge routes from common inference: {e}")
        return False

def include_normative_routes():
    """Import and include Normative routes from common inference"""
    try:
        # Import the FastAPI app from common_inference.py
        from common_inference import normative_app
        logger.info("Successfully imported Normative routes from common inference")
        
        # Include all routes from normative_app
        for route in normative_app.routes:
            app.router.routes.append(route)
            
        return True
    except Exception as e:
        logger.error(f"Failed to import Normative routes from common inference: {e}")
        return False

# Include routes from inference files
brainage_loaded = include_brainage_routes()
normative_loaded = include_normative_routes()

# Add fallback routes if modules failed to load
if not brainage_loaded:
    from fastapi import File, UploadFile, Form, HTTPException
    from pydantic import BaseModel
    from typing import Dict
    import tempfile
    import shutil
    import uuid
    
    class BrainAgeResponse(BaseModel):
        job_id: str
        participant_id: str
        status: str
        chronological_age: float
        predicted_brain_age: float
        brain_age_gap: float
        processing_time_seconds: float
        volumetric_features: Dict
        metadata: Dict
    
    @app.post("/brain-age", response_model=BrainAgeResponse)
    async def brain_age_fallback(
        nifti_file: UploadFile = File(..., description="NIfTI MRI file (.nii or .nii.gz)"),
        age: float = Form(..., description="Chronological age in years"),
        gender: str = Form(..., description="Gender (M/F)")
    ):
        """Fallback Brain Age Prediction Route"""
        import random
        from datetime import datetime
        
        start_time = datetime.now()
        job_id = str(uuid.uuid4())
        participant_id = Path(nifti_file.filename).stem.replace('.nii', '')
        
        # Simulate processing
        predicted_age = age + random.uniform(-5, 8)
        brain_age_gap = predicted_age - age
        
        # Dummy volumes
        volumes = {
            'total_brain': 95000000.0,
            'csf': 50000000.0,
            'gray_matter': 22000000.0,
            'white_matter': 23000000.0
        }
        
        metadata = {
            'participant_id': participant_id,
            'age': age,
            'sex': gender.upper(),
            'timestamp': datetime.now().isoformat()
        }
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return BrainAgeResponse(
            job_id=job_id,
            participant_id=participant_id,
            status="fallback_success",
            chronological_age=age,
            predicted_brain_age=round(predicted_age, 1),
            brain_age_gap=round(brain_age_gap, 1),
            processing_time_seconds=processing_time,
            volumetric_features=volumes,
            metadata=metadata
        )
    
    logger.info("Added fallback /brain-age route")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    available_routes = []
    
    if brainage_loaded:
        available_routes.append("/brain-age")
    if normative_loaded:
        available_routes.append("/normative")
    
    return {
        "message": "MRBrain Processing API",
        "description": "Unified API with routes from inference modules",
        "available_routes": available_routes,
        "brainage_loaded": brainage_loaded,
        "normative_loaded": normative_loaded,
        "version": "1.0.0"
    }

@app.get("/status")
async def status():
    """Status endpoint"""
    return {
        "status": "healthy",
        "brainage_module": "loaded" if brainage_loaded else "failed",
        "normative_module": "loaded" if normative_loaded else "failed",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    logger.info("Starting MRBrain Processing API...")
    logger.info(f"BrainAge module: {'loaded' if brainage_loaded else 'failed'}")
    logger.info(f"Normative module: {'loaded' if normative_loaded else 'failed'}")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)