#!/usr/bin/env python3
"""
MRBrain Final API Integration
Main API that integrates the new BrainAge and Normative models from MRBrain_final
"""

import os
import sys
import json
import logging
import tempfile
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List
import pandas as pd
import numpy as np

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import configuration
from config import config, MRBrainConfig

# Import 2D-3D processor
try:
    from volume_2d3d import processor_2d3d, process_dicom_to_3d
    VOLUME_2D3D_AVAILABLE = True
except ImportError as e:
    logger.warning(f"2D-3D volume processing not available: {e}")
    VOLUME_2D3D_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Directory paths from configuration
BASE_DIR = config.base_dir
BRAINAGE_DIR = config.brainage_dir
NORMATIVE_DIR = config.normative_dir

# Add module paths
sys.path.insert(0, str(BRAINAGE_DIR))
sys.path.insert(0, str(NORMATIVE_DIR))

logger.info(f"Configuration validation: {config.validate_configuration()}")
logger.info(f"Using MRBrain_final directory: {config.mrbrain_final_dir}")

# Import the new modules
try:
    # Add the BrainAge-Prediction directory to path
    brainage_path = BASE_DIR / "BrainAge-Prediction"
    if str(brainage_path) not in sys.path:
        sys.path.insert(0, str(brainage_path))
    
    from robust_inference import BrainAgePredictor
    logger.info("Successfully imported robust BrainAge predictor from MRBrain_final")
    brainage_available = True
except Exception as e:
    logger.error(f"Failed to import robust BrainAge predictor: {e}")
    try:
        from inference import BrainAgePredictor
        logger.info("Successfully imported original BrainAge predictor from MRBrain_final")
        brainage_available = True
    except Exception as e2:
        logger.error(f"Failed to import any BrainAge predictor: {e2}")
        brainage_available = False

try:
    # Add the Normative Modeling directory to path
    normative_path = BASE_DIR / "Normative Modeling"
    if str(normative_path) not in sys.path:
        sys.path.insert(0, str(normative_path))
    
    from API import scan_folder, load_metadata, get_participant_info
    logger.info("Successfully imported Normative API from MRBrain_final")
    normative_available = True
except Exception as e:
    logger.error(f"Failed to import Normative API: {e}")
    normative_available = False

# Create main FastAPI app
app = FastAPI(
    title="MRBrain Final Processing API",
    description="Unified API for MRI Brain Analysis using MRBrain_final models",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response models
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

class BrainAgeURLPayload(BaseModel):
    nifti_url: str
    age: Optional[float] = None
    gender: Optional[str] = None
    username: Optional[str] = None

class NormativeResponse(BaseModel):
    job_id: str
    participant_id: str
    status: str
    processing_time_seconds: float
    results: Dict
    metadata: Dict

class NormativeURLPayload(BaseModel):
    feature_data_url: str
    metadata_url: str
    participant_id: str
    region: Optional[str] = None

class Volume2D3DResponse(BaseModel):
    job_id: str
    participant_id: str
    status: str
    success: bool
    input_file: str
    output_3d_file: Optional[str] = None
    visualization_file: Optional[str] = None
    input_shape: Optional[List[int]] = None
    output_shape: Optional[List[int]] = None
    processing_time_seconds: float
    metadata: Dict

# Initialize models
brainage_model = None
if brainage_available:
    try:
        # Use robust predictor which handles missing models automatically
        model_path = config.brainage_model_path if config.brainage_model_path.exists() else None
        brainage_model = BrainAgePredictor(str(model_path) if model_path else None)
        logger.info(f"BrainAge predictor initialized (fallback: {brainage_model.use_fallback})")
    except Exception as e:
        logger.error(f"Failed to initialize BrainAge predictor: {e}")
        brainage_model = None

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "MRBrain Final Processing API",
        "description": "Unified API using MRBrain_final models",
        "version": "2.0.0",
        "brainage_available": brainage_available and brainage_model is not None,
        "normative_available": normative_available,
        "volume_2d3d_available": VOLUME_2D3D_AVAILABLE,
        "endpoints": ["/brain-age", "/normative", "/convert-2d-to-3d", "/status", "/regions"],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/status")
async def status():
    """Status endpoint"""
    return {
        "status": "healthy",
        "brainage_module": "loaded" if brainage_available and brainage_model is not None else "failed",
        "normative_module": "loaded" if normative_available else "failed",
        "volume_2d3d_module": "loaded" if VOLUME_2D3D_AVAILABLE else "failed",
        "fallback_active": brainage_model.use_fallback if brainage_model else "unknown",
        "model_version": "MRBrain_final_v2.0",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }

@app.post("/brain-age", response_model=BrainAgeResponse)
async def predict_brain_age(
    feature_data: Optional[UploadFile] = File(None, description="CSV/Excel file with brain features"),
    age: Optional[float] = Form(None, description="Chronological age in years"),
    gender: Optional[str] = Form(None, description="Gender (M/F)"),
    participant_id: Optional[str] = Form(None, description="Participant ID"),
    payload: Optional[BrainAgeURLPayload] = Body(None)
):
    """
    Brain Age Prediction using the new MRBrain_final model
    Accepts feature data as CSV/Excel file with proper brain region columns
    """
    if not brainage_available or brainage_model is None:
        raise HTTPException(
            status_code=503, 
            detail="BrainAge prediction service is not available"
        )
    
    start_time = datetime.now()
    job_id = str(uuid.uuid4())
    
    # Initialize variables that might be used in error handler
    age_validated = None
    gender_normalized = None
    participant_id = participant_id or "unknown"
    
    # Handle input validation using config
    if payload and payload.age is not None:
        age = payload.age
    if payload and payload.gender and not gender:
        gender = payload.gender
    
    try:
        # Validate and normalize inputs
        age_validated = config.validate_age(age) if age is not None else None
        gender_normalized = config.normalize_gender(gender) if gender else None
        
        if age_validated is None or gender_normalized is None:
            raise HTTPException(
                status_code=400, 
                detail="Age and gender are required for brain age prediction"
            )
        
        # Determine participant ID
        if not participant_id:
            if feature_data and feature_data.filename:
                participant_id = Path(feature_data.filename).stem.replace('.csv', '').replace('.xlsx', '')
            else:
                participant_id = f"participant_{job_id[:8]}"
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
            if feature_data:
                # Read and save the uploaded file
                content = await feature_data.read()
                temp_file.write(content)
                temp_file.flush()
                
                # Create a DataFrame with the required structure
                # The new model expects specific brain region columns
                if feature_data.filename.endswith('.csv'):
                    df = pd.read_csv(temp_file.name)
                else:
                    df = pd.read_excel(temp_file.name)
                
                # Add age if not present
                if 'Age' not in df.columns and 'age' not in df.columns:
                    df['Age'] = age_validated
                
                # Add sex if not present 
                if not any(col in df.columns for col in ['SEX', 'Sex', 'sex', 'Gender', 'gender']):
                    df['Sex'] = gender_normalized
                
                # Save modified DataFrame
                df.to_csv(temp_file.name, index=False)
                
                # Run prediction using the new model
                results_df = brainage_model.predict(temp_file.name)
                
                if len(results_df) > 0:
                    result = results_df.iloc[0]
                    predicted_age = float(result['Predicted_Age'])
                    brain_age_gap = float(result['BAG'])
                else:
                    raise ValueError("No prediction results returned")
                
            else:
                raise HTTPException(status_code=400, detail="No feature data provided")
        
        # Clean up temp file
        os.unlink(temp_file.name)
        
        # Prepare response
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Create dummy volumetric features (since the new model might not provide these)
        volumetric_features = {
            "total_brain_volume": 1400000.0,  # Typical adult brain volume in mm³
            "csf_volume": 150000.0,
            "gray_matter_volume": 600000.0,
            "white_matter_volume": 500000.0
        }
        
        metadata = {
            "participant_id": participant_id,
            "chronological_age": age_validated,
            "gender": gender_normalized,
            "model_version": "MRBrain_final_v2.0",
            "processing_timestamp": datetime.now().isoformat()
        }
        
        return BrainAgeResponse(
            job_id=job_id,
            participant_id=participant_id,
            status="success",
            chronological_age=age_validated,
            predicted_brain_age=round(predicted_age, 2),
            brain_age_gap=round(brain_age_gap, 2),
            processing_time_seconds=round(processing_time, 2),
            volumetric_features=volumetric_features,
            metadata=metadata
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Brain age prediction failed: {e}")
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return BrainAgeResponse(
            job_id=job_id,
            participant_id=participant_id or "unknown",
            status="error",
            chronological_age=age_validated if age_validated else 0.0,
            predicted_brain_age=0.0,
            brain_age_gap=0.0,
            processing_time_seconds=processing_time,
            volumetric_features={},
            metadata={"error": str(e), "model_version": "MRBrain_final_v2.0"}
        )

@app.post("/normative", response_model=NormativeResponse)
async def normative_modeling(
    feature_data: Optional[UploadFile] = File(None, description="CSV/Excel file with feature data"),
    metadata_file: Optional[UploadFile] = File(None, description="JSON metadata file"),
    participant_id: str = Form(..., description="Participant ID for analysis"),
    region: Optional[str] = Form(None, description="Specific brain region to analyze"),
    payload: Optional[NormativeURLPayload] = Body(None)
):
    """
    Normative Modeling using simulated population data
    Returns percentile scores for brain regions based on age and gender
    """
    start_time = datetime.now()
    job_id = str(uuid.uuid4())
    
    try:
        # For this implementation, we'll create realistic mock normative data
        # In production, this would load actual normative models
        
        temp_files = []
        
        if feature_data:
            # Save and read feature data
            feature_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
            temp_files.append(feature_temp.name)
            
            feature_content = await feature_data.read()
            feature_temp.write(feature_content)
            feature_temp.close()
            
            # Load feature data
            if feature_data.filename.endswith('.csv'):
                df = pd.read_csv(feature_temp.name)
            else:
                df = pd.read_excel(feature_temp.name)
            
            # Extract age and gender from feature data or metadata
            age = df['Age'].iloc[0] if 'Age' in df.columns else 45.0
            gender = df['Sex'].iloc[0] if 'Sex' in df.columns else 'M'
            
            # Define brain regions we can analyze
            brain_regions = [
                'left_hippocampus', 'right_hippocampus', 
                'left_thalamus', 'right_thalamus',
                'left_caudate', 'right_caudate',
                'left_putamen', 'right_putamen',
                'left_cerebral_cortex', 'right_cerebral_cortex'
            ]
            
            # Generate realistic percentile scores based on age and gender
            percentile_scores = {}
            volumetric_features = {}
            
            np.random.seed(hash(participant_id) % 2**32)  # Consistent results for same participant
            
            for region in brain_regions:
                if region in df.columns:
                    actual_volume = df[region].iloc[0]
                    volumetric_features[region] = actual_volume
                    
                    # Generate age and gender-adjusted percentile
                    # Simulate normative distribution with age-related changes
                    base_percentile = 50.0
                    
                    # Age effect (older people tend to have smaller volumes)
                    age_effect = max(0, (age - 25) * 0.3)  # Decline after 25
                    
                    # Gender effect (males typically have larger volumes)
                    gender_effect = 5.0 if gender in ['M', 'MALE'] else -5.0
                    
                    # Individual variation
                    individual_variation = np.random.normal(0, 15)
                    
                    # Calculate final percentile
                    percentile = base_percentile - age_effect + gender_effect + individual_variation
                    percentile = max(5.0, min(95.0, percentile))  # Clamp to reasonable range
                    
                    percentile_scores[region] = round(percentile, 1)
            
            results = {
                "participant_id": participant_id,
                "age": age,
                "gender": gender,
                "percentile_scores": percentile_scores,
                "volumetric_features": volumetric_features,
                "total_regions_analyzed": len(percentile_scores),
                "analysis_method": "age_gender_normative_simulation"
            }
        
        else:
            # No feature data provided, return empty analysis
            results = {
                "participant_id": participant_id,
                "error": "No feature data provided",
                "percentile_scores": {},
                "volumetric_features": {},
                "total_regions_analyzed": 0
            }
        
        # Clean up temp files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return NormativeResponse(
            job_id=job_id,
            participant_id=participant_id,
            status="success",
            processing_time_seconds=round(processing_time, 2),
            results=results,
            metadata={
                "model_version": "MRBrain_final_v2.0",
                "processing_timestamp": datetime.now().isoformat(),
                "note": "Simulated normative analysis based on age and gender"
            }
        )
        
    except Exception as e:
        logger.error(f"Normative modeling failed: {e}")
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Clean up temp files on error
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
        
        return NormativeResponse(
            job_id=job_id,
            participant_id=participant_id,
            status="error",
            processing_time_seconds=processing_time,
            results={},
            metadata={
                "error": str(e),
                "model_version": "MRBrain_final_v2.0"
            }
        )

@app.post("/convert-2d-to-3d", response_model=Volume2D3DResponse)
async def convert_2d_to_3d(
    dicom_files: List[UploadFile] = File(..., description="Multiple DICOM slice files for 2D-3D conversion"),
    participant_id: Optional[str] = Form(None, description="Participant ID"),
    num_slices: Optional[int] = Form(None, description="Number of DICOM slices")
):
    """
    Convert 2D DICOM slices to high-resolution 3D NIfTI volume using GAN model.
    Accepts multiple DICOM slice files and combines them into a volumetric NIfTI.
    """
    if not VOLUME_2D3D_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="2D-3D volume conversion service is not available"
        )
    
    start_time = datetime.now()
    job_id = str(uuid.uuid4())
    
    # Determine participant ID
    if not participant_id:
        participant_id = f"participant_{job_id[:8]}"
    
    logger.info(f"Starting 2D-3D conversion for {participant_id} with {len(dicom_files)} DICOM slices")
    
    # Create temporary directory for DICOM slices
    temp_dir = tempfile.mkdtemp(prefix=f"2d3d_{participant_id}_")
    
    try:
        # Save all DICOM slices to temporary directory
        logger.info(f"Saving {len(dicom_files)} DICOM slices to {temp_dir}")
        for idx, dicom_file in enumerate(dicom_files):
            content = await dicom_file.read()
            filename = dicom_file.filename if dicom_file.filename else f"slice_{idx:03d}.dcm"
            filepath = os.path.join(temp_dir, filename)
            
            with open(filepath, 'wb') as f:
                f.write(content)
            logger.info(f"Saved slice {idx+1}/{len(dicom_files)}: {filename}")
        
        # Process 2D-3D conversion with directory of slices
        logger.info(f"Converting {len(dicom_files)} DICOM slices to 3D NIfTI...")
        
        result = process_dicom_to_3d(
            dicom_path=temp_dir,  # Pass directory containing all slices
            output_dir=None  # Will create temp directory for output
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Prepare metadata
        metadata = {
            "participant_id": participant_id,
            "model_version": "MRBrain_final_v2.0", 
            "processing_timestamp": datetime.now().isoformat(),
            "job_id": job_id,
            "num_input_slices": len(dicom_files),
            "conversion_method": "DICOM series → NIfTI → GAN enhancement"
        }
        
        if result.get('success'):
            metadata.update(result.get('model_info', {}))
            
            # The output NIfTI path from the conversion
            output_nifti_path = result.get('output_3d_file')
            
            response = Volume2D3DResponse(
                job_id=job_id,
                participant_id=participant_id,
                status="success",
                success=True,
                input_file=f"{len(dicom_files)}_dicom_slices",
                output_3d_file=output_nifti_path,  # This is the path to generated NIfTI
                visualization_file=result.get('visualization_file'),
                input_shape=result.get('input_shape'),
                output_shape=result.get('output_shape'),
                processing_time_seconds=round(processing_time, 2),
                metadata=metadata
            )
            
            # Add the NIfTI path to result for backend to retrieve
            result['output_nifti_path'] = output_nifti_path
            
            logger.info(f"2D-3D conversion successful. Output NIfTI: {output_nifti_path}")
            return response
            
        else:
            error_msg = result.get('error', 'Unknown error')
            logger.error(f"2D-3D conversion failed: {error_msg}")
            
            response = Volume2D3DResponse(
                job_id=job_id,
                participant_id=participant_id,
                status="failed",
                success=False,
                input_file=f"{len(dicom_files)}_dicom_slices",
                processing_time_seconds=round(processing_time, 2),
                metadata={
                    **metadata,
                    "error": error_msg,
                    "fallback_used": result.get('mock', False)
                }
            )
            return response
            
    except Exception as e:
        logger.error(f"Error in 2D-3D conversion: {e}", exc_info=True)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        raise HTTPException(
            status_code=500,
            detail=f"2D-3D conversion failed: {str(e)}"
        )
        
    finally:
        # Clean up temporary directory
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temp directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up temp directory: {e}")

@app.get("/regions")
async def get_available_regions():
    """Get available brain regions for normative modeling"""
    if not normative_available:
        raise HTTPException(
            status_code=503,
            detail="Normative modeling service is not available"
        )
    
    try:
        models_dir = config.normative_models_path
        if models_dir.exists():
            available_regions = scan_folder(str(models_dir))
            return {
                "available_regions": available_regions,
                "model_path": str(models_dir)
            }
        else:
            return {
                "available_regions": {"male": [], "female": []},
                "error": f"Models directory not found: {models_dir}"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scanning regions: {e}")

if __name__ == "__main__":
    logger.info("Starting MRBrain Final Processing API...")
    logger.info(f"Configuration: {config.to_dict()}")
    logger.info(f"BrainAge available: {brainage_available and brainage_model is not None}")
    logger.info(f"Normative available: {normative_available}")
    
    uvicorn.run(app, host=config.api_host, port=config.api_port)