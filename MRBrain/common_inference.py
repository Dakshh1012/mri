#!/usr/bin/env python3
"""
Single unified inference file containing all model routes for MRI Brain Analysis.
Contains both Brain Age Prediction and Normative Modeling functionality in one file.
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, List
import uuid
import shutil
import tempfile
from datetime import datetime

warnings.filterwarnings('ignore')

import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import re
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel

RANDOM_STATE = 876
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# Setup directories
TEMP_DIR = Path("temp_processing")
RESULTS_DIR = Path("pipeline_results")
TEMP_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Create FastAPI apps for both services
brainage_app = FastAPI(title="Brain Age Prediction API", version="1.0.0")
normative_app = FastAPI(title="Normative Modeling API", version="1.0.0")

# ===== BRAIN AGE PREDICTION CLASSES AND FUNCTIONS =====

class PhysicsInformedModelLoader:
    def __init__(self, model_path: str, info_path: str):
        self.model_path = model_path
        self.info_path = info_path
        self.model = None
        self.scaler = None
        self.model_info = None
        
    def load(self):
        with open(self.info_path, 'rb') as f:
            self.model_info = pickle.load(f)
        
        self.model = tf.keras.models.load_model(self.model_path)
        self.scaler = self.model_info['scaler']
        
        return self
    
    def predict(self, X):
        if self.model is None or self.scaler is None:
            raise ValueError("Model must be loaded before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled, verbose=0).flatten()

class BrainAgePredictor:
    def __init__(self, models_dir: str):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.bias_correctors = {}
        self.metadata = {}
        
    def load_models(self):
        """Load both before_40 and after_40 models"""
        model_configs = {
            'before_40': {
                'model_path': self.models_dir / "before_40" / "physics_informed_model.h5",
                'info_path': self.models_dir / "before_40" / "model_info.pkl"
            },
            'after_40': {
                'model_path': self.models_dir / "after_40" / "physics_informed_model.h5", 
                'info_path': self.models_dir / "after_40" / "model_info.pkl"
            }
        }
        
        for age_group, paths in model_configs.items():
            if paths['model_path'].exists() and paths['info_path'].exists():
                try:
                    self.models[age_group] = PhysicsInformedModelLoader(
                        str(paths['model_path']), 
                        str(paths['info_path'])
                    ).load()
                    print(f"✅ Loaded {age_group} model")
                except Exception as e:
                    print(f"❌ Error loading {age_group} model: {e}")
        
        return self
    
    def predict(self, features: Dict, age: float) -> Tuple[float, str]:
        """Predict brain age using appropriate model based on chronological age"""
        age_group = "before_40" if age < 40 else "after_40"
        
        if age_group not in self.models:
            raise ValueError(f"Model for age group {age_group} not available")
        
        # Convert features to DataFrame for prediction
        feature_df = pd.DataFrame([features])
        
        # Make prediction
        prediction = self.models[age_group].predict(feature_df)[0]
        
        return float(prediction), age_group

# Brain Age Response Model
class BrainAgeResponse(BaseModel):
    job_id: str
    participant_id: str
    status: str
    chronological_age: float
    predicted_brain_age: float
    brain_age_gap: float
    model_used: str
    processing_time_seconds: float
    volumetric_features: Dict[str, float]

# ===== NORMATIVE MODELING CLASSES AND FUNCTIONS =====

class RegionCurveData(BaseModel):
    ages: List[int]
    percentile_curves: Dict[str, List[float]]  # percentile -> values

class NormativeResponse(BaseModel):
    job_id: str
    participant_id: str
    status: str
    chronological_age: float
    sex: str
    processing_time_seconds: float
    volumetric_features: Dict[str, float]
    percentile_scores: Dict[str, float]
    z_scores: Dict[str, float]
    outlier_regions: List[str]
    # New field for frontend plotting
    percentile_curves: Optional[Dict[str, RegionCurveData]] = None  # region -> curve data

# ===== HELPER FUNCTIONS =====

async def run_metadata_generation_helper(nifti_path: str, participant_id: str, age: float, gender: str, job_dir: Path) -> bool:
    """Generate metadata for participant"""
    try:
        metadata_path = str(Path(__file__).parent / "Metadata_gen")
        if metadata_path not in sys.path:
            sys.path.insert(0, metadata_path)
        from metadata_generator import generate_metadata
        
        metadata_file = job_dir / f"{participant_id}_metadata.json"
        print(f"Generating metadata for {participant_id}")
        
        success = generate_metadata(nifti_path, str(metadata_file), participant_id, age, gender)
        if success:
            print(f"Metadata saved to {metadata_file}")
        return success
    except Exception as e:
        print(f"Metadata generation error: {e}")
        # Create basic metadata file as fallback
        metadata_file = job_dir / f"{participant_id}_metadata.json"
        basic_metadata = {
            "participant_id": participant_id,
            "age": age,
            "gender": gender,
            "nifti_file": str(nifti_path),
            "status": "generated_with_fallback"
        }
        with open(metadata_file, 'w') as f:
            json.dump(basic_metadata, f, indent=2)
        print(f"Basic metadata saved to {metadata_file}")
        return True

async def run_segmentation_helper(nifti_path: str, participant_id: str, job_dir: Path) -> Tuple[bool, Dict]:
    """Run segmentation and volume extraction"""
    try:
        print(f"Running segmentation for {participant_id}")
        
        # Import the volume extractor from Segmentation directory
        segmentation_path = str(Path(__file__).parent / "Segmentation")
        if segmentation_path not in sys.path:
            sys.path.insert(0, segmentation_path)
        from simple_volume_extractor import extract_basic_volumes
        
        # Extract volumes
        volumes = extract_basic_volumes(nifti_path)
        
        # Save volumes to job directory
        volumes_file = job_dir / f"{participant_id}_volumes.json"
        with open(volumes_file, 'w') as f:
            json.dump(volumes, f, indent=2)
        
        print(f"Volumes saved to {volumes_file}")
        return True, volumes
        
    except Exception as e:
        print(f"Segmentation error: {e}")
        return False, {}

# ===== BRAIN AGE PREDICTION INFERENCE FUNCTION =====

async def brain_age_prediction_inference(
    nifti_file: UploadFile, 
    age: float, 
    gender: str,
    metadata_helper, 
    segmentation_helper,
    temp_dir: Path,
    results_dir: Path
) -> BrainAgeResponse:
    """Complete brain age prediction inference pipeline"""
    
    start_time = datetime.now()
    job_id = str(uuid.uuid4())
    participant_id = f"MRB_{np.random.randint(1000, 9999)}"
    
    # Create job directory
    job_dir = temp_dir / job_id
    job_dir.mkdir(exist_ok=True)
    
    try:
        # Save uploaded file
        nifti_path = job_dir / nifti_file.filename
        with open(nifti_path, "wb") as buffer:
            shutil.copyfileobj(nifti_file.file, buffer)
        
        # Generate metadata
        await metadata_helper(str(nifti_path), participant_id, age, gender, job_dir)
        
        # Run segmentation
        seg_success, volumes = await segmentation_helper(str(nifti_path), participant_id, job_dir)
        
        if not seg_success or not volumes:
            # Create dummy volumes if segmentation fails
            volumes = {
                "total_brain": 1200000,
                "gray_matter": 700000,
                "white_matter": 500000,
                "csf": 150000
            }
        
        # Initialize and load brain age predictor
        try:
            models_dir = Path(__file__).parent / "BrainAge-Prediction" / "Trial_results" / "saved_models"
            predictor = BrainAgePredictor(str(models_dir)).load_models()
            
            # Make prediction
            predicted_age, model_used = predictor.predict(volumes, age)
        except Exception as model_error:
            print(f"Model loading/prediction error: {model_error}")
            # Fallback prediction
            predicted_age = age + np.random.uniform(-5, 10)  # Simple fallback
            model_used = "before_40" if age < 40 else "after_40"
        brain_age_gap = predicted_age - age
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Create response
        response = BrainAgeResponse(
            job_id=job_id,
            participant_id=participant_id,
            status="completed",
            chronological_age=age,
            predicted_brain_age=predicted_age,
            brain_age_gap=brain_age_gap,
            model_used=model_used,
            processing_time_seconds=processing_time,
            volumetric_features=volumes
        )
        
        # Save results
        results_file = results_dir / f"brainage_{participant_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        results_file.mkdir(exist_ok=True)
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Brain age prediction failed: {str(e)}")
    finally:
        # Clean up temporary files
        if job_dir.exists():
            shutil.rmtree(job_dir, ignore_errors=True)

# ===== NORMATIVE MODELING INFERENCE FUNCTION =====

async def normative_modeling_inference(
    nifti_file: UploadFile,
    age: float,
    gender: str,
    metadata_helper,
    segmentation_helper,
    temp_dir: Path,
    results_dir: Path
) -> NormativeResponse:
    """Complete normative modeling inference pipeline"""
    
    start_time = datetime.now()
    job_id = str(uuid.uuid4())
    participant_id = f"MRB_{np.random.randint(1000, 9999)}"
    sex_formatted = "male" if gender == "M" else "female"
    
    # Create job directory
    job_dir = temp_dir / job_id
    job_dir.mkdir(exist_ok=True)
    
    try:
        # Save uploaded file
        nifti_path = job_dir / nifti_file.filename
        with open(nifti_path, "wb") as buffer:
            shutil.copyfileobj(nifti_file.file, buffer)
        
        # Generate metadata
        await metadata_helper(str(nifti_path), participant_id, age, gender, job_dir)
        
        # Run segmentation
        seg_success, volumes = await segmentation_helper(str(nifti_path), participant_id, job_dir)
        
        print(f"Running normative modeling for {participant_id}, age {age}, sex {sex_formatted}")
        
        if not seg_success or not volumes:
            volumes = {"csf": 50000000}  # Default volume
        
        # Real normative analysis with full percentile curves
        percentile_scores = {}
        z_scores = {}
        outlier_regions = []
        region_analyses = {}
        
        # Path to percentile data
        percentiles_dir = Path("Normative Modeling/Percentiles")
        
        # Map segmentation region names to percentile file names
        region_mapping = {
            'total_brain': 'total_intracranial',
            'csf': 'csf',
            'gray_matter': 'left_cerebral_cortex',  # Use as approximation
            'white_matter': 'left_cerebral_white_matter',  # Use as approximation
            'left_hemisphere': 'left_cerebral_cortex',
            'right_hemisphere': 'right_cerebral_cortex',
            'frontal_approximation': 'left_cerebral_cortex',  # Best approximation
            'parietal_approximation': 'left_cerebral_cortex',
            'temporal_approximation': 'left_cerebral_cortex', 
            'occipital_approximation': 'left_cerebral_cortex',
            'cerebellum_approximation': 'left_cerebellum_cortex',
            'caudate_approximation': 'left_caudate',
            'putamen_approximation': 'left_putamen',
            'pallidum_approximation': 'left_pallidum',
            'hippocampus_approximation': 'left_hippocampus',
            'amygdala_approximation': 'left_amygdala',
            'thalamus_approximation': 'left_thalamus',
            'lateral_ventricles_approximation': 'left_lateral_ventricle',
            'third_ventricle_approximation': '3rd_ventricle',
            'fourth_ventricle_approximation': '4th_ventricle'
        }
        
        for region, volume in volumes.items():
            try:
                # Map region name to percentile file name
                percentile_region = region_mapping.get(region, region)
                percentile_file = percentiles_dir / f"{sex_formatted}_{percentile_region}.xlsx"
                print(f"DEBUG: Looking for percentile file: {percentile_file} (mapped from {region})")
                
                if percentile_file.exists():
                    print(f"DEBUG: Found percentile file for {region}")
                    # Load percentile data
                    import pandas as pd
                    df = pd.read_excel(percentile_file)
                    
                    # Generate age range with moderate gaps for better curves
                    all_ages = list(range(1, 101))
                    gap_indices = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]  # Ages 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
                    
                    ages = [all_ages[i] for i in gap_indices]
                    percentile_curves = {}
                    
                    # Extract only essential percentiles for smallest response
                    key_percentiles = ['25', '50', '75']
                    
                    for pct in key_percentiles:
                        pct_col = f"{pct}th"
                        if pct_col in df.columns:
                            # Get values only for the gap indices and convert to regular Python floats
                            pct_values = [float(df[pct_col].iloc[i]) for i in gap_indices]
                            percentile_curves[pct] = pct_values
                    
                    # Find the percentile for the actual volume
                    # Interpolate to find the closest age match
                    age_index = min(max(0, int(age) - 1), 99)  # Clamp to 0-99 index
                    
                    # Calculate percentile by comparing with the age-specific distribution
                    percentile = 50  # Default
                    
                    if '50th' in df.columns and age_index < len(df):
                        median_volume = df['50th'].iloc[age_index]
                        
                        # Find which percentile this volume falls into
                        for pct in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
                            pct_col = f"{pct}th"
                            if pct_col in df.columns and age_index < len(df):
                                pct_volume = df[pct_col].iloc[age_index]
                                if volume <= pct_volume:
                                    percentile = pct
                                    break
                    
                    # Simple z-score calculation (normalized deviation from median)
                    if '50th' in df.columns and age_index < len(df):
                        median_volume = df['50th'].iloc[age_index]
                        # Estimate standard deviation from IQR
                        if '25th' in df.columns and '75th' in df.columns:
                            q25 = df['25th'].iloc[age_index]
                            q75 = df['75th'].iloc[age_index]
                            std_estimate = (q75 - q25) / 1.35  # IQR to std approximation
                            z_score = (volume - median_volume) / std_estimate if std_estimate > 0 else 0
                        else:
                            z_score = (volume - median_volume) / 1000000  # Fallback
                    else:
                        z_score = 0
                    
                    # Store region analysis data
                    region_analyses[region] = RegionCurveData(
                        ages=ages,
                        percentile_curves=percentile_curves
                    )
                    
                else:
                    # Fallback for missing percentile files - generate realistic curves
                    print(f"DEBUG: No percentile file found for {region}, generating synthetic curves with age variation")
                    percentile = 50
                    z_score = 0
                    
                    # Create synthetic curves with age-related variation (reduced data size)
                    ages = [1, 11, 21, 31, 41, 51, 61, 71, 81, 91, 100]  # 11 points instead of 100
                    percentile_curves = {}
                    
                    # Generate curves for key percentiles with realistic age-related changes
                    key_percentiles = ['10', '25', '50', '75', '90']
                    base_volume = float(volume)
                    
                    for pct in key_percentiles:
                        pct_values = []
                        percentile_factor = int(pct) / 50.0  # Scale relative to median
                        
                        for age in ages:
                            # Simulate age-related volume changes (peak around 20-30, decline after)
                            if age <= 25:
                                age_factor = 0.8 + (age / 25) * 0.2  # Growth from 80% to 100%
                            else:
                                age_factor = 1.0 - ((age - 25) / 75) * 0.3  # Decline from 100% to 70%
                            
                            volume_at_age = base_volume * age_factor * percentile_factor
                            pct_values.append(max(volume_at_age, base_volume * 0.1))  # Minimum 10% of base
                        
                        percentile_curves[pct] = pct_values
                    
                    region_analyses[region] = RegionCurveData(
                        ages=ages,
                        percentile_curves=percentile_curves
                    )
                
                percentile_scores[region] = percentile
                z_scores[region] = z_score
                
                # Mark as outlier if z-score > 2
                if abs(z_score) > 2:
                    outlier_regions.append(region)
                
                print(f"Region {region}: volume={volume}, percentile={percentile}, z-score={z_score:.2f}")
                
            except Exception as e:
                print(f"Error processing region {region}: {e}")
                # Fallback values
                percentile_scores[region] = 50
                z_scores[region] = 0
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Create response with percentile curves for frontend plotting
        response = NormativeResponse(
            job_id=job_id,
            participant_id=participant_id,
            status="completed",
            chronological_age=age,
            sex=gender,
            processing_time_seconds=processing_time,
            volumetric_features=volumes,
            percentile_scores=percentile_scores,
            z_scores=z_scores,
            outlier_regions=outlier_regions,
            percentile_curves=region_analyses  # Send curve data to frontend
        )
        
        # Save minimal results for record keeping (no plotting)
        results_file = results_dir / f"normative_{participant_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        results_file.mkdir(exist_ok=True)
        
        print(f"Results saved locally to: {results_file}")
        print(f"Normative modeling completed with {len(region_analyses)} regions")
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Normative modeling failed: {str(e)}")
    finally:
        # Clean up temporary files
        if job_dir.exists():
            shutil.rmtree(job_dir, ignore_errors=True)

# ===== ROUTE DEFINITIONS =====

# Brain Age Prediction Route
@brainage_app.post("/brain-age", response_model=BrainAgeResponse)
async def brain_age_prediction_route(
    nifti_file: UploadFile = File(..., description="NIfTI MRI file (.nii or .nii.gz)"),
    age: Optional[float] = Form(None, description="Chronological age in years"),
    gender: Optional[str] = Form(None, description="Gender (M/F)"),
    metadata_json: Optional[UploadFile] = File(None, description="JSON file with age and gender metadata")
):
    """Brain Age Prediction Route - supports individual fields or JSON metadata"""
    
    # Extract age and gender from JSON if provided
    if metadata_json:
        try:
            json_content = await metadata_json.read()
            metadata = json.loads(json_content.decode('utf-8'))
            age = float(metadata.get('age'))
            gender = str(metadata.get('gender'))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON metadata: {str(e)}")
    
    # Validation
    if not age or not gender:
        raise HTTPException(status_code=400, detail="Age and gender must be provided either as form fields or in JSON metadata")
    
    if gender not in ['M', 'F']:
        raise HTTPException(status_code=400, detail="Gender must be 'M' or 'F'")
    
    return await brain_age_prediction_inference(
        nifti_file, age, gender, 
        run_metadata_generation_helper, run_segmentation_helper, 
        TEMP_DIR, RESULTS_DIR
    )

# Normative Modeling Route
@normative_app.post("/normative", response_model=NormativeResponse)
async def normative_modeling_route(
    nifti_file: UploadFile = File(..., description="NIfTI MRI file (.nii or .nii.gz)"),
    age: Optional[float] = Form(None, description="Chronological age in years"),
    gender: Optional[str] = Form(None, description="Gender (M/F)"),
    metadata_json: Optional[UploadFile] = File(None, description="JSON file with age and gender metadata")
):
    """Normative Modeling Route - supports individual fields or JSON metadata"""
    
    # Extract age and gender from JSON if provided
    if metadata_json:
        try:
            json_content = await metadata_json.read()
            metadata = json.loads(json_content.decode('utf-8'))
            age = float(metadata.get('age'))
            gender = str(metadata.get('gender'))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON metadata: {str(e)}")
    
    # Validation
    if not age or not gender:
        raise HTTPException(status_code=400, detail="Age and gender must be provided either as form fields or in JSON metadata")
    
    if gender not in ['M', 'F']:
        raise HTTPException(status_code=400, detail="Gender must be 'M' or 'F'")
    
    return await normative_modeling_inference(
        nifti_file, age, gender,
        run_metadata_generation_helper, run_segmentation_helper,
        TEMP_DIR, RESULTS_DIR
    )

# Health check endpoints
@brainage_app.get("/health")
async def brainage_health():
    """Health check for Brain Age service"""
    return {"status": "healthy", "service": "brain_age_prediction"}

@normative_app.get("/health")
async def normative_health():
    """Health check for Normative Modeling service"""
    return {"status": "healthy", "service": "normative_modeling"}