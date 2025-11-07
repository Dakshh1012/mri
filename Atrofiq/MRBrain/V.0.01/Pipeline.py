#!/usr/bin/env python3
"""
Enhanced MRI Processing Pipeline v2.0 - WITH NORMATIVE MODELING
Integrates metadata generation, segmentation, brain age prediction, and normative modeling
Supports: Metadata_gen -> Segmentation -> BrainAge-Prediction -> Normative Modeling
"""

import os
import sys
import argparse
import subprocess
import json
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_pipeline_v2.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class EnhancedMRIPipelineV2:
    def __init__(self, base_dir: str = "."):
        """Initialize pipeline with base directory containing all modules"""
        self.base_dir = Path(base_dir).resolve()
        self.metadata_gen_dir = self.base_dir / "Metadata_gen"
        self.segmentation_dir = self.base_dir / "Segmentation"
        self.brainage_dir = self.base_dir / "BrainAge-Prediction"
        self.normative_dir = self.base_dir / "Normative Modeling"
        self.twod_threed_dir = self.base_dir / "2D-3D"
        
        # Default paths based on your structure
        self.default_results_dir = self.base_dir / "New_results_all"
        
        # Verify required directories exist
        self._verify_directories()
    
    def _verify_directories(self):
        """Verify that required directories and scripts exist"""
        required_dirs = [
            self.metadata_gen_dir, 
            self.segmentation_dir, 
            self.brainage_dir,
            self.normative_dir
        ]
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                logger.warning(f"Directory not found: {dir_path}")
        
        # Check for required scripts
        self.metadata_script = self.metadata_gen_dir / "metadata_generator.py"
        self.segmentation_script = self.segmentation_dir / "mri_pipeline_clean.py"
        self.brainage_script = self.brainage_dir / "Inference.py"
        self.normative_script = self.normative_dir / "API.py"
        
        # Log which scripts are available
        scripts = [
            ("Metadata script", self.metadata_script),
            ("Segmentation script", self.segmentation_script),
            ("Brain age script", self.brainage_script),
            ("Normative modeling script", self.normative_script)
        ]
        
        for name, script_path in scripts:
            if script_path.exists():
                logger.info(f"✓ {name} found: {script_path}")
            else:
                logger.warning(f"✗ {name} not found: {script_path}")
    
    def extract_participant_ids(self, metadata_json: str) -> List[str]:
        """Extract participant IDs from metadata JSON file"""
        try:
            with open(metadata_json, 'r') as f:
                metadata = json.load(f)
            
            # Handle different metadata structures
            if isinstance(metadata, dict):
                if "metadata" in metadata and "patient ids" in metadata["metadata"]:
                    participant_ids = metadata["metadata"]["patient ids"]
                elif "patient ids" in metadata:
                    participant_ids = metadata["patient ids"]
                elif "participants" in metadata:
                    participant_ids = metadata["participants"]
                else:
                    # Try to find any list that looks like participant IDs
                    for key, value in metadata.items():
                        if isinstance(value, list) and len(value) > 0:
                            # Check if it looks like participant IDs
                            if isinstance(value[0], str) and any(char.isalnum() for char in value[0]):
                                participant_ids = value
                                break
                    else:
                        logger.error("Could not find participant IDs in metadata")
                        return []
            else:
                logger.error("Unexpected metadata structure")
                return []
            
            logger.info(f"Found {len(participant_ids)} participant IDs: {participant_ids[:5]}...")
            return participant_ids
            
        except Exception as e:
            logger.error(f"Error extracting participant IDs: {e}")
            return []
    
    def validate_csv_format(self, csv_file: str) -> bool:
        """Validate CSV file format"""
        try:
            separators = ['\t', ',', ' ']
            
            for sep in separators:
                try:
                    df = pd.read_csv(csv_file, sep=sep, header=None)
                    if len(df.columns) >= 3:
                        logger.info(f"CSV loaded with separator '{sep}', shape: {df.shape}")
                        logger.info(f"First few rows:\n{df.head()}")
                        return True
                except:
                    continue
            
            logger.error("Could not read CSV file with any common separator")
            return False
            
        except Exception as e:
            logger.error(f"Error validating CSV file: {e}")
            return False
    
    def run_metadata_generation(self, mri_dir: str, csv_file: str, 
                              output_file: str = "metadata.json") -> bool:
        """Run metadata generation step"""
        try:
            logger.info("="*60)
            logger.info("STEP 1: METADATA GENERATION")
            logger.info("="*60)
            
            mri_dir = str(Path(mri_dir).resolve())
            csv_file = str(Path(csv_file).resolve())
            
            if not Path(mri_dir).exists():
                logger.error(f"MRI directory not found: {mri_dir}")
                return False
            
            if not Path(csv_file).exists():
                logger.error(f"CSV file not found: {csv_file}")
                return False
            
            if not self.validate_csv_format(csv_file):
                logger.error("CSV file format validation failed")
                return False
            
            logger.info(f"MRI directory: {mri_dir}")
            logger.info(f"CSV file: {csv_file}")
            logger.info(f"Output file: {output_file}")
            
            cmd = [
                sys.executable, str(self.metadata_script),
                "--mri_dir", mri_dir,
                "--csv", csv_file
            ]
            
            if "--output" in self._get_script_help(self.metadata_script):
                cmd.extend(["--output", output_file])
            
            logger.info(f"Running command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.metadata_gen_dir)
            
            if result.stdout:
                logger.info(f"Metadata stdout:\n{result.stdout}")
            
            if result.stderr:
                logger.info(f"Metadata stderr:\n{result.stderr}")
            
            if result.returncode != 0:
                logger.error(f"Metadata generation failed with return code: {result.returncode}")
                return False
            
            # Check if output file was created
            output_path = Path(output_file)
            if not output_path.is_absolute():
                output_path = self.metadata_gen_dir / output_file
            
            if output_path.exists():
                logger.info(f"✓ Metadata file created: {output_path}")
                try:
                    with open(output_path, 'r') as f:
                        metadata = json.load(f)
                    logger.info("Sample metadata structure:")
                    if isinstance(metadata, dict) and "metadata" in metadata:
                        meta_data = metadata["metadata"]
                        if "patient ids" in meta_data:
                            logger.info(f"Patient IDs found: {len(meta_data['patient ids'])}")
                            logger.info(f"First 3 patients: {meta_data['patient ids'][:3]}")
                except Exception as e:
                    logger.warning(f"Could not parse metadata file: {e}")
            else:
                logger.warning(f"Expected output file not found: {output_path}")
            
            logger.info("✓ Metadata generation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Unexpected error in metadata generation: {e}")
            return False
    
    def run_segmentation(self, mri_dir: str, output_dir: str, 
                        volumes_csv: str = "volumes.csv", 
                        qc_csv: str = "qc_scores.csv",
                        threads: int = 5, parc: bool = True) -> bool:
        """Run segmentation step"""
        try:
            logger.info("="*60)
            logger.info("STEP 2: SEGMENTATION")
            logger.info("="*60)
            
            mri_dir = str(Path(mri_dir).resolve())
            output_dir = str(Path(output_dir).resolve())
            
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            if not Path(volumes_csv).is_absolute():
                volumes_csv = str(Path.cwd() / volumes_csv)
            if not Path(qc_csv).is_absolute():
                qc_csv = str(Path.cwd() / qc_csv)
            
            logger.info(f"Input directory: {mri_dir}")
            logger.info(f"Output directory: {output_dir}")
            logger.info(f"Volumes CSV: {volumes_csv}")
            logger.info(f"QC CSV: {qc_csv}")
            logger.info(f"Threads: {threads}")
            
            if not self.segmentation_script.exists():
                logger.error(f"Segmentation script not found: {self.segmentation_script}")
                return False
            
            cmd = [
                sys.executable, str(self.segmentation_script.name),
                "--i", mri_dir,
                "--o", output_dir,
                "--vol", volumes_csv,
                "--qc", qc_csv,
                "--threads", str(threads)
            ]
            
            if parc:
                cmd.append("--parc")
            
            logger.info(f"Running command: {' '.join(cmd)}")
            
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.segmentation_dir) + os.pathsep + env.get('PYTHONPATH', '')
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=self.segmentation_dir,
                env=env
            )
            
            if result.stdout:
                logger.info(f"Segmentation stdout:\n{result.stdout}")
            
            if result.stderr and result.stderr.strip():
                logger.info(f"Segmentation stderr:\n{result.stderr}")
            
            if result.returncode != 0:
                logger.error(f"Segmentation failed with return code: {result.returncode}")
                return False
            
            # Check outputs
            output_checks = [
                (Path(output_dir), "Segmentation output directory"),
                (Path(volumes_csv), "Volumes CSV"),
                (Path(qc_csv), "QC scores CSV")
            ]
            
            for path, description in output_checks:
                if path.exists():
                    logger.info(f"✓ {description} created: {path}")
                    if path.suffix == '.csv':
                        try:
                            df = pd.read_csv(path)
                            logger.info(f"  - Shape: {df.shape}")
                            logger.info(f"  - Columns: {list(df.columns)[:5]}...")
                        except:
                            logger.info(f"  - File exists but couldn't read CSV structure")
                else:
                    logger.warning(f"Expected output not found: {path}")
            
            logger.info("✓ Segmentation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Unexpected error in segmentation: {e}")
            return False
    
    def run_brain_age_prediction(self, volumes_csv: str, metadata_json: str, 
                                models_dir: str = "Trial_results/saved_models/",
                                output_csv: str = "brain_age_results.csv") -> bool:
        """Run brain age prediction step"""
        try:
            logger.info("="*60)
            logger.info("STEP 3: BRAIN AGE PREDICTION")
            logger.info("="*60)
            
            volumes_csv = str(Path(volumes_csv).resolve())
            metadata_json = str(Path(metadata_json).resolve())
            
            if not Path(volumes_csv).exists():
                logger.error(f"Volumes CSV not found: {volumes_csv}")
                return False
            
            if not Path(metadata_json).exists():
                logger.error(f"Metadata JSON not found: {metadata_json}")
                return False
            
            models_path = Path(models_dir)
            if not models_path.is_absolute():
                models_path = self.brainage_dir / models_dir
            
            if not models_path.exists():
                logger.error(f"Models directory not found: {models_path}")
                return False
            
            logger.info(f"Volumes CSV: {volumes_csv}")
            logger.info(f"Metadata JSON: {metadata_json}")
            logger.info(f"Models directory: {models_path}")
            
            cmd = [
                sys.executable, str(self.brainage_script),
                "--models_dir", str(models_path),
                "--volumes", volumes_csv,
                "--metadata", metadata_json
            ]
            
            if "--output" in self._get_script_help(self.brainage_script):
                cmd.extend(["--output", output_csv])
            
            logger.info(f"Running command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.brainage_dir)
            
            if result.stdout:
                logger.info(f"Brain age stdout:\n{result.stdout}")
            
            if result.stderr and result.stderr.strip():
                logger.info(f"Brain age stderr:\n{result.stderr}")
            
            if result.returncode != 0:
                logger.error(f"Brain age prediction failed with return code: {result.returncode}")
                return False
            
            logger.info("✓ Brain age prediction completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Unexpected error in brain age prediction: {e}")
            return False
    
    def run_normative_modeling(self, metadata_json: str, 
                              importance_folder: str = None,
                              percentiles_folder: str = None,
                              percentiles: List[int] = [25, 50, 75],
                              output_dir: str = "normative_results") -> bool:
        """
        Run normative modeling for all participants automatically
        
        Args:
            metadata_json: Path to metadata JSON file
            importance_folder: Path to feature importance folder (auto-detect if None)
            percentiles_folder: Path to percentiles folder (auto-detect if None)
            percentiles: List of percentiles to calculate
            output_dir: Output directory for normative modeling results
        
        Returns:
            bool: Success status
        """
        try:
            logger.info("="*60)
            logger.info("STEP 4: NORMATIVE MODELING")
            logger.info("="*60)
            
            # Extract participant IDs from metadata
            participant_ids = self.extract_participant_ids(metadata_json)
            if not participant_ids:
                logger.error("Could not extract participant IDs for normative modeling")
                return False
            
            # Auto-detect paths if not provided
            if importance_folder is None:
                importance_folder = self.brainage_dir / "Trial_results" / "feature_importance"
                if not importance_folder.exists():
                    # Try alternative locations
                    alt_paths = [
                        self.brainage_dir / "Trial_results" / "feature_importance",
                        self.brainage_dir / "Results" / "feature_importance",
                        self.brainage_dir / "feature_importance"
                    ]
                    for alt_path in alt_paths:
                        if alt_path.exists():
                            importance_folder = alt_path
                            break
                    else:
                        logger.error(f"Could not find feature importance folder. Tried: {[str(p) for p in [importance_folder] + alt_paths]}")
                        return False
            
            if percentiles_folder is None:
                percentiles_folder = self.normative_dir / "Percentiles"
                if not percentiles_folder.exists():
                    logger.error(f"Percentiles folder not found: {percentiles_folder}")
                    return False
            
            # Convert to absolute paths
            metadata_json = str(Path(metadata_json).resolve())
            importance_folder = str(Path(importance_folder).resolve())
            percentiles_folder = str(Path(percentiles_folder).resolve())
            
            # Create output directory
            output_path = Path(output_dir).resolve()
            output_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Processing {len(participant_ids)} participants")
            logger.info(f"Metadata JSON: {metadata_json}")
            logger.info(f"Importance folder: {importance_folder}")
            logger.info(f"Percentiles folder: {percentiles_folder}")
            logger.info(f"Output directory: {output_path}")
            logger.info(f"Percentiles: {percentiles}")
            
            # Check if normative script exists
            if not self.normative_script.exists():
                logger.error(f"Normative modeling script not found: {self.normative_script}")
                return False
            
            successful_participants = []
            failed_participants = []
            
            # Process each participant
            for i, participant_id in enumerate(participant_ids, 1):
                logger.info(f"\nProcessing participant {i}/{len(participant_ids)}: {participant_id}")
                
                # Prepare output file for this participant
                participant_output = output_path / f"{participant_id}_normative_results.json"
                
                # Build command
                cmd = [
                    sys.executable, str(self.normative_script),
                    "--participant-id", participant_id,
                    "--importance-folder", importance_folder,
                    "--percentiles-folder", percentiles_folder,
                    "--metadata", metadata_json,
                    "--pretty"
                ]
                
                # Add percentiles
                for p in percentiles:
                    cmd.extend(["--percentiles", str(p)])
                
                # Add output parameter if supported
                help_text = self._get_script_help(self.normative_script)
                if "--output" in help_text:
                    cmd.extend(["--output", str(participant_output)])
                
                logger.info(f"Running: {' '.join(cmd)}")
                
                try:
                    result = subprocess.run(
                        cmd, 
                        capture_output=True, 
                        text=True, 
                        cwd=self.normative_dir,
                        timeout=300  # 5 minute timeout per participant
                    )
                    
                    if result.returncode == 0:
                        successful_participants.append(participant_id)
                        logger.info(f"✓ Successfully processed {participant_id}")
                        
                        # Save output if not automatically saved
                        if result.stdout and not participant_output.exists():
                            try:
                                # Try to parse as JSON and save
                                import json
                                if result.stdout.strip().startswith('{'):
                                    output_data = json.loads(result.stdout)
                                    with open(participant_output, 'w') as f:
                                        json.dump(output_data, f, indent=2)
                                else:
                                    # Save as text
                                    with open(participant_output.with_suffix('.txt'), 'w') as f:
                                        f.write(result.stdout)
                            except:
                                # Save raw output
                                with open(participant_output.with_suffix('.txt'), 'w') as f:
                                    f.write(result.stdout)
                    else:
                        failed_participants.append(participant_id)
                        logger.error(f"✗ Failed to process {participant_id}")
                        if result.stderr:
                            logger.error(f"Error: {result.stderr}")
                        
                        # Save error log
                        error_file = output_path / f"{participant_id}_error.log"
                        with open(error_file, 'w') as f:
                            f.write(f"Command: {' '.join(cmd)}\n")
                            f.write(f"Return code: {result.returncode}\n")
                            f.write(f"Stdout:\n{result.stdout}\n")
                            f.write(f"Stderr:\n{result.stderr}\n")
                
                except subprocess.TimeoutExpired:
                    failed_participants.append(participant_id)
                    logger.error(f"✗ Timeout processing {participant_id}")
                except Exception as e:
                    failed_participants.append(participant_id)
                    logger.error(f"✗ Error processing {participant_id}: {e}")
            
            # Summary
            logger.info("="*60)
            logger.info("NORMATIVE MODELING SUMMARY")
            logger.info("="*60)
            logger.info(f"Total participants: {len(participant_ids)}")
            logger.info(f"Successful: {len(successful_participants)}")
            logger.info(f"Failed: {len(failed_participants)}")
            
            if successful_participants:
                logger.info(f"✓ Successful participants: {successful_participants}")
            
            if failed_participants:
                logger.info(f"✗ Failed participants: {failed_participants}")
            
            # Create summary file
            summary = {
                "total_participants": len(participant_ids),
                "successful_participants": successful_participants,
                "failed_participants": failed_participants,
                "success_rate": len(successful_participants) / len(participant_ids) * 100,
                "parameters": {
                    "importance_folder": importance_folder,
                    "percentiles_folder": percentiles_folder,
                    "percentiles": percentiles
                }
            }
            
            summary_file = output_path / "normative_modeling_summary.json"
            import json
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"📊 Summary saved to: {summary_file}")
            
            # Consider it successful if at least 80% of participants were processed
            success_rate = len(successful_participants) / len(participant_ids)
            if success_rate >= 0.8:
                logger.info("✓ Normative modeling completed successfully")
                return True
            else:
                logger.error(f"Normative modeling had low success rate: {success_rate:.1%}")
                return False
            
        except Exception as e:
            logger.error(f"Unexpected error in normative modeling: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _get_script_help(self, script_path: Path) -> str:
        """Get help text from a script to check available parameters"""
        try:
            result = subprocess.run([sys.executable, str(script_path), "--help"], 
                                  capture_output=True, text=True, timeout=10)
            return result.stdout + result.stderr
        except:
            return ""
    
    def run_full_pipeline(self, mri_dir: str, csv_file: str, 
                         output_base_dir: str = "New_results_all",
                         threads: int = 90, parc: bool = True,
                         run_brainage: bool = True,
                         run_normative: bool = True,
                         percentiles: List[int] = [25, 50, 75]) -> bool:
        """
        Run the complete pipeline including normative modeling
        """
        try:
            # Create output directory structure
            output_path = Path(output_base_dir).resolve()
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Define output paths
            metadata_output = output_path / "metadata.json"
            segmentation_output = output_path / "segmentations"
            volumes_csv = output_path / "volumes.csv"
            qc_csv = output_path / "qc_scores.csv"
            brainage_output = output_path / "brain_age_results.csv"
            normative_output = output_path / "normative_results"
            
            logger.info("="*70)
            logger.info("STARTING ENHANCED MRI PROCESSING PIPELINE v2.0")
            logger.info("WITH AUTOMATED NORMATIVE MODELING")
            logger.info("="*70)
            logger.info(f"Input MRI directory: {mri_dir}")
            logger.info(f"Input CSV file: {csv_file}")
            logger.info(f"Output base directory: {output_base_dir}")
            logger.info(f"Threads: {threads}")
            logger.info(f"Parcellation: {parc}")
            logger.info(f"Run brain age: {run_brainage}")
            logger.info(f"Run normative modeling: {run_normative}")
            logger.info(f"Percentiles: {percentiles}")
            
            # Step 1: Generate metadata
            if not self.run_metadata_generation(mri_dir, csv_file, str(metadata_output)):
                logger.error("Pipeline failed at metadata generation step")
                return False
            
            # Step 2: Run segmentation
            if not self.run_segmentation(mri_dir, str(segmentation_output), 
                                       str(volumes_csv), str(qc_csv), threads, parc):
                logger.error("Pipeline failed at segmentation step")
                return False
            
            # Step 3: Brain age prediction
            if run_brainage:
                if not self.run_brain_age_prediction(str(volumes_csv), str(metadata_output), 
                                                   output_csv=str(brainage_output)):
                    logger.error("Pipeline failed at brain age prediction step")
                    return False
            
            # Step 4: Normative modeling (NEW!)
            if run_normative:
                if not self.run_normative_modeling(str(metadata_output), 
                                                 output_dir=str(normative_output),
                                                 percentiles=percentiles):
                    logger.error("Pipeline failed at normative modeling step")
                    return False
            
            # Pipeline completed successfully
            logger.info("="*70)
            logger.info("PIPELINE V2.0 COMPLETED SUCCESSFULLY!")
            logger.info("="*70)
            logger.info("Output files:")
            logger.info(f"📁 Base directory: {output_path}")
            logger.info(f"📋 Metadata: {metadata_output}")
            logger.info(f"🧠 Segmentations: {segmentation_output}")
            logger.info(f"📊 Volumes CSV: {volumes_csv}")
            logger.info(f"📈 QC Scores: {qc_csv}")
            if run_brainage:
                logger.info(f"🎯 Brain Age Results: {brainage_output}")
            if run_normative:
                logger.info(f"📐 Normative Modeling: {normative_output}")
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced MRI Processing Pipeline v2.0 - Now with Automated Normative Modeling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full pipeline (all 4 steps including normative modeling)
    python enhanced_pipeline_v2.py --mri_dir /path/to/mri/files --csv patient_info.csv --output_dir New_results_all
    
    # Run full pipeline with custom percentiles
    python enhanced_pipeline_v2.py --mri_dir /path/to/mri/files --csv patient_info.csv --percentiles 10 25 50 75 90
    
    # Run without normative modeling
    python enhanced_pipeline_v2.py --mri_dir /path/to/mri/files --csv patient_info.csv --no_normative
    
    # Run only normative modeling (requires existing metadata.json)
    python enhanced_pipeline_v2.py --metadata New_results_all/metadata.json --normative_only
    
    # Use existing New_results_all structure
    python enhanced_pipeline_v2.py --mri_dir ../SynthSeg/Post-contrast-Data/ --csv ../SynthSeg/Post-contrast-subjects.csv

Expected CSV format:
    MRB_0097	27	W
    MRB_0099	20	F
    (tab-separated: subject_id, age, sex)
        """
    )
    
    parser.add_argument('--mri_dir', type=str,
                       help='Directory containing MRI files (.nii format)')
    parser.add_argument('--csv', type=str,
                       help='CSV file with patient info (subject_id, age, sex columns, tab-separated)')
    parser.add_argument('--output_dir', type=str, default='New_results_all',
                       help='Output directory for all results (default: New_results_all)')
    parser.add_argument('--threads', type=int, default=90,
                       help='Number of threads for segmentation (default: 90)')
    parser.add_argument('--no_parc', action='store_true',
                       help='Disable parcellation in segmentation')
    parser.add_argument('--no_brainage', action='store_true',
                       help='Skip brain age prediction step')
    parser.add_argument('--no_normative', action='store_true',
                       help='Skip normative modeling step')
    
    # Normative modeling specific options
    parser.add_argument('--percentiles', type=int, nargs='+', default=[25, 50, 75],
                       help='Percentiles for normative modeling (default: 25 50 75)')
    parser.add_argument('--importance_folder', type=str,
                       help='Custom path to feature importance folder (auto-detected if not provided)')
    parser.add_argument('--percentiles_folder', type=str,
                       help='Custom path to percentiles folder (auto-detected if not provided)')
    
    # Individual step options
    parser.add_argument('--metadata_only', action='store_true',
                       help='Run only metadata generation step')
    parser.add_argument('--segmentation_only', action='store_true',
                       help='Run only segmentation step')
    parser.add_argument('--brainage_only', action='store_true',
                       help='Run only brain age prediction step')
    parser.add_argument('--normative_only', action='store_true',
                       help='Run only normative modeling step')
    
    # Individual step file inputs
    parser.add_argument('--volumes', type=str,
                       help='Volumes CSV file (for brain age prediction only)')
    parser.add_argument('--metadata', type=str,
                       help='Metadata JSON file (for brain age or normative modeling only)')
    parser.add_argument('--models_dir', type=str, default='Trial_results/saved_models/',
                       help='Directory containing brain age models')
    
    parser.add_argument('--base_dir', type=str, default='.',
                       help='Base directory containing all module folders (default: . for current directory)')
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = EnhancedMRIPipelineV2(base_dir=args.base_dir)
        
        # Create output directory
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        success = False
        
        if args.metadata_only:
            # Run only metadata generation
            if not args.mri_dir or not args.csv:
                logger.error("--mri_dir and --csv are required for metadata generation")
                sys.exit(1)
            
            metadata_output = output_path / "metadata.json"
            success = pipeline.run_metadata_generation(args.mri_dir, args.csv, str(metadata_output))
        
        elif args.segmentation_only:
            # Run only segmentation
            if not args.mri_dir:
                logger.error("--mri_dir is required for segmentation")
                sys.exit(1)
            
            segmentation_output = output_path / "segmentations"
            volumes_csv = output_path / "volumes.csv"
            qc_csv = output_path / "qc_scores.csv"
            success = pipeline.run_segmentation(
                args.mri_dir, str(segmentation_output), 
                str(volumes_csv), str(qc_csv), 
                args.threads, not args.no_parc
            )
        
        elif args.brainage_only:
            # Run only brain age prediction
            if not args.volumes or not args.metadata:
                logger.error("--volumes and --metadata are required for brain age prediction")
                sys.exit(1)
            
            brainage_output = output_path / "brain_age_results.csv"
            success = pipeline.run_brain_age_prediction(
                args.volumes, args.metadata, args.models_dir, str(brainage_output)
            )
        
        elif args.normative_only:
            # Run only normative modeling
            if not args.metadata:
                logger.error("--metadata is required for normative modeling")
                sys.exit(1)
            
            normative_output = output_path / "normative_results"
            success = pipeline.run_normative_modeling(
                args.metadata,
                importance_folder=args.importance_folder,
                percentiles_folder=args.percentiles_folder,
                percentiles=args.percentiles,
                output_dir=str(normative_output)
            )
        
        else:
            # Run full pipeline
            if not args.mri_dir or not args.csv:
                logger.error("--mri_dir and --csv are required for full pipeline")
                sys.exit(1)
            
            success = pipeline.run_full_pipeline(
                args.mri_dir, args.csv, args.output_dir,
                args.threads, not args.no_parc, 
                not args.no_brainage, not args.no_normative,
                args.percentiles
            )
        
        if success:
            logger.info("🎉 Pipeline execution completed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ Pipeline execution failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()