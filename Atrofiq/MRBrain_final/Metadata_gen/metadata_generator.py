#!/usr/bin/env python3
"""
MRI Metadata Generator v0.04 - Fixed Subject ID Extraction
Generates metadata for MRI files by matching filenames with CSV data
"""

import os
import sys
import argparse
import json
import logging
import pandas as pd
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('metadata_generator.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class MRIMetadataGenerator:
    def __init__(self):
        """Initialize metadata generator"""
        self.supported_extensions = ['.nii.gz', '.nii', '.dcm']
        logger.info("MRI Metadata Generator initialized")
    
    def find_mri_files(self, mri_dir: str) -> List[Path]:
        """
        Find all MRI files in directory with supported extensions
        
        Args:
            mri_dir: Directory containing MRI files
        
        Returns:
            List[Path]: List of MRI file paths
        """
        mri_dir_path = Path(mri_dir)
        if not mri_dir_path.exists():
            raise FileNotFoundError(f"MRI directory not found: {mri_dir}")
        
        mri_files = []
        for ext in self.supported_extensions:
            pattern = f"*{ext}"
            files = list(mri_dir_path.glob(pattern))
            mri_files.extend(files)
        
        # Sort files for consistent ordering
        mri_files.sort()
        
        logger.info(f"Found {len(mri_files)} MRI files in {mri_dir}")
        return mri_files
    
    def extract_subject_id_from_filename(self, filename: str) -> str:
        """
        Extract subject ID from filename using robust pattern matching
        
        Args:
            filename: Name of the MRI file
        
        Returns:
            str: Extracted subject ID (e.g., MRB_0134)
        """
        # Remove extension first to get clean filename
        base_name = filename
        for ext in self.supported_extensions:
            if filename.endswith(ext):
                base_name = filename[:-len(ext)]
                break
        
        # Try multiple patterns to extract subject ID
        patterns = [
            r'(MRB_\d{4})',  # Match MRB_#### pattern
            r'([A-Z]+_\d{4})',  # Match any letters_#### pattern
            r'([A-Z]+\d{4})',  # Match any letters#### pattern (no underscore)
            r'^([A-Za-z0-9_]+)',  # Take everything until first non-alphanumeric/underscore
        ]
        
        for pattern in patterns:
            match = re.search(pattern, base_name)
            if match:
                subject_id = match.group(1)
                logger.debug(f"Extracted subject ID '{subject_id}' from filename '{filename}' using pattern '{pattern}'")
                return subject_id
        
        # Fallback: take first 8 characters if no pattern matches
        subject_id = base_name[:8] if len(base_name) >= 8 else base_name
        logger.warning(f"No pattern matched for '{filename}', using fallback: '{subject_id}'")
        return subject_id
    
    def load_csv_data(self, csv_file: str) -> Dict[str, Dict[str, any]]:
        """
        Load patient data from CSV file
        
        Args:
            csv_file: Path to CSV file with Subject ID, Age, SEX columns
        
        Returns:
            Dict: Dictionary mapping subject_id to {age, sex}
        """
        try:
            # Try different separators and header configurations
            separators = ['\t', ',', ';']
            df = None
            
            # First, try reading with headers
            for sep in separators:
                try:
                    df_with_header = pd.read_csv(csv_file, sep=sep, encoding='utf-8')
                    logger.info(f"Trying with header, separator '{sep}': shape={df_with_header.shape}, columns={list(df_with_header.columns)}")
                    
                    # Check if this looks like it has actual data (not just headers)
                    if len(df_with_header) > 0 and len(df_with_header.columns) >= 3:
                        df = df_with_header
                        logger.info(f"Successfully read CSV with header and separator '{sep}'")
                        break
                except Exception as e:
                    logger.debug(f"Failed to read with header, separator '{sep}': {e}")
                    continue
            
            # If no success with headers or no data rows, try without header
            if df is None or len(df) == 0:
                logger.info("Trying to read CSV without header...")
                for sep in separators:
                    try:
                        df_no_header = pd.read_csv(csv_file, sep=sep, header=None, encoding='utf-8')
                        logger.info(f"Trying without header, separator '{sep}': shape={df_no_header.shape}")
                        
                        if len(df_no_header) > 0 and len(df_no_header.columns) >= 3:
                            df = df_no_header
                            df.columns = ['subject_id', 'age', 'sex'] + [f'col_{i}' for i in range(3, len(df.columns))]
                            logger.info(f"Successfully read CSV without header using separator '{sep}'")
                            break
                    except Exception as e:
                        logger.debug(f"Failed to read without header, separator '{sep}': {e}")
                        continue
            
            if df is None or len(df) == 0:
                raise ValueError("Could not read CSV file with any separator or no data found")
            
            # Clean column names (remove whitespace and normalize)
            if hasattr(df.columns, 'str'):
                df.columns = df.columns.str.strip()
            
            logger.info(f"CSV columns found: {list(df.columns)}")
            logger.info(f"CSV shape: {df.shape}")
            
            # Special handling for files that look like they have data as column names
            # If we have 3 columns and shape is (0, 3), the data became column headers
            if len(df) == 0 and len(df.columns) == 3:
                # Try to detect if columns contain actual data
                potential_data = list(df.columns)
                if (any(re.match(r'MRB_\d{4}', str(col)) for col in potential_data) and 
                    any(str(col).isdigit() or (str(col).replace('.','').isdigit()) for col in potential_data) and
                    any(str(col).upper() in ['M', 'F', 'W', 'MALE', 'FEMALE', 'WOMEN', 'MEN'] for col in potential_data)):
                    
                    logger.info("Detected that column names are actually data - reconstructing DataFrame")
                    # Create new dataframe with this data as first row
                    df = pd.DataFrame([potential_data], columns=['subject_id', 'age', 'sex'])
                    logger.info(f"Reconstructed CSV shape: {df.shape}")
                    logger.info(f"Reconstructed data: {df.iloc[0].to_dict()}")
            
            # Create mapping from actual column names to standard names
            col_mapping = {}
            
            # If columns are already standardized (subject_id, age, sex)
            if 'subject_id' in df.columns and 'age' in df.columns and 'sex' in df.columns:
                col_mapping = {'subject_id': 'subject_id', 'age': 'age', 'sex': 'sex'}
                logger.info("Using standard column names")
            else:
                # Try to map columns by name similarity
                actual_cols_lower = [col.lower() for col in df.columns]
                
                # Map subject ID column
                subject_id_candidates = ['subject id', 'subject_id', 'subjectid', 'id', 'patient_id', 'patient id', 'subject', 'participant_id']
                for candidate in subject_id_candidates:
                    if candidate.lower() in actual_cols_lower:
                        idx = actual_cols_lower.index(candidate.lower())
                        col_mapping['subject_id'] = df.columns[idx]
                        break
                
                # If no specific column found, use first column
                if 'subject_id' not in col_mapping:
                    col_mapping['subject_id'] = df.columns[0]
                    logger.warning(f"No subject ID column found, using first column: {col_mapping['subject_id']}")
                
                # Map age column
                age_candidates = ['age']
                for candidate in age_candidates:
                    if candidate.lower() in actual_cols_lower:
                        idx = actual_cols_lower.index(candidate.lower())
                        col_mapping['age'] = df.columns[idx]
                        break
                
                # If no age column found, use second column if available
                if 'age' not in col_mapping and len(df.columns) > 1:
                    col_mapping['age'] = df.columns[1]
                    logger.warning(f"No age column found, using second column: {col_mapping['age']}")
                
                # Map sex column
                sex_candidates = ['sex', 'gender']
                for candidate in sex_candidates:
                    if candidate.lower() in actual_cols_lower:
                        idx = actual_cols_lower.index(candidate.lower())
                        col_mapping['sex'] = df.columns[idx]
                        break
                
                # If no sex column found, use third column if available
                if 'sex' not in col_mapping and len(df.columns) > 2:
                    col_mapping['sex'] = df.columns[2]
                    logger.warning(f"No sex column found, using third column: {col_mapping['sex']}")
            
            logger.info(f"Column mapping: {col_mapping}")
            
            # Convert to dictionary using the mapped column names
            patient_data = {}
            for _, row in df.iterrows():
                # Get subject ID from CSV
                if 'subject_id' in col_mapping:
                    subject_id = str(row[col_mapping['subject_id']]).strip()
                    
                    # Try to extract MRB_#### pattern from the subject ID
                    patterns = [
                        r'(MRB_\d{4})',  # Match MRB_#### pattern
                        r'([A-Z]+_\d{4})',  # Match any letters_#### pattern
                        r'([A-Z]+\d{4})',  # Match any letters#### pattern (no underscore)
                    ]
                    
                    extracted_id = subject_id
                    for pattern in patterns:
                        match = re.search(pattern, subject_id)
                        if match:
                            extracted_id = match.group(1)
                            break
                    
                    # Handle gender mapping: W -> F, M -> M
                    if 'sex' in col_mapping:
                        gender_value = str(row[col_mapping['sex']]).strip().upper()
                        if gender_value == 'W':
                            sex = 'F'  # Women -> Female
                        elif gender_value == 'M':
                            sex = 'M'  # Men -> Male
                        else:
                            sex = gender_value if pd.notna(row[col_mapping['sex']]) else None
                    else:
                        sex = None
                    
                    # Get age
                    if 'age' in col_mapping:
                        try:
                            age_val = row[col_mapping['age']]
                            age = int(float(age_val)) if pd.notna(age_val) and str(age_val).replace('.','').isdigit() else None
                        except (ValueError, TypeError):
                            age = None
                    else:
                        age = None
                    
                    patient_data[extracted_id] = {
                        'age': age,
                        'sex': sex
                    }
                    logger.info(f"Loaded subject: {extracted_id}, Age: {age}, Sex: {sex}")
            
            logger.info(f"Loaded data for {len(patient_data)} subjects from CSV")
            logger.info(f"Subject IDs in CSV: {list(patient_data.keys())}")
            return patient_data
            
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            logger.error(f"CSV file path: {csv_file}")
            return {}
    
    def generate_metadata(self, mri_dir: str, csv_file: Optional[str] = None, 
                         preview: bool = False) -> Dict:
        """
        Generate metadata for MRI files
        
        Args:
            mri_dir: Directory containing MRI files
            csv_file: Optional CSV file with patient data
            preview: If True, only show what would be processed
        
        Returns:
            Dict: Metadata dictionary in the specified format
        """
        try:
            # Find MRI files
            mri_files = self.find_mri_files(mri_dir)
            
            if not mri_files:
                logger.warning("No MRI files found")
                return {"metadata": {"patient ids": [], "age": [], "Sex": []}}
            
            # Load CSV data if provided
            csv_data = {}
            if csv_file and os.path.exists(csv_file):
                csv_data = self.load_csv_data(csv_file)
            elif csv_file:
                logger.warning(f"CSV file not found: {csv_file}")
            
            # Extract subject IDs and match with CSV data
            patient_ids = []
            ages = []
            sexes = []
            
            processed_subjects = set()  # Avoid duplicates
            
            logger.info(f"Processing {len(mri_files)} MRI files...")
            
            for mri_file in mri_files:
                logger.info(f"Processing file: {mri_file.name}")
                
                # Extract subject ID from filename
                subject_id = self.extract_subject_id_from_filename(mri_file.name)
                
                if not subject_id:
                    logger.warning(f"Could not extract subject ID from: {mri_file.name}")
                    continue
                
                logger.info(f"Extracted subject ID: '{subject_id}' from file: {mri_file.name}")
                
                # Avoid duplicate subject IDs
                if subject_id in processed_subjects:
                    logger.info(f"Duplicate subject ID found, skipping: {subject_id}")
                    continue
                
                processed_subjects.add(subject_id)
                patient_ids.append(subject_id)
                
                # Get age and sex from CSV if available
                if subject_id in csv_data:
                    age = csv_data[subject_id]['age']
                    sex = csv_data[subject_id]['sex']
                    logger.info(f"Found CSV data for {subject_id}: Age={age}, Sex={sex}")
                else:
                    logger.warning(f"No CSV data found for subject: {subject_id}")
                    age = None
                    sex = None
                
                ages.append(age)
                sexes.append(sex)
            
            # Create metadata structure
            metadata = {
                "metadata": {
                    "patient ids": patient_ids,
                    "age": ages,
                    "Sex": sexes
                }
            }
            
            logger.info(f"Final metadata contains {len(patient_ids)} subjects:")
            for i, pid in enumerate(patient_ids):
                logger.info(f"  {pid}: Age={ages[i]}, Sex={sexes[i]}")
            
            if preview:
                logger.info("PREVIEW MODE - Extracted subject information:")
                logger.info(f"Found {len(patient_ids)} unique subjects:")
                for i, pid in enumerate(patient_ids):
                    logger.info(f"  {pid}: Age={ages[i]}, Sex={sexes[i]}")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error generating metadata: {e}")
            return {"metadata": {"patient ids": [], "age": [], "Sex": []}}
    
    def save_metadata(self, metadata: Dict, output_file: str):
        """
        Save metadata to JSON file
        
        Args:
            metadata: Metadata dictionary
            output_file: Output JSON file path
        """
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Metadata saved to: {output_file}")
            
            # Log what was saved
            patient_ids = metadata.get("metadata", {}).get("patient ids", [])
            logger.info(f"Saved metadata for subjects: {patient_ids}")
            
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def run_metadata_generation(self, mri_dir: str, output_file: str = "metadata.json", 
                              csv_file: Optional[str] = None, preview: bool = False) -> bool:
        """
        Run complete metadata generation process
        
        Args:
            mri_dir: Directory containing MRI files
            output_file: Output JSON file path
            csv_file: Optional CSV file with subject info
            preview: Preview mode flag
        
        Returns:
            bool: Success status
        """
        try:
            logger.info("Starting metadata generation...")
            logger.info(f"MRI directory: {mri_dir}")
            logger.info(f"CSV file: {csv_file}")
            logger.info(f"Output file: {output_file}")
            
            # Generate metadata
            metadata = self.generate_metadata(mri_dir, csv_file, preview)
            
            if not preview:
                # Save to file
                self.save_metadata(metadata, output_file)
            
            patient_count = len(metadata.get("metadata", {}).get("patient ids", []))
            logger.info(f"Metadata generation completed. Processed {patient_count} subjects.")
            
            return True
            
        except Exception as e:
            logger.error(f"Metadata generation failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

def main():
    parser = argparse.ArgumentParser(
        description="MRI Metadata Generator v0.04 - Generate metadata from MRI files and CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate metadata with CSV data
    python metadata_generator.py --mri_dir /path/to/mri/files --output metadata.json --csv subjects.csv
    
    # Preview mode - see what subject IDs will be extracted
    python metadata_generator.py --mri_dir /path/to/mri/files --csv subjects.csv --preview
    
    # Generate metadata without CSV (ages and sex will be null)
    python metadata_generator.py --mri_dir /path/to/mri/files --output metadata.json
    
CSV file should have columns: Subject ID (or similar), Age, SEX
- SEX values: W (women) -> F, M (men) -> M
- Subject IDs should match MRI filename patterns (e.g., MRB_0134)

Example CSV content:
Subject ID,Age,SEX
MRB_0003,18,M
MRB_0004,27,W

The script will automatically detect:
- Tab-separated or comma-separated values
- Files with or without headers
- Various column name formats
- Subject ID patterns in filenames
        """
    )
    
    parser.add_argument('--mri_dir', type=str, required=True,
                       help='Directory containing MRI files (supports .nii.gz, .nii, .dcm)')
    parser.add_argument('--output', type=str, default='metadata.json',
                       help='Output JSON file (default: metadata.json)')
    parser.add_argument('--csv', type=str,
                       help='CSV file with subject info (Subject ID, Age, SEX columns)')
    parser.add_argument('--preview', action='store_true',
                       help='Preview extracted subject IDs without saving file')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize generator
        generator = MRIMetadataGenerator()
        
        # Run metadata generation
        success = generator.run_metadata_generation(
            args.mri_dir, args.output, args.csv, args.preview
        )
        
        if success:
            if not args.preview:
                logger.info(f"Metadata generation completed successfully!")
                logger.info(f"Output saved to: {args.output}")
            else:
                logger.info("Preview completed successfully!")
            sys.exit(0)
        else:
            logger.error("Metadata generation failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Process failed with error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()