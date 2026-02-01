"""
MRBrain Final Configuration
Centralized configuration for MRBrain_final integration
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

class MRBrainConfig:
    """Configuration class for MRBrain Final models"""
    
    def __init__(self):
        # Base directories
        self.base_dir = Path(__file__).parent
        self.mrbrain_final_dir = self.get_mrbrain_final_path()
        
        # Model paths
        self.brainage_dir = self.mrbrain_final_dir / "BrainAge-Prediction"
        self.normative_dir = self.mrbrain_final_dir / "Normative Modeling"
        
        # API Configuration
        self.api_host = os.getenv('MRBRAIN_API_HOST', '0.0.0.0')
        self.api_port = int(os.getenv('MRBRAIN_API_PORT', '8000'))
        self.api_base_url = os.getenv('MRBRAIN_API_URL', f'http://localhost:{self.api_port}')
        
        # Model specific configurations
        self.brainage_model_path = self.brainage_dir / "saved_models" / "brain_age_pipeline.pkl"
        self.normative_models_path = self.normative_dir / "Models"
        
        # Processing configurations
        self.processing_timeout = int(os.getenv('PROCESSING_TIMEOUT', '1800'))  # 30 minutes
        self.max_file_size_mb = int(os.getenv('MAX_FILE_SIZE_MB', '100'))
        self.temp_dir = Path(os.getenv('TEMP_DIR', '/tmp'))
        
        # Feature configurations
        self.required_brain_regions = [
            'left_cerebral_white_matter', 'left_cerebral_cortex', 'left_lateral_ventricle',
            'left_thalamus', 'left_caudate', 'left_putamen', 'left_pallidum',
            'left_hippocampus', 'left_amygdala',
            'right_cerebral_white_matter', 'right_cerebral_cortex', 'right_lateral_ventricle',
            'right_thalamus', 'right_caudate', 'right_putamen', 'right_pallidum',
            'right_hippocampus', 'right_amygdala'
        ]
        
        # Validation configurations
        self.min_age = float(os.getenv('MIN_AGE', '18'))
        self.max_age = float(os.getenv('MAX_AGE', '95'))
        self.supported_genders = ['M', 'F', 'MALE', 'FEMALE']
        
        # Logging configuration
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        self.enable_debug = os.getenv('ENABLE_DEBUG', 'false').lower() == 'true'
        
    def get_mrbrain_final_path(self) -> Path:
        """Get the path to MRBrain_final directory"""
        # Try different possible locations
        possible_paths = [
            Path(os.getenv('MRBRAIN_FINAL_PATH', '')),  # Explicit env var
            Path('/app/MRBrain_final'),                  # Docker path
            self.base_dir / 'MRBrain_final',             # Relative to config
            self.base_dir.parent / 'MRBrain_final',      # Parent directory
            Path.cwd() / 'MRBrain_final',                # Current working directory
        ]
        
        for path in possible_paths:
            if path.exists() and path.is_dir():
                return path
                
        # Fallback to first path even if it doesn't exist
        return possible_paths[1]  # Docker path as default
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate the configuration and return status"""
        status = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'paths': {}
        }
        
        # Check paths
        status['paths']['mrbrain_final'] = str(self.mrbrain_final_dir)
        status['paths']['brainage_dir'] = str(self.brainage_dir)
        status['paths']['normative_dir'] = str(self.normative_dir)
        
        if not self.mrbrain_final_dir.exists():
            status['errors'].append(f"MRBrain_final directory not found: {self.mrbrain_final_dir}")
            status['valid'] = False
        
        if not self.brainage_dir.exists():
            status['warnings'].append(f"BrainAge directory not found: {self.brainage_dir}")
        
        if not self.normative_dir.exists():
            status['warnings'].append(f"Normative directory not found: {self.normative_dir}")
        
        # Check model files
        if not self.brainage_model_path.exists():
            status['warnings'].append(f"BrainAge model file not found: {self.brainage_model_path}")
        
        if not self.normative_models_path.exists():
            status['warnings'].append(f"Normative models directory not found: {self.normative_models_path}")
        
        return status
    
    def get_feature_defaults(self) -> Dict[str, float]:
        """Get default values for missing brain region features"""
        # Based on typical adult brain volumes (in mm³)
        return {
            'left_cerebral_white_matter': 250000,
            'left_cerebral_cortex': 230000,
            'left_lateral_ventricle': 12000,
            'left_thalamus': 8500,
            'left_caudate': 3800,
            'left_putamen': 5200,
            'left_pallidum': 1800,
            'left_hippocampus': 4200,
            'left_amygdala': 1600,
            'right_cerebral_white_matter': 250000,
            'right_cerebral_cortex': 230000,
            'right_lateral_ventricle': 12000,
            'right_thalamus': 8500,
            'right_caudate': 3800,
            'right_putamen': 5200,
            'right_pallidum': 1800,
            'right_hippocampus': 4200,
            'right_amygdala': 1600,
        }
    
    def normalize_gender(self, gender: str) -> str:
        """Normalize gender input to M or F"""
        if not gender:
            raise ValueError("Gender is required")
            
        gender = gender.strip().upper()
        if gender in ['M', 'MALE']:
            return 'M'
        elif gender in ['F', 'FEMALE']:
            return 'F'
        else:
            raise ValueError(f"Invalid gender: {gender}. Must be one of {self.supported_genders}")
    
    def validate_age(self, age: float) -> float:
        """Validate age input"""
        if not isinstance(age, (int, float)):
            try:
                age = float(age)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid age: {age}. Must be a number")
        
        if age < self.min_age or age > self.max_age:
            raise ValueError(f"Age {age} is outside valid range ({self.min_age}-{self.max_age})")
        
        return float(age)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'base_dir': str(self.base_dir),
            'mrbrain_final_dir': str(self.mrbrain_final_dir),
            'brainage_dir': str(self.brainage_dir),
            'normative_dir': str(self.normative_dir),
            'api_host': self.api_host,
            'api_port': self.api_port,
            'api_base_url': self.api_base_url,
            'brainage_model_path': str(self.brainage_model_path),
            'normative_models_path': str(self.normative_models_path),
            'processing_timeout': self.processing_timeout,
            'max_file_size_mb': self.max_file_size_mb,
            'temp_dir': str(self.temp_dir),
            'min_age': self.min_age,
            'max_age': self.max_age,
            'supported_genders': self.supported_genders,
            'log_level': self.log_level,
            'enable_debug': self.enable_debug,
        }

# Global configuration instance
config = MRBrainConfig()

# Environment-specific configurations
def get_docker_config() -> Dict[str, str]:
    """Get Docker-specific environment variables"""
    return {
        'MRBRAIN_FINAL_PATH': '/app/MRBrain_final',
        'MRBRAIN_API_HOST': '0.0.0.0',
        'MRBRAIN_API_PORT': '8000',
        'TEMP_DIR': '/tmp',
        'LOG_LEVEL': 'INFO',
    }

def get_development_config() -> Dict[str, str]:
    """Get development environment configuration"""
    return {
        'MRBRAIN_FINAL_PATH': str(Path.cwd() / 'MRBrain_final'),
        'MRBRAIN_API_HOST': 'localhost',
        'MRBRAIN_API_PORT': '8000',
        'LOG_LEVEL': 'DEBUG',
        'ENABLE_DEBUG': 'true',
    }