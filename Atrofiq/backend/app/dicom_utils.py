"""
DICOM utilities for Orthanc integration.
Handles DICOM file detection and upload to Orthanc server.
"""

import os
import tempfile
import logging
from typing import Dict, Optional, Tuple
from pathlib import Path

import requests
from requests.auth import HTTPBasicAuth
import pydicom
from pydicom.errors import InvalidDicomError


logger = logging.getLogger(__name__)


class OrthancClient:
    """Client for interacting with Orthanc DICOM server."""
    
    def __init__(self, endpoint: str, username: str, password: str):
        """Initialize Orthanc client.
        
        Args:
            endpoint: Orthanc server endpoint (e.g., 'localhost:8042')
            username: Orthanc username
            password: Orthanc password
        """
        self.base_url = f"http://{endpoint}"
        self.auth = HTTPBasicAuth(username, password)
        self.session = requests.Session()
        self.session.auth = self.auth
        
    def is_available(self) -> bool:
        """Check if Orthanc server is available."""
        try:
            response = self.session.get(f"{self.base_url}/system")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Orthanc server not available: {e}")
            return False
    
    def upload_dicom(self, dicom_data: bytes, metadata: Optional[Dict] = None) -> Optional[str]:
        """Upload DICOM file to Orthanc.
        
        Args:
            dicom_data: Raw DICOM file data
            metadata: Optional metadata to store with the DICOM
            
        Returns:
            Instance ID if successful, None otherwise
        """
        try:
            # Upload DICOM file
            response = self.session.post(
                f"{self.base_url}/instances",
                data=dicom_data,
                headers={'Content-Type': 'application/dicom'}
            )
            
            if response.status_code == 200:
                result = response.json()
                instance_id = result.get('ID')
                
                # Add custom metadata if provided
                if metadata and instance_id:
                    self._add_metadata(instance_id, metadata)
                
                logger.info(f"Successfully uploaded DICOM to Orthanc, instance ID: {instance_id}")
                return instance_id
            else:
                logger.error(f"Failed to upload DICOM to Orthanc: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error uploading DICOM to Orthanc: {e}")
            return None
    
    def _add_metadata(self, instance_id: str, metadata: Dict) -> None:
        """Add custom metadata to a DICOM instance."""
        try:
            # Add AtrofIQ-specific metadata using custom tags
            if 'study_folder' in metadata:
                self.session.put(
                    f"{self.base_url}/instances/{instance_id}/metadata/1025",  # AtrofiqFolderId
                    data=metadata['study_folder'].encode('utf-8')
                )
            
            if 'uploaded_by' in metadata:
                self.session.put(
                    f"{self.base_url}/instances/{instance_id}/metadata/1026",  # AtrofiqUploadedBy
                    data=metadata['uploaded_by'].encode('utf-8')
                )
                
        except Exception as e:
            logger.warning(f"Failed to add metadata to DICOM instance {instance_id}: {e}")
    
    def get_instance_info(self, instance_id: str) -> Optional[Dict]:
        """Get information about a DICOM instance."""
        try:
            response = self.session.get(f"{self.base_url}/instances/{instance_id}")
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.error(f"Error getting instance info: {e}")
            return None


def is_dicom_file(file_data: bytes, filename: str = None) -> Tuple[bool, Optional[Dict]]:
    """Check if file data represents a DICOM file.
    
    Args:
        file_data: Raw file data
        filename: Optional filename for additional context
        
    Returns:
        Tuple of (is_dicom: bool, dicom_info: Optional[Dict])
    """
    try:
        # Write to temporary file and try to read with pydicom
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file_data)
            temp_path = temp_file.name
        
        try:
            # Try to read as DICOM
            ds = pydicom.dcmread(temp_path, force=True)
            
            # Extract basic DICOM info
            dicom_info = {
                'patient_id': getattr(ds, 'PatientID', 'Unknown'),
                'patient_name': str(getattr(ds, 'PatientName', 'Unknown')),
                'study_date': getattr(ds, 'StudyDate', None),
                'study_time': getattr(ds, 'StudyTime', None),
                'study_description': getattr(ds, 'StudyDescription', None),
                'series_description': getattr(ds, 'SeriesDescription', None),
                'modality': getattr(ds, 'Modality', None),
                'sop_instance_uid': getattr(ds, 'SOPInstanceUID', None),
                'study_instance_uid': getattr(ds, 'StudyInstanceUID', None),
                'series_instance_uid': getattr(ds, 'SeriesInstanceUID', None),
            }
            
            logger.info(f"Detected DICOM file: {filename or 'unknown'}")
            return True, dicom_info
            
        except (InvalidDicomError, Exception) as e:
            # Also check by file extension as fallback
            if filename and filename.lower().endswith(('.dcm', '.dicom')):
                logger.warning(f"File {filename} has DICOM extension but failed DICOM parsing: {e}")
                return True, {'error': f'DICOM parsing failed: {str(e)}'}
            return False, None
            
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass
                
    except Exception as e:
        logger.error(f"Error checking if file is DICOM: {e}")
        return False, None


def get_orthanc_client() -> Optional[OrthancClient]:
    """Get configured Orthanc client from environment variables."""
    endpoint = os.getenv("ORTHANC_ENDPOINT")
    username = os.getenv("ORTHANC_USERNAME", "orthanc")
    password = os.getenv("ORTHANC_PASSWORD", "orthanc")
    
    if not endpoint:
        logger.warning("ORTHANC_ENDPOINT not configured")
        return None
        
    return OrthancClient(endpoint, username, password)