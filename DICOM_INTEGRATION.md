# DICOM Integration Guide

## Overview

AtrofIQ now includes comprehensive DICOM (Digital Imaging and Communications in Medicine) support through integration with Orthanc, a lightweight DICOM server. This allows users to upload, store, view, and manage DICOM files within the AtrofIQ workflow.

## Features

### Backend Features
- **Automatic DICOM Detection**: Files are automatically detected as DICOM during upload
- **Dual Storage**: DICOM files are stored both in Orthanc (for DICOM operations) and MinIO (for compatibility)
- **DICOM Metadata Extraction**: Patient information, study details, and modality data are extracted
- **REST API**: Complete API for managing DICOM studies, series, and instances

### Frontend Features
- **DICOM Viewer**: Interactive viewer to browse DICOM files by study and series
- **Preview Images**: PNG previews of DICOM instances (when supported by Orthanc)
- **File Downloads**: Download individual DICOM files
- **Study Information**: View patient details, study dates, and modality information

## API Endpoints

### Orthanc Status
- `GET /orthanc/status` - Check Orthanc server availability

### Studies Management
- `GET /orthanc/studies` - List all DICOM studies in Orthanc
- `GET /orthanc/studies/{study_id}` - Get detailed study information including series

### Instance Operations
- `GET /orthanc/instances/{instance_id}/file` - Download a DICOM file
- `GET /orthanc/instances/{instance_id}/preview` - Get PNG preview of DICOM instance

### Folder Integration
- `GET /folders/{folder}/dicom-studies` - Get DICOM studies associated with a specific upload folder

## Usage

### Uploading DICOM Files

1. Navigate to the worklist page
2. Click "Upload Study Files"
3. Select one or more DICOM files (.dcm, .dicom extensions)
4. Optionally provide patient age and gender
5. Click "Upload to Worklist"

The system will:
- Automatically detect DICOM files
- Upload them to Orthanc server
- Store them in MinIO for backup
- Extract and store metadata

### Viewing DICOM Files

1. In the worklist, click the purple "DICOM" button for any study
2. The DICOM viewer will open showing:
   - List of all DICOM files in the study
   - Patient information and study details
   - Preview images (if available)
   - Download buttons for individual files

### DICOM File Information Displayed

- **Patient Information**: Patient ID, Name
- **Study Information**: Study Date, Study Description, Study Instance UID
- **Series Information**: Series Description, Modality, Series Number
- **Instance Information**: SOP Instance UID, Instance status

## Configuration

### Environment Variables

```bash
# Orthanc Configuration
ORTHANC_ENDPOINT=orthanc:8042           # Orthanc server endpoint
ORTHANC_USERNAME=orthanc                # Orthanc username
ORTHANC_PASSWORD=orthanc                # Orthanc password
ORTHANC_ENABLED=true                    # Enable/disable Orthanc integration
```

### Docker Compose Integration

The Orthanc server is automatically configured in the docker-compose.yml:

```yaml
orthanc:
  image: orthancteam/orthanc:latest
  container_name: atrofiq_orthanc
  ports:
    - "8042:8042"  # Web UI and REST API
    - "4242:4242"  # DICOM protocol
  volumes:
    - orthanc_data:/var/lib/orthanc/db
    - ./orthanc-config:/etc/orthanc
```

### Orthanc Web Interface

Access the Orthanc web interface directly at: http://localhost:8042
- Username: orthanc
- Password: orthanc

## File Support

### Supported Formats
- **.dcm** - Standard DICOM files
- **.dicom** - DICOM files with .dicom extension
- **Content-based detection** - Files are validated by content, not just extension

### ZIP File Support
DICOM files within ZIP archives are automatically extracted and processed during upload.

## Error Handling

### Orthanc Unavailable
If Orthanc is not available during upload:
- DICOM files are still stored in MinIO
- A warning is logged but upload continues
- Frontend shows "Orthanc unavailable - limited functionality"

### File Processing Errors
- Invalid DICOM files are skipped with warning messages
- Upload continues for valid files
- Error details are available in server logs

## Troubleshooting

### DICOM Files Not Appearing
1. Check Orthanc server status: `GET /orthanc/status`
2. Verify files are valid DICOM format
3. Check server logs for upload errors
4. Ensure Orthanc container is running: `docker ps`

### Preview Images Not Loading
1. Verify Orthanc is available
2. Check if the DICOM file contains image data
3. Some DICOM files (non-image) don't support previews

### Download Issues
1. Ensure browser allows downloads
2. Check if instance ID is valid
3. Verify Orthanc connectivity

## Integration with Existing Workflow

### Backward Compatibility
- Non-DICOM files (.nii, .nii.gz) continue to work as before
- Existing MinIO storage is preserved
- Database structure remains compatible

### Processing Pipeline
- DICOM files bypass automatic MRI inference processing
- They require different processing workflows
- Manual analysis tools are available through the DICOM viewer

## Security Considerations

### Access Control
- Orthanc uses HTTP Basic Authentication
- Default credentials should be changed in production
- API endpoints require backend authentication

### Data Privacy
- Patient information is extracted and stored
- Consider HIPAA compliance requirements
- Implement appropriate access logging

## Performance

### File Size Limits
- No specific DICOM file size limits (handled by MinIO)
- Large studies with many instances may take time to load
- Consider implementing pagination for large studies

### Storage Usage
- Files are stored in both Orthanc and MinIO
- Monitor disk usage for both services
- Orthanc provides compression options

## Development

### Adding New DICOM Features

1. **Backend**: Extend `dicom_utils.py` for new DICOM operations
2. **API**: Add new endpoints in `main.py`
3. **Frontend**: Extend `DicomViewer.js` for new UI features

### Testing DICOM Functionality

1. Upload test DICOM files
2. Verify files appear in Orthanc web interface
3. Test DICOM viewer functionality
4. Check API endpoints with curl/Postman

Example API test:
```bash
curl -X GET "http://localhost:7000/orthanc/studies" 
curl -X GET "http://localhost:7000/folders/study-20240101-120000/dicom-studies"
```

## Migration

### Existing Installations
- Run `docker-compose up --build` to add Orthanc service
- No database migrations required
- Existing studies remain accessible

### Data Migration
- Previous uploads without DICOM support remain in MinIO
- New uploads will have DICOM integration
- Manual migration scripts can be developed if needed