# Enhanced MRI Processing Pipeline v.0.01

## Overview

The Enhanced MRI Processing Pipeline v.0.01 is a comprehensive, automated pipeline for processing MRI brain scans. It integrates multiple neuroimaging analysis steps into a single workflow, including metadata generation, brain segmentation, brain age prediction, and normative modeling.

## Pipeline Components

The pipeline consists of four main processing steps:

1. **Metadata Generation** - Extracts and organizes patient information from MRI files and CSV data
2. **Brain Segmentation** - Performs automated brain segmentation using SynthSeg
3. **Brain Age Prediction** - Predicts brain age from volumetric features using trained models
4. **Normative Modeling** - Compares individual brain metrics against normative population data

## Prerequisites

### Directory Structure

The pipeline expects the following directory structure:

```
your_base_directory/
├── Pipeline.py          # Main pipeline script
├── Metadata_gen/
│   └── metadata_generator.py
├── Segmentation/
│   └── mri_pipeline_clean.py
|   ├── models/
│       ├── synthseg_2.0.h5
|       ├── synthseg_robust_2.0.h5
|       ├── synthseg_parc_2.0.h5
|       ├── synthseg_qc_2.0.h5
|       ├── *.npy
|       └── ...
├── BrainAge-Prediction/
│   ├── Inference.py
│   └── Trial_results/saved_models/
└── Normative Modeling/
    ├── API.py
    ├── dashboard.py
    ├── Plot_API.py
    └── Percentiles/
```
Google Drive ( Ask team lead for access) : [link](https://drive.google.com/drive/folders/1ME4PiizHXKgcMndFkqzGdDa5zn230583?usp=drive_link)
### Required Files

1. **MRI Directory**: Directory containing `.nii` format MRI files
2. **CSV File**: Tab-separated file with patient information in format:
   ```
   MRB_0097	27	W
   MRB_0099	20	F
   ```
   Columns: `subject_id`, `age`, `sex`

### Dependencies

- Python 3.7+
- pandas
- pathlib
- subprocess
- logging
- json

## Usage

### Basic Usage

```bash
# Run complete pipeline (all 4 steps)
python Pipeline.py --mri_dir /path/to/mri/files --csv patient_info.csv

# Specify custom output directory
python Pipeline.py --mri_dir /path/to/mri/files --csv patient_info.csv --output_dir my_results
```

### Advanced Options

```bash
# Run with custom number of threads and percentiles
python Pipeline.py \
    --mri_dir /path/to/mri/files \
    --csv patient_info.csv \
    --threads 50 \
    --percentiles 10 25 50 75 90

# Skip certain steps
python Pipeline.py \
    --mri_dir /path/to/mri/files \
    --csv patient_info.csv \
    --no_brainage \
    --no_normative
```

### Individual Step Execution

Run only specific pipeline steps:

```bash
# Only metadata generation
python Pipeline.py --metadata_only --mri_dir /path/to/mri --csv patient_info.csv

# Only segmentation
python Pipeline.py --segmentation_only --mri_dir /path/to/mri --threads 90

# Only brain age prediction
python Pipeline.py --brainage_only --volumes volumes.csv --metadata metadata.json

# Only normative modeling
python Pipeline.py --normative_only --metadata metadata.json
```

## Command Line Arguments

### Required Arguments (for full pipeline)
- `--mri_dir`: Directory containing MRI files (.nii format)
- `--csv`: CSV file with patient information

### Optional Arguments
- `--output_dir`: Output directory (default: `New_results_all`)
- `--threads`: Number of threads for segmentation (default: 90)
- `--percentiles`: Percentiles for normative modeling (default: 25 50 75)
- `--base_dir`: Base directory containing module folders (default: current directory)

### Pipeline Control
- `--no_parc`: Disable parcellation in segmentation
- `--no_brainage`: Skip brain age prediction
- `--no_normative`: Skip normative modeling

### Individual Step Options
- `--metadata_only`: Run only metadata generation
- `--segmentation_only`: Run only segmentation
- `--brainage_only`: Run only brain age prediction
- `--normative_only`: Run only normative modeling

### Step-Specific Inputs
- `--volumes`: Volumes CSV file (for brain age only)
- `--metadata`: Metadata JSON file (for brain age/normative only)
- `--models_dir`: Brain age models directory
- `--importance_folder`: Feature importance folder path
- `--percentiles_folder`: Percentiles folder path

## Output Structure

The pipeline creates a comprehensive output directory structure:

```
New_results_all/                     # Main output directory
├── metadata.json                    # Patient metadata
├── volumes.csv                      # Brain region volumes
├── qc_scores.csv                   # Quality control scores
├── brain_age_results.csv           # Brain age predictions
├── segmentations/                  # Individual segmentation results
│   ├── participant_001/
│   ├── participant_002/
│   └── ...
├── normative_results/              # Normative modeling results
│   ├── participant_001_normative_results.json
│   ├── participant_002_normative_results.json
│   ├── ...
│   └── normative_modeling_summary.json
└── enhanced_pipeline_v2.log       # Processing log
```

### Key Output Files

1. **metadata.json**: Contains structured patient information and IDs
2. **volumes.csv**: Brain region volumes for all participants
3. **qc_scores.csv**: Quality control metrics for each scan
4. **brain_age_results.csv**: Predicted vs chronological age data
5. **normative_results/**: Individual normative modeling results for each participant
6. **normative_modeling_summary.json**: Overall success/failure summary

## Processing Steps Detail

### Step 1: Metadata Generation
- Reads MRI files and patient CSV
- Creates structured metadata JSON
- Maps patient IDs to MRI files
- Validates data consistency

### Step 2: Brain Segmentation
- Performs automated brain segmentation
- Extracts volumetric measurements
- Generates quality control metrics
- Supports optional parcellation

### Step 3: Brain Age Prediction
- Uses trained models to predict brain age
- Compares predicted vs chronological age
- Generates brain age gap metrics
- Provides confidence intervals

### Step 4: Normative Modeling
- Compares individual metrics to population norms
- Calculates percentile rankings
- Identifies outliers and abnormalities
- Provides feature importance analysis

## Error Handling and Monitoring

The pipeline includes comprehensive error handling:

- **Logging**: All steps logged to `enhanced_pipeline_v2.log`
- **Validation**: Input file format and existence checking
- **Recovery**: Individual participant failures don't stop pipeline
- **Summary**: Success/failure rates reported at completion
- **Error Files**: Individual error logs saved for failed participants

## Performance Considerations

### Resource Usage
- **Memory**: Depends on MRI file sizes and batch processing
- **CPU**: Configurable threading (default: 90 threads for segmentation)
- **Storage**: Requires significant space for intermediate and final results
- **Time**: Full pipeline typically takes several hours for large datasets

### Optimization Tips
- Adjust `--threads` based on available CPU cores
- Monitor disk space before processing large datasets
- Use `--no_parc` to speed up segmentation if parcellation not needed
- Process subsets using individual step modes for debugging

### Debug Mode

For detailed debugging, check the log file:
```bash
tail -f enhanced_pipeline_v2.log
```

### Individual Step Testing

Test each step individually to isolate issues:
```bash
# Test metadata generation
python Pipeline.py --metadata_only --mri_dir test_data --csv test.csv

# Test segmentation
python Pipeline.py --segmentation_only --mri_dir test_data
```

## Examples

### Basic Example
```bash
# Process MRI data with default settings
python Pipeline.py \
    --mri_dir ../SynthSeg/Post-contrast-Data/ \
    --csv ../SynthSeg/Post-contrast-subjects.csv
```

### Advanced Example
```bash
# Custom configuration with specific percentiles and threading
python Pipeline.py \
    --mri_dir /data/mri_scans/ \
    --csv patient_data.csv \
    --output_dir results_2024 \
    --threads 50 \
    --percentiles 5 25 50 75 95 \
    --importance_folder /models/feature_importance \
    --percentiles_folder /models/percentiles
```

### Partial Processing Example
```bash
# Run everything except normative modeling
python Pipeline.py \
    --mri_dir /data/mri_scans/ \
    --csv patient_data.csv \
    --no_normative
```

## Support

For issues or questions:
1. Check the log file: `enhanced_pipeline_v2.log`
2. Verify input file formats and directory structure
3. Test individual pipeline steps to isolate problems
4. Ensure all dependencies and module scripts are available
5. Run the `Dashboard.py` using `streamlit run V.0.01/Dashboard.py`. The directory is important.