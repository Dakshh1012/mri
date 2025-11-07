
# MRI Metadata Generator v0.04

A Python tool that generates metadata for MRI files by extracting patient IDs from filenames and matching them with patient information from CSV files.

---

## Project Structure

```
metadata_gen/
│
├── metadata_generator.py    # Main metadata generator script
└── README.md               # Documentation
```

---

## Features

- **Automatic patient ID extraction** from MRI filenames (supports `.nii.gz`, `.nii`, `.dcm`)
- **Flexible CSV parsing** with automatic detection of separators and column formats
- **Robust pattern matching** for subject IDs (MRB_####, etc.)
- **Preview mode** to verify extracted patient IDs before processing
- **Comprehensive logging** with detailed processing information
- **Error handling** for malformed files and missing data

---

## System Requirements

- Python 3.8+
- Dependencies:
  ```bash
  pip install pandas
  ```

---

## Usage

### 1. Generate Metadata with CSV Data

```bash
python metadata_generator.py --mri_dir /path/to/mri/files --output metadata.json --csv subjects.csv
```

### 2. Preview Mode (Recommended First Step)

Check which patient IDs will be extracted without generating the file:

```bash
python metadata_generator.py --mri_dir /path/to/mri/files --csv subjects.csv --preview
```

### 3. Generate Metadata Without CSV

Creates metadata with null age/sex values:

```bash
python metadata_generator.py --mri_dir /path/to/mri/files --output metadata.json
```

### 4. Debug Mode

Enable detailed logging for troubleshooting:

```bash
python metadata_generator.py --mri_dir /path/to/mri/files --csv subjects.csv --debug
```

---

## CSV File Format

The script automatically detects various CSV formats. Supported formats include:

**Standard format:**
```csv
Subject ID,Age,SEX
MRB_0003,18,M
MRB_0004,27,W
```

**Alternative formats:**
- Tab-separated values
- Files without headers
- Various column names (subject_id, patient_id, id, etc.)
- Gender codes: W (women) → F, M (men) → M

---

## Arguments

- `--mri_dir` : Directory containing MRI files (required)
- `--output` : Output JSON file path (default: metadata.json)
- `--csv` : CSV file with subject information
- `--preview` : Preview extracted subject IDs without saving
- `--debug` : Enable debug logging

---

## Output Format

Generated `metadata.json`:

```json
{
  "metadata": {
    "patient ids": ["MRB_0003", "MRB_0004", "MRB_0007"],
    "age": [18, 27, 35],
    "Sex": ["M", "F", "M"]
  }
}
```

---

## Patient ID Extraction

The script uses multiple patterns to extract patient IDs from filenames:

1. `MRB_####` pattern (primary)
2. `LETTERS_####` pattern
3. `LETTERS####` pattern (no underscore)
4. First alphanumeric sequence (fallback)

Example filename extractions:
- `MRB_0134.nii.gz` → `MRB_0134`
- `subject_MRB_0134_scan.nii` → `MRB_0134`
- `PATIENT0001.dcm` → `PATIENT0001`

---

## Logging

The script generates detailed logs in `metadata_generator.log` including:
- File processing status
- Patient ID extractions
- CSV matching results
- Error messages and warnings

---

## Error Handling

- Handles malformed CSV files with automatic format detection
- Continues processing when individual files fail
- Provides detailed error messages and suggestions
- Skips duplicate patient IDs automatically

---

## Notes

- Processes only unique patient IDs (duplicates are skipped)
- Supports various MRI file formats (.nii.gz, .nii, .dcm)
- CSV data is optional - metadata can be generated with null values
- Use preview mode to verify patient ID extraction before final processing
