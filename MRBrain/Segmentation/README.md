# SynthSeg Pipeline

This repository contains a cleaned version of the **SynthSeg** pipeline for brain MRI segmentation.  
All comments and docstrings have been removed, leaving only the executable code in `mri_pipeline_clean.py`.

---

## Setup

1. Clone or download this repository.
2. Download the pretrained models from the provided Google Drive link.
3. Place the downloaded models into a folder named **`models/`** located in the same directory as `mri_pipeline_clean.py`.

The final structure should look like this:
```
Segmentation/
│
├── mri_pipeline_clean.py
├── models/
│   ├── synthseg_2.0.h5
│   ├── synthseg_robust_2.0.h5
│   ├── synthseg_parc_2.0.h5
│   ├── synthseg_qc_2.0.h5
│   ├── *.npy
│   └── ...
```

---

## Usage

Run the pipeline with the following command:

```bash
python mri_pipeline_clean.py --i <input_image_or_folder> --o <output_folder> [options]
```

---

## Arguments

- `--i` : Input image(s) to segment. Can be a path to an image or a folder.
- `--o` : Output segmentation(s). Must be a folder if `--i` is a folder.
- `--parc` : Perform cortical parcellation (optional).
- `--robust` : Use robust predictions (slower but more stable).
- `--fast` : Skip some processing for faster predictions.
- `--ct` : Clip CT scans in Hounsfield scale to [0, 80].
- `--vol` : Output CSV file with volumes for all structures and subjects.
- `--qc` : Output CSV file with QC scores for all subjects.
- `--post` : Save posteriors (requires output folder if multiple inputs).
- `--resample` : Save resampled images (requires output folder if multiple inputs).
- `--crop` : Analyse only an image patch of the given size (provide 3 integers).
- `--threads` : Number of CPU threads to use (default: 1).
- `--cpu` : Force CPU mode (disable GPU).
- `--v1` : Use SynthSeg 1.0 instead of 2.0.
- `--photo` : Use Photo-SynthSeg (`left`, `right`, or `both`).

---

## Example

Segment a single MRI scan and save results in `output/`:

```bash
python mri_pipeline_clean.py --i input.nii.gz --o output/
```

Run robust segmentation on a folder of scans with 4 CPU threads:

```bash
python mri_pipeline_clean.py --i input_folder/ --o output_folder/ --robust --threads 4 --cpu
```

Save volumes and QC scores:

```bash
python mri_pipeline_clean.py --i input.nii.gz --o output/ --vol volumes.csv --qc qc_scores.csv
```

---

## Citation

If you use this tool in your research, please cite the original SynthSeg papers:

- **SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining**  
  B. Billot, D.N. Greve, O. Puonti, A. Thielscher, K. Van Leemput, B. Fischl, A.V. Dalca, J.E. Iglesias.  
  *Medical Image Analysis, accepted for publication.*

- **Robust machine learning segmentation for large-scale analysis of heterogeneous clinical brain MRI datasets**  
  B. Billot, C. Magdamo, Y. Cheng, S.E. Arnold, S. Das, J.E. Iglesias.  
  *PNAS, accepted for publication.*

---
