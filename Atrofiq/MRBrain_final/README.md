# AtrofIQ: Clinical MR Analysis Pipeline

**A Comprehensive Framework for Brain MRI Analysis, Age Prediction, and Normative Modeling.**

## 🧠 Overview
**AtrofIQ** (formerly MRBrain) is an integrated research platform designed to process clinical-grade 2D MRI scans and extract high-value biomarkers. It unifies state-of-the-art Deep Learning models for super-resolution, segmentation, and statistical analysis into a sleek, clinician-friendly dashboard.

## 🚀 Key Features

*   **2D-to-3D Super-Resolution**: Reconstructs high-fidelity 3D volumes from low-resolution 2D DICOM slices using a GAN-based architecture (`2D-3D`).
*   **Automated Segmentation**: Robust segmentation of subcortical structures (HPC, Thalamus, etc.) powered by `SynthSeg` / `FastSurfer` variants (`Segmentation`).
*   **Brain Age Prediction**: Estimates "Brain Age" vs. Chronological Age using 3D CNNs to identify accelerated aging or neurodegeneration (`BrainAge-Prediction`).
*   **Normative Modeling**: Maps patient volumes against population centiles (Lifespan Charts) to visualize deviations (`Normative Modeling`).
*   **Interactive Web Interface**: A premium, "Glassmorphism" UI for easy data upload, visualization, and report generation (`V.0.01`).

## 📂 Repository Structure

| Directory | Description |
| :--- | :--- |
| **`V.0.01/`** | **Start Here**. Contains the comprehensive Pipeline Integration and the **Web Interface** (Flask App). |
| `2D-3D/` | GAN-based model for generating 3D MRI volumes from 2D slices. |
| `Segmentation/` | Scripts for anatomical segmentation of brain structures. |
| `BrainAge-Prediction/` | Models for predicting biological brain age (Post-Contrast). |
| `BrainAge-Prediction_Pre/` | Models for predicting brain age (Pre-Contrast). |
| `Normative Modeling/` | Reference charts and scripts for population centile analysis. |
| `Metadata_gen/` | Utilities for dataset normalization and metadata generation. |

## 🛠️ Installation & Setup

### Prerequisites
*   Linux OS (Ubuntu 20.04+ recommended)
*   Python 3.8+
*   NVIDIA GPU (CUDA 11+) for inference acceleration.

### Installation
1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Rad-AI-Private/MRBrain.git
    cd MRBrain
    ```

2.  **Install Dependencies**
    It is recommended to use a virtual environment.
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

## 🖥️ Usage

The easiest way to use the system is via the Web Dashboard.

### Launching the Dashboard
Move to the application directory and run the Flask server:
```bash
cd V.0.01
python3 app.py
```
> Access the interface at **`http://localhost:5000`** in your web browser.

### Pipeline Logic (Automated)
1.  **Input**: User uploads a folder of DICOM files.
2.  **2D-3D**: The pipeline converts DICOMs to NIfTI and generates a 3D High-Res volume.
3.  **Segmentation**: The 3D volume is segmented into anatomical ROIs.
4.  **Analysis**:
    *   Volumes are calculated.
    *   Brain Age is predicted.
    *   Normative centiles are computed.
5.  **Output**: Results are displayed on the dashboard with interactive charts and viewers.


## 📜 License
Private Research Codebase. All rights reserved.
