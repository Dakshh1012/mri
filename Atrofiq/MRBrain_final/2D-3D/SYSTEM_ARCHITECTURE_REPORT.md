# System Architecture and Execution Report

## Overview
This document outlines the architecture of the **Region-Focused 2D-to-3D Training Pipeline**, explaining how different components interact and how execution flows during training.

## 1. The Orchestrator: `train_region_focused_v2.py`
This is the master script that runs continuously. It manages the training lifecycle, checkpointing, validation, and visualization.

### Execution Loop
The script runs a loop from `start_epoch` to `total_epochs`:

1.  **Training Step** (Calls `Train.py`)
2.  **Weight Update Step** (Calls `generate_volumes_subset.py` and Segmentation)
3.  **Validation Step** (Calls `Inference.py` and `BrainAge-Prediction`)
4.  **Milestone Saving** (Every 50 epochs)
5.  **Visualization Step** (Every 20 epochs, calls `visualize_progress.py`)

## 2. Component Interaction via Subprocesses
A key design feature is that components are called as **independent subprocesses**. This keeps memory clean and allows for **hot-swapping of code** (changes to `Train.py` take effect immediately in the next epoch loop).

### A. Training (`Train.py`)
*   **Trigger:** Called every epoch cycle (e.g., train for 1 epoch).
*   **Mechanism:** `subprocess.check_call([python, "Train.py", ...])`
*   **Input:** Data directory, Current Checkpoint (to resume).
*   **Output:** Updated model weights (`latest_model.pth`).
*   **Note:** Since it restarts every cycle, it re-loads the dataset and re-imports libraries. This allows modifying `Train.py` logic (like sampling balance) without killing the main orchestrator.

### B. Region Weight Update
*   **Trigger:** After training step.
*   **Mechanism:**
    1.  Call `generate_volumes_subset.py` to create 3D volumes for a subset of training data.
    2.  Call `Segmentation/mri_pipeline_pytorch.py` to segment these volumes.
    3.  Calculate error (MAPE) between generated segmentation and Global Ground Truth.
    4.  Update `region_weights.json` based on errors.
*   **Effect:** The next `Train.py` run reads the updated JSON and adjusts loss weights (focusing on poorly reconstructed regions).

### C. Validation (Brain Age)
*   **Trigger:** Every epoch cycle.
*   **Mechanism:**
    1.  Call `Inference.py` to generate volumes for Test Set.
    2.  Call `Segmentation` pipeline.
    3.  Call `BrainAge-Prediction/inference.py` to predict age from segmentation.
    4.  Log MAE (Mean Absolute Error) to `test_bag_history.csv`.

### D. Visualization (`visualize_progress.py`)
*   **Trigger:** Every 20 epochs (`if target % 20 == 0`).
*   **Mechanism:** `subprocess.check_call([python, "visualize_progress.py", ...])`
*   **Function:**
    1.  Loads the latest generator.
    2.  Generates full volumes for 3 fixed subjects.
    3.  Segments them.
    4.  Produces a **3-View Plot** (Axial, Coronal, Sagittal) with segmentation overlays.
    5.  Saves to `Region_Focused_Results/Plots/epoch_XX`.
*   **Purpose:** Allows visual debugging of:
    *   **3D Consistency:** Are all 3 views looking like brains?
    *   **Orientation:** Is "Coronal" actually Coronal? (If not, dataset orientation is mixed).
    *   **Segmentation Quality:** Is the segmentation matching the anatomy?

## 3. Directory Structure
```
MRBrain/2D-3D/
├── train_region_focused_v2.py  # MASTER
├── Train.py                    # Subprocess (Training)
├── Preprocessing_pipeline.py   # Utility
├── visualize_progress.py       # Subprocess (Vis)
├── generate_volumes_subset.py  # Utility
├── Inference.py                # Subprocess (Valid)
└── Region_Focused_Results/     # OUTPUTS
    ├── weights/                # Checkpoints
    ├── Plots/                  # Visualization images
    ├── Generated_Volumes_Train/# Temp vols for weight update
    └── Test_Validation/        # Validation outputs
```

## 4. Current Debugging Status
*   **Orientation Issue:** Detected mixed orientations (some Sagittal inputs labeled as Axial).
*   **Fix Applied:** `Train.py` modified to sample views continuously (Axial/Coronal/Sagittal balanced 33% each). This forces the model to verify 3D consistency during training.
*   **Visualization:** The existing `visualize_progress.py` IS configured to show all 3 views. Check `Region_Focused_Results/Plots` every 20 epochs to verify the fix works.
