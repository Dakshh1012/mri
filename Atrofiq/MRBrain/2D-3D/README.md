

## 📂 Repository Structure

```

2D-3D/
│
├── Main.py                   # Core model & dataset classes (UNet3D, dataset loaders)
├── Trainer.py                # Baseline GAN-UNET trainer (memory-efficient)
├── Better_trainer.py         # Progressive GAN trainer (multi-stage slice growth)
├── Eval.py                   # Evaluation utilities
├── Run_eval.py               # Evaluation runners
├── Final.py                  # High-level evaluation script (single, batch, compare)
├── synthseg_eval.py          # SynthSeg v2.0 training & evaluation wrapper
├── Generate.py               # Generate 3D NIfTI from LR slices
├── Preprocessing_pipeline.py # Build Alignment_LR_HR_slices dataset (LR + HR slices)
│
├── Alignment_LR_HR_slices/   # Preprocessed dataset with LR (2D) + HR (3D) slices
├── evaluation_batch*/ # Evaluated and generated output with Error scores and differences
├── Models/ # Model directory trained using Better_trainer.py 
````
[Drive link](https://drive.google.com/drive/folders/1u_voT4n7CJO2u6q4tw32g7z_sLnC4Dtr?usp=drive_link) to get the raw 2D mris and the updated model that to be replaced on `Models/`
---

## 🚀 Quickstart

### 1. Preprocessing (Build LR/HR Slices)

Use `Preprocessing_pipeline.py` to prepare **paired LR and HR slices** from raw 2D and 3D MRIs:

```bash
python Preprocessing_pipeline.py \
    --input_2d_dir /path/to/2d_scans \
    --input_3d_dir /path/to/3d_scans \
    --output_dir Alignment_LR_HR_slices \
    --mode both \
    --target_shape 256,256
````

**Arguments:**

* `--mode` : `both` (default), `lr`, or `hr` → which slice sets to generate
* `--input_2d_dir` : directory with 2D MRIs (low-res, \~25 slices)
* `--input_3d_dir` : directory with 3D MRIs (high-res, \~250 slices)
* `--output_dir` : directory to save LR/HR slice `.npy` files
* `--target_shape` : in-plane shape (default `256,256`)

---

### 2. Baseline Training

Run the standard GAN-UNET trainer:

```bash
python Trainer.py \
    --data_dir Alignment_LR_HR_slices \
    --mode train \
    --epochs 50 \
    --batch_size 2 \
    --lr_g 2e-4 \
    --lr_d 1e-4 \
    --target_slices 128 \
    --target_size 256 \
    --device cuda
```

Other modes:

* `test` – run inference with a trained checkpoint
* `gif` – create GIFs of generated slices
* `best_worst` – visualize best/worst generated slices
* `multistage` – staged training cycles
* `range_gen` – generate a specific slice range

---

### 3. Progressive Training

Train progressively from partial to full volumes:

```bash
python Better_trainer.py \
    --data_dir Alignment_LR_HR_slices \
    --mode progressive_train \
    --epochs 100 \
    --batch_size 2 \
    --start_slices 64 \
    --full_volume_slices 256 \
    --increment_ratio 0.2 \
    --target_size 256 \
    --device cuda
```

---

### 4. Generation

Generate **synthetic 3D MRI volumes** from LR slices:

```bash
python Generate.py \
    --model Models/best_model_progressive.pth \
    --slices Alignment_LR_HR_slices \
    --num_participants 10 \
    --output_dir generated_volumes \
    --device cuda
```

**Arguments:**

* `--model` : path to trained `best_model_progressive.pth`
* `--slices` : directory with LR slice `.npy` files (from preprocessing)
* `--num_participants` : number of participants to process
* `--output_dir` : directory to save generated `.nii.gz` volumes
* `--device` : `cuda` or `cpu`

---

### 5. Evaluation

Evaluate trained models and compare reconstructions:

```bash
# Single participant
python Final.py \
    --mode single \
    --model checkpoints/model.pth \
    --data Alignment_LR_HR_slices \
    --output results/ \
    --participant 0

# Batch evaluation
python Final.py \
    --mode batch \
    --model checkpoints/model.pth \
    --data Alignment_LR_HR_slices \
    --output results/ \
    --max_participants 50

# Compare generated vs ground truth vs hybrid
python Final.py \
    --mode compare \
    --gen results/generated_volume.nii.gz \
    --hr results/ground_truth_volume.nii.gz \
    --hybrid results/hybrid_volume.nii.gz \
    --compare_output results/comparison.png
```

---

### 6. SynthSeg Evaluation

To evaluate reconstruction quality using **segmentation consistency**:

```bash
python synthseg_eval.py \
    --evaluation_dir evaluation_batch/ \
    --synthseg_path SynthSeg/ \
    --output_dir synthseg_training_output \
    --epochs 50 \
    --batch_size 2
```

---

## 📊 Outputs

* **Checkpoints** (`.pth`) – saved models
* **Logs** – training curves and loss values
* **GIFs/PNGs** – visualization of slices
* **NIfTI volumes** – generated MRI outputs
* **JSON metrics** – segmentation evaluation reports

---

## 📝 Notes

* `Trainer.py` → baseline GAN-UNET (single stage).
* `Better_trainer.py` → progressive training (multi-stage slice growth).
* `Eval.py`, `Run_eval.py`, `Final.py` → evaluation workflows.
* `Preprocessing_pipeline.py` → builds LR/HR slice dataset (`Alignment_LR_HR_slices`).
* `Generate.py` → generates synthetic 3D MRI volumes from LR slices.
* `synthseg_eval.py` → segmentation-based evaluation (SynthSeg v2.0).


