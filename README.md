# ASL Recognition Project

An American Sign Language (ASL) recognition project using a CNN classifier, MediaPipe hand tracking, and a local Tkinter desktop app for real-time webcam inference.

This repo includes scripts for:
- training a base ASL image classifier,
- collecting and merging custom images,
- fine-tuning the model,
- evaluating performance,
- and running live inference (CLI and GUI).

---

## Dataset (Kaggle)

Primary dataset used:
- **ASL Alphabet** by akash: https://www.kaggle.com/datasets/grassknoted/asl-alphabet

> Note: several scripts currently include Windows-specific absolute paths (`C:\\Users\\...`). You can update those path constants (or environment variables where supported) to match your machine.

---

## Project Structure and File Guide

### Core training + inference

- **`CNN_test1.py`**  
  Main model training script for a CNN on ASL alphabet images (29 classes: A–Z + `del`, `space`, `nothing`). Saves:
  - `asl_model.keras`
  - `asl_model.tflite`
  - training history JSON
  - class names JSON

- **`fineTune.py`**  
  Fine-tunes the base Keras model on a merged dataset (your own captured hand images + sampled Kaggle images). Exports fine-tuned model + metrics/plots to `asl_finetuned/`.

- **`full_pipe.py`**  
  Live webcam inference pipeline (OpenCV window) using:
  - MediaPipe Hand Landmarker (`hand_landmarker.task`)
  - trained Keras model

- **`web_app/webapp.py`**  
  Local Tkinter desktop ASL interpreter UI with:
  - camera feed,
  - detected letter view,
  - hold/progress logic,
  - sentence builder (`Delete` / `Clear`).

### Data collection + dataset prep

- **`collectFineTune.py`**  
  Captures cropped hand images from webcam and stores them into per-class folders to build your custom dataset.

- **`addPhotos.py`**  
  Samples images from Kaggle class folders and copies them into your custom dataset folders (for balancing/expansion).

- **`test.py`**  
  Utility script to create a test split by copying a fixed number of images per class from train source to test destination.

### Evaluation scripts

- **`modeltest.py`**  
  Loads a saved model and evaluates it on an external test directory; prints accuracy, confusion matrix, and classification report.

### Saved model + metadata artifacts

- **`asl_model.keras`** – saved base Keras model.
- **`asl_model.tflite`** – TensorFlow Lite export of base model.
- **`class_names.json`** – label ordering used by the base model.
- **`hand_landmarker.task`** – MediaPipe hand landmark model file required for webcam hand detection.

### Fine-tuned output directory

- **`asl_finetuned/asl_model_finetuned.keras`** – fine-tuned model.
- **`asl_finetuned/class_names_finetuned.json`** – class order for fine-tuned model.
- **`asl_finetuned/finetune_history.json`** – fine-tune metric history.
- **`asl_finetuned/finetune_curves.png`** – accuracy/loss plots.
- **`asl_finetuned/finetune_confusion_matrix.png`** – confusion matrix image.

### App sample input assets

- **`web_app/input_data/*.jpg`**  
  Letter/sample images used by the web app folder.

### Notebook

- **`CNNtest1Colab.ipynb`**  
  Notebook version/experiments related to model training.

---

## Setup

## 1) Clone repo

```bash
git clone <your-repo-url>
cd asl_recognition
```

## 2) Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
```

- Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

- macOS/Linux:

```bash
source .venv/bin/activate
```

## 3) Install dependencies

```bash
pip install tensorflow opencv-python mediapipe numpy pillow matplotlib scikit-learn
```

---

## How to Run

### A) Train the base model

1. Download/unzip Kaggle ASL dataset.
2. Point `DATA_DIR` in `CNN_test1.py` to your training folder.
3. Run:

```bash
python CNN_test1.py
```

### B) Collect your own ASL images (optional, for fine-tuning)

1. Update `OUTPUT_ROOT` in `collectFineTune.py`.
2. Ensure `hand_landmarker.task` is in repo root (or update path constant).
3. Run:

```bash
python collectFineTune.py
```

Controls are printed in terminal (`g`, `p`, `n`, `r`, `q`).

### C) Merge sampled Kaggle images into your custom dataset (optional)

1. Update `KAGGLE_DIR` and `TARGET_DIR` in `addPhotos.py`.
2. Run:

```bash
python addPhotos.py
```

### D) Fine-tune the base model

1. Update `BASE_MODEL_PATH`, `DATA_DIR`, and `SAVE_DIR` in `fineTune.py`.
2. Run:

```bash
python fineTune.py
```

### E) Evaluate model on test folder

1. Update model/test paths in `modeltest.py`.
2. Run:

```bash
python modeltest.py
```

### F) Real-time webcam inference (terminal/OpenCV)

1. Update `MODEL_PATH` in `full_pipe.py`.
2. Make sure `hand_landmarker.task` path is correct.
3. Run:

```bash
python full_pipe.py
```

### G) Run desktop GUI interpreter

```bash
python web_app/webapp.py
```

Optional env vars for model paths:

- `ASL_MODEL_PATH`
- `ASL_HAND_LANDMARKER`

---

## Quick Notes

- The model expects class ordering to match the saved `class_names*.json` files.
- Inference scripts assume MediaPipe hand detection is available via `hand_landmarker.task`.
- If running on a non-Windows machine, path constants in scripts will likely need editing.

---

## Possible Next Improvements

- Add a `requirements.txt`.
- Replace hardcoded paths with CLI args (`argparse`) or config files.
- Add automated train/eval pipelines.
- Add WER/CER or sentence-level metrics for live spelling quality.
