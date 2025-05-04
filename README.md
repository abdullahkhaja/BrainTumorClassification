# Brain Tumor Classification

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

This project fine-tunes a pretrained **ResNet-18** in PyTorch to classify brain MRI scans into tumor types, exports the trained model to **ONNX**, and deploys a fully static, in-browser demo via **ONNX Runtime Web**.

---

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Export to ONNX](#export-to-onnx)
- [Static Demo](#static-demo)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [License](#license)

---

## Installation

```bash
git clone https://github.com/abdullahkhaja/BrainTumorClassification.git
cd BrainTumorClassification
pip install torch torchvision kagglehub numpy scikit-learn matplotlib seaborn tqdm onnx onnxruntime
```

---

## Dataset

The **Brain Tumor MRI Scans** dataset from Kaggle contains four classes:

| Class      | Description      |
|------------|------------------|
| Glioma     | Tumor            |
| Healthy    | No tumor present |
| Meningioma | Tumor            |
| Pituitary  | Tumor            |

The dataset is downloaded automatically in `model/train.py`:

```python
import kagglehub
path = kagglehub.dataset_download("rm1000/brain-tumor-mri-scans")
print("Data downloaded to:", path)
```

---

## Model Architecture

We start with a pretrained **ResNet-18** and adapt it in two phases:

| Phase | What we train            | Details                                                                 |
|-------|--------------------------|-------------------------------------------------------------------------|
| 1     | **Final fully connected layer only**     | - Freeze all convolutional layers<br>- Replace the final fully connected layer for 4 classes<br>- Train this new output layer for 5 epochs at learning rate **0.001** |
| 2     | **Last convolutional block + output layer**    | - Unfreeze ResNet's last convolutional block<br>- Train both the last block and the output layer for 15 epochs at learning rate **0.00001** |

### Why two phases?

1. **Quick head training**  
   We first train only the new output layer (the “head”). It learns to map existing features to our classes without altering the backbone, so it’s fast and stable.

2. **Gentle fine-tuning**  
   Then we unfreeze the deepest convolutional block and train at a smaller learning rate. This refines high-level MRI-specific features without overwriting the backbone's general knowledge.

---

## Training

```bash
python model/train.py
```

This script will:

1. Download & extract the dataset  
2. Split into 70% train / 15% val / 15% test  
3. Run Phase 1 and Phase 2 training  
4. Save a PyTorch checkpoint **and** export the model to ONNX (`docs/model.onnx`)

**Sample log**

```text
🚀 Training on device: cuda:0
[Head 1/5] loss=0.5234 | Val Acc=0.8125
…
[FT 15/15] loss=0.1123 | Val Acc=0.9458
✅ model.onnx exported to docs/model.onnx
```

---

## Export to ONNX

```bash
python export_onnx.py
```

Generates `docs/model.onnx` for the static demo.

---

## Static Demo

A pure-static demo lives under the `docs/` folder. To test locally:

```bash
cd docs
python -m http.server 8000
```

Open **http://localhost:8000** in your browser.

---

## Usage

1. **Download test images**  
   Go to the **Releases** page and download `test_samples.zip`.  
   Unzip it to any folder.

2. **Open the demo**  
   Visit the live site:  
   https://abdullahkhaja.github.io/BrainTumorClassification/

3. **Upload an image**  
   Click **Choose File**, select an MRI image from your unzipped folder, then click **Predict**.

4. **See the result**  
   The page will display the predicted tumor class and confidence percentage.

---

## Project Structure

```text
BrainTumorClassification/
├── docs/                ← static demo (index.html, model.onnx, css/)
│   ├── index.html
│   ├── model.onnx
│   └── css/
│       ├── bootstrap.min.css
│       └── static.css
├── model/
│   ├── dataset.py       ← custom Dataset & transforms
│   ├── predictor.py     ← PyTorch inference code
│   └── train.py         ← two-phase training script
├── export_onnx.py       ← exports model to ONNX
├── extract_test_samples.py
├── static/              ← Flask assets (legacy)
├── templates/           ← Flask templates (optional)
├── app.py               ← optional Flask server
└── requirements.txt     ← Python dependencies
```

---

## License

Distributed under the [MIT License](LICENSE).
