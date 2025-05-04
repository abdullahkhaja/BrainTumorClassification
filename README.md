# Brain Tumor Classification

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

This project fine‑tunes a pretrained **ResNet‑18** in PyTorch to classify brain MRI scans into tumor types, exports the trained model to **ONNX**, and deploys a fully static in‑browser demo via **ONNX Runtime Web** on **GitHub Pages**.

---

## Table of Contents

- [Installation](#installation)  
- [Dataset](#dataset)  
- [Model Architecture](#model-architecture)  
- [Training](#training)  
- [Export to ONNX](#export-to-onnx)  
- [Static Demo](#static-demo)  
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

The **Brain Tumor MRI Scans** dataset (Kaggle) contains four classes:

| Class      | Description       |
|------------|-------------------|
| Glioma     | Tumor             |
| Healthy    | No tumor present  |
| Meningioma | Tumor             |
| Pituitary  | Tumor             |

The dataset is downloaded automatically inside `model/train.py`:

```python
import kagglehub
path = kagglehub.dataset_download("rm1000/brain-tumor-mri-scans")
print("Data downloaded to:", path)
```

---

## Model Architecture

| Phase | Layers Trained   | Epochs | Learning Rate |
|-------|------------------|--------|---------------|
| 1     | `fc` only        |   5    | 1 × 10⁻³      |
| 2     | `layer4` + `fc`  |  15    | 1 × 10⁻⁵      |

---

## Training

```bash
python model/train.py
```

The script:

1. Downloads & extracts the dataset  
2. Splits into **70 % train / 15 % val / 15 % test**  
3. Phase 1: train head (5 epochs)  
4. Phase 2: fine‑tune layer4 + head (15 epochs)  
5. Saves checkpoint **and** exports to ONNX  

**Sample log**

```text
🚀 Training on device: cuda:0
[Head 1/5] loss=0.5234 | Val Acc=0.8125
…
[FT 15/15] loss=0.1123 | Val Acc=0.9458
✅ model.onnx exported to docs/model.onnx
```

---

## Export to ONNX

```bash
python export_onnx.py
```

Creates `docs/model.onnx`, ready for the browser demo.

---

## Static Demo

### Live demo  
🔗 **https://abdullahkhaja.github.io/BrainTumorClassification/** – runs 100 % in the browser.

### Run locally

```bash
cd docs
python -m http.server 8000
```

Open **http://localhost:8000**, upload an MRI scan, and get the prediction instantly.

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
│   └── train.py         ← two‑phase training script
├── export_onnx.py       ← checkpoint → ONNX
├── extract_test_samples.py
├── static/              ← Flask assets (legacy)
├── templates/           ← Flask templates (optional)
├── app.py               ← optional Flask server
└── requirements.txt
```

---

## License

Distributed under the [MIT License](LICENSE).
