<div align="center">

# 🌿 EcoSort
### Smart Waste Sorting System

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Accuracy](https://img.shields.io/badge/Accuracy-94.99%25-22c55e?style=flat)](#results)
[![License](https://img.shields.io/badge/License-MIT-6366f1?style=flat)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-TrashNet-f59e0b?style=flat)](https://www.kaggle.com/datasets/feyzazkefe/trashnet)

An end-to-end AI-powered waste classification system that identifies waste into **6 categories** using **EfficientNetV2-M** with a real-time desktop GUI.

</div>

---

## 📌 Overview

EcoSort uses deep learning to classify waste images into 6 categories — **cardboard, glass, metal, paper, plastic, and trash** — and provides recycling guidance, bin color, and CO₂ impact for each classification. Built with PyTorch and a custom Tkinter GUI with live webcam support.

---

## ✨ Features

- 🧠 **EfficientNetV2-M** — 94.99% validation accuracy on TrashNet
- 📷 **Live Webcam Feed** — 30 FPS real-time capture and classification
- 📊 **Top-3 Predictions** — Animated confidence bars for top results
- ♻️ **Recycling Guidance** — Bin color, disposal tips, and CO₂ savings
- 🕘 **Session History** — Color-coded chips for last 10 classifications
- ⚡ **GPU Accelerated** — CUDA + Mixed Precision (AMP) training

---

## 🗂️ Project Structure

```
EcoSort/
├── train.py              # Full training pipeline
├── inference.py          # Model inference + recycling metadata
├── app.py                # Tkinter GUI application
├── requirements.txt      # Dependencies
├── README.md
├── docs/                 # Screenshots and plots
│   ├── gui_screenshot.png
│   ├── training_history.png
│   └── confusion_matrix.png
└── ecosort_output/       # Auto-generated after training
    ├── ecosort_best.pth
    ├── training_history.png
    └── confusion_matrix.png
```

---

## 🧠 Model

| Model | Params | ImageNet Acc | TrashNet Acc | Speed |
|---|---|---|---|---|
| ResNet-50 | 25M | 76.1% | ~88% | Fast |
| EfficientNet-B4 | 19M | 83.4% | ~91% | Fast |
| **EfficientNetV2-M** ✅ | **54M** | **85.1%** | **94.99%** | Medium |
| ViT-L/16 | 307M | 88.7% | ~96% | Slow |

**EfficientNetV2-M** was selected for the best accuracy-to-compute tradeoff.


## 📊 Dataset

| Class | Samples |
|---|---|
| Paper | 594 |
| Glass | 501 |
| Plastic | 482 |
| Metal | 410 |
| Cardboard | 403 |
| Trash | 137 |
| **Total** | **~2,527** |

Source: [TrashNet on Kaggle](https://www.kaggle.com/datasets/feyzazkefe/trashnet)

Class imbalance (especially `trash`) is handled via **WeightedRandomSampler**.

---

## 📈 Results

<div align="center">

| Metric | Value |
|---|---|
| Validation Accuracy | **94.99%** |
| Training Epochs | 17 (early stopping) |
| Device | NVIDIA RTX 3050 |
| Training Time | ~40 minutes |

</div>


## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Asmita2587/EcoSort.git
cd EcoSort
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

For NVIDIA GPU (recommended):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4. Download Dataset

```bash
kaggle datasets download -d feyzazkefe/trashnet
```
Extract to a folder named `dataset-resized/` inside the project root.

### 5. Train the Model

```bash
python train.py
```


### 6. Launch GUI

```bash
python app.py
```

### 7. Command-Line Inference

```bash
python inference.py path/to/image.jpg
```

---

## 🖥️ GUI Usage

1. Click **📂 Open Image** to load a waste photo
2. Or click **📷 Camera** to open the live webcam preview
3. Click **🔍 CLASSIFY WASTE**
4. View result: class, confidence %, bin color, recycling tip, CO₂ savings

---

## 🔧 Configuration

Edit `CONFIG` at the top of `train.py`:

```python
CONFIG = {
    "data_dir":            "dataset-resized/",
    "img_size":            224,     # increase to 384 for higher accuracy
    "batch_size":          64,      # reduce to 32 if GPU runs out of memory
    "num_epochs":          30,
    "lr_init":             1e-3,
    "early_stop_patience": 8,
}
```

---

## 📦 Requirements

```
torch >= 2.1.0
torchvision >= 0.16.0
Pillow >= 10.0.0
numpy >= 1.24.0
scikit-learn >= 1.3.0
matplotlib >= 3.7.0
seaborn >= 0.12.0
opencv-python >= 4.8.0
```

---

## 🔗 Model Download

The trained model weights (`ecosort_best.pth`) exceed GitHub's file size limit.

👉 **[Download from Google Drive](https://drive.google.com/file/d/1kivMPdHTLe-I3nCnXM00Y4q0irWZT7Vj/view?usp=sharingNK)**

Place it at: `ecosort_output/ecosort_best.pth`

---

## 📜 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 👩‍💻 Author

**Asmita**
- GitHub: [Asmita2587](https://github.com/Asmita2587)

---
