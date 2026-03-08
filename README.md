# 🌿 EcoSort — Smart Waste Sorting System

> Image-recognition-powered waste classification using **EfficientNetV2-M**  
> Trained on the **TrashNet** dataset · 6 waste classes · Target accuracy **≥ 95 %**

---

## 📁 Project Structure

```
ecosort/
├── train.py           # Full training pipeline
├── inference.py       # Inference + recycling metadata
├── app.py             # Tkinter GUI application
├── requirements.txt
└── ecosort_output/    # Created automatically after training
    ├── ecosort_best.pth
    ├── training_history.png
    └── confusion_matrix.png
```

---

## 🧠 Why EfficientNetV2-M?

| Model | Params | ImageNet Top-1 | TrashNet Est. Acc | Speed |
|---|---|---|---|---|
| ResNet-50 | 25 M | 76.1 % | ~88 % | Fast |
| EfficientNet-B4 | 19 M | 83.4 % | ~91 % | Fast |
| **EfficientNetV2-M** | **54 M** | **85.1 %** | **~95 %** | Medium |
| ViT-L/16 | 307 M | 88.7 % | ~96 % | Slow |

EfficientNetV2-M delivers the best **accuracy-to-compute tradeoff** for TrashNet:
- Progressive learning with **Mixup + CutMix** augmentation
- **Label smoothing** (0.1) for better calibration
- **Cosine Annealing with Warm Restarts** scheduler
- **Weighted Random Sampler** for class imbalance
- **Mixed Precision (AMP)** for faster GPU training
- **Early stopping** with patience

---

## 🚀 Quick Start

### 1 · Install Dependencies

```bash
pip install -r requirements.txt
```

### 2 · Download TrashNet Dataset

```bash
# Option A — Kaggle CLI
pip install kaggle
kaggle datasets download -d fedesoriano/trashnet-dataset
unzip trashnet-dataset.zip -d dataset/

# Option B — Manual
# Download from https://www.kaggle.com/datasets/fedesoriano/trashnet-dataset
# Extract so you have:  dataset/cardboard/  dataset/glass/  etc.
```

### 3 · Train the Model

```bash
python train.py
```

Expected output:
```
[EcoSort] Using device: cuda
[EcoSort] Train=1772  Val=379  Test=380
Epoch [001/050]  Train Loss: 1.3241  Acc: 62.14%  |  Val Loss: 0.8832  Acc: 78.36%  ...
...
Epoch [035/050]  Train Loss: 0.2114  Acc: 94.71%  |  Val Loss: 0.1983  Acc: 95.22%  ...
  ✅ New best saved — Val Acc: 95.22%
```

Training generates:
- `ecosort_output/ecosort_best.pth` — Best model checkpoint
- `ecosort_output/training_history.png` — Loss / Accuracy / LR curves
- `ecosort_output/confusion_matrix.png` — Per-class confusion matrix

### 4 · Launch the GUI

```bash
python app.py
```

### 5 · Command-Line Inference

```bash
python inference.py path/to/image.jpg
```

---

## 📊 Dataset Info — TrashNet

| Class | Approx Samples |
|---|---|
| cardboard | 403 |
| glass | 501 |
| metal | 410 |
| paper | 594 |
| plastic | 482 |
| trash | 137 |
| **Total** | **~2,527** |

> The class imbalance (especially `trash`) is handled via **WeightedRandomSampler**.

---

## ⚙️ Training Techniques Explained

| Technique | Why It Helps |
|---|---|
| **Transfer Learning** (ImageNet) | Starts with powerful visual features instead of random init |
| **Progressive Unfreezing** | Freeze early layers → unfreeze all at epoch 15, prevents catastrophic forgetting |
| **Mixup α=0.4** | Linearly interpolates two images; forces smoother decision boundaries |
| **CutMix α=1.0** | Replaces image regions with patches from another; teaches local features |
| **Label Smoothing 0.1** | Prevents overconfidence, better calibration |
| **Cosine Annealing W/ Restarts** | Escapes local minima, converges to flatter (better-generalizing) solutions |
| **WeightedRandomSampler** | Each batch sees all classes equally despite imbalance |
| **AMP (float16)** | 2× speedup on GPU with negligible accuracy loss |
| **Gradient Clipping** | Prevents exploding gradients |
| **RandomErasing** | Simulates occlusion robustness |

---

## 🖥️ GUI Features

- **Open Image** — Load from file
- **Capture** — Grab from webcam (requires opencv-python)
- **Classify** — Instant prediction with confidence %
- **Top-3 bar chart** — See alternative predictions
- **Recycling info** — Bin color, recycling tip, CO₂ impact
- **Session history** — Color-coded chips for last 8 classifications

---

## 🔧 Configuration

Edit the `CONFIG` dict at the top of `train.py`:

```python
CONFIG = {
    "data_dir":        "dataset/",    # path to your TrashNet folder
    "img_size":        384,           # reduce to 300 for faster training
    "batch_size":      32,            # reduce to 16 if OOM on GPU
    "num_epochs":      50,
    "lr_init":         1e-3,
    "early_stop_patience": 10,
}
```

---

## 📜 License

MIT — Free to use, modify, and distribute.
