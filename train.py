"""
EcoSort - Smart Waste Sorting System
Training Script using EfficientNetV2 on TrashNet Dataset
=========================================================
Dataset: https://www.kaggle.com/datasets/fedesoriano/trashnet-dataset
Classes: cardboard, glass, metal, paper, plastic, trash
"""

import os
import json
import time
import multiprocessing
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import classification_report, confusion_matrix

# ─── CONFIG ──────────────────────────────────────────────────────────────────

CONFIG = {
    "data_dir":            "dataset-resized/",
    "output_dir":          "ecosort_output/",
    "num_classes":         6,
    "batch_size":          64,    # increased for speed
    "num_epochs":          30,    # reduced — early stopping handles the rest
    "img_size":            224,   # reduced for speed (was 384)
    "lr_init":             1e-3,
    "lr_min":              1e-6,
    "weight_decay":        1e-4,
    "label_smoothing":     0.1,
    "mixup_alpha":         0.4,
    "cutmix_alpha":        1.0,
    "early_stop_patience": 8,
    "seed":                42,
    "num_workers":         0,     # MUST be 0 on Windows
    "classes": ["cardboard", "glass", "metal", "paper", "plastic", "trash"],
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── TRANSFORMS ──────────────────────────────────────────────────────────────

def get_transforms():
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    sz   = CONFIG["img_size"]

    train_tf = transforms.Compose([
        transforms.Resize((sz + 32, sz + 32)),
        transforms.RandomCrop(sz),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_tf, val_tf


# ─── DATASET ─────────────────────────────────────────────────────────────────

def load_datasets():
    train_tf, val_tf = get_transforms()
    data_path = Path(CONFIG["data_dir"])

    sub_dirs   = [d for d in data_path.iterdir() if d.is_dir()]
    has_splits = any(d.name in ["train", "val", "test"] for d in sub_dirs)

    if has_splits:
        train_ds = datasets.ImageFolder(data_path / "train", transform=train_tf)
        val_ds   = datasets.ImageFolder(data_path / "val",   transform=val_tf)
        test_ds  = datasets.ImageFolder(data_path / "test",  transform=val_tf)
    else:
        full_train = datasets.ImageFolder(data_path, transform=train_tf)
        full_val   = datasets.ImageFolder(data_path, transform=val_tf)

        n       = len(full_train)
        n_train = int(0.70 * n)
        n_val   = int(0.15 * n)
        n_test  = n - n_train - n_val

        indices  = torch.randperm(n, generator=torch.Generator().manual_seed(CONFIG["seed"])).tolist()
        train_ds = Subset(full_train, indices[:n_train])
        val_ds   = Subset(full_val,   indices[n_train:n_train + n_val])
        test_ds  = Subset(full_val,   indices[n_train + n_val:])

    print(f"[EcoSort] Train={len(train_ds)}  Val={len(val_ds)}  Test={len(test_ds)}", flush=True)
    return train_ds, val_ds, test_ds


def make_loaders(train_ds, val_ds, test_ds):
    if hasattr(train_ds, "indices"):
        targets = [train_ds.dataset.targets[i] for i in train_ds.indices]
    else:
        targets = train_ds.targets

    class_counts   = np.bincount(targets)
    weights        = 1.0 / class_counts
    sample_weights = torch.tensor([weights[t] for t in targets], dtype=torch.float)
    sampler        = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], sampler=sampler, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"], shuffle=False,   num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=CONFIG["batch_size"], shuffle=False,   num_workers=0, pin_memory=False)
    return train_loader, val_loader, test_loader


# ─── MIXUP / CUTMIX ──────────────────────────────────────────────────────────

def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0)).to(x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0)).to(x.device)
    _, _, H, W = x.size()
    cut_rat = np.sqrt(1 - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1 = max(cx - cut_w // 2, 0); x2 = min(cx + cut_w // 2, W)
    y1 = max(cy - cut_h // 2, 0); y2 = min(cy + cut_h // 2, H)
    mixed = x.clone()
    mixed[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    return mixed, y, y[idx], lam


def mixed_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ─── MODEL ───────────────────────────────────────────────────────────────────

def build_model():
    model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1)

    for name, param in model.features.named_parameters():
        try:
            block_id = int(name.split(".")[0])
            param.requires_grad = block_id >= 5
        except ValueError:
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, 512),
        nn.SiLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.3),
        nn.Linear(512, CONFIG["num_classes"]),
    )
    return model.to(DEVICE)


# ─── TRAIN / EVAL ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, scaler, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    use_cutmix = np.random.rand() > 0.5

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        if use_cutmix:
            imgs, y_a, y_b, lam = cutmix_data(imgs, labels, CONFIG["cutmix_alpha"])
        else:
            imgs, y_a, y_b, lam = mixup_data(imgs, labels, CONFIG["mixup_alpha"])

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(DEVICE.type == "cuda")):
            outputs = model(imgs)
            loss    = mixed_criterion(criterion, outputs, y_a, y_b, lam)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * imgs.size(0)
        _, predicted = outputs.max(1)
        correct += (lam * predicted.eq(y_a).float() + (1 - lam) * predicted.eq(y_b).float()).sum().item()
        total   += labels.size(0)

        if (batch_idx + 1) % 5 == 0:
            print(f"  Batch [{batch_idx+1}/{len(loader)}]  Loss: {loss.item():.4f}", flush=True)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        with torch.amp.autocast('cuda', enabled=(DEVICE.type == "cuda")):
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


# ─── PLOTS ───────────────────────────────────────────────────────────────────

def plot_history(history):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#0f1117")
    for ax in axes:
        ax.set_facecolor("#1a1d2e")
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], color="#40c4ff", label="Train")
    axes[0].plot(epochs, history["val_loss"],   color="#00e676", label="Val")
    axes[0].set_title("Loss", color="white"); axes[0].legend()
    axes[1].plot(epochs, [a*100 for a in history["train_acc"]], color="#40c4ff", label="Train")
    axes[1].plot(epochs, [a*100 for a in history["val_acc"]],   color="#00e676", label="Val")
    axes[1].set_title("Accuracy (%)", color="white"); axes[1].legend()
    axes[2].plot(epochs, history["lr"], color="#ff6d00")
    axes[2].set_title("Learning Rate", color="white"); axes[2].set_yscale("log")
    for ax in axes:
        ax.tick_params(colors="white")
        for spine in ax.spines.values(): spine.set_edgecolor("#333")
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], "training_history.png"), dpi=150)
    plt.close()
    print("[EcoSort] Training history plot saved.", flush=True)


def plot_confusion(labels, preds):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGn",
                xticklabels=CONFIG["classes"], yticklabels=CONFIG["classes"])
    plt.title("EcoSort — Confusion Matrix")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], "confusion_matrix.png"), dpi=150)
    plt.close()
    print("[EcoSort] Confusion matrix saved.", flush=True)


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60, flush=True)
    print("        EcoSort — EfficientNetV2-M Training", flush=True)
    print(f"        Device     : {DEVICE}", flush=True)
    print(f"        Image Size : {CONFIG['img_size']}x{CONFIG['img_size']}", flush=True)
    print(f"        Batch Size : {CONFIG['batch_size']}", flush=True)
    print(f"        Max Epochs : {CONFIG['num_epochs']}", flush=True)
    print("=" * 60, flush=True)

    train_ds, val_ds, test_ds             = load_datasets()
    train_loader, val_loader, test_loader = make_loaders(train_ds, val_ds, test_ds)

    model     = build_model()
    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["lr_init"], weight_decay=CONFIG["weight_decay"]
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=CONFIG["lr_min"])
    scaler    = torch.amp.GradScaler('cuda', enabled=(DEVICE.type == "cuda"))

    best_val_acc = 0.0
    patience_cnt = 0
    history      = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}

    print(f"\n[EcoSort] Starting training...\n", flush=True)

    for epoch in range(1, CONFIG["num_epochs"] + 1):
        t0 = time.time()

        if epoch == 10:
            print("\n[EcoSort] Unfreezing ALL layers for full fine-tuning...", flush=True)
            for param in model.parameters():
                param.requires_grad = True
            optimizer.add_param_group({
                "params": [p for p in model.features.parameters() if not p.requires_grad],
                "lr": CONFIG["lr_init"] * 0.1
            })

        print(f"\n--- Epoch {epoch}/{CONFIG['num_epochs']} ---", flush=True)
        tr_loss, tr_acc       = train_one_epoch(model, train_loader, optimizer, criterion, scaler, epoch)
        vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)
        history["lr"].append(lr_now)

        print(
            f"Epoch [{epoch:03d}/{CONFIG['num_epochs']}]  "
            f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc*100:.2f}%  |  "
            f"Val Loss: {vl_loss:.4f}  Acc: {vl_acc*100:.2f}%  |  "
            f"LR: {lr_now:.2e}  [{elapsed:.1f}s]",
            flush=True
        )

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            patience_cnt = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": best_val_acc,
                "config": CONFIG,
            }, os.path.join(CONFIG["output_dir"], "ecosort_best.pth"))
            print(f"  ✅ New best saved — Val Acc: {best_val_acc*100:.2f}%", flush=True)
        else:
            patience_cnt += 1
            print(f"  No improvement ({patience_cnt}/{CONFIG['early_stop_patience']})", flush=True)
            if patience_cnt >= CONFIG["early_stop_patience"]:
                print(f"\n[EcoSort] Early stopping at epoch {epoch}.", flush=True)
                break

    # ── Final Test Evaluation ──
    print("\n[EcoSort] Loading best model for final test evaluation...", flush=True)
    ckpt = torch.load(os.path.join(CONFIG["output_dir"], "ecosort_best.pth"), map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    _, test_acc, preds, labels = evaluate(model, test_loader, criterion)

    print(f"\n{'='*60}")
    print(f"  FINAL TEST ACCURACY: {test_acc*100:.2f}%")
    print(f"{'='*60}\n")
    print(classification_report(labels, preds, target_names=CONFIG["classes"]))

    with open(os.path.join(CONFIG["output_dir"], "history.json"), "w") as f:
        json.dump(history, f)

    plot_history(history)
    plot_confusion(labels, preds)
    print(f"\n[EcoSort] Done! All outputs saved to: {CONFIG['output_dir']}", flush=True)


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()