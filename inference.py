"""
EcoSort — Inference Utility
Loads the trained EfficientNetV2-M checkpoint and returns
predictions with confidence scores for a single image or batch.
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms, models


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_LABELS = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Recycling tips for each class
RECYCLING_INFO = {
    "cardboard": {
        "icon": "📦",
        "bin_color": "Blue",
        "tip": "Flatten boxes, remove tape & staples. Keep dry!",
        "recyclable": True,
        "co2_saved": "1.1 kg CO₂ per kg recycled",
    },
    "glass": {
        "icon": "🫙",
        "bin_color": "Green / Brown",
        "tip": "Rinse containers. Separate by color if required.",
        "recyclable": True,
        "co2_saved": "0.3 kg CO₂ per kg recycled",
    },
    "metal": {
        "icon": "🥫",
        "bin_color": "Yellow",
        "tip": "Rinse cans. Crush to save space. Remove lids.",
        "recyclable": True,
        "co2_saved": "9.1 kg CO₂ per kg recycled (aluminium)",
    },
    "paper": {
        "icon": "📄",
        "bin_color": "Blue",
        "tip": "Remove plastic windows from envelopes. No greasy paper!",
        "recyclable": True,
        "co2_saved": "0.9 kg CO₂ per kg recycled",
    },
    "plastic": {
        "icon": "🧴",
        "bin_color": "Yellow / Recycling",
        "tip": "Check resin code (1-7). Rinse & dry. No films/bags.",
        "recyclable": True,
        "co2_saved": "1.5 kg CO₂ per kg recycled",
    },
    "trash": {
        "icon": "🗑️",
        "bin_color": "Black / General Waste",
        "tip": "General waste. Cannot be recycled. Minimize this!",
        "recyclable": False,
        "co2_saved": "N/A",
    },
}


def get_transform(img_size=384):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def build_model(num_classes=6):
    model = models.efficientnet_v2_m(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, 512),
        nn.SiLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )
    return model


def load_model(checkpoint_path: str = "ecosort_output/ecosort_best.pth"):
    """Load trained EcoSort model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt  = torch.load(checkpoint_path, map_location=DEVICE)
    cfg   = ckpt.get("config", {})
    nc    = cfg.get("num_classes", 6)
    model = build_model(nc)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()

    val_acc = ckpt.get("val_acc", 0) * 100
    epoch   = ckpt.get("epoch", "?")
    print(f"[EcoSort] Loaded model — Epoch {epoch}, Val Acc: {val_acc:.2f}%")
    return model


@torch.no_grad()
def predict(model, image: Image.Image, img_size: int = 384, topk: int = 3):
    """
    Run inference on a PIL Image.

    Returns:
        dict with keys:
            label       – top-1 class name
            confidence  – top-1 confidence (0-1)
            topk        – list of (label, prob) for top-k classes
            info        – recycling metadata dict
    """
    tf     = get_transform(img_size)
    tensor = tf(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    logits = model(tensor)                      # (1, C)
    probs  = torch.softmax(logits, dim=1)[0]    # (C,)

    topk_probs, topk_ids = probs.topk(topk)
    topk_results = [(CLASS_LABELS[i], float(p)) for i, p in zip(topk_ids, topk_probs)]

    top1_label = topk_results[0][0]
    top1_conf  = topk_results[0][1]

    return {
        "label":      top1_label,
        "confidence": top1_conf,
        "topk":       topk_results,
        "info":       RECYCLING_INFO[top1_label],
    }


@torch.no_grad()
def predict_batch(model, image_paths: list, img_size: int = 384):
    """Run inference on a list of image paths."""
    tf = get_transform(img_size)
    results = []
    for path in image_paths:
        img    = Image.open(path).convert("RGB")
        tensor = tf(img).unsqueeze(0).to(DEVICE)
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
        pred_id = probs.argmax().item()
        results.append({
            "path":       str(path),
            "label":      CLASS_LABELS[pred_id],
            "confidence": float(probs[pred_id]),
        })
    return results


if __name__ == "__main__":
    # Quick smoke-test
    import sys
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path> [checkpoint_path]")
        sys.exit(1)

    img_path  = sys.argv[1]
    ckpt_path = sys.argv[2] if len(sys.argv) > 2 else "ecosort_output/ecosort_best.pth"

    model = load_model(ckpt_path)
    img   = Image.open(img_path)
    out   = predict(model, img)

    print(f"\n{'='*50}")
    print(f"  Prediction : {out['info']['icon']}  {out['label'].upper()}")
    print(f"  Confidence : {out['confidence']*100:.1f}%")
    print(f"  Bin        : {out['info']['bin_color']}")
    print(f"  Tip        : {out['info']['tip']}")
    print(f"\n  Top-3:")
    for label, prob in out["topk"]:
        bar = "█" * int(prob * 20)
        print(f"    {label:<12} {prob*100:5.1f}%  {bar}")
    print("=" * 50)
