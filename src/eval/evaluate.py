import json
from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.data.dataset import build_dataloaders
from src.models.baseline_cnn import BaselineCNN
from src.models.resnet import build_resnet18
from src.utils import ensure_dir, get_device, setup_logger

logger = setup_logger("evaluate")


def _build_model(model_name: str, num_classes: int, pretrained: bool, freeze_backbone: bool) -> torch.nn.Module:
    if model_name == "baseline_cnn":
        return BaselineCNN(num_classes=num_classes)
    if model_name == "resnet18":
        return build_resnet18(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)
    raise ValueError(f"Unknown model: {model_name}")


def evaluate_checkpoint(config_path: str, checkpoint_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    model_cfg = config["model"]
    output_cfg = config["output"]

    loaders, class_names = build_dataloaders(
        data_dir=data_cfg["data_dir"],
        image_size=data_cfg["image_size"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        val_ratio=data_cfg["val_ratio"],
        seed=config["project"]["seed"],
        use_weighted_sampler=False,
        imbalance_ratio_threshold=data_cfg["imbalance_ratio_threshold"],
    )

    device = get_device()
    model = _build_model(
        model_name=model_cfg["name"],
        num_classes=len(class_names),
        pretrained=model_cfg.get("pretrained", False),
        freeze_backbone=model_cfg.get("freeze_backbone", False),
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in loaders["test"]:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)

    ensure_dir(output_cfg["figures_dir"])
    report_path = Path(output_cfg["class_report_path"])
    ensure_dir(report_path.parent)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    metrics = {"test_accuracy": acc}
    metrics_path = Path(output_cfg["metrics_path"])
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    fig_path = Path(output_cfg["figures_dir"]) / "confusion_matrix.png"
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues", cbar=True)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    logger.info("Test accuracy: %.4f", acc)
    return {"accuracy": acc, "confusion_matrix": cm, "report": report}
