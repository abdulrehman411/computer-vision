from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.dataset import build_dataloaders
from src.models.baseline_cnn import BaselineCNN
from src.models.resnet import build_resnet18
from src.utils import ensure_dir, get_device, setup_logger

logger = setup_logger("error_analysis")


def _build_model(model_name: str, num_classes: int, pretrained: bool, freeze_backbone: bool) -> torch.nn.Module:
    if model_name == "baseline_cnn":
        return BaselineCNN(num_classes=num_classes)
    if model_name == "resnet18":
        return build_resnet18(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)
    raise ValueError(f"Unknown model: {model_name}")


def save_misclassifications(
    config,
    checkpoint_path: str,
    max_samples: int = 16,
) -> List[Path]:
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

    saved = []
    ensure_dir(output_cfg["figures_dir"])

    with torch.no_grad():
        for images, labels, paths in loaders["test"]:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()
            labels = labels.numpy()
            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    path = paths[i]
                    if path is None:
                        continue
                    fig = plt.figure(figsize=(3, 3))
                    img = plt.imread(path)
                    plt.imshow(img)
                    plt.title(f"true={class_names[labels[i]]}\npred={class_names[preds[i]]}")
                    plt.axis("off")
                    out_path = Path(output_cfg["figures_dir"]) / f"misclassified_{len(saved)}.png"
                    fig.savefig(out_path, bbox_inches="tight")
                    plt.close(fig)
                    saved.append(out_path)
                    if len(saved) >= max_samples:
                        logger.info("Saved %d misclassifications", len(saved))
                        return saved

    logger.info("Saved %d misclassifications", len(saved))
    return saved
