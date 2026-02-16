import json
from pathlib import Path
from typing import Dict, Tuple

import torch
import yaml
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import build_dataloaders
from src.models.baseline_cnn import BaselineCNN
from src.models.resnet import build_resnet18
from src.utils import CheckpointMeta, ensure_dir, get_device, seed_everything, setup_logger

logger = setup_logger("train")


def _build_model(model_name: str, num_classes: int, pretrained: bool, freeze_backbone: bool) -> nn.Module:
    if model_name == "baseline_cnn":
        return BaselineCNN(num_classes=num_classes)
    if model_name == "resnet18":
        return build_resnet18(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)
    raise ValueError(f"Unknown model: {model_name}")


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool = True,
) -> Tuple[float, float]:
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in tqdm(loader, desc="train" if train else "eval"):
        images = images.to(device)
        labels = labels.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / max(1, total)
    accuracy = correct / max(1, total)
    return avg_loss, accuracy


def train_from_config(config_path: str, model_override: str = None) -> Path:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed_everything(config["project"]["seed"])

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    output_cfg = config["output"]

    model_name = model_override or model_cfg["name"]
    loaders, class_names = build_dataloaders(
        data_dir=data_cfg["data_dir"],
        image_size=data_cfg["image_size"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        val_ratio=data_cfg["val_ratio"],
        seed=config["project"]["seed"],
        use_weighted_sampler=data_cfg["use_weighted_sampler"],
        imbalance_ratio_threshold=data_cfg["imbalance_ratio_threshold"],
    )

    device = get_device()
    model = _build_model(
        model_name=model_name,
        num_classes=len(class_names),
        pretrained=model_cfg.get("pretrained", False),
        freeze_backbone=model_cfg.get("freeze_backbone", False),
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg.get("label_smoothing", 0.0))
    optimizer = Adam(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])

    ensure_dir(output_cfg["checkpoints_dir"])
    best_acc = 0.0
    best_path = None

    for epoch in range(train_cfg["epochs"]):
        logger.info("Epoch %d/%d", epoch + 1, train_cfg["epochs"])
        train_loss, train_acc = _run_epoch(model, loaders["train"], criterion, optimizer, device, train=True)
        val_loss, val_acc = _run_epoch(model, loaders["val"], criterion, optimizer, device, train=False)

        logger.info(
            "Train loss %.4f acc %.4f | Val loss %.4f acc %.4f",
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )

        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint = {
                "model_state": model.state_dict(),
                "meta": CheckpointMeta(
                    model_name=model_name,
                    class_names=class_names,
                    image_size=data_cfg["image_size"],
                    pretrained=model_cfg.get("pretrained", False),
                    freeze_backbone=model_cfg.get("freeze_backbone", False),
                ).__dict__,
            }
            best_path = Path(output_cfg["checkpoints_dir"]) / f"{model_name}_best.pth"
            torch.save(checkpoint, best_path)
            logger.info("Saved best checkpoint to %s", best_path)

    metrics_path = Path(output_cfg["metrics_path"])
    ensure_dir(metrics_path.parent)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"best_val_accuracy": best_acc, "model": model_name}, f, indent=2)

    if best_path is None:
        best_path = Path(output_cfg["checkpoints_dir"]) / f"{model_name}_last.pth"
        torch.save({"model_state": model.state_dict()}, best_path)

    return best_path
