from typing import Dict

import torch
from PIL import Image
from torchvision import transforms

from src.models.baseline_cnn import BaselineCNN
from src.models.resnet import build_resnet18
from src.utils import get_device, setup_logger

logger = setup_logger("predict")


def _build_model(model_name: str, num_classes: int, pretrained: bool, freeze_backbone: bool) -> torch.nn.Module:
    if model_name == "baseline_cnn":
        return BaselineCNN(num_classes=num_classes)
    if model_name == "resnet18":
        return build_resnet18(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)
    raise ValueError(f"Unknown model: {model_name}")


def load_checkpoint(checkpoint_path: str) -> Dict:
    device = get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    meta = checkpoint.get("meta", {})
    model_name = meta.get("model_name", "resnet18")
    class_names = meta.get("class_names", [])
    image_size = meta.get("image_size", 224)
    pretrained = meta.get("pretrained", False)
    freeze_backbone = meta.get("freeze_backbone", False)

    model = _build_model(
        model_name=model_name,
        num_classes=len(class_names) if class_names else 37,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return {
        "model": model,
        "class_names": class_names,
        "image_size": image_size,
        "device": device,
    }


def preprocess_image(image_path: str, image_size: int) -> torch.Tensor:
    tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    return tf(image).unsqueeze(0)


def predict_image(image_path: str, checkpoint_path: str) -> Dict:
    bundle = load_checkpoint(checkpoint_path)
    model = bundle["model"]
    class_names = bundle["class_names"]
    image_size = bundle["image_size"]
    device = bundle["device"]

    tensor = preprocess_image(image_path, image_size).to(device)
    with torch.no_grad():
        logits = model(tensor)
        prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(prob.argmax())

    pred_label = class_names[pred_idx] if class_names else str(pred_idx)
    return {"label": pred_label, "confidence": float(prob[pred_idx])}
