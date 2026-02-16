#!/usr/bin/env python
"""Validate project setup and dependencies."""

import sys
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 60)
print("CV Assessment Project - Setup Validation")
print("=" * 60)

# Check Python version
print(f"\n✓ Python version: {sys.version.split()[0]}")

# Check imports
print("\nChecking imports...")
try:
    import torch
    print(f"  ✓ torch {torch.__version__}")
except ImportError as e:
    print(f"  ✗ torch: {e}")
    sys.exit(1)

try:
    import torchvision
    print(f"  ✓ torchvision {torchvision.__version__}")
except ImportError as e:
    print(f"  ✗ torchvision: {e}")
    sys.exit(1)

try:
    import numpy
    print(f"  ✓ numpy {numpy.__version__}")
except ImportError as e:
    print(f"  ✗ numpy: {e}")
    sys.exit(1)

try:
    import yaml
    print(f"  ✓ yaml (pyyaml)")
except ImportError as e:
    print(f"  ✗ yaml: {e}")
    sys.exit(1)

try:
    from src.utils import get_device, setup_logger, seed_everything
    print("  ✓ src.utils")
except ImportError as e:
    print(f"  ✗ src.utils: {e}")
    sys.exit(1)

try:
    from src.data.dataset import build_dataloaders
    print("  ✓ src.data.dataset")
except ImportError as e:
    print(f"  ✗ src.data.dataset: {e}")
    sys.exit(1)

try:
    from src.models.baseline_cnn import BaselineCNN
    print("  ✓ src.models.baseline_cnn")
except ImportError as e:
    print(f"  ✗ src.models.baseline_cnn: {e}")
    sys.exit(1)

try:
    from src.models.resnet import build_resnet18
    print("  ✓ src.models.resnet")
except ImportError as e:
    print(f"  ✗ src.models.resnet: {e}")
    sys.exit(1)

try:
    from src.training.train import train_from_config
    print("  ✓ src.training.train")
except ImportError as e:
    print(f"  ✗ src.training.train: {e}")
    sys.exit(1)

try:
    from src.eval.evaluate import evaluate_checkpoint
    print("  ✓ src.eval.evaluate")
except ImportError as e:
    print(f"  ✗ src.eval.evaluate: {e}")
    sys.exit(1)

try:
    from src.inference.predict import predict_image
    print("  ✓ src.inference.predict")
except ImportError as e:
    print(f"  ✗ src.inference.predict: {e}")
    sys.exit(1)

# Check directories
print("\nChecking directories...")
dirs = ["data", "configs", "outputs", "scripts", "src"]
for d in dirs:
    if Path(d).exists():
        print(f"  ✓ {d}/")
    else:
        print(f"  ✗ {d}/ (missing)")

# Check files
print("\nChecking key files...")
files = [
    "configs/config.yaml",
    "README.md",
    "requirements.txt",
    ".gitignore",
]
for f in files:
    if Path(f).exists():
        print(f"  ✓ {f}")
    else:
        print(f"  ✗ {f} (missing)")

# Check device
print("\nDevice info:")
device = get_device()
print(f"  ✓ torch device: {device}")

print("\n" + "=" * 60)
print("✓ All checks passed!")
print("=" * 60)
print("\nNext steps:")
print("1. Download dataset: ./run.sh scripts/data_report.py")
print("2. Train baseline:   ./run.sh scripts/train_baseline.py")
print("3. Train stronger:   ./run.sh scripts/train_resnet.py")
print("4. Evaluate:         ./run.sh scripts/evaluate.py --checkpoint outputs/checkpoints/resnet18_best.pth")
print("5. Infer:            ./run.sh scripts/infer.py --image <path> --checkpoint outputs/checkpoints/resnet18_best.pth")
print()
