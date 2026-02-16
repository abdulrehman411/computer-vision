# Oxford-IIIT Pet Classification (CV Assessment)

## Overview
This project implements a clean, end-to-end computer vision workflow for **image classification** using the **Oxford-IIIT Pet** dataset. It includes data inspection, preprocessing, training two models (baseline CNN and fine-tuned ResNet18), evaluation with error analysis, and a production-ready inference script.

**Why this dataset?**
- Non-trivial size and class count (37 classes, 7,349 images).
- Real-world variability (pose, lighting, backgrounds).
- Public, well-documented, and commonly used for CV benchmarking.

## Setup

### Prerequisites
- Python 3.9+
- macOS/Linux (tested on macOS)

### Installation

1. Clone and setup venv:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Validate setup:
```bash
./run.sh scripts/validate_setup.py
```

## Dataset
- **Name:** Oxford-IIIT Pet
- **Size:** 7,349 images (trainval + test)
- **Classes:** 37 (breeds)
- **Label format:** Image-level category

### Data understanding
Generate dataset summary and class distribution:

- Run: `./run.sh scripts/data_report.py --data-dir data --output-dir outputs`

Outputs:
- `outputs/dataset_summary.json`
- `outputs/class_distribution.csv`

## Project Structure
- data/ (downloaded dataset)
- configs/ (YAML configuration)
- src/
  - data/ (dataset loading, transforms, analysis)
  - models/ (baseline CNN, ResNet18)
  - training/ (training loop)
  - eval/ (metrics + error analysis)
  - inference/ (inference logic)
- scripts/ (entrypoints)
- outputs/ (metrics, figures, checkpoints)

## Training
### Baseline model
- Simple CNN (3 conv blocks + GAP)
- CPU-friendly

Run:
- `./run.sh scripts/train_baseline.py --config configs/config.yaml`

### Stronger model
- ResNet18 (ImageNet pretrained)
- Optional backbone freezing in config

Run:
- `./run.sh scripts/train_resnet.py --config configs/config.yaml`

## Evaluation
Evaluate a checkpoint on the test set:
- `./run.sh scripts/evaluate.py --config configs/config.yaml --checkpoint outputs/checkpoints/resnet18_best.pth`

This also saves a confusion matrix to `outputs/figures/confusion_matrix.png`.

## Error Analysis
Misclassification visualization:
- `./run.sh scripts/error_analysis.py --config configs/config.yaml --checkpoint outputs/checkpoints/resnet18_best.pth --max-samples 16`

## Inference
Run inference on a single image:
- `./run.sh scripts/infer.py --image /path/to/image.jpg --checkpoint outputs/checkpoints/resnet18_best.pth`

## Engineering Notes
- Stratified train/val split from the official trainval set.
- Weighted sampling is enabled when imbalance ratio exceeds threshold.
- Basic augmentations applied in training (crop, flip, jitter, rotation).
- Clean separation between training and inference.

## If More Compute Were Available

### Scaling Training
- **Multi-GPU training**: Implement DistributedDataParallel for faster epoch times on larger models
- **Mixed precision (FP16)**: Use automatic mixed precision to reduce memory and speed up training
- **Larger batch sizes**: Scale to 128-256 with learning rate adjustments (linear scaling rule)
- **Extended training**: Increase epochs to 30-50 with learning rate scheduling (cosine annealing, warmup)
- **Cross-validation**: Run 5-fold CV to get more robust performance estimates and reduce variance

### Model Improvements
- **Modern architectures**: EfficientNetV2, ConvNeXt, Vision Transformers (ViT, Swin)
- **Higher resolution**: Train at 384x384 or 512x512 for finer-grained breed discrimination
- **Architecture search**: Explore depth, width, and compound scaling for optimal capacity
- **Advanced regularization**: Stochastic depth, mixup/cutmix, dropout tuning
- **Fine-grained techniques**: Attention mechanisms, bilinear pooling for subtle breed features
- **Ensembling**: Train multiple models with different initializations and architectures

### Data Improvements
- **Data augmentation search**: AutoAugment, RandAugment, TrivialAugment for optimal policy
- **Advanced augmentations**: CutOut, GridMask, AugMax for improved generalization
- **External data**: Leverage additional dog/cat images from ImageNet or COCO for pretraining
- **Hard example mining**: Identify and oversample frequently misclassified breeds
- **Label noise analysis**: Review borderline cases, potential annotation errors in difficult classes
- **Test-time augmentation**: Average predictions over multiple augmented views at inference
- **Class balancing strategies**: Compare focal loss, class-balanced loss, vs weighted sampling

### Analysis & Debugging
- **Per-class error analysis**: Deep dive into confused breed pairs (e.g., similar terriers)
- **Feature visualization**: Grad-CAM/attention maps to verify model focuses on discriminative regions
- **Learning curve analysis**: Plot training curves to diagnose underfitting vs overfitting
- **Confusion patterns**: Identify systematic errors (e.g., all longhaired breeds confused)
- **Data subset experiments**: Train on progressively larger subsets to estimate saturation point

## Reproducibility
- Seeded for deterministic results (where possible).
- All key parameters in `configs/config.yaml`.

## License
This project uses the Oxford-IIIT Pet dataset; see its official license for usage terms.
