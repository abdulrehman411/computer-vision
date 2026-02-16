# Oxford-IIIT Pet Classification (CV Assessment)

## Overview
This project implements a clean, end-to-end computer vision workflow for **image classification** using the **Oxford-IIIT Pet** dataset. It includes data inspection, preprocessing, training two models (baseline CNN and fine-tuned ResNet18), evaluation with error analysis, and a production-ready inference script.

**Why this dataset?**
- Non-trivial size and class count (37 classes, 7,349 images).
- Real-world variability (pose, lighting, backgrounds).
- Public, well-documented, and commonly used for CV benchmarking.

## Dataset
- **Name:** Oxford-IIIT Pet
- **Size:** 7,349 images (trainval + test)
- **Classes:** 37 (breeds)
- **Label format:** Image-level category

### Data understanding
Generate dataset summary and class distribution:

- Run: `python scripts/data_report.py --data-dir data --output-dir outputs`

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
- `python scripts/train_baseline.py --config configs/config.yaml`

### Stronger model
- ResNet18 (ImageNet pretrained)
- Optional backbone freezing in config

Run:
- `python scripts/train_resnet.py --config configs/config.yaml`

## Evaluation
Evaluate a checkpoint on the test set:
- `python scripts/evaluate.py --config configs/config.yaml --checkpoint outputs/checkpoints/resnet18_best.pth`

This also saves a confusion matrix to `outputs/figures/confusion_matrix.png`.

## Error Analysis
Misclassification visualization:
- `python scripts/error_analysis.py --config configs/config.yaml --checkpoint outputs/checkpoints/resnet18_best.pth --max-samples 16`

## Inference
Run inference on a single image:
- `python scripts/infer.py --image /path/to/image.jpg --checkpoint outputs/checkpoints/resnet18_best.pth`

## Engineering Notes
- Stratified train/val split from the official trainval set.
- Weighted sampling is enabled when imbalance ratio exceeds threshold.
- Basic augmentations applied in training (crop, flip, jitter, rotation).
- Clean separation between training and inference.

## If More Compute Were Available
- Train larger backbones (EfficientNet, ConvNeXt) and tune resolution.
- Extensive hyperparameter search and regularization sweeps.
- Hard example mining and label noise inspection.
- Ensembling and test-time augmentation.

## Reproducibility
- Seeded for deterministic results (where possible).
- All key parameters in `configs/config.yaml`.

## License
This project uses the Oxford-IIIT Pet dataset; see its official license for usage terms.
