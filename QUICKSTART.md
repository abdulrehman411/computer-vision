# Quick Start Guide

## Verify Setup (First Time)
```bash
./run.sh scripts/validate_setup.py
```

## 1. Download Dataset & Analyze
```bash
./run.sh scripts/data_report.py
```
**Outputs:**
- `outputs/dataset_summary.json` - Dataset metadata
- `outputs/class_distribution.csv` - Class counts

## 2. Train Models
### Baseline CNN (fast, CPU-friendly)
```bash
./run.sh scripts/train_baseline.py --config configs/config.yaml
```

### ResNet18 (stronger, pretrained ImageNet)
```bash
./run.sh scripts/train_resnet.py --config configs/config.yaml
```

**Checkpoint saved to:** `outputs/checkpoints/{model_name}_best.pth`

## 3. Evaluate Model
```bash
./run.sh scripts/evaluate.py \
  --config configs/config.yaml \
  --checkpoint outputs/checkpoints/resnet18_best.pth
```

**Outputs:**
- `outputs/metrics.json` - Test accuracy
- `outputs/class_report.csv` - Per-class metrics
- `outputs/figures/confusion_matrix.png` - Confusion matrix

## 4. Error Analysis
```bash
./run.sh scripts/error_analysis.py \
  --config configs/config.yaml \
  --checkpoint outputs/checkpoints/resnet18_best.pth \
  --max-samples 16
```

**Outputs:** `outputs/figures/misclassified_*.png` - Misclassified samples

## 5. Single Image Inference
```bash
./run.sh scripts/infer.py \
  --image /path/to/pet/image.jpg \
  --checkpoint outputs/checkpoints/resnet18_best.pth
```

**Output:** JSON with predicted label and confidence

---

## Key Files
- `configs/config.yaml` - All hyperparameters (edit here)
- `src/training/train.py` - Training loop
- `src/models/` - Model definitions (baseline CNN, ResNet18)
- `src/data/dataset.py` - Data loading and preprocessing
- `src/inference/predict.py` - Inference logic

## Troubleshooting
- **Import errors?** Run: `./run.sh scripts/validate_setup.py`
- **Out of memory?** Reduce `batch_size` in `configs/config.yaml`
- **Slow training?** Reduce `epochs` or `image_size` in config

---

See `README.md` for full documentation.
