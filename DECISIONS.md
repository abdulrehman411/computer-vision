# Project Decision Log

## Task Selection: Image Classification on Oxford-IIIT Pet Dataset

### Why This Dataset?
1. **Non-trivial complexity**: 37 classes (dog/cat breeds) with 7,349 images—enough to test real ML concerns without being massive.
2. **Real-world challenges**: Varied poses, lighting, backgrounds, occlusions; not a toy dataset like MNIST.
3. **Accessibility**: Public, well-documented, torchvision integration; downloads automatically.
4. **Classification framing**: Clear task boundaries enable focused evaluation on core CV fundamentals.

### Why Not Other Tasks?
- **Object Detection**: Adds complexity (bounding boxes, NMS) beyond assessment scope; annotation overhead.
- **Segmentation**: Higher-complexity annotations; typically requires larger datasets; less common baseline comparisons.
- Classification was chosen for clarity and bandwidth constraints.

---

## Model Choices

### Baseline: Simple CNN
- **Design**: 3 conv blocks (32→64→128 channels), adaptive GAP, small MLP.
- **Rationale**: CPU-friendly, interpretable, sets a lower bound. Tests core convolution/pooling concepts.
- **Expected performance**: ~60-70% top-1 accuracy (reasonable for 37 classes).

### Stronger: ResNet18 + Transfer Learning
- **Design**: ImageNet-pretrained backbone, tunable head. Unfrozen by default (fine-tuning).
- **Rationale**: Modern standard; transfer learning is realistic; modest compute footprint for laptops.
- **Expected performance**: ~85-92% top-1 accuracy.

### Why Not Larger Models?
- EfficientNet, ConvNeXt would add marginal gains (<5%); assessment prioritizes engineering, not SOTA.
- Training on CPU feasible; GPU access unnecessary.

---

## Data Strategy

### Train/Val/Test Split
- **Approach**: Stratified 70/15/15 from official trainval + test split.
- **Rationale**: Stratification preserves class distribution; reduces variance in metrics.

### Augmentation
- **Applied**: RandomResizedCrop, HorizontalFlip, ColorJitter, RandomRotation.
- **Rationale**: Prevents overfitting; improves robustness to real-world variations (pose, lighting).
- **Not applied to val/test**: Consistent evaluation baseline.

### Class Imbalance Handling
- **Detection**: Compute imbalance ratio (max_count / min_count).
- **Mitigation**: WeightedRandomSampler when ratio > threshold (default 1.5).
- **Rationale**: Oxford-IIIT has moderate imbalance; weighted sampling is simpler than losses, still effective.

---

## Hyperparameter Selection

| Parameter | Baseline CNN | ResNet18 | Rationale |
|-----------|--------------|----------|-----------|
| Epochs | 8 | 8 | Fast iteration; CPU constraint. |
| LR | 0.001 | 0.0003 | ResNet fine-tuning uses smaller LR. |
| Batch size | 32 | 32 | CPU + memory trade-off. |
| Optimizer | Adam | Adam | Robust, adaptive; no manual scheduling needed. |
| Weight decay | 0.0001 | 0.0001 | Mild L2 regularization. |
| Image size | 224 | 224 | ImageNet standard; ResNet backbone trained on this. |

### Tuning Approach
- **Grid search scope**: Limited (epochs, batch size, LR only).
- **Rationale**: Time/compute constraints; diminishing returns beyond simple sweep.

---

## Evaluation Design

### Metrics
- **Accuracy**: Simple, interpretable for multi-class classification.
- **Confusion matrix**: Reveals per-class patterns, failure modes.
- **Classification report**: Precision, recall, F1 per class.

### Error Analysis
- **Misclassification samples**: Visualize 10–20 failure cases.
- **Rationale**: Qualitative insight into model blindness (e.g., pose confusion, hard negatives).

### Test Set Use
- **Frozen**: Used only after training. No validation/tuning on test data.
- **Rationale**: Honest assessment; avoid overfitting to test set.

---

## Engineering Decisions

### Project Structure
```
src/
  data/      → dataset loading, transforms, analysis
  models/    → model definitions (reusable)
  training/  → training logic (batch loop, checkpointing)
  eval/      → evaluation, metrics, error analysis
  inference/ → inference pipeline (separate from training)
configs/     → YAML config (reproducibility)
scripts/     → entry points (run.sh wrapper, CLI tools)
```

**Rationale**: Clean separation of concerns; easy to reuse, test, and productionize.

### Checkpoint Metadata
- Saved with model: class names, image size, pretrained flag, freeze status.
- **Rationale**: Inference reproducibility; no need to manually track config.

### Inference Decoupling
- `src/inference/predict.py`: Standalone module, no training dependencies.
- `scripts/infer.py`: CLI wrapper, no YAML config required (metadata from checkpoint).
- **Rationale**: Deployment-ready; can ship checkpoint + inference script independently.

### Configuration (YAML)
- Centralized `configs/config.yaml` for all hyperparameters.
- **Rationale**: Version control, reproducibility, easy sweeps.

---

## If More Compute Were Available

### Scale Up (Compute-Aware)
1. **Larger models**: EfficientNet-B3/B4, ConvNeXt-S/B.
2. **Higher resolution**: 384–448px (quadratic memory cost).
3. **Larger batch sizes**: 64, 128, 256 (better gradient estimates).
4. **Longer training**: 20–50 epochs with LR scheduling (cosine annealing, warmup).

### Data & Augmentation
1. **Stronger augmentations**: RandAugment, MixUp, CutMix.
2. **Pseudo-labeling**: Train on larger unlabeled pet datasets.
3. **Hard example mining**: Identify mispredictions, retrain on hard cases.

### Ensembling & TTA
1. **Model ensemble**: Average predictions from baseline + ResNet18 + larger model.
2. **Test-time augmentation**: Multi-crop, multi-scale inference.

### Hyperparameter Optimization
1. **Systematic search**: Grid/random search on LR, weight decay, warmup schedule.
2. **AutoML**: Population-based training (PBT) or Optuna for efficient tuning.

### Methodology
1. **Distributed training**: DataParallel or DistributedDataParallel on multi-GPU.
2. **Mixed precision**: Automatic mixed precision (AMP) for 2–3x speedup.
3. **Label smoothing & regularization**: Sweep label_smoothing, dropout, cutout.

---

## Reproducibility

- **Seeding**: torch, numpy, random all seeded to `project.seed` (42).
- **Deterministic ops**: `torch.backends.cudnn.deterministic = True` (slight perf penalty, deterministic output).
- **Config versioning**: Commit all configs to git; link results to commit hash.
- **Checkpoint metadata**: Every checkpoint includes training config; can recreate environment.

---

## Known Limitations & Future Work

1. **CPU bottleneck**: No GPU; training takes 2–5 min per model. Acceptable for assessment; scales linearly with compute.
2. **Class imbalance**: Mild (max ~2.5x ratio); weighted sampling sufficient. Could explore focal loss or cost-sensitive learning.
3. **Limited hyperparameter search**: 8 epochs + single LR; space is small. More compute → fuller sweep.
4. **No online hard example mining**: Could boost OOD robustness; deferred due to time.
5. **No external data**: Pure supervised learning. Pretraining on larger datasets (ImageNet) used via transfer learning.

---

## Summary

This project balances **ML correctness** (proper train/val/test, stratification, weighted sampling) with **engineering discipline** (modular code, reproducibility, clean interfaces) under realistic constraints. Models are modest (baseline CNN, ResNet18), training is CPU-feasible, and evaluation is thorough (metrics + error analysis). The codebase is deployment-ready and easily extensible.
