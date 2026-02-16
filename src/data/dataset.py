from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.datasets import OxfordIIITPet

from src.data.transforms import build_transforms
from src.utils import setup_logger

logger = setup_logger("dataset")


@dataclass
class DatasetSplits:
    train: Dataset
    val: Dataset
    test: Dataset
    class_names: List[str]


class IndexedPetDataset(Dataset):
    def __init__(self, base_dataset: OxfordIIITPet, indices: List[int], transform=None):
        self.base = base_dataset
        self.indices = indices
        self.transform = transform

        self._images = None
        if hasattr(base_dataset, "_images"):
            self._images = base_dataset._images
        elif hasattr(base_dataset, "images"):
            self._images = base_dataset.images

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        base_idx = self.indices[idx]
        image, label = self.base[base_idx]
        if self.transform is not None:
            image = self.transform(image)
        path = None
        if self._images is not None:
            path = self._images[base_idx]
        return image, label, path


def _get_targets(dataset: OxfordIIITPet) -> List[int]:
    if hasattr(dataset, "targets"):
        return list(dataset.targets)
    if hasattr(dataset, "_labels"):
        return list(dataset._labels)
    if hasattr(dataset, "_targets"):
        return list(dataset._targets)
    return [dataset[i][1] for i in range(len(dataset))]


def _get_class_names(dataset: OxfordIIITPet) -> List[str]:
    if hasattr(dataset, "classes"):
        return list(dataset.classes)
    if hasattr(dataset, "class_to_idx"):
        return list(dataset.class_to_idx.keys())
    return [str(i) for i in range(len(set(_get_targets(dataset))))]


def build_datasets(data_dir: str, val_ratio: float, seed: int) -> DatasetSplits:
    trainval = OxfordIIITPet(root=data_dir, split="trainval", target_type="category", download=True)
    test = OxfordIIITPet(root=data_dir, split="test", target_type="category", download=True)

    targets = _get_targets(trainval)
    indices = np.arange(len(trainval))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_ratio,
        random_state=seed,
        stratify=targets,
    )

    class_names = _get_class_names(trainval)

    train_ds = IndexedPetDataset(trainval, train_idx.tolist())
    val_ds = IndexedPetDataset(trainval, val_idx.tolist())
    test_ds = IndexedPetDataset(test, list(range(len(test))))

    return DatasetSplits(train=train_ds, val=val_ds, test=test_ds, class_names=class_names)


def build_dataloaders(
    data_dir: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    seed: int,
    use_weighted_sampler: bool,
    imbalance_ratio_threshold: float,
) -> Tuple[Dict[str, DataLoader], List[str]]:
    splits = build_datasets(data_dir, val_ratio=val_ratio, seed=seed)

    train_tf = build_transforms(image_size=image_size, train=True)
    eval_tf = build_transforms(image_size=image_size, train=False)

    splits.train.transform = train_tf
    splits.val.transform = eval_tf
    splits.test.transform = eval_tf

    targets = [label for _, label, _ in splits.train]
    class_counts = np.bincount(targets)
    max_count = class_counts.max()
    min_count = class_counts.min()
    imbalance_ratio = max_count / max(1, min_count)

    sampler = None
    if use_weighted_sampler and imbalance_ratio >= imbalance_ratio_threshold:
        weights = 1.0 / class_counts
        sample_weights = [weights[label] for label in targets]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        logger.info("Using WeightedRandomSampler due to imbalance ratio %.2f", imbalance_ratio)
    else:
        logger.info("Imbalance ratio %.2f; using standard shuffling", imbalance_ratio)

    train_loader = DataLoader(
        splits.train,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        splits.val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        splits.test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}, splits.class_names
