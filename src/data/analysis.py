import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import pandas as pd
from torchvision.datasets import OxfordIIITPet

from src.utils import ensure_dir, setup_logger

logger = setup_logger("data_analysis")


def dataset_summary(data_dir: str) -> Dict:
    trainval = OxfordIIITPet(root=data_dir, split="trainval", target_types="category", download=True)
    test = OxfordIIITPet(root=data_dir, split="test", target_types="category", download=True)

    targets = None
    if hasattr(trainval, "targets"):
        targets = list(trainval.targets)
    elif hasattr(trainval, "_labels"):
        targets = list(trainval._labels)
    else:
        targets = [trainval[i][1] for i in range(len(trainval))]

    class_names = list(trainval.classes) if hasattr(trainval, "classes") else None
    counts = Counter(targets)

    summary = {
        "dataset": "Oxford-IIIT Pet",
        "trainval_size": len(trainval),
        "test_size": len(test),
        "num_classes": len(counts),
        "label_format": "image-level categorical",
        "class_distribution": counts,
        "class_names": class_names,
    }
    return summary


def write_report(data_dir: str, output_dir: str) -> None:
    ensure_dir(output_dir)
    summary = dataset_summary(data_dir)
    output_path = Path(output_dir) / "dataset_summary.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    counts = summary["class_distribution"]
    df = pd.DataFrame({"class_id": list(counts.keys()), "count": list(counts.values())})
    df = df.sort_values("count", ascending=False)
    df.to_csv(Path(output_dir) / "class_distribution.csv", index=False)

    logger.info("Saved dataset summary to %s", output_path)
