"""Adapt OUYANG's datasets to the canonical training factory contract."""

import hashlib
from pathlib import Path

from torch.utils.data import Dataset

from src.datasets import DataConfig, GenImageDataset
from src.training.data import balanced_subset


class OrderedDataset(Dataset):
    def __init__(self, dataset, limit, seed):
        self.dataset = dataset
        rows = balanced_subset([{**row, "label": int(row["label"])} for row in dataset.rows], limit, seed)
        self.image_ids = [row["image_id"] for row in rows]
        positions = {row["image_id"]: i for i, row in enumerate(dataset.rows)}
        self.indices = [positions[image_id] for image_id in self.image_ids]

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        self.dataset.set_epoch(epoch)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


def build_datasets(config):
    """Return ordered train/clean-val datasets and checkpoint metadata.

    data_config must explicitly name the versioned OUYANG JSON file.
    The training entry point owns shuffling; this adapter only orders samples.
    """
    if not config.get("data_config"):
        raise ValueError("OUYANG factory requires an explicit data_config JSON path")
    path = Path(config["data_config"])
    data = DataConfig.from_json(path)
    shared = dict(manifest=config["data_manifest"], root=config["data_root"],
                  seed=config["seed"], config=data)
    train = GenImageDataset(split="train", condition=config["condition"], **shared)
    val = GenImageDataset(split="val", condition="A", suite="clean", **shared)
    metadata = {
        "provider": "src.datasets", "version": data.version,
        "data_config": data.to_dict(), "config_fingerprint": data.fingerprint,
        "config_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rgb": True, "exif_transpose": True, "resize_short_edge": data.resize_size,
        "center_crop": data.image_size, "interpolation": "bicubic",
        "mean": list(data.mean), "std": list(data.std), "validation_suite": "clean",
        "paired_view": "JPEG/downsample/both, parameters from data_config",
    }
    return (OrderedDataset(train, config["train_limit"], config["seed"]),
            OrderedDataset(val, config["val_limit"], config["seed"]), metadata)
