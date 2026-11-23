"""Strict manifest-based datasets and reproducible epoch sampling."""
import csv
import hashlib
from pathlib import Path, PurePosixPath

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler

from .config import DataConfig
from .transforms import EvalTransform, PairedViewTransform, stable_seed

SPLITS = ("train", "val", "test_seen", "test_unseen_ADM", "test_unseen_GLIDE", "fixture")
GENERATORS = {
    "train": {"real", "BigGAN", "stable_diffusion_v_1_4"},
    "val": {"real", "BigGAN", "stable_diffusion_v_1_4"},
    "test_seen": {"real", "BigGAN", "stable_diffusion_v_1_4"},
    "test_unseen_ADM": {"real", "ADM"},
    "test_unseen_GLIDE": {"real", "GLIDE"},
    "fixture": {"real", "BigGAN", "stable_diffusion_v_1_4"},
}


def resolve_image_path(root, rel_path):
    root = Path(root).resolve()
    rel = PurePosixPath(rel_path)
    if rel.is_absolute() or ".." in rel.parts or "\\" in rel_path or ":" in rel_path:
        raise ValueError(f"Unsafe manifest path: {rel_path}")
    path = (root / str(rel)).resolve()
    if path == root or root not in path.parents:
        raise ValueError(f"Image path escapes root: {rel_path}")
    return path


class GenImageDataset(Dataset):
    def __init__(self, manifest, split, root=None, condition="A", seed=5421,
                 suite="clean", config=None):
        self.manifest = Path(manifest).resolve()
        self.root = Path(root).resolve() if root is not None else self.manifest.parent
        self.config = config or DataConfig()
        self.split, self.condition, self.seed = split, condition, int(seed)
        self.suite = suite
        if split not in SPLITS:
            raise ValueError(f"Unknown split: {split}")
        if condition not in ("A", "B", "C"):
            raise ValueError("condition must be A, B or C")
        if condition != "A" and split not in ("train", "fixture"):
            raise ValueError("Paired views are training-only; evaluate all models with condition='A'")
        if split in ("train", "fixture") and condition != "A" and suite not in ("clean", "bias_control_jpeg95"):
            raise ValueError("B/C training allows clean or bias_control_jpeg95 base only")
        self.manifest_sha256 = hashlib.sha256(self.manifest.read_bytes()).hexdigest()
        with self.manifest.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"image_id", "split", "label", "generator", "rel_path"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Manifest missing columns: {sorted(required - set(reader.fieldnames or []))}")
            all_rows = list(reader)
        ids, paths = set(), set()
        for row in all_rows:
            row_split = row["split"]
            if row_split not in SPLITS or row["generator"] not in GENERATORS[row_split]:
                raise ValueError(f"Unexpected split/generator: {row['image_id']}")
            if row["label"] not in ("0", "1") or int(row["label"]) != int(row["generator"] != "real"):
                raise ValueError(f"Invalid label/generator: {row['image_id']}")
            if not row["image_id"] or row["image_id"] in ids:
                raise ValueError(f"Empty or duplicate image_id: {row['image_id']}")
            path = resolve_image_path(self.root, row["rel_path"])
            if path in paths:
                raise ValueError(f"Duplicate image path: {row['rel_path']}")
            ids.add(row["image_id"])
            paths.add(path)
        self.rows = [row for row in all_rows if row["split"] == split]
        if not self.rows:
            raise ValueError(f"No rows for split {split!r} in {self.manifest}")
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        self.eval_transform = EvalTransform(self.config, suite)
        self.paired_transform = PairedViewTransform(self.config, suite == "bias_control_jpeg95")

    @property
    def epoch(self):
        return int(self._epoch.item())

    def set_epoch(self, epoch):
        if int(epoch) != epoch or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self._epoch.fill_(int(epoch))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = resolve_image_path(self.root, row["rel_path"])
        if not path.is_file():
            raise FileNotFoundError(f"Missing image {row['image_id']}: {path}. Obtain the original Images/ dataset.")
        try:
            with Image.open(path) as opened:
                opened.load()
                if self.condition in ("B", "C"):
                    output = self.paired_transform(opened, stable_seed(self.seed, self.epoch, row["image_id"]))
                else:
                    output = {"image": self.eval_transform(opened), "view_params": "{}"}
        except (OSError, ValueError) as error:
            raise RuntimeError(f"Cannot decode/transform {row['image_id']}: {path}") from error
        output.update({key: row[key] for key in ("image_id", "generator", "split", "rel_path")})
        output["label"] = torch.tensor(float(row["label"]), dtype=torch.float32)
        output["transformation"] = self.suite
        return output


class EpochSampler(Sampler):
    def __init__(self, dataset, shuffle=True):
        self.dataset, self.shuffle = dataset, shuffle

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        if not self.shuffle:
            return iter(range(len(self.dataset)))
        generator = torch.Generator()
        generator.manual_seed(stable_seed(self.dataset.seed, self.dataset.epoch, "batch-order"))
        return iter(torch.randperm(len(self.dataset), generator=generator).tolist())


def make_dataloader(dataset, batch_size=16, shuffle=None, num_workers=0,
                    persistent_workers=False, drop_last=False, pin_memory=False):
    if persistent_workers and num_workers == 0:
        raise ValueError("persistent_workers requires num_workers > 0")
    if shuffle is None:
        shuffle = dataset.split in ("train", "fixture")
    worker_generator = torch.Generator().manual_seed(dataset.seed)
    return DataLoader(dataset, batch_size=batch_size, sampler=EpochSampler(dataset, shuffle),
                      num_workers=num_workers, persistent_workers=persistent_workers,
                      drop_last=drop_last, pin_memory=pin_memory, generator=worker_generator)
