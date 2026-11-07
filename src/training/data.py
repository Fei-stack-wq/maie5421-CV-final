"""Manifest adapter and deterministic PIL views for workflow pilots.

OUYANG owns the formal data/transform module. A dataset factory hook is exposed
by src.train so that module can replace this pilot adapter without editing the
model, loss, or checkpoint interfaces.
"""

import csv
import hashlib
import io
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
PREPROCESS = {
    "rgb": True, "resize_short_edge": 256, "center_crop": 224,
    "interpolation": "bilinear", "mean": list(MEAN), "std": list(STD),
}


def sample_seed(seed, epoch, image_id):
    value = f"{seed}:{epoch}:{image_id}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def clean_image(image):
    return TF.center_crop(TF.resize(image.convert("RGB"), 256), [224, 224])


def normalize(image):
    return TF.normalize(TF.to_tensor(image), MEAN, STD)


def pilot_view(image, rng):
    """Temporary JPEG/downsample views, not team-approved formal transforms."""
    operation = rng.choice(("jpeg", "downsample", "both"))
    if operation in ("downsample", "both"):
        image = image.resize((112, 112), Image.Resampling.BILINEAR).resize(
            (224, 224), Image.Resampling.BILINEAR
        )
    if operation in ("jpeg", "both"):
        encoded = io.BytesIO()
        image.save(encoded, format="JPEG", quality=rng.choice((50, 70)))
        encoded.seek(0)
        with Image.open(encoded) as decoded:
            image = decoded.convert("RGB").copy()
    return image


def read_manifest(path, root):
    root = Path(root).resolve()
    with open(path, newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"image_id", "rel_path", "split", "label", "generator"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"manifest requires columns {sorted(required)}")
        rows = list(reader)
    ids, paths = set(), set()
    for row in rows:
        if row["label"] not in ("0", "1"):
            raise ValueError("manifest labels must be 0 or 1")
        image_path = (root / row["rel_path"]).resolve()
        if not image_path.is_relative_to(root):
            raise ValueError("rel_path escapes data_root")
        if row["image_id"] in ids or str(image_path) in paths:
            raise ValueError("duplicate image_id or image path in manifest")
        ids.add(row["image_id"])
        paths.add(str(image_path))
        row["resolved_path"] = str(image_path)
        row["label"] = int(row["label"])
    return rows


def validate_train_val(rows):
    splits = {split: [r for r in rows if r["split"] == split] for split in ("train", "val")}
    for split, selected in splits.items():
        if {r["label"] for r in selected} != {0, 1}:
            raise ValueError(f"{split} must contain both real and AI labels")
        if any(r["generator"] in ("ADM", "GLIDE") for r in selected):
            raise ValueError("held-out generators must not occur in train/val")
        if any(not Path(r["resolved_path"]).is_file() for r in selected):
            raise FileNotFoundError(f"images are missing in {split}")
    # Prefer content hashes over file names when available.
    for key in ("sha256", "source_md5"):
        train = {r[key] for r in splits["train"] if r.get(key)}
        val = {r[key] for r in splits["val"] if r.get(key)}
        if train & val:
            raise ValueError(f"train/val content leakage in {key}")
    return splits


def verify_declared_hashes(rows, require_hashes=False):
    """Rehash manifest-declared images across all splits without model evaluation.

    This checks exact bytes only, not perceptual near-duplicates. Synthetic unit
    test fixtures may omit hashes; formal project runs may not.
    """
    seen, verified = {}, 0
    for row in rows:
        expected = row.get("sha256")
        if not expected:
            if require_hashes:
                raise ValueError("formal data requires sha256 for every manifest row")
            continue
        digest = hashlib.sha256()
        with open(row["resolved_path"], "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != expected:
            raise ValueError(f"image hash mismatch: {row['image_id']}")
        if actual in seen:
            raise ValueError(f"duplicate image bytes: {seen[actual]} and {row['image_id']}")
        seen[actual] = row["image_id"]
        verified += 1
    return {"manifest_rows": len(rows), "sha256_verified": verified,
            "missing_hashes": len(rows) - verified, "perceptual_duplicates_checked": False}


def balanced_subset(rows, limit, seed):
    if limit is None:
        return sorted(rows, key=lambda r: r["image_id"])
    if limit < 2 or limit % 2:
        raise ValueError("pilot limits must be even and at least two")
    rng = random.Random(seed)
    selected = []
    for label in (0, 1):
        group = sorted((r for r in rows if r["label"] == label), key=lambda r: r["image_id"])
        if len(group) < limit // 2:
            raise ValueError("not enough rows for balanced pilot limit")
        selected.extend(rng.sample(group, limit // 2))
    return sorted(selected, key=lambda r: r["image_id"])


class ManifestPilotDataset(Dataset):
    def __init__(self, rows, seed=5421, paired=False):
        self.rows = rows
        self.seed = seed
        self.paired = paired
        self.epoch = 0

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["resolved_path"]) as image:
            clean = clean_image(image)
        result = {
            "image": normalize(clean),
            "label": torch.tensor(row["label"], dtype=torch.float32),
            "image_id": row["image_id"],
        }
        if self.paired:
            rng = random.Random(sample_seed(self.seed, self.epoch, row["image_id"]))
            result["view"] = normalize(pilot_view(clean, rng))
        return result


class ContractDataset(Dataset):
    """Validate an external loader's identity, labels, and tensor interface."""

    def __init__(self, dataset, rows, paired=False):
        self.dataset, self.rows, self.paired = dataset, rows, paired

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

    def __getitem__(self, index):
        sample = self.dataset[index]
        expected = self.rows[index]
        if not isinstance(sample, dict) or sample.get("image_id") != expected["image_id"]:
            raise ValueError("factory sample image_id disagrees with ordered manifest IDs")
        label = sample.get("label")
        if not isinstance(label, torch.Tensor) or label.ndim != 0 or label.item() != expected["label"]:
            raise ValueError("factory sample label disagrees with manifest")
        for name in (("image", "view") if self.paired else ("image",)):
            value = sample.get(name)
            if not isinstance(value, torch.Tensor) or value.shape != (3, 224, 224) or not value.is_floating_point():
                raise ValueError(f"factory {name} must be a floating point (3,224,224) tensor")
            if not torch.isfinite(value).all():
                raise ValueError(f"factory {name} contains nonfinite values")
        return sample
