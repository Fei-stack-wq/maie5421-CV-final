"""Versioned data settings shared by training, evaluation and CLI."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    version: str = "ouyang-v1"
    image_size: int = 224
    resize_size: int = 256
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)
    jpeg_quality_min: int = 50
    jpeg_quality_max: int = 95
    downsample_min: float = 0.5
    downsample_max: float = 0.9
    control_size: int = 256
    control_quality: int = 95

    def __post_init__(self):
        for name in ("image_size", "resize_size", "jpeg_quality_min", "jpeg_quality_max", "control_size", "control_quality"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.image_size < 1 or self.resize_size < self.image_size:
            raise ValueError("Require resize_size >= image_size > 0")
        if (len(self.mean) != 3 or len(self.std) != 3
                or not all(math.isfinite(v) for v in (*self.mean, *self.std))
                or any(s <= 0 for s in self.std)):
            raise ValueError("Require three normalization means and positive stds")
        if not 1 <= self.jpeg_quality_min <= self.jpeg_quality_max <= 95:
            raise ValueError("JPEG training quality must be in [1, 95]")
        if not 0 < self.downsample_min <= self.downsample_max < 1:
            raise ValueError("Require 0 < downsample_min <= downsample_max < 1")
        if self.control_size < 1 or not 1 <= self.control_quality <= 95:
            raise ValueError("Invalid bias control settings")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_json(cls, path):
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in ("mean", "std"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)

    @property
    def fingerprint(self):
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()
