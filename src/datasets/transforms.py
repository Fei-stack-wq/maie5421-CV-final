"""PIL pixel-domain transforms followed by shared ImageNet normalization.

All corruption operates on the same 224px canvas, before normalization.
Training randomness is local; it never consumes a global RNG or sees a label.
"""
import hashlib
import io
import json
import random

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps

from .config import DataConfig

EVAL_SUITES = (
    "clean", "jpeg_q70", "jpeg_q50", "downsample_0.5",
    "gaussian_blur_1.0", "bias_control_jpeg95",
)


def stable_seed(seed, epoch, image_id):
    payload = json.dumps([int(seed), int(epoch), str(image_id)], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def jpeg_roundtrip(image, quality):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality), subsampling=2,
               optimize=False, progressive=False)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB").copy()


def downsample_roundtrip(image, factor):
    w, h = image.size
    size = (max(1, int(w * factor)), max(1, int(h * factor)))
    return image.resize(size, Image.Resampling.BICUBIC).resize(
        (w, h), Image.Resampling.BICUBIC)


def prepare_image(image, config=None, bias_control=False):
    config = config or DataConfig()
    image = ImageOps.exif_transpose(image).convert("RGB")
    if bias_control:
        image = image.resize((config.control_size, config.control_size),
                             Image.Resampling.BICUBIC)
        image = jpeg_roundtrip(image, config.control_quality)
    w, h = image.size
    size = config.resize_size
    # Fix the short edge exactly; floor the long edge, like torchvision Resize.
    target = (size, int(h * size / w)) if w <= h else (int(w * size / h), size)
    image = image.resize(target, Image.Resampling.BICUBIC)
    left = round((image.width - config.image_size) / 2)
    top = round((image.height - config.image_size) / 2)
    return image.crop((left, top, left + config.image_size, top + config.image_size))


def to_tensor(image, config=None):
    config = config or DataConfig()
    array = np.array(image, dtype=np.float32, copy=True) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    mean = torch.tensor(config.mean, dtype=torch.float32)[:, None, None]
    std = torch.tensor(config.std, dtype=torch.float32)[:, None, None]
    return (tensor - mean) / std


def perturb_image(image, suite):
    """Apply a fixed evaluation corruption to an already prepared RGB image."""
    if suite in ("clean", "bias_control_jpeg95"):
        return image.copy()
    if suite == "jpeg_q70":
        return jpeg_roundtrip(image, 70)
    if suite == "jpeg_q50":
        return jpeg_roundtrip(image, 50)
    if suite == "downsample_0.5":
        return downsample_roundtrip(image, 0.5)
    if suite == "gaussian_blur_1.0":
        return image.filter(ImageFilter.GaussianBlur(radius=1.0))
    raise ValueError(f"Unknown evaluation suite {suite!r}; choose from {EVAL_SUITES}")


class EvalTransform:
    def __init__(self, config=None, suite="clean"):
        if suite not in EVAL_SUITES:
            raise ValueError(f"Unknown evaluation suite: {suite}")
        self.config = config or DataConfig()
        self.suite = suite

    def __call__(self, image):
        base = prepare_image(image, self.config, self.suite == "bias_control_jpeg95")
        return to_tensor(perturb_image(base, self.suite), self.config)


class PairedViewTransform:
    def __init__(self, config=None, bias_control=False):
        self.config = config or DataConfig()
        self.bias_control = bias_control

    def __call__(self, image, seed):
        cfg = self.config
        rng = random.Random(int(seed))
        mode = rng.choice(("jpeg", "downsample", "both"))
        params = {"mode": mode, "jpeg_quality": None, "downsample_factor": None}
        base = prepare_image(image, cfg, self.bias_control)
        view = base.copy()
        if mode in ("downsample", "both"):
            factor = rng.uniform(cfg.downsample_min, cfg.downsample_max)
            params["downsample_factor"] = factor
            view = downsample_roundtrip(view, factor)
        if mode in ("jpeg", "both"):
            quality = rng.randint(cfg.jpeg_quality_min, cfg.jpeg_quality_max)
            params["jpeg_quality"] = quality
            view = jpeg_roundtrip(view, quality)
        return {"image": to_tensor(base, cfg), "view": to_tensor(view, cfg),
                "view_params": json.dumps(params, sort_keys=True)}
