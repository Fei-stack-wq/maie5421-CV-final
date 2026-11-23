"""Versioned, atomic state_dict checkpoints with epoch-boundary resume."""

import os
import random
from pathlib import Path

import numpy as np
import torch

from src.models import BinaryResNet18

SCHEMA_VERSION = 1


def rng_state():
    np_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (np_state[0], np_state[1].tolist(), *np_state[2:]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    name, keys, pos, has_gauss, cached_gaussian = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), pos, has_gauss, cached_gaussian))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temp)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def read_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema_version")
    if payload.get("architecture") != "resnet18_binary":
        raise ValueError("unsupported checkpoint architecture")
    if payload.get("label_mapping") != {"real": 0, "AI": 1}:
        raise ValueError("unexpected checkpoint label mapping")
    return payload


def load_model(path, device="cpu"):
    """Return (eval model, metadata), with no pretrained-weight download."""
    payload = read_checkpoint(path)
    model = BinaryResNet18(weights=None)
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    metadata = {key: payload[key] for key in (
        "schema_version", "architecture", "label_mapping", "threshold", "config",
        "preprocessing", "epoch", "best_epoch", "best_val_auc", "provenance",
    )}
    return model, metadata
