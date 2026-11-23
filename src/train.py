"""Run TONG's A/B training, or integrate WANG's C loss with the same loop.

Example: python -m src.train --config examples/tong/pilot_a.yaml
"""

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torchvision
import yaml
from torch.utils.data import DataLoader

from src.models import BinaryResNet18
from src.training.checkpoint import SCHEMA_VERSION, atomic_save, read_checkpoint, restore_rng, rng_state
from src.training.data import (
    PREPROCESS, ManifestPilotDataset, ContractDataset, balanced_subset, read_manifest, validate_train_val,
    verify_declared_hashes,
)
from src.training.engine import train_epoch, validate


@dataclass
class TrainConfig:
    condition: str = "A"
    seed: int = 5421
    data_manifest: str = "Images/manifest.csv"
    data_root: str = "Images"
    run_kind: str = "pilot"
    protocol_confirmed: bool = False
    dataset_factory: str | None = None
    data_config: str | None = None
    weights: str | None = "IMAGENET1K_V1"
    freeze_policy: str = "layer4_head_frozen_bn"
    lr: float = 1e-4
    weight_decay: float = 0.01
    batch: int = 4
    accumulation_steps: int = 4
    epochs: int = 8
    early_stop: int = 2
    min_delta: float = 0.0
    amp: bool = True
    amp_init_scale: float = 65536.0
    deterministic: bool = True
    device: str = "auto"
    num_workers: int = 0
    cpu_threads: int = 2
    train_limit: int | None = 32
    val_limit: int | None = 16
    consistency_weight: float = 1.0

    def validate(self):
        if self.condition not in ("A", "B", "C"):
            raise ValueError("condition must be A, B, or C")
        if self.run_kind not in ("pilot", "formal"):
            raise ValueError("run_kind must be pilot or formal")
        if self.freeze_policy != "layer4_head_frozen_bn":
            raise ValueError("unsupported freeze_policy")
        if self.weights not in (None, "IMAGENET1K_V1"):
            raise ValueError("unsupported pretrained weights")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu, or cuda")
        for key in ("batch", "accumulation_steps", "epochs", "early_stop", "cpu_threads"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if isinstance(self.num_workers, bool) or not isinstance(self.num_workers, int) or self.num_workers < 0:
            raise ValueError("num_workers must be a nonnegative integer")
        for key in ("lr", "weight_decay", "min_delta", "consistency_weight", "amp_init_scale"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if self.lr == 0:
            raise ValueError("lr must be positive")
        if self.amp_init_scale == 0:
            raise ValueError("amp_init_scale must be positive")
        for key in ("amp", "deterministic", "protocol_confirmed"):
            if not isinstance(getattr(self, key), bool):
                raise ValueError(f"{key} must be a YAML boolean")
        for key in ("train_limit", "val_limit"):
            value = getattr(self, key)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 2 or value % 2):
                raise ValueError(f"{key} must be null or an even integer >= 2")
        if self.run_kind == "formal":
            if not self.protocol_confirmed or not self.dataset_factory:
                raise ValueError("formal runs require agreed protocol and OUYANG's dataset_factory")
            if self.train_limit is not None or self.val_limit is not None:
                raise ValueError("formal runs must use complete train and val splits")
            if self.weights != "IMAGENET1K_V1":
                raise ValueError("formal runs require ImageNet-pretrained weights")
            if self.batch * self.accumulation_steps != 16 or self.epochs > 8:
                raise ValueError("formal proposal uses effective batch 16 and at most eight epochs")
        return self


def load_config(path):
    with open(path, encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("config must be a YAML mapping")
    unknown = set(raw) - set(TrainConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown config fields: {sorted(unknown)}")
    return TrainConfig(**raw).validate()


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        return {"commit": commit, "working_tree_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "working_tree_dirty": None}


def seed_everything(seed, deterministic):
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_datasets(config):
    rows = read_manifest(config.data_manifest, config.data_root)
    splits = validate_train_val(rows)
    audit = verify_declared_hashes(rows, require_hashes=config.run_kind == "formal")
    if config.dataset_factory:
        # A factory is explicit project code, not a serialized object to unpickle.
        module_name, function_name = config.dataset_factory.split(":", 1)
        factory = getattr(importlib.import_module(module_name), function_name)
        train, val, preprocessing = factory(asdict(config))
        if not isinstance(preprocessing, dict):
            raise ValueError("factory preprocessing metadata must be a mapping")
        json.dumps(preprocessing, allow_nan=False)
        expected_train = balanced_subset(splits["train"], config.train_limit, config.seed)
        expected_val = balanced_subset(splits["val"], config.val_limit, config.seed)
        # The loader must expose these IDs without decoding or transforming the images.
        if list(train.image_ids) != [r["image_id"] for r in expected_train] or len(train) != len(expected_train):
            raise ValueError("factory train IDs must match the canonical ordered manifest split")
        if list(val.image_ids) != [r["image_id"] for r in expected_val] or len(val) != len(expected_val):
            raise ValueError("factory val IDs must match the canonical ordered manifest split")
        if not callable(getattr(train, "set_epoch", None)):
            raise ValueError("factory training dataset must implement set_epoch(epoch)")
        return (ContractDataset(train, expected_train, paired=config.condition != "A"),
                ContractDataset(val, expected_val), preprocessing, audit)
    train_rows = balanced_subset(splits["train"], config.train_limit, config.seed)
    val_rows = balanced_subset(splits["val"], config.val_limit, config.seed)
    return (
        ManifestPilotDataset(train_rows, config.seed, config.condition != "A"),
        ManifestPilotDataset(val_rows, config.seed), dict(PREPROCESS), audit,
    )


def write_history(output, history):
    temp = output / "history.csv.tmp"
    with open(temp, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    temp.replace(output / "history.csv")


def write_predictions(output, predictions):
    temp = output / "val_predictions.csv.tmp"
    with open(temp, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("image_id", "label", "p_ai"))
        writer.writerows(predictions)
    temp.replace(output / "val_predictions.csv")


def run(config, output, resume=None):
    config.validate()
    output = Path(output).resolve()
    if resume is None and output.exists() and any(output.iterdir()):
        raise FileExistsError("output directory is nonempty; choose another run or --resume last.pt")
    if resume is not None and Path(resume).resolve() != output / "last.pt":
        raise ValueError("resume must use this output directory's last.pt")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.cpu_threads)
    seed_everything(config.seed, config.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if config.device == "auto" else torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    amp = config.amp and device.type == "cuda"
    train_data, val_data, preprocessing, data_audit = make_datasets(config)
    manifest_hash = file_hash(config.data_manifest)
    # External factory setup must not influence the shared model initialization.
    seed_everything(config.seed, config.deterministic)
    model = BinaryResNet18(weights=None if resume else config.weights).to(device)
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=config.lr, weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=config.amp_init_scale, enabled=amp)
    start_epoch, best_epoch, best_auc, bad_epochs, history = 0, -1, -math.inf, 0, []
    provenance = {
        **git_revision(), "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
        "device": str(device), "amp_enabled": amp, "manifest_sha256": manifest_hash,
        "run_kind": config.run_kind,
        "purpose": "workflow validation, not final results" if config.run_kind == "pilot" else "formal experiment",
        "train_images": len(train_data), "val_images": len(val_data),
        "data_audit": data_audit,
    }
    if resume:
        state = read_checkpoint(resume)
        previous = dict(state["config"])
        # Checkpoints written before this field used GradScaler's default.
        previous.setdefault("amp_init_scale", 65536.0)
        previous.setdefault("data_config", None)
        current = asdict(config)
        previous.pop("epochs")
        current.pop("epochs")
        if previous != current or state["provenance"]["manifest_sha256"] != manifest_hash:
            raise ValueError("resume config or manifest changed; only increasing epochs is supported")
        if state["preprocessing"] != preprocessing:
            raise ValueError("resume preprocessing changed")
        if state["provenance"]["device"] != str(device):
            raise ValueError("resume device changed; use the original device")
        # last.pt is the commit point. Versioned best checkpoints permit recovery
        # if execution stopped between writing best.pt and committing last.pt.
        best_version = output / f"best_epoch_{state['best_epoch']:03d}.pt"
        best = read_checkpoint(best_version)
        if best["epoch"] != state["best_epoch"]:
            raise ValueError("versioned best checkpoint is inconsistent")
        atomic_save(best, output / "best.pt")
        write_predictions(output, best["val_predictions"])
        model.load_state_dict(state["model_state"], strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        scaler.load_state_dict(state["scaler_state"])
        restore_rng(state["rng_state"])
        start_epoch, best_epoch = state["epoch"] + 1, state["best_epoch"]
        best_auc, bad_epochs, history = state["best_val_auc"], state["bad_epochs"], state["history"]
        provenance["resumed_from_epoch"] = state["epoch"]
        if bad_epochs >= config.early_stop or start_epoch >= config.epochs:
            raise ValueError("run already stopped or reached requested epochs")
    (output / "config.yaml").write_text(yaml.safe_dump(asdict(config), sort_keys=False), encoding="utf-8")
    for epoch in range(start_epoch, config.epochs):
        started = time.perf_counter()
        train_data.set_epoch(epoch)
        if hasattr(val_data, "set_epoch"):
            val_data.set_epoch(0)
        # Reset epoch generators, so order and worker randomness do not depend on resume.
        generator = torch.Generator().manual_seed(config.seed + epoch)
        loader = DataLoader(train_data, batch_size=config.batch, shuffle=True, generator=generator,
                            num_workers=config.num_workers, worker_init_fn=seed_worker)
        val_loader = DataLoader(val_data, batch_size=config.batch, shuffle=False,
                               num_workers=config.num_workers, worker_init_fn=seed_worker,
                               generator=torch.Generator().manual_seed(config.seed))
        metrics = train_epoch(model, loader, optimizer, scaler, device, config.condition,
                              config.accumulation_steps, amp, config.consistency_weight)
        validation, predictions = validate(model, val_loader, device)
        metrics.update(validation)
        improved = metrics["val_auc"] > best_auc + config.min_delta
        if improved:
            best_auc, best_epoch, bad_epochs = metrics["val_auc"], epoch, 0
        else:
            bad_epochs += 1
        record = {"epoch": epoch, **metrics, "seconds": time.perf_counter() - started,
                  "is_best": improved, "bad_epochs": bad_epochs}
        history.append(record)
        payload = {
            "schema_version": SCHEMA_VERSION, "architecture": "resnet18_binary",
            "label_mapping": {"real": 0, "AI": 1}, "threshold": 0.5,
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(), "rng_state": rng_state(),
            "config": asdict(config), "preprocessing": preprocessing, "provenance": provenance,
            "epoch": epoch, "best_epoch": best_epoch, "best_val_auc": best_auc,
            "bad_epochs": bad_epochs, "history": history,
            "val_predictions": predictions,
        }
        if improved:
            atomic_save(payload, output / f"best_epoch_{epoch:03d}.pt")
            atomic_save(payload, output / "best.pt")
            write_predictions(output, predictions)
        atomic_save(payload, output / "last.pt")
        write_history(output, history)
        print(json.dumps(record, allow_nan=False), flush=True)
        if bad_epochs >= config.early_stop:
            break
    summary = {"condition": config.condition, "seed": config.seed, "best_epoch": best_epoch,
               "best_val_auc": best_auc, "completed_epochs": len(history),
               "early_stopped": bad_epochs >= config.early_stop, "provenance": provenance}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", help="Resume the output directory's last.pt at the next epoch")
    args = parser.parse_args()
    run(load_config(args.config), args.output_dir, args.resume)


if __name__ == "__main__":
    main()
