"""Independently reload formal C checkpoints using clean validation only."""

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.train import load_config, make_datasets, seed_everything
from src.training.checkpoint import load_model, read_checkpoint
from src.training.engine import validate


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(2)
    results = {"purpose": "formal C checkpoint and clean-validation reproduction",
               "device": args.device, "runs": {}, "ok": False}
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("verification requested CUDA but it is unavailable")
        results["gpu"] = torch.cuda.get_device_name(0)
    for seed in (5421, 2026):
        directory = args.runs_root / f"C_{seed}"
        config = load_config(directory / "config.yaml")
        seed_everything(config.seed, config.deterministic)
        state = read_checkpoint(directory / "best.pt")
        summary = json.loads((directory / "summary.json").read_text())
        if config.condition != "C" or config.seed != seed or config.run_kind != "formal":
            raise ValueError("expected a complete formal C run")
        if state["config"] != asdict(config):
            raise ValueError("checkpoint configuration differs from saved config")
        if state["epoch"] != summary["best_epoch"] or state["best_val_auc"] != summary["best_val_auc"]:
            raise ValueError("best checkpoint disagrees with run summary")
        if state["provenance"]["manifest_sha256"] != sha256(config.data_manifest):
            raise ValueError("manifest changed since training")
        _, val_data, preprocessing, audit = make_datasets(config)
        if len(val_data) != 1500 or audit["sha256_verified"] != 12000:
            raise ValueError("complete project data required")
        model, metadata = load_model(directory / "best.pt", device=args.device)
        if preprocessing != metadata["preprocessing"]:
            raise ValueError("checkpoint preprocessing does not match the data module")
        metrics, predictions = validate(model, DataLoader(val_data, batch_size=config.batch), torch.device(args.device))
        with (directory / "val_predictions.csv").open(newline="") as stream:
            expected = list(csv.DictReader(stream))
        if len(expected) != len(predictions):
            raise ValueError("prediction record count mismatch")
        errors = []
        for actual, saved in zip(predictions, expected):
            if actual[0] != saved["image_id"] or actual[1] != float(saved["label"]):
                raise ValueError("prediction ID or label mismatch")
            errors.append(abs(actual[2] - float(saved["p_ai"])))
        if max(errors) > 1e-5:
            raise ValueError("reloaded probabilities differ by more than 1e-5")
        results["runs"][f"C_{seed}"] = {
            "best_checkpoint_sha256": sha256(directory / "best.pt"),
            "last_checkpoint_sha256": sha256(directory / "last.pt"),
            "config_sha256": sha256(directory / "config.yaml"),
            "summary": summary, "reloaded_validation_metrics": metrics,
            "validation_images": len(predictions), "max_probability_error": max(errors),
            "probability_tolerance": 1e-5,
        }
        print(f"C_{seed}: best checkpoint and all 1500 validation probabilities verified", flush=True)
    results["ok"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
