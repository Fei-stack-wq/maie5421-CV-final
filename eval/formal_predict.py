"""Evaluate the six published checkpoints on the manifest's held-out images."""
import argparse
import csv
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path

import torch
from PIL import __version__ as pillow_version

from cli.download_models import MODELS, ensure_file
from cli.predict import load_prediction_model
from src.datasets import EVAL_SUITES, GenImageDataset, make_dataloader
from src.datasets.manifest import resolve_image_path

SPLITS = ("test_seen", "test_unseen_ADM", "test_unseen_GLIDE")
COUNTS = {
    "test_seen": {"real": 750, "BigGAN": 375, "stable_diffusion_v_1_4": 375},
    "test_unseen_ADM": {"real": 750, "ADM": 750},
    "test_unseen_GLIDE": {"real": 750, "GLIDE": 750},
}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_data(args):
    manifest_hash = sha256(args.manifest)
    with args.manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] in SPLITS]
    for split in SPLITS:
        if Counter(row["generator"] for row in rows if row["split"] == split) != COUNTS[split]:
            raise ValueError(f"Unexpected generator counts in {split}")
    for row in rows:
        path = resolve_image_path(args.images, row["rel_path"])
        if sha256(path) != row["sha256"]:
            raise ValueError(f"Image hash mismatch: {row['image_id']}")
    print(f"Verified {len(rows)} held-out images", flush=True)
    return manifest_hash


def predict(args):
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_hash = verify_data(args)
    # Git may materialize the same CSV with CRLF on Windows.
    normalized_manifest_hash = hashlib.sha256(args.manifest.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    models = {}
    for name, spec in MODELS.items():
        files = spec["files"]
        checkpoint = ensure_file(files["checkpoint"], verify_only=True)
        config_path = ensure_file(files["config"], verify_only=True)
        model, transform, meta = load_prediction_model(checkpoint, config_path, False)
        run_config = meta["config"]
        if (meta["model_status"] != "formal" or run_config["condition"] != spec["condition"]
                or run_config["seed"] != spec["seed"]):
            raise ValueError(f"Unexpected model identity: {name}")
        if meta["training_provenance"]["manifest_sha256"] not in {manifest_hash, normalized_manifest_hash}:
            raise ValueError(f"Training manifest differs from the evaluation manifest: {name}")
        if models and transform.config.fingerprint != data_config.fingerprint:
            raise ValueError(f"Preprocessing mismatch: {name}")
        data_config = transform.config
        model = model.to(device, memory_format=torch.channels_last)
        models[name] = model
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = dict(manifest_sha256=manifest_hash, manifest_lf_sha256=normalized_manifest_hash,
                    preprocessing_sha256=data_config.fingerprint,
                    python=platform.python_version(), torch=torch.__version__, pillow=pillow_version,
                    device=str(device), batch_size=args.batch_size,
                    models={name: spec["files"]["checkpoint"]["sha256"] for name, spec in MODELS.items()})
    if device.type == "cuda":
        metadata["gpu"] = torch.cuda.get_device_name(device)
        metadata["cuda"] = torch.version.cuda
    fields = ("image_id", "label", "p_ai", "generator", "transformation", "seed",
              "condition", "split", "rel_path")
    for split in SPLITS:
        for suite in EVAL_SUITES:
            dataset = GenImageDataset(args.manifest, split, root=args.images, suite=suite,
                                      config=data_config)
            loader = make_dataloader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                                     pin_memory=device.type == "cuda")
            with (args.output / f"{split}__{suite}.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                with torch.inference_mode():
                    for batch in loader:
                        images = batch["image"].to(device, memory_format=torch.channels_last)
                        for name, model in models.items():
                            probabilities = model(images).flatten().sigmoid().cpu()
                            if len(probabilities) != len(images) or not probabilities.isfinite().all():
                                raise ValueError(f"Invalid probabilities from {name}")
                            spec = MODELS[name]
                            writer.writerows(
                                dict(image_id=image_id, label=int(label), p_ai=float(probability),
                                     generator=generator, transformation=suite, seed=spec["seed"],
                                     condition=spec["condition"], split=split, rel_path=path)
                                for image_id, label, probability, generator, path in zip(
                                    batch["image_id"], batch["label"].tolist(), probabilities.tolist(),
                                    batch["generator"], batch["rel_path"])
                            )
            print(f"Completed {split} / {suite}", flush=True)
    shards = [args.output / f"{split}__{suite}.csv" for split in SPLITS for suite in EVAL_SUITES]
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8") as combined:
        for index, shard in enumerate(shards):
            with shard.open(newline="", encoding="utf-8") as source:
                if index:
                    next(source)
                for line in source:
                    combined.write(line)
            shard.unlink()
    (args.output / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("Images/manifest.csv"))
    parser.add_argument("--images", type=Path, default=Path("Images"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    predict(parser.parse_args())
