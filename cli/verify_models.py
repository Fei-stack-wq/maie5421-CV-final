"""Compare six CLI models on four fixed validation images and recompute CSV AUC."""

import argparse
import csv
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from itertools import groupby
from pathlib import Path

import PIL
import torch
import torchvision
from PIL import Image

from cli.download_models import MODELS, ROOT, ensure_file
from cli.predict import file_hash, write_result
from src.datasets import DataConfig, EvalTransform
from src.training.checkpoint import load_model


SAMPLE_IDS = ("val-001510c78c44", "val-001cd7d5a0a5", "val-002c37a9130b", "val-008618d5511b")
DATA_COMMIT = "e7db18d0657865d5f344632534a3248cb49a6355"


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def prediction_auc(rows):
    """Mann-Whitney AUC with half credit for equal predicted probabilities."""
    positives = sum(int(float(row["label"])) for row in rows)
    negatives = len(rows) - positives
    negative_before = 0
    pairs = 0.0
    for _, group in groupby(sorted(rows, key=lambda row: float(row["p_ai"])),
                            key=lambda row: float(row["p_ai"])):
        tied = list(group)
        positive = sum(int(float(row["label"])) for row in tied)
        negative = len(tied) - positive
        pairs += positive * (negative_before + 0.5 * negative)
        negative_before += negative
    return pairs / (positives * negatives)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/six_models")
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    manifest = read_rows(ROOT / "Images/manifest.csv")
    validation = {row["image_id"]: row for row in manifest if row["split"] == "val"}
    samples = {image_id: validation[image_id] for image_id in SAMPLE_IDS}
    image_paths = {}
    for image_id, sample in samples.items():
        image_paths[image_id] = ensure_file({
            "path": str(output_dir / "samples" / sample["filename"]),
            "sha256": sample["sha256"],
            "url": f"https://raw.githubusercontent.com/Fei-stack-wq/maie5421-CV-final/{DATA_COMMIT}/Images/{sample['rel_path']}",
        })

    comparisons = []
    runs = {}
    for run_id, spec in MODELS.items():
        paths = {name: ensure_file(artifact, verify_only=True) for name, artifact in spec["files"].items()}
        published = read_rows(paths["predictions"])
        labels = {row["image_id"]: int(float(row["label"])) for row in published}
        if len(published) != len(validation) or labels != {
            image_id: int(row["label"]) for image_id, row in validation.items()
        }:
            raise ValueError(f"{run_id} 验证预测的图片 ID 或标签与 manifest 不一致。")
        references = {row["image_id"]: float(row["p_ai"]) for row in published}
        model, metadata = load_model(paths["checkpoint"], device="cpu")
        if (metadata["config"]["condition"], metadata["config"]["seed"]) != (spec["condition"], spec["seed"]):
            raise ValueError(f"{run_id} 模型身份不匹配。")
        transform = EvalTransform(DataConfig(**metadata["preprocessing"]["data_config"]), suite="clean")
        csv_auc = prediction_auc(published)
        run_rows = []
        for image_id, image_path in image_paths.items():
            output = output_dir / "predictions" / f"{run_id}_{image_id}.json"
            command = [sys.executable, "-m", "cli.predict", "--image", str(image_path),
                       "--checkpoint", str(paths["checkpoint"]), "--config", str(paths["config"]),
                       "--output", str(output)]
            subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True, timeout=60)
            actual = json.loads(output.read_text())
            with Image.open(image_path) as image, torch.inference_mode():
                shared_probability = model.predict_proba(transform(image).unsqueeze(0)).item()
            row = {
                "run_id": run_id, "image_id": image_id, "split": "val", "transformation": "clean",
                "true_label": int(samples[image_id]["label"]),
                "image_sha256": actual["image_sha256"], "checkpoint_sha256": actual["checkpoint_sha256"],
                "published_p_ai": references[image_id], "cli_p_ai": actual["p_ai"],
                "shared_api_p_ai": shared_probability,
                "published_abs_difference": abs(actual["p_ai"] - references[image_id]),
                "shared_api_abs_difference": abs(actual["p_ai"] - shared_probability),
                "published_label": int(references[image_id] >= 0.5), "cli_label": actual["predicted_label_id"],
            }
            if (row["image_sha256"] != samples[image_id]["sha256"]
                    or row["checkpoint_sha256"] != spec["files"]["checkpoint"]["sha256"]):
                raise ValueError(f"{run_id}/{image_id} 推理输出的文件哈希不匹配。")
            run_rows.append(row)
        runs[run_id] = {
            "validation_csv_rows": len(published), "validation_auc_from_csv": csv_auc,
            "checkpoint_best_val_auc": metadata["best_val_auc"],
            "auc_abs_difference": abs(csv_auc - metadata["best_val_auc"]),
            "sample_predictions": len(run_rows),
            "label_agreements": sum(row["published_label"] == row["cli_label"] for row in run_rows),
            "max_published_probability_difference": max(row["published_abs_difference"] for row in run_rows),
            "max_shared_api_difference": max(row["shared_api_abs_difference"] for row in run_rows),
            "checkpoint_sha256": spec["files"]["checkpoint"]["sha256"],
        }
        comparisons.extend(run_rows)
        print(f"{run_id}: {runs[run_id]['label_agreements']}/4 labels match; CSV AUC={csv_auc:.10f}", flush=True)
    with (output_dir / "sample_predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=comparisons[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(comparisons)
    summary = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Six CPU CLI models, four fixed clean-validation images each; AUC recomputed from published validation CSVs.",
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "torch": torch.__version__, "torchvision": torchvision.__version__, "pillow": PIL.__version__},
        "source_sha256": {name: file_hash(ROOT / name) for name in (
            "cli/predict.py", "cli/verify_models.py", "cli/formal_models.json",
            "src/training/checkpoint.py", "src/datasets/transforms.py", "Images/manifest.csv")},
        "sample_ids": list(SAMPLE_IDS), "runs": runs,
        "ok": all(run["label_agreements"] == 4 and run["max_shared_api_difference"] <= 1e-7
                  and run["auc_abs_difference"] <= 1e-12 for run in runs.values()),
    }
    write_result(output_dir / "summary.json", summary)
    print(f"Report: {output_dir / 'summary.json'}")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
