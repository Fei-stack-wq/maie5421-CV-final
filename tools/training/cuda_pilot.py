"""Small real-image CUDA checks for C, B/C-zero equivalence and epoch resume.

Run from the repository root with python -m tools.training.cuda_pilot.
Outputs stay under ignored runs/. No formal experiment or test-set scoring.
"""

import argparse
import csv
import gc
import hashlib
import io
import json
import platform
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

import PIL
import torch
import torchvision
import yaml
from torch.utils.data import DataLoader

from src.models import BinaryResNet18
from src.train import load_config, make_datasets, run, seed_everything
from src.training.checkpoint import load_model, read_checkpoint
from src.training.data import balanced_subset, read_manifest
from src.training.engine import validate


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(config, output):
    """Preserve source IDs/hashes in a train/val-only pilot manifest."""
    rows = read_manifest(config.data_manifest, config.data_root)
    with open(config.data_manifest, encoding="utf-8-sig", newline="") as stream:
        fields = csv.DictReader(stream).fieldnames
    selected = []
    for split, limit in (("train", config.train_limit), ("val", config.val_limit)):
        if limit is None:
            raise ValueError("the CUDA pilot requires explicit small sample limits")
        selected.extend(balanced_subset([r for r in rows if r["split"] == split], limit, config.seed))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(selected)
    text = buffer.getvalue()
    manifest = output / "pilot_manifest.csv"
    if manifest.exists() and manifest.read_text(encoding="utf-8") != text:
        raise ValueError("existing pilot manifest differs from selected source rows")
    manifest.write_text(text, encoding="utf-8")
    (output / "sparse-paths.txt").write_text(
        "".join("/Images/" + r["rel_path"] + "\n" for r in selected), encoding="utf-8"
    )
    return manifest


def identical(first, second):
    if isinstance(first, torch.Tensor):
        return isinstance(second, torch.Tensor) and torch.equal(first, second)
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(identical(first[k], second[k]) for k in first)
    if isinstance(first, (tuple, list)):
        return len(first) == len(second) and all(identical(a, b) for a, b in zip(first, second))
    return first == second


def compare_checkpoints(first, second):
    a, b = read_checkpoint(first / "last.pt"), read_checkpoint(second / "last.pt")
    keys = ("model_state", "optimizer_state", "scaler_state", "rng_state",
            "epoch", "best_epoch", "best_val_auc", "bad_epochs", "val_predictions")
    checks = {key: identical(a[key], b[key]) for key in keys}
    histories = [[{k: v for k, v in row.items() if k != "seconds"} for row in s["history"]]
                 for s in (a, b)]
    checks["history_except_seconds"] = identical(*histories)
    if not all(checks.values()):
        raise AssertionError(f"checkpoint comparison failed: {checks}")
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="examples/wang/pilot_c_cuda.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    if not output.is_relative_to(Path("runs").resolve()):
        raise ValueError("output-dir must be inside runs/")
    output.mkdir(parents=True, exist_ok=True)
    if any(p.name not in {"pilot_manifest.csv", "sparse-paths.txt"} for p in output.iterdir()):
        raise FileExistsError("choose a new output-dir; previous results are never overwritten")
    base = load_config(args.config)
    if base.run_kind != "pilot" or base.protocol_confirmed or base.device != "cuda" or not base.amp:
        raise ValueError("requires an unconfirmed CUDA AMP pilot config")
    manifest = prepare(base, output)
    if args.prepare_only:
        print(f"Prepared {manifest}; fetch image paths listed in {output / 'sparse-paths.txt'}")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    config = replace(base, data_manifest=manifest.relative_to(Path.cwd()).as_posix())
    torch.set_num_threads(config.cpu_threads)
    report = {
        "purpose": "CUDA integration pilot, not formal model results",
        "protocol_confirmed": False, "formal_c_models_completed": 0,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {str(p): sha256(p) for folder in ("src", "tools/training")
                          for p in sorted(Path(folder).rglob("*.py"))},
        "config_sha256": sha256(args.config),
        "source_manifest_sha256": sha256(base.data_manifest),
        "pilot_manifest_sha256": sha256(manifest),
        "dataset_factory": config.dataset_factory, "data_config": config.data_config,
        "environment": {
            "python": platform.python_version(), "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__), "pillow": PIL.__version__,
            "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        },
        "runs": {}, "checks": {}, "ok": False,
    }

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    save()
    if config.dataset_factory:
        train_b, val_b, metadata_b, _ = make_datasets(replace(config, condition="B"))
        train_c, val_c, metadata_c, _ = make_datasets(config)
        assert metadata_b == metadata_c
        checked = 0
        for epoch in range(config.epochs):
            train_b.set_epoch(epoch)
            train_c.set_epoch(epoch)
            for i in range(len(train_b)):
                b, c = train_b[i], train_c[i]
                assert b["image_id"] == c["image_id"]
                assert all(torch.equal(b[k], c[k]) for k in ("image", "view", "label"))
                assert b["view_params"] == c["view_params"]
                checked += 1
        for i in range(len(val_c)):
            b, c = val_b[i], val_c[i]
            assert "view" not in c and b["image_id"] == c["image_id"]
            assert torch.equal(b["image"], c["image"])
        report["checks"]["factory_views"] = {"paired_samples_checked": checked,
            "clean_val_samples_checked": len(val_c), "equal": True}
        report["preprocessing"] = metadata_c
        save()
    seed_everything(config.seed, config.deterministic)
    initial = BinaryResNet18(weights=config.weights)
    initial_state = {k: v.clone() for k, v in initial.state_dict().items()}
    trainable = {name for name, param in initial.named_parameters() if param.requires_grad}
    del initial

    def case(name, case_config, resume=False):
        directory = output / name
        config_file = output / (name + ("_resume" if resume else "") + ".yaml")
        config_file.write_text(yaml.safe_dump(asdict(case_config), sort_keys=False), encoding="utf-8")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        summary = run(case_config, directory, directory / "last.pt" if resume else None)
        torch.cuda.synchronize()
        record = {
            "summary": summary, "wall_seconds": time.perf_counter() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        last = read_checkpoint(directory / "last.pt")
        frozen = [key for key in initial_state if key not in trainable]
        record["frozen_tensors_unchanged"] = all(torch.equal(initial_state[k], last["model_state"][k]) for k in frozen)
        record["frozen_tensor_count"] = len(frozen)
        record["changed_trainable_tensors"] = sum(not torch.equal(initial_state[k], last["model_state"][k]) for k in trainable)
        record["checkpoint_sha256"] = sha256(directory / "best.pt")
        _, val_data, _, _ = make_datasets(case_config)
        model, _ = load_model(directory / "best.pt", device="cuda")
        _, predictions = validate(model, DataLoader(val_data, batch_size=case_config.batch), torch.device("cuda"))
        best = read_checkpoint(directory / "best.pt")
        assert [row[:2] for row in predictions] == [row[:2] for row in best["val_predictions"]]
        record["reload_max_probability_error"] = max(abs(a[2] - b[2]) for a, b in zip(predictions, best["val_predictions"]))
        assert record["reload_max_probability_error"] <= 1e-6
        assert record["frozen_tensors_unchanged"] and record["changed_trainable_tensors"] > 0
        assert sum(row["optimizer_steps"] for row in last["history"]) > 0
        report["runs"][name + ("_resumed" if resume else "")] = record
        save()
        print(f"{name}: CUDA AMP run and checkpoint checks passed", flush=True)

    try:
        case("B", replace(config, condition="B"))
        case("C_zero", replace(config, consistency_weight=0.0))
        report["checks"]["b_matches_c_zero"] = compare_checkpoints(output / "B", output / "C_zero")
        save()
        case("C", config)
        case("C_resume", replace(config, epochs=1))
        case("C_resume", config, resume=True)
        report["checks"]["c_epoch_resume_matches_full"] = compare_checkpoints(output / "C", output / "C_resume")
        report["ok"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save()
    print(f"All GPU pilot checks passed. Evidence: {output / 'report.json'}")


if __name__ == "__main__":
    main()
