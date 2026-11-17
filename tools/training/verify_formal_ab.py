"""Reload completed formal A/B runs without modifying checkpoints or run records.

Run from the repository root. Verification may use CPU even when training used
CUDA; probability tolerance is 1e-5. Only clean validation is evaluated.
"""

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.train import load_config, make_datasets, seed_everything
from src.training.checkpoint import load_model, read_checkpoint
from src.training.engine import validate

RUN_IDS = ("A_5421", "B_5421", "A_2026", "B_2026")
PROBABILITY_TOLERANCE = 1e-5


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def read_json(path):
    def reject(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def finite(value, name, low=None, high=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric data")
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{name} is outside its valid range")
    return value


def integer(value, name, low=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < low:
        raise ValueError(f"{name} must be an integer >= {low}")
    return value


def check_protocol(config, run_id, reference_config):
    """Compare with the same seed's published C protocol, not a pilot config."""
    if run_id not in RUN_IDS:
        raise ValueError("unknown formal A/B run ID")
    condition, seed = run_id.split("_")
    config.validate()
    reference_config.validate()
    expected = asdict(reference_config)
    if reference_config.condition != "C" or reference_config.seed != int(seed):
        raise ValueError("reference must be the matching formal C configuration")
    required = {
        "run_kind": "formal", "protocol_confirmed": True, "train_limit": None,
        "val_limit": None, "dataset_factory": "src.training.ouyang_factory:build_datasets",
        "data_config": "configs/ouyang_data_v1.json", "weights": "IMAGENET1K_V1",
        "freeze_policy": "layer4_head_frozen_bn", "lr": 1e-4, "weight_decay": 0.01,
        "batch": 4, "accumulation_steps": 4, "epochs": 8, "early_stop": 2,
        "min_delta": 0.0, "amp": True, "amp_init_scale": 1024.0,
        "deterministic": True, "device": "cuda", "num_workers": 0, "consistency_weight": 1.0,
    }
    if any(expected.get(key) != value for key, value in required.items()):
        raise ValueError("reference C configuration differs from the agreed protocol")
    expected.update(condition=condition, consistency_weight=0.0)
    if asdict(config) != expected:
        raise ValueError("A/B configuration differs from the shared C protocol")


def check_provenance(provenance, manifest_hash):
    if provenance.get("run_kind") != "formal" or provenance.get("manifest_sha256") != manifest_hash:
        raise ValueError("checkpoint manifest or formal provenance mismatch")
    if provenance.get("train_images") != 6000 or provenance.get("val_images") != 1500:
        raise ValueError("formal provenance requires 6000 train and 1500 validation images")
    audit = provenance.get("data_audit", {})
    if (audit.get("manifest_rows") != 12000 or audit.get("sha256_verified") != 12000
            or audit.get("missing_hashes") != 0):
        raise ValueError("formal provenance requires all 12000 hashes verified")
    if provenance.get("working_tree_dirty") is not False or not re.fullmatch(
            r"[0-9a-f]{40}", str(provenance.get("commit", ""))):
        raise ValueError("training must identify a clean committed source revision")
    if provenance.get("device") != "cuda" or provenance.get("amp_enabled") is not True:
        raise ValueError("formal training must use the shared CUDA AMP protocol")


def check_records(config, best, last, summary, history, manifest_hash):
    """Cross-check completion, best selection and records without loading a model."""
    n = len(history)
    if not 1 <= n <= config.epochs:
        raise ValueError("invalid completed epoch count")
    best_auc, best_epoch, bad_epochs = -math.inf, -1, 0
    for epoch, row in enumerate(history):
        if integer(row.get("epoch"), "epoch") != epoch:
            raise ValueError("history must contain consecutive zero-based epochs")
        for key in ("train_loss", "val_loss", "seconds"):
            finite(row.get(key), key, low=0)
        auc = finite(row.get("val_auc"), "val_auc", 0, 1)
        finite(row.get("val_accuracy"), "val_accuracy", 0, 1)
        if row.get("train_images") != 6000 or row.get("val_images") != 1500:
            raise ValueError("history does not describe full train/validation splits")
        updates = integer(row.get("optimizer_steps"), "optimizer_steps")
        skips = integer(row.get("amp_skipped_steps"), "amp_skipped_steps")
        if updates == 0:
            raise ValueError('epoch has zero actual optimizer updates; investigate AMP failure')
        if updates + skips != 375:
            raise ValueError("history update attempts differ from the shared batch budget")
        improved = auc > best_auc + config.min_delta
        if improved:
            best_auc, best_epoch, bad_epochs = auc, epoch, 0
        else:
            bad_epochs += 1
        if row.get("is_best") is not improved or row.get("bad_epochs") != bad_epochs:
            raise ValueError("history checkpoint-selection or early-stop state mismatch")
        if bad_epochs >= config.early_stop and epoch != n - 1:
            raise ValueError("history continues after early stopping")
    stopped = bad_epochs >= config.early_stop
    if not stopped and n != config.epochs:
        raise ValueError("run is incomplete: neither budget reached nor early stopped")
    expected_summary = dict(condition=config.condition, seed=config.seed,
                            best_epoch=best_epoch, best_val_auc=best_auc,
                            completed_epochs=n, early_stopped=stopped)
    for key, value in expected_summary.items():
        if summary.get(key) != value:
            raise ValueError(f"summary {key} disagrees with checkpoint history")
    for label, state, epoch, prefix in (("best", best, best_epoch, history[:best_epoch + 1]),
                                      ("last", last, n - 1, history)):
        if state.get("config") != asdict(config):
            raise ValueError(f"{label} checkpoint configuration mismatch")
        if (state.get("epoch") != epoch or state.get("best_epoch") != best_epoch
                or state.get("best_val_auc") != best_auc or state.get("history") != prefix
                or state.get("bad_epochs") != prefix[-1]["bad_epochs"]):
            raise ValueError(f"{label} checkpoint selection/history mismatch")
        if state.get("threshold") != 0.5:
            raise ValueError("checkpoint threshold must be 0.5")
        check_provenance(state.get("provenance", {}), manifest_hash)
    for key in ("commit", "torch", "torchvision", "device", "amp_enabled", "manifest_sha256"):
        if best["provenance"].get(key) != last["provenance"].get(key):
            raise ValueError(f"best/last training provenance differs: {key}")
    if best.get("preprocessing") != last.get("preprocessing"):
        raise ValueError("best/last preprocessing mismatch")
    if summary.get("provenance") != last["provenance"]:
        raise ValueError("summary provenance differs from last checkpoint")
    return {"epochs": n, "optimizer_steps": sum(r["optimizer_steps"] for r in history),
            "amp_skipped_steps": sum(r["amp_skipped_steps"] for r in history)}


def read_history(path, checkpoint_history):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = reader.fieldnames
    if not checkpoint_history or fields != list(checkpoint_history[0]):
        raise ValueError("history CSV columns differ from last checkpoint")
    expected = [{key: str(value) for key, value in row.items()} for row in checkpoint_history]
    if rows != expected:
        raise ValueError("history CSV differs from last checkpoint")
    return checkpoint_history


def compare_predictions(actual, recorded, count=1500, tolerance=PROBABILITY_TOLERANCE):
    integer(count, "prediction count", 1)
    finite(tolerance, "tolerance", 0, PROBABILITY_TOLERANCE)
    if len(actual) != count or len(recorded) != count:
        raise ValueError("prediction record count mismatch")
    ids = set()
    errors = []
    for current, saved in zip(actual, recorded):
        image_id, label, probability = current
        if not isinstance(image_id, str) or not image_id or image_id in ids:
            raise ValueError("prediction IDs must be nonempty and unique")
        ids.add(image_id)
        saved_label, saved_probability = float(saved["label"]), float(saved["p_ai"])
        finite(label, "label", 0, 1)
        finite(saved_label, "saved label", 0, 1)
        if label not in (0, 1) or image_id != saved["image_id"] or label != saved_label:
            raise ValueError("prediction ID or label mismatch")
        finite(probability, "probability", 0, 1)
        finite(saved_probability, "saved probability", 0, 1)
        errors.append(abs(probability - saved_probability))
    if max(errors) > tolerance:
        raise ValueError("reloaded probabilities exceed tolerance")
    return max(errors)


def verify_run(directory, run_id, device="cpu", reference_config=None):
    directory = Path(directory)
    config = load_config(directory / "config.yaml")
    reference_config = reference_config or load_config(f"configs/wang/c_{run_id.split('_')[-1]}.yaml")
    check_protocol(config, run_id, reference_config)
    seed_everything(config.seed, config.deterministic)
    best, last = (read_checkpoint(directory / name) for name in ("best.pt", "last.pt"))
    summary = read_json(directory / "summary.json")
    history = read_history(directory / "history.csv", last["history"])
    budget = check_records(config, best, last, summary, history, sha256(config.data_manifest))
    train, val, preprocessing, audit = make_datasets(config)
    if (len(train) != 6000 or len(val) != 1500 or audit.get("manifest_rows") != 12000
            or audit.get("sha256_verified") != 12000 or audit.get("missing_hashes") != 0):
        raise ValueError("complete project data and all declared hashes are required")
    if preprocessing != best["preprocessing"]:
        raise ValueError("checkpoint preprocessing does not match the data module")
    model, metadata = load_model(directory / "best.pt", device=device)
    if metadata["preprocessing"] != preprocessing:
        raise ValueError("loaded-model preprocessing mismatch")
    metrics, predictions = validate(model, DataLoader(val, batch_size=config.batch), torch.device(device))
    best_record = history[best["epoch"]]
    for name in ("val_loss", "val_auc", "val_accuracy"):
        value = finite(metrics.get(name), name, 0, None if name == "val_loss" else 1)
        if abs(value - best_record[name]) > PROBABILITY_TOLERANCE:
            raise ValueError(f"reloaded {name} differs from best history beyond tolerance")
    if metrics.get("val_images") != 1500:
        raise ValueError("reloaded validation metric count mismatch")
    with (directory / "val_predictions.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["image_id", "label", "p_ai"]:
            raise ValueError("validation CSV schema mismatch")
        saved = list(reader)
    compare_predictions(best["val_predictions"], saved, tolerance=0.0)
    # Last may be a different epoch: its probabilities need not equal best.
    # Still validate every stored value, ID and label against the same val split.
    last_predictions = last["val_predictions"]
    last_records = [{"image_id": row["image_id"], "label": row["label"], "p_ai": pred[2]}
                    for row, pred in zip(saved, last_predictions)]
    compare_predictions(last_predictions, last_records, tolerance=0.0)
    error = compare_predictions(predictions, saved)
    return {"status": "verified", "summary": summary, "budget": budget,
            "budget_review_required": budget["amp_skipped_steps"] > 0,
            "verification_scope": "best.pt inference reload; last.pt records/hash only, not a resume test",
            "validation_images": len(predictions), "reloaded_validation_metrics": metrics,
            "max_probability_error": error, "probability_tolerance": PROBABILITY_TOLERANCE,
            "sha256": {name: sha256(directory / name) for name in
                       ("best.pt", "last.pt", "config.yaml", "summary.json", "history.csv", "val_predictions.csv")}}


def verify_runs(runs_root, run_ids, device="cpu"):
    if not run_ids or len(set(run_ids)) != len(run_ids) or any(x not in RUN_IDS for x in run_ids):
        raise ValueError("select unique formal A/B run IDs")
    if device not in ("cpu", "cuda"):
        raise ValueError("verification device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("verification requested CUDA but it is unavailable")
    torch.set_num_threads(2)
    results = {"purpose": "formal A/B checkpoint and clean-validation reproduction",
               "device": device, "requested_runs": list(run_ids), "runs": {},
               "not_verified_runs": [x for x in RUN_IDS if x not in run_ids],
               "all_four_verified": False, "selected_verified": False}
    for run_id in run_ids:
        results["runs"][run_id] = verify_run(Path(runs_root) / run_id, run_id, device)
    results["selected_verified"] = True
    results["all_four_verified"] = set(run_ids) == set(RUN_IDS)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--run-id", action="append", choices=RUN_IDS)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        parser.error("output already exists; choose a new verification report")
    root = args.runs_root.resolve()
    if output == root or root in output.parents:
        parser.error("verification report must be outside the run directory tree")
    results = verify_runs(root, args.run_id or RUN_IDS, args.device)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"selected_verified": True, "all_four_verified": results["all_four_verified"],
                      "verified_runs": list(results["runs"])}))


if __name__ == "__main__":
    main()
