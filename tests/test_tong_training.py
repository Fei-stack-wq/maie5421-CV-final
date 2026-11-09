"""TONG model, accumulation, data and resume checks; no downloads required."""

import copy
import csv
import json
import random
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from src.models import BinaryResNet18
from src.train import TrainConfig, load_config, make_datasets, run
from src.training.checkpoint import atomic_save, load_model, read_checkpoint, restore_rng, rng_state
from src.training.data import (
    PREPROCESS, ContractDataset, ManifestPilotDataset, balanced_subset,
    read_manifest, validate_train_val, verify_declared_hashes,
)
from src.training.engine import binary_auc, per_image_loss, train_epoch, validate

torch.set_num_threads(2)


class TinyModel(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.fc = nn.Linear(3, 1)

    def forward(self, images):
        return self.fc(images.mean(dim=(-1, -2))).squeeze(-1)


def fixture(root):
    rows = []
    for split, count in (("train", 6), ("val", 4)):
        for index in range(count):
            name = f"{split}_{index}.png"
            Image.new("RGB", (36, 40), (index * 30, 45, 200 - index * 20)).save(root / name)
            rows.append({"image_id": name, "rel_path": name, "split": split,
                         "label": index % 2, "generator": "BigGAN" if index % 2 else "real"})
    manifest = root / "manifest.csv"
    with open(manifest, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


class ConfigTests(unittest.TestCase):
    def test_formal_requires_protocol_and_factory(self):
        with self.assertRaisesRegex(ValueError, "agreed protocol"):
            TrainConfig(run_kind="formal").validate()

    def test_formal_rejects_limits_and_random_init(self):
        base = TrainConfig(run_kind="formal", protocol_confirmed=True, dataset_factory="data:build")
        with self.assertRaisesRegex(ValueError, "complete"):
            base.validate()
        with self.assertRaisesRegex(ValueError, "ImageNet"):
            replace(base, train_limit=None, val_limit=None, weights=None).validate()

    def test_invalid_numeric_settings(self):
        for values in ({"batch": 0}, {"batch": True}, {"epochs": 1.5}, {"lr": float("nan")},
                       {"num_workers": -1}, {"seed": -1}, {"train_limit": 3}, {"amp": "false"},
                       {"amp_init_scale": 0}, {"amp_init_scale": -1}, {"amp_init_scale": float("nan")}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                TrainConfig(**values).validate()

    def test_unknown_config_field(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.yaml"
            path.write_text("lern_rate: 1\n")
            with self.assertRaisesRegex(ValueError, "unknown"):
                load_config(path)


class ModelTests(unittest.TestCase):
    def test_freeze_batchnorm_and_gradient_policy(self):
        model = BinaryResNet18(weights=None).train()
        before = {key: tensor.clone() for key, tensor in model.state_dict().items() if "running_" in key}
        logits = model(torch.randn(2, 3, 64, 64))
        self.assertEqual(logits.shape, (2,))
        logits.sum().backward()
        for name, parameter in model.backbone.named_parameters():
            expected = name.startswith(("layer4.", "fc."))
            self.assertEqual(parameter.requires_grad, expected, name)
            self.assertEqual(parameter.grad is not None, expected, name)
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.assertFalse(module.training)
        for key, value in before.items():
            self.assertTrue(torch.equal(value, model.state_dict()[key]), key)

    def test_checkpoint_load_has_no_download_and_same_predictions(self):
        model = BinaryResNet18(weights=None).eval()
        image = torch.randn(2, 3, 64, 64)
        expected = model.predict_proba(image)
        payload = {"schema_version": 1, "architecture": "resnet18_binary", "model_state": model.state_dict(),
                   "label_mapping": {"real": 0, "AI": 1}, "threshold": 0.5, "config": {},
                   "preprocessing": {}, "epoch": 0, "best_epoch": 0, "best_val_auc": 0.5, "provenance": {}}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "test.pt"
            atomic_save(payload, path)
            loaded, metadata = load_model(path)
            self.assertTrue(torch.equal(expected, loaded.predict_proba(image)))
            self.assertEqual(metadata["threshold"], 0.5)

    def test_predict_requires_eval(self):
        with self.assertRaises(RuntimeError):
            BinaryResNet18(weights=None).predict_proba(torch.zeros(1, 3, 32, 32))


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.manifest = fixture(self.root)
        self.rows = read_manifest(self.manifest, self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_manifest_train_val(self):
        splits = validate_train_val(self.rows)
        self.assertEqual(len(splits["train"]), 6)
        self.assertEqual(len(splits["val"]), 4)

    def test_heldout_generator_rejected(self):
        self.rows[0]["generator"] = "ADM"
        with self.assertRaisesRegex(ValueError, "held-out"):
            validate_train_val(self.rows)

    def test_hash_leakage_rejected(self):
        self.rows[0]["sha256"] = "same"
        self.rows[-1]["sha256"] = "same"
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_train_val(self.rows)

    def test_path_escape_rejected(self):
        text = self.manifest.read_text().replace("train_0.png", "../outside.png")
        self.manifest.write_text(text)
        with self.assertRaisesRegex(ValueError, "escapes"):
            read_manifest(self.manifest, self.root)

    def test_duplicate_id_rejected(self):
        text = self.manifest.read_text().splitlines()
        self.manifest.write_text("\n".join(text + [text[1]]) + "\n")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            read_manifest(self.manifest, self.root)

    def test_subset_is_balanced_and_deterministic(self):
        selected = balanced_subset(self.rows, 4, 7)
        self.assertEqual(sum(row["label"] for row in selected), 2)
        self.assertEqual(selected, balanced_subset(self.rows, 4, 7))

    def test_paired_views_ignore_global_rng(self):
        dataset = ManifestPilotDataset(self.rows, seed=7, paired=True)
        dataset.set_epoch(3)
        first = dataset[0]
        random.seed(999)
        random.random()
        second = dataset[0]
        self.assertTrue(torch.equal(first["view"], second["view"]))
        self.assertEqual(first["image"].shape, (3, 224, 224))
        self.assertEqual(first["label"].ndim, 0)

    def test_declared_hash_verification_and_required_hashes(self):
        with self.assertRaisesRegex(ValueError, "requires sha256"):
            verify_declared_hashes(self.rows, require_hashes=True)
        self.rows[0]["sha256"] = "incorrect"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            verify_declared_hashes(self.rows)

    def factory(self, config):
        splits = validate_train_val(self.rows)
        datasets = []
        for split in ("train", "val"):
            selected = balanced_subset(splits[split], None, config["seed"])
            dataset = ManifestPilotDataset(selected, config["seed"],
                                           split == "train" and config["condition"] != "A")
            dataset.image_ids = [row["image_id"] for row in selected]
            datasets.append(dataset)
        return *datasets, dict(PREPROCESS)

    def test_factory_requires_canonical_order(self):
        config = TrainConfig(data_manifest=str(self.manifest), data_root=str(self.root),
                             train_limit=None, val_limit=None, dataset_factory="fixture:build")
        module = SimpleNamespace(build=self.factory)
        with patch("src.train.importlib.import_module", return_value=module):
            train, val, _, _ = make_datasets(config)
            self.assertEqual(train[0]["image_id"], "train_0.png")
            self.assertEqual(len(val), 4)
        def reversed_factory(config):
            train, val, preprocessing = self.factory(config)
            train.image_ids.reverse()
            return train, val, preprocessing
        with patch("src.train.importlib.import_module", return_value=SimpleNamespace(build=reversed_factory)):
            with self.assertRaisesRegex(ValueError, "canonical ordered"):
                make_datasets(config)

    def test_factory_sample_schema_and_labels_are_checked(self):
        row = self.rows[0]
        good = ManifestPilotDataset([row], paired=True)[0]
        wrapped = ContractDataset([good], [row], paired=True)
        self.assertEqual(wrapped[0]["image_id"], row["image_id"])
        for key, value in (("image_id", "wrong"), ("label", torch.tensor(1.)),
                           ("image", torch.zeros(3, 10, 10)),
                           ("view", torch.full((3, 224, 224), float("nan")))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                ContractDataset([{**good, key: value}], [row], paired=True)[0]

    def test_factory_rng_does_not_change_initialization(self):
        config = TrainConfig(data_manifest=str(self.manifest), data_root=str(self.root),
                             train_limit=None, val_limit=None, weights=None, epochs=1,
                             dataset_factory="fixture:build", amp=False)
        def noisy_factory(config):
            torch.rand(37)
            random.random()
            np.random.rand(19)
            return self.factory(config)
        with patch("src.train.BinaryResNet18", TinyModel):
            for name, factory in (("normal", self.factory), ("noisy", noisy_factory)):
                with patch("src.train.importlib.import_module", return_value=SimpleNamespace(build=factory)):
                    run(config, self.root / name)
        one = read_checkpoint(self.root / "normal/last.pt")
        two = read_checkpoint(self.root / "noisy/last.pt")
        for key in one["model_state"]:
            self.assertTrue(torch.equal(one["model_state"][key], two["model_state"][key]))


class EngineTests(unittest.TestCase):
    def test_auc_ties_and_order(self):
        self.assertEqual(binary_auc([0, 1], [0.5, 0.5]), 0.5)
        self.assertEqual(binary_auc([0, 1], [0.0, 1.0]), 1.0)
        self.assertEqual(binary_auc([0, 1], [1.0, 0.0]), 0.0)
        with self.assertRaises(ValueError):
            binary_auc([1, 1], [0.2, 0.3])

    def test_b_is_mean_of_two_bces_and_c_zero_matches(self):
        logits, views, labels = torch.tensor([0.2, -0.5]), torch.tensor([1.0, 0.1]), torch.tensor([1., 0.])
        expected = 0.5 * (per_image_loss(logits, labels, "A") + per_image_loss(views, labels, "A"))
        actual = per_image_loss(logits, labels, "B", views)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(actual, per_image_loss(logits, labels, "C", views, 0.0)))

    def test_accumulation_matches_large_batch_including_remainder(self):
        torch.manual_seed(1)
        samples = [{"image": torch.randn(3, 4, 4), "label": torch.tensor(float(i % 2)), "image_id": str(i)} for i in range(7)]
        one = TinyModel()
        two = copy.deepcopy(one)
        optim_one = torch.optim.SGD(one.parameters(), lr=0.1)
        optim_two = torch.optim.SGD(two.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        train_epoch(one, DataLoader(samples, batch_size=6), optim_one, scaler, torch.device("cpu"), "A", 1)
        metrics = train_epoch(two, DataLoader(samples, batch_size=2), optim_two, scaler, torch.device("cpu"), "A", 3)
        self.assertEqual(metrics["optimizer_steps"], 2)
        for p1, p2 in zip(one.parameters(), two.parameters()):
            torch.testing.assert_close(p1, p2, rtol=1e-6, atol=1e-7)

    def test_rng_round_trip(self):
        state = rng_state()
        expected = (random.random(), np.random.rand(), torch.rand(2))
        restore_rng(state)
        actual = (random.random(), np.random.rand(), torch.rand(2))
        self.assertEqual(expected[:2], actual[:2])
        self.assertTrue(torch.equal(expected[2], actual[2]))

    def test_validation_uses_clean_images(self):
        model = TinyModel()
        samples = [{"image": torch.ones(3, 2, 2) * i, "view": torch.full((3, 2, 2), float("nan")),
                    "label": torch.tensor(float(i)), "image_id": str(i)} for i in (0, 1)]
        result, predictions = validate(model, DataLoader(samples, batch_size=2), torch.device("cpu"))
        self.assertTrue(np.isfinite(result["val_loss"]))
        self.assertEqual(len(predictions), 2)

    def test_auc_preserves_saturated_logit_order(self):
        class Logits(nn.Module):
            def forward(self, images):
                return images[:, 0, 0, 0]
        samples = [{"image": torch.full((3, 2, 2), 20. + i),
                    "label": torch.tensor(float(i)), "image_id": str(i)} for i in (0, 1)]
        metrics, _ = validate(Logits(), DataLoader(samples, batch_size=2), torch.device("cpu"))
        self.assertEqual(metrics["val_auc"], 1.0)

    def test_scaler_overflow_skips_and_reduces_scale(self):
        model = TinyModel()
        before = copy.deepcopy(model.state_dict())
        hook = model.fc.weight.register_hook(lambda grad: grad * float("inf"))
        scaler = torch.amp.GradScaler("cpu", enabled=True)
        initial_scale = scaler.get_scale()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        samples = [{"image": torch.ones(3, 2, 2), "label": torch.tensor(0.), "image_id": "a"}]
        metrics = train_epoch(model, DataLoader(samples), optimizer, scaler, torch.device("cpu"), "A")
        hook.remove()
        self.assertEqual(metrics["optimizer_steps"], 0)
        self.assertEqual(metrics["amp_skipped_steps"], 1)
        self.assertLess(scaler.get_scale(), initial_scale)
        for key in before:
            self.assertTrue(torch.equal(before[key], model.state_dict()[key]))


class ResumeTests(unittest.TestCase):
    def test_resume_checkpoint_without_amp_init_scale(self):
        with tempfile.TemporaryDirectory() as temp, patch("src.train.BinaryResNet18", TinyModel):
            root = Path(temp)
            manifest = fixture(root)
            config = TrainConfig(data_manifest=str(manifest), data_root=str(root), weights=None,
                                 train_limit=None, val_limit=None, epochs=1, amp=False, early_stop=4)
            run(config, root / "run")
            old = read_checkpoint(root / "run/last.pt")
            old["config"].pop("amp_init_scale")
            atomic_save(old, root / "run/last.pt")
            run(replace(config, epochs=2), root / "run", root / "run/last.pt")
            last = read_checkpoint(root / "run/last.pt")
            self.assertEqual(last["epoch"], 1)
            self.assertEqual(last["config"]["amp_init_scale"], 65536.0)

    def test_epoch_boundary_resume_matches_uninterrupted_run(self):
        with tempfile.TemporaryDirectory() as temp, patch("src.train.BinaryResNet18", TinyModel):
            root = Path(temp)
            manifest = fixture(root)
            config = TrainConfig(data_manifest=str(manifest), data_root=str(root), weights=None,
                                 train_limit=None, val_limit=None, batch=2, accumulation_steps=2,
                                 epochs=2, early_stop=4, amp=False)
            run(config, root / "full")
            run(replace(config, epochs=1), root / "resumed")
            run(config, root / "resumed", root / "resumed/last.pt")
            full = read_checkpoint(root / "full/last.pt")
            resumed = read_checkpoint(root / "resumed/last.pt")
            for key in full["model_state"]:
                self.assertTrue(torch.equal(full["model_state"][key], resumed["model_state"][key]), key)
            self.assertEqual(full["best_val_auc"], resumed["best_val_auc"])
            for one, two in zip(full["history"], resumed["history"]):
                one, two = dict(one), dict(two)
                one.pop("seconds")
                two.pop("seconds")
                self.assertEqual(one, two)

    def test_existing_output_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / "teammate.txt").write_text("keep")
            with self.assertRaises(FileExistsError):
                run(TrainConfig(), path)
            self.assertEqual((path / "teammate.txt").read_text(), "keep")

    def test_changed_resume_config_rejected(self):
        with tempfile.TemporaryDirectory() as temp, patch("src.train.BinaryResNet18", TinyModel):
            root = Path(temp)
            manifest = fixture(root)
            config = TrainConfig(data_manifest=str(manifest), data_root=str(root), weights=None,
                                 train_limit=None, val_limit=None, epochs=1, amp=False)
            run(config, root / "run")
            with self.assertRaisesRegex(ValueError, "config or manifest changed"):
                run(replace(config, epochs=2, seed=12), root / "run", root / "run/last.pt")

    def test_resume_recovers_best_file_ahead_of_committed_last(self):
        with tempfile.TemporaryDirectory() as temp, patch("src.train.BinaryResNet18", TinyModel):
            root = Path(temp)
            manifest = fixture(root)
            config = TrainConfig(data_manifest=str(manifest), data_root=str(root), weights=None,
                                 train_limit=None, val_limit=None, epochs=1, amp=False, early_stop=4)
            run(config, root / "run")
            ahead = read_checkpoint(root / "run/best.pt")
            ahead["epoch"] = 7
            atomic_save(ahead, root / "run/best.pt")
            run(replace(config, epochs=2), root / "run", root / "run/last.pt")
            best = read_checkpoint(root / "run/best.pt")
            last = read_checkpoint(root / "run/last.pt")
            self.assertEqual(best["epoch"], last["best_epoch"])


if __name__ == "__main__":
    unittest.main()
