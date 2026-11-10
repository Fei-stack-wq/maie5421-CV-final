import csv
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.datasets import DataConfig, EvalTransform
from src.train import TrainConfig, make_datasets
from src.training.ouyang_factory import build_datasets


class OuyangFactoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rows = []
        for split in ("train", "val"):
            for i in reversed(range(6)):
                name = f"{split}_{i}.png"
                pixels = (np.arange(40 * 48 * 3).reshape(40, 48, 3) + i * 7).astype(np.uint8)
                Image.fromarray(pixels).save(self.root / name)
                rows.append(dict(image_id=name, rel_path=name, split=split, label=i % 2,
                                 generator="BigGAN" if i % 2 else "real"))
        manifest = self.root / "manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        self.data_config = self.root / "data.json"
        self.data_config.write_text(json.dumps(DataConfig().to_dict()), encoding="utf-8")
        self.config = TrainConfig(condition="C", data_manifest=str(manifest), data_root=str(self.root),
                                 dataset_factory="src.training.ouyang_factory:build_datasets",
                                 data_config=str(self.data_config), train_limit=4, val_limit=4)

    def test_canonical_order_and_balanced_subset_through_training_contract(self):
        for limit in (4, None):
            train, val, metadata, _ = make_datasets(replace(self.config, train_limit=limit, val_limit=limit))
            ids = [train[i]["image_id"] for i in range(len(train))]
            self.assertEqual(ids, sorted(ids))
            self.assertEqual(len(train), limit or 6)
            self.assertEqual(sum(train[i]["label"].item() for i in range(len(train))), len(train) / 2)
            self.assertNotIn("view", val[0])
            self.assertEqual(metadata["config_fingerprint"], DataConfig().fingerprint)

    def test_bc_views_and_epoch_forwarding(self):
        b, _, metadata_b = build_datasets(asdict(replace(self.config, condition="B")))
        c, _, metadata_c = build_datasets(asdict(self.config))
        self.assertEqual(metadata_b, metadata_c)
        self.assertEqual(b.image_ids, c.image_ids)
        params = []
        for epoch in (0, 1):
            b.set_epoch(epoch)
            c.set_epoch(epoch)
            self.assertEqual(c.dataset.epoch, epoch)
            params.append(c[0]["view_params"])
            for i in range(len(b)):
                for key in ("image", "view", "label"):
                    self.assertTrue(torch.equal(b[i][key], c[i][key]))
                self.assertEqual(b[i]["view_params"], c[i]["view_params"])
        self.assertNotEqual(params[0], params[1])

    def test_clean_validation_matches_ouyang_transform(self):
        _, val, metadata = build_datasets(asdict(self.config))
        sample = val[0]
        with Image.open(self.root / sample["rel_path"]) as image:
            expected = EvalTransform(DataConfig(), "clean")(image)
        self.assertTrue(torch.equal(sample["image"], expected))
        self.assertEqual(metadata["interpolation"], "bicubic")

    def test_explicit_versioned_config_and_metadata_change(self):
        with self.assertRaisesRegex(ValueError, "explicit data_config"):
            build_datasets(asdict(replace(self.config, data_config=None)))
        _, _, before = build_datasets(asdict(self.config))
        changed = DataConfig(jpeg_quality_max=94)
        self.data_config.write_text(json.dumps(changed.to_dict()), encoding="utf-8")
        _, _, after = build_datasets(asdict(self.config))
        self.assertNotEqual(before["config_fingerprint"], after["config_fingerprint"])
        self.assertNotEqual(before["config_file_sha256"], after["config_file_sha256"])


if __name__ == "__main__":
    unittest.main()
