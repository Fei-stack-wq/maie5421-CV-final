"""Check the CLI against the shared inference API using six released weights."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from cli.download_models import MODELS
from src.datasets import DataConfig, EvalTransform
from src.training.checkpoint import load_model


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = {run_id: ROOT / spec["files"]["checkpoint"]["path"] for run_id, spec in MODELS.items()}
SAMPLE = ROOT / "Images/fixture/real/036_31cf018b81.JPEG"


@unittest.skipUnless(all(path.is_file() for path in CHECKPOINTS.values()) and SAMPLE.is_file(),
                     "Run python -m cli.download_models and obtain the fixture image before these checks.")
class FormalCliTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.directory = Path(cls.temp.name)
        torch.set_num_threads(2)
        cls.models = {run_id: load_model(path, device="cpu") for run_id, path in CHECKPOINTS.items()}

    def run_cli(self, image=SAMPLE, run_id="C_5421", *extra):
        return subprocess.run([
            sys.executable, "-m", "cli.predict", "--image", str(image),
            "--checkpoint", str(CHECKPOINTS[run_id]), *extra,
        ], cwd=ROOT, text=True, capture_output=True, timeout=60)

    def reference_probability(self, image, run_id):
        model, metadata = self.models[run_id]
        transform = EvalTransform(DataConfig(**metadata["preprocessing"]["data_config"]), suite="clean")
        with Image.open(image) as opened, torch.inference_mode():
            return model.predict_proba(transform(opened).unsqueeze(0)).item()

    def test_six_checkpoints_match_shared_interface_and_optional_config(self):
        for run_id, spec in MODELS.items():
            with self.subTest(run_id=run_id):
                output = self.directory / f"{run_id}.json"
                run = self.run_cli(SAMPLE, run_id, "--output", str(output))
                self.assertEqual(run.returncode, 0, run.stderr)
                first = json.loads(output.read_text())
                self.assertEqual(first["purpose"], "single_image_inference")
                self.assertEqual(first["model_status"], "formal")
                self.assertEqual((first["condition"], first["seed"]), (spec["condition"], spec["seed"]))
                self.assertNotIn("warning", first)
                self.assertAlmostEqual(first["p_ai"], self.reference_probability(SAMPLE, run_id), places=7)
                self.assertEqual(first["predicted_label_id"], int(first["p_ai"] >= 0.5))
                config = ROOT / spec["files"]["config"]["path"]
                again = self.run_cli(SAMPLE, run_id, "--config", str(config), "--output", str(output))
                self.assertEqual(again.returncode, 0, again.stderr)
                self.assertEqual(first, json.loads(output.read_text()))

    def test_exif_orientation_and_grayscale_match_shared_preprocessing(self):
        oriented = self.directory / "rotated.jpg"
        grayscale = self.directory / "gray.png"
        with Image.open(SAMPLE) as source:
            exif = source.getexif()
            exif[274] = 6
            source.save(oriented, exif=exif)
            source.convert("L").save(grayscale)
        for image in (oriented, grayscale):
            with self.subTest(image=image.name):
                output = self.directory / f"{image.stem}.json"
                run = self.run_cli(image, "C_5421", "--output", str(output))
                self.assertEqual(run.returncode, 0, run.stderr)
                actual = json.loads(output.read_text())["p_ai"]
                self.assertAlmostEqual(actual, self.reference_probability(image, "C_5421"), places=7)

    def test_wrong_seed_or_condition_config_preserves_existing_output(self):
        output = self.directory / "existing.json"
        output.write_text('{"previous": true}\n')
        for run_id, other in (("C_5421", "C_2026"), ("A_5421", "B_5421")):
            with self.subTest(run_id=run_id, other=other):
                config = ROOT / MODELS[other]["files"]["config"]["path"]
                run = self.run_cli(SAMPLE, run_id, "--config", str(config), "--output", str(output))
                self.assertEqual(run.returncode, 2)
                self.assertIn("配置", run.stderr)
                self.assertNotIn("P(AI)=", run.stdout)
                self.assertEqual(output.read_text(), '{"previous": true}\n')

    def test_optional_config_output_protection(self):
        for path in (SAMPLE, CHECKPOINTS["C_5421"]):
            with self.subTest(path=path.name):
                run = self.run_cli(SAMPLE, "C_5421", "--output", str(path))
                self.assertEqual(run.returncode, 2)
                self.assertIn("不能覆盖", run.stderr)


if __name__ == "__main__":
    unittest.main()
