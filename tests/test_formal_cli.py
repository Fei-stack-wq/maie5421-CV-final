"""Check the CLI against the shared inference API using released C weights."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from src.datasets import DataConfig, EvalTransform
from src.training.checkpoint import load_model


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = {seed: ROOT / f"checkpoints/wang/C_{seed}_best.pt" for seed in (5421, 2026)}
SAMPLE = ROOT / "Images/fixture/real/036_31cf018b81.JPEG"


@unittest.skipUnless(all(path.is_file() for path in CHECKPOINTS.values()) and SAMPLE.is_file(),
                     "Download the two C checkpoints and fixture image before running these checks.")
class FormalCliTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.directory = Path(cls.temp.name)
        torch.set_num_threads(2)
        cls.models = {seed: load_model(path, device="cpu") for seed, path in CHECKPOINTS.items()}

    def run_cli(self, image=SAMPLE, seed=5421, *extra):
        return subprocess.run([
            sys.executable, "-m", "cli.predict", "--image", str(image),
            "--checkpoint", str(CHECKPOINTS[seed]), *extra,
        ], cwd=ROOT, text=True, capture_output=True, timeout=60)

    def reference_probability(self, image, seed):
        model, metadata = self.models[seed]
        transform = EvalTransform(DataConfig(**metadata["preprocessing"]["data_config"]), suite="clean")
        with Image.open(image) as opened, torch.inference_mode():
            return model.predict_proba(transform(opened).unsqueeze(0)).item()

    def test_both_checkpoints_match_shared_interface_and_optional_config(self):
        for seed in CHECKPOINTS:
            with self.subTest(seed=seed):
                output = self.directory / f"{seed}.json"
                run = self.run_cli(SAMPLE, seed, "--output", str(output))
                self.assertEqual(run.returncode, 0, run.stderr)
                first = json.loads(output.read_text())
                self.assertEqual(first["purpose"], "single_image_inference")
                self.assertEqual(first["model_status"], "formal")
                self.assertEqual((first["condition"], first["seed"]), ("C", seed))
                self.assertNotIn("warning", first)
                self.assertAlmostEqual(first["p_ai"], self.reference_probability(SAMPLE, seed), places=7)
                self.assertEqual(first["predicted_label_id"], int(first["p_ai"] >= 0.5))
                config = ROOT / f"docs/wang-formal-results/C_{seed}/config.yaml"
                again = self.run_cli(SAMPLE, seed, "--config", str(config), "--output", str(output))
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
                run = self.run_cli(image, 5421, "--output", str(output))
                self.assertEqual(run.returncode, 0, run.stderr)
                actual = json.loads(output.read_text())["p_ai"]
                self.assertAlmostEqual(actual, self.reference_probability(image, 5421), places=7)

    def test_wrong_seed_config_preserves_existing_output(self):
        output = self.directory / "existing.json"
        output.write_text('{"previous": true}\n')
        config = ROOT / "docs/wang-formal-results/C_2026/config.yaml"
        run = self.run_cli(SAMPLE, 5421, "--config", str(config), "--output", str(output))
        self.assertEqual(run.returncode, 2)
        self.assertIn("配置", run.stderr)
        self.assertNotIn("P(AI)=", run.stdout)
        self.assertEqual(output.read_text(), '{"previous": true}\n')

    def test_optional_config_output_protection(self):
        for path in (SAMPLE, CHECKPOINTS[5421]):
            with self.subTest(path=path.name):
                run = self.run_cli(SAMPLE, 5421, "--output", str(path))
                self.assertEqual(run.returncode, 2)
                self.assertIn("不能覆盖", run.stderr)


if __name__ == "__main__":
    unittest.main()
