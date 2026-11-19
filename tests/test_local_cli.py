"""Integration checks for the local-only CLI and its untrained-model guard."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image

from cli.smoke_model import create_smoke_checkpoint


class LocalCliTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.checkpoint, cls.config = create_smoke_checkpoint(cls.root / "model")
        cls.image = cls.root / "sample.png"
        Image.new("RGB", (240, 320), color=(60, 100, 140)).save(cls.image)

    def run_cli(self, *extra, image=None, checkpoint=None, config=None):
        return subprocess.run([
            sys.executable, "-m", "cli.predict",
            "--image", str(image or self.image),
            "--checkpoint", str(checkpoint or self.checkpoint),
            "--config", str(config or self.config), *extra,
        ], text=True, capture_output=True, timeout=60)

    def test_explicit_smoke_result_is_saved_and_reproducible(self):
        first, second = self.root / "first.json", self.root / "second.json"
        for output in [first, second]:
            run = self.run_cli("--allow-untrained", "--output", str(output))
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn("仅流程测试", run.stdout)
        a, b = json.loads(first.read_text()), json.loads(second.read_text())
        self.assertEqual(a, b)
        self.assertEqual(a["model_status"], "untrained")
        self.assertEqual(a["purpose"], "pipeline_test_only")
        self.assertGreaterEqual(a["p_ai"], 0)
        self.assertLessEqual(a["p_ai"], 1)
        self.assertIn("不是有效", a["warning"])

    def test_untrained_model_is_rejected_by_default(self):
        run = self.run_cli()
        self.assertEqual(run.returncode, 2)
        self.assertIn("--allow-untrained", run.stderr)
        self.assertNotIn("P(AI)=", run.stdout)

    def test_missing_image_or_checkpoint_fails_without_prediction(self):
        for override in [{"image": self.root / "missing.png"},
                         {"checkpoint": self.root / "missing.pt"}]:
            with self.subTest(override=override):
                run = self.run_cli("--allow-untrained", **override)
                self.assertEqual(run.returncode, 2)
                self.assertIn("文件不存在", run.stderr)
                self.assertNotIn("P(AI)=", run.stdout)

    def test_corrupt_image_fails_without_json(self):
        corrupt, output = self.root / "corrupt.png", self.root / "bad-result.json"
        corrupt.write_bytes(b"not an image")
        run = self.run_cli("--allow-untrained", "--output", str(output), image=corrupt)
        self.assertEqual(run.returncode, 2)
        self.assertFalse(output.exists())

    def test_changed_config_is_rejected(self):
        changed = self.root / "changed.yaml"
        config = yaml.safe_load(self.config.read_text())
        config["threshold"] = 0.7
        changed.write_text(yaml.safe_dump(config))
        run = self.run_cli("--allow-untrained", config=changed)
        self.assertEqual(run.returncode, 2)
        self.assertNotIn("P(AI)=", run.stdout)

    def test_output_cannot_overwrite_input(self):
        original = self.image.read_bytes()
        run = self.run_cli("--allow-untrained", "--output", str(self.image))
        self.assertEqual(run.returncode, 2)
        self.assertEqual(self.image.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
