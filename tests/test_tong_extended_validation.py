"""Stopping and invalid-device guards for TONG's training entry point."""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from test_tong_training import TinyModel, fixture
from src.train import TrainConfig, run
from src.training.checkpoint import read_checkpoint
from src.training.engine import binary_auc


class TrainingGuardTests(unittest.TestCase):
    def test_patience_stops_after_two_non_improvements_and_rejects_resume(self):
        with tempfile.TemporaryDirectory() as temp, patch('src.train.BinaryResNet18', TinyModel):
            root = Path(temp)
            manifest = fixture(root)
            config = TrainConfig(data_manifest=str(manifest), data_root=str(root), weights=None,
                                 train_limit=None, val_limit=None, epochs=8, early_stop=2,
                                 min_delta=1.0, amp=False)
            # AUC <= 1, so no later epoch can exceed best_auc + 1.0.
            summary = run(config, root/'run')
            self.assertTrue(summary['early_stopped'])
            self.assertEqual(summary['completed_epochs'], 3)
            self.assertEqual(summary['best_epoch'], 0)
            last = read_checkpoint(root/'run/last.pt')
            best = read_checkpoint(root/'run/best.pt')
            self.assertEqual(last['bad_epochs'], 2)
            self.assertEqual(last['epoch'], 2)
            self.assertEqual(best['epoch'], 0)
            with self.assertRaisesRegex(ValueError, 'already stopped'):
                run(replace(config, epochs=9), root/'run', root/'run/last.pt')

    def test_auc_matches_independent_pairwise_definition(self):
        # Independent positive-negative pair counting, including repeated scores.
        generator = torch.Generator().manual_seed(2026)
        labels = [0, 1] * 12
        for _ in range(20):
            scores = torch.randint(-3, 4, (len(labels),), generator=generator).tolist()
            positive = [score for label, score in zip(labels, scores) if label == 1]
            negative = [score for label, score in zip(labels, scores) if label == 0]
            expected = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative)
            expected /= len(positive) * len(negative)
            self.assertAlmostEqual(binary_auc(labels, scores), expected, places=14)

    @unittest.skipIf(torch.cuda.is_available(), 'CUDA is available in this environment')
    def test_unavailable_cuda_is_reported_before_data_or_weight_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch('src.train.make_datasets') as make_data:
                with self.assertRaisesRegex(RuntimeError, 'CUDA was requested but is unavailable'):
                    run(TrainConfig(device='cuda'), Path(temp)/'run')
                make_data.assert_not_called()


if __name__ == '__main__':
    unittest.main()
