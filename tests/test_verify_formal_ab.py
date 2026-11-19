"""Offline guards for formal A/B verification; no real-data/model claims."""

import copy
import csv
import hashlib
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import yaml

from src.train import TrainConfig
from tools.training import verify_formal_ab as verifier


def records():
    config = TrainConfig(condition="A", consistency_weight=0.0, run_kind="formal",
                         protocol_confirmed=True, dataset_factory="src.training.ouyang_factory:build_datasets",
                         data_config="configs/ouyang_data_v1.json", device="cuda", amp_init_scale=1024.0,
                         train_limit=None, val_limit=None)
    provenance = dict(commit="a" * 40, working_tree_dirty=False, device="cuda", amp_enabled=True,
                      run_kind="formal", manifest_sha256="manifest", train_images=6000, val_images=1500,
                      data_audit=dict(manifest_rows=12000, sha256_verified=12000, missing_hashes=0))
    history = []
    for epoch, (auc, improved, bad) in enumerate(((.7, True, 0), (.8, True, 0), (.75, False, 1), (.78, False, 2))):
        history.append(dict(epoch=epoch, train_loss=.4, optimizer_steps=375, amp_skipped_steps=0,
                            train_images=6000, val_loss=.5, val_auc=auc, val_accuracy=.7, val_images=1500,
                            seconds=1.2, is_best=improved, bad_epochs=bad))
    common = dict(config=asdict(config), best_epoch=1, best_val_auc=.8, threshold=.5,
                  provenance=provenance, preprocessing={"provider": "test"})
    best = dict(copy.deepcopy(common), epoch=1, history=copy.deepcopy(history[:2]), bad_epochs=0)
    last = dict(copy.deepcopy(common), epoch=3, history=copy.deepcopy(history), bad_epochs=2)
    summary = dict(condition="A", seed=5421, best_epoch=1, best_val_auc=.8,
                   completed_epochs=4, early_stopped=True, provenance=copy.deepcopy(provenance))
    return config, best, last, summary, history


class FormalRecordTests(unittest.TestCase):
    def setUp(self):
        self.config, self.best, self.last, self.summary, self.history = records()

    def check(self):
        return verifier.check_records(self.config, self.best, self.last, self.summary, self.history, "manifest")

    def test_accepts_early_stop_with_best_before_last(self):
        self.assertEqual(self.check(), {"epochs": 4, "optimizer_steps": 1500, "amp_skipped_steps": 0})

    def test_rejects_unfinished_run(self):
        self.history.pop()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.check()

    def test_rejects_summary_mismatch(self):
        self.summary["best_epoch"] = 3
        with self.assertRaisesRegex(ValueError, "summary"):
            self.check()

    def test_rejects_nonfinite_history(self):
        self.history[0]["val_auc"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            self.check()

    def test_rejects_zero_actual_updates(self):
        self.history[0]["optimizer_steps"] = 0
        self.history[0]["amp_skipped_steps"] = 375
        with self.assertRaisesRegex(ValueError, "zero actual"):
            self.check()

    def test_rejects_wrong_update_budget(self):
        self.history[0]["optimizer_steps"] = 374
        with self.assertRaisesRegex(ValueError, "budget"):
            self.check()

    def test_rejects_wrong_best_selection(self):
        self.history[1]["is_best"] = False
        with self.assertRaisesRegex(ValueError, "selection"):
            self.check()

    def test_rejects_extra_epoch_after_stopping(self):
        self.history.append(dict(self.history[-1], epoch=4))
        with self.assertRaisesRegex(ValueError, "continues after"):
            self.check()

    def test_rejects_wrong_best_checkpoint(self):
        self.best["epoch"] = 3
        with self.assertRaisesRegex(ValueError, "best checkpoint"):
            self.check()

    def test_rejects_partial_audit(self):
        self.best["provenance"]["data_audit"]["sha256_verified"] = 48
        with self.assertRaisesRegex(ValueError, "12000"):
            self.check()

    def test_rejects_dirty_training_code(self):
        self.best["provenance"]["working_tree_dirty"] = True
        with self.assertRaisesRegex(ValueError, "clean committed"):
            self.check()

    def test_rejects_mixed_source_revisions(self):
        self.best["provenance"]["commit"] = "b" * 40
        with self.assertRaisesRegex(ValueError, "provenance differs"):
            self.check()

    def test_rejects_changed_manifest(self):
        self.last["provenance"]["manifest_sha256"] = "other"
        with self.assertRaisesRegex(ValueError, "manifest"):
            self.check()

    def test_rejects_changed_preprocessing(self):
        self.best["preprocessing"] = {"provider": "wrong"}
        with self.assertRaisesRegex(ValueError, "preprocessing"):
            self.check()

    def test_rejects_mixed_checkpoint_config(self):
        self.best["config"]["seed"] = 2026
        with self.assertRaisesRegex(ValueError, "configuration"):
            self.check()

    def test_history_csv_must_match_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(self.history[0]))
                writer.writeheader()
                writer.writerows(self.history)
            self.assertEqual(verifier.read_history(path, self.history), self.history)
            path.write_text(path.read_text().replace("0.7", "0.6", 1))
            with self.assertRaisesRegex(ValueError, "CSV differs"):
                verifier.read_history(path, self.history)


class ProtocolAndPredictionTests(unittest.TestCase):
    def test_four_protocols_match_c_except_condition_and_lambda(self):
        config, *_ = records()
        for run_id in verifier.RUN_IDS:
            condition, seed = run_id.split("_")
            current = replace(config, condition=condition, seed=int(seed))
            reference = replace(current, condition="C", consistency_weight=1.0)
            verifier.check_protocol(current, run_id, reference)
            with self.assertRaisesRegex(ValueError, "differs"):
                verifier.check_protocol(replace(current, amp_init_scale=65536.0), run_id, reference)

    def test_rejects_pilot_and_unagreed_reference(self):
        config, *_ = records()
        reference = replace(config, condition="C", consistency_weight=1.0)
        with self.assertRaises(ValueError):
            verifier.check_protocol(replace(config, run_kind="pilot"), "A_5421", reference)
        with self.assertRaisesRegex(ValueError, "reference C"):
            verifier.check_protocol(config, "A_5421", replace(reference, amp=False))

    def test_exact_predictions_and_bounded_probability_error(self):
        actual = [("x", 0, .25), ("y", 1, .75)]
        saved = [{"image_id": x, "label": str(y), "p_ai": str(p)} for x, y, p in actual]
        self.assertEqual(verifier.compare_predictions(actual, saved, count=2), 0)
        saved[1]["p_ai"] = ".750001"
        self.assertLess(verifier.compare_predictions(actual, saved, count=2), 1e-5)
        saved[1]["p_ai"] = ".8"
        with self.assertRaisesRegex(ValueError, "tolerance"):
            verifier.compare_predictions(actual, saved, count=2)

    def test_rejects_nan_duplicate_id_and_bad_label(self):
        actual = [("x", 0, .25), ("y", 1, .75)]
        saved = [{"image_id": x, "label": str(y), "p_ai": str(p)} for x, y, p in actual]
        for replacement in ("nan", "inf", "-0.1", "1.1"):
            wrong = copy.deepcopy(saved)
            wrong[0]["p_ai"] = replacement
            with self.assertRaises(ValueError):
                verifier.compare_predictions(actual, wrong, count=2)
        with self.assertRaisesRegex(ValueError, "unique"):
            verifier.compare_predictions([actual[0], actual[0]], [saved[0], saved[0]], count=2)
        with self.assertRaisesRegex(ValueError, "label"):
            verifier.compare_predictions([("x", .5, .25), actual[1]], saved, count=2)

    def test_json_rejects_nonfinite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('{"value": NaN}')
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                verifier.read_json(path)


class VerificationWorkflowTests(unittest.TestCase):
    def test_mocked_complete_run_checks_all_1500_predictions_and_hashes(self):
        config, best, last, summary, history = records()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            manifest.write_text("synthetic manifest; no real data asserted")
            config = replace(config, data_manifest=str(manifest))
            for state in (best, last):
                state["config"] = asdict(config)
                state["provenance"]["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
            summary["provenance"] = last["provenance"]
            predictions = [(str(i), i % 2, .75 if i % 2 else .25) for i in range(1500)]
            best["val_predictions"] = predictions
            last["val_predictions"] = predictions
            for filename in ("best.pt", "last.pt"):
                (root / filename).write_bytes(b"mock checkpoint: not a trained model")
            (root / "config.yaml").write_text(yaml.safe_dump(asdict(config)))
            (root / "summary.json").write_text(json.dumps(summary))
            with (root / "history.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(history[0]))
                writer.writeheader(); writer.writerows(history)
            with (root / "val_predictions.csv").open("w", newline="") as stream:
                writer = csv.writer(stream); writer.writerow(("image_id", "label", "p_ai")); writer.writerows(predictions)
            audit = dict(manifest_rows=12000, sha256_verified=12000, missing_hashes=0)
            with patch.object(verifier, "read_checkpoint", side_effect=[best, last]), \
                 patch.object(verifier, "make_datasets", return_value=(range(6000), range(1500), best["preprocessing"], audit)), \
                 patch.object(verifier, "load_model", return_value=(object(), {"preprocessing": best["preprocessing"]})), \
                 patch.object(verifier, "validate", return_value=({"val_auc": .8, "val_loss": .5, "val_accuracy": .7, "val_images": 1500}, predictions)):
                result = verifier.verify_run(root, "A_5421", reference_config=replace(config, condition="C", consistency_weight=1.0))
            self.assertEqual(result["validation_images"], 1500)
            self.assertEqual(result["max_probability_error"], 0)
            self.assertEqual(len(result["sha256"]), 6)

    def test_subset_never_claims_all_four_verified(self):
        with patch.object(verifier, "verify_run", return_value={"status": "verified"}):
            result = verifier.verify_runs("unused", ["A_5421"])
        self.assertTrue(result["selected_verified"])
        self.assertFalse(result["all_four_verified"])
        self.assertEqual(len(result["not_verified_runs"]), 3)

    def test_all_four_and_duplicate_guard(self):
        with patch.object(verifier, "verify_run", return_value={"status": "verified"}):
            self.assertTrue(verifier.verify_runs("unused", verifier.RUN_IDS)["all_four_verified"])
        with self.assertRaisesRegex(ValueError, "unique"):
            verifier.verify_runs("unused", ["A_5421", "A_5421"])

    def test_existing_report_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text("keep")
            with patch.object(verifier, "verify_runs") as verify, self.assertRaises(SystemExit):
                verifier.main(["--output", str(path)])
            verify.assert_not_called()
            self.assertEqual(path.read_text(), "keep")

    def test_report_cannot_overwrite_run_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(verifier, "verify_runs") as verify, self.assertRaises(SystemExit):
                verifier.main(["--runs-root", directory, "--output", str(Path(directory) / "A_5421/new.json")])
            verify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
