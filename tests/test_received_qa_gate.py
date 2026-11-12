import copy
import json
import unittest
from pathlib import Path

from tools.training.check_received_qa import check_reports


class ReceivedQAGateTests(unittest.TestCase):
    def setUp(self):
        self.qa = json.loads(Path("artifacts/ouyang/qa_received.json").read_text())
        self.review = json.loads(Path("artifacts/ouyang/qa_visual_review.json").read_text())

    def test_accepts_existing_review_without_rewriting_raw_status(self):
        check_reports(self.qa, self.qa, self.review)
        self.assertFalse(self.qa["full_qa_pass"])

    def test_rejects_incomplete_or_changed_screening(self):
        changes = [("images", "images_verified", 12000),
                   ("near_duplicates", "candidate_pair_count", 2),
                   ("near_duplicates", "coverage_complete", False),
                   ("near_duplicates", "threshold_hamming", 5)]
        for group, key, value in changes:
            qa = copy.deepcopy(self.qa)
            qa[group][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                check_reports(qa, self.qa, self.review)
