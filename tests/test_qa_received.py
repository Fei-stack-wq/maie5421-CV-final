"""Receipt QA must distinguish metadata, bytes, and perceptual coverage."""

import csv
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


_SPEC = importlib.util.spec_from_file_location(
    "qa_received", Path(__file__).resolve().parents[1] / "tools/dataset/qa_received.py"
)
qa = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(qa)


def make_row(root, split, name, *, pixels=None, compress_level=6):
    path = root / split / "real" / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    if pixels is None:
        seed = sum(map(ord, name))
        pixels = np.random.default_rng(seed).integers(0, 256, size=(40, 50, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    image.save(path, compress_level=compress_level)
    data = path.read_bytes()
    return {
        "image_id": f"{split}-{name}", "split": split, "label": "0",
        "generator": "real", "rel_path": path.relative_to(root).as_posix(),
        "source_md5": hashlib.md5(data).hexdigest(),
        "sha256": hashlib.sha256(data).hexdigest(), "width": "50", "height": "40",
        "format": "PNG", "bytes_file": str(len(data)), "class_id": "0",
    }


def write_manifest(root, rows, name="manifest.csv"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(qa.REQUIRED))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_metadata_success_never_claims_full_receipt(tmp_path):
    row = make_row(tmp_path, "train", "a")
    manifest = write_manifest(tmp_path, [row])
    (tmp_path / row["rel_path"]).unlink()
    result = qa.run_qa(tmp_path, manifest, None, project_counts=False)
    assert result["metadata"]["ok"]
    assert result["status"] == "metadata_pass_images_unchecked"
    assert result["full_qa_pass"] is False
    assert "images" not in result


def test_missing_bytes_are_incomplete(tmp_path):
    row = make_row(tmp_path, "train", "a")
    manifest = write_manifest(tmp_path, [row])
    (tmp_path / row["rel_path"]).unlink()
    result = qa.run_qa(tmp_path, manifest, None, check_images=True, phash=True, project_counts=False)
    assert result["status"] == "incomplete_missing_images"
    assert result["exit_code"] == 2
    assert result["images"]["missing_image_count"] == 1
    assert result["near_duplicates"]["coverage_complete"] is False
    assert not result["full_qa_pass"]


def test_complete_bytes_without_phash_still_pending(tmp_path):
    manifest = write_manifest(tmp_path, [make_row(tmp_path, "train", "a")])
    result = qa.run_qa(tmp_path, manifest, None, check_images=True, project_counts=False)
    assert result["images"]["complete"]
    assert result["status"] == "image_integrity_pass_phash_pending"
    assert not result["full_qa_pass"]


def test_modified_bytes_fail_integrity(tmp_path):
    row = make_row(tmp_path, "train", "a")
    manifest = write_manifest(tmp_path, [row])
    (tmp_path / row["rel_path"]).write_bytes(b"not an image")
    result = qa.run_qa(tmp_path, manifest, None, check_images=True, project_counts=False)
    assert result["status"] == "failed"
    assert result["images"]["error_count"] == 1
    assert result["images"]["images_verified"] == 0


@pytest.mark.parametrize("path", ["../secret.png", "train/../../secret.png", "C:/secret.png", "/secret.png", "\\\\server\\secret.png"])
def test_path_escape_rejected(tmp_path, path):
    with pytest.raises(ValueError):
        qa.safe_image_path(tmp_path, path)


def test_fixture_exact_leakage_is_detected(tmp_path):
    row = make_row(tmp_path, "train", "a")
    fixture = {**row, "image_id": "fixture-a", "split": "fixture", "rel_path": "fixture/real/a.png"}
    result = qa.metadata_qa([row], tmp_path, project_counts=False, fixture_rows=[fixture])
    assert not result["ok"]
    assert result["duplicate_groups"]["sha256"][0]["cross_split"]


def test_held_out_generator_and_bad_class_are_reported(tmp_path):
    row = make_row(tmp_path, "train", "a")
    row.update(generator="ADM", label="1", class_id="not-an-int")
    result = qa.metadata_qa([row], tmp_path, project_counts=False)
    assert not result["ok"]
    assert "held-out generator leakage" in " ".join(result["errors"])


def test_identical_pixels_different_encodings_need_review(tmp_path):
    pixels = np.random.default_rng(42).integers(0, 256, size=(40, 50, 3), dtype=np.uint8)
    a = make_row(tmp_path, "train", "a", pixels=pixels, compress_level=0)
    b = make_row(tmp_path, "test_seen", "b", pixels=pixels, compress_level=9)
    assert a["sha256"] != b["sha256"]
    manifest = write_manifest(tmp_path, [a, b])
    result = qa.run_qa(tmp_path, manifest, None, phash=True, project_counts=False)
    assert result["status"] == "review_required"
    assert result["exit_code"] == 3
    assert result["near_duplicates"]["candidate_pair_count"] == 1
    assert result["near_duplicates"]["candidates"][0]["hamming_distance"] == 0
    assert (tmp_path / a["rel_path"]).exists()  # no automated deletion


def test_full_pass_requires_complete_images_and_zero_candidates(tmp_path):
    rows = [make_row(tmp_path, "train", "a"), make_row(tmp_path, "test_seen", "b")]
    manifest = write_manifest(tmp_path, rows)
    result = qa.run_qa(tmp_path, manifest, None, phash=True, project_counts=False, phash_threshold=0)
    assert result["status"] == "passed"
    assert result["full_qa_pass"]


def test_candidate_count_remains_complete_when_examples_are_capped():
    rows = [
        {"image_id": str(i), "split": split, "label": "0", "generator": "real", "rel_path": str(i)}
        for i, split in enumerate(["train"] * 3 + ["val"] * 2 + ["fixture"])
    ]
    result = qa.screen_near_duplicates([(row, 17) for row in rows], max_candidates=2)
    assert result["candidate_pair_count"] == 3 * 2 + 3 * 1 + 2 * 1
    assert len(result["candidates"]) == 2
    assert result["candidate_examples_truncated"]


def test_bktree_matches_brute_force_cross_split_screen():
    rng = np.random.default_rng(1)
    values = list(map(int, rng.integers(0, 2**16, size=40, dtype=np.uint64)))
    rows = [
        {"image_id": str(i), "split": ["train", "val", "fixture"][i % 3],
         "label": "0", "generator": "real", "rel_path": str(i)}
        for i in range(len(values))
    ]
    expected = sum(
        rows[i]["split"] != rows[j]["split"] and bin(values[i] ^ values[j]).count("1") <= 4
        for i in range(len(values)) for j in range(i)
    )
    result = qa.screen_near_duplicates(list(zip(rows, values)), threshold=4)
    assert result["candidate_pair_count"] == expected
