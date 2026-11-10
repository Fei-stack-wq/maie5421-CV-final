"""Contract checks for OUYANG's data handoff, using generated local images."""

import csv
import hashlib
import json

import numpy as np
import pytest
import torch
from PIL import Image

from src.datasets import (
    DataConfig,
    EVAL_SUITES,
    EvalTransform,
    GenImageDataset,
    PairedViewTransform,
    make_dataloader,
)


@pytest.fixture
def image_root(tmp_path):
    """A tiny manifest with the real field names and all images outside git."""
    root = tmp_path / "Images"
    root.mkdir()
    yy, xx = np.indices((310, 420))
    pattern = np.stack(
        [(xx * 19 + yy * 3) % 256, (yy * 23) % 256, ((xx ^ yy) * 7) % 256],
        axis=-1,
    ).astype(np.uint8)
    records = []
    for i, (split, label, mode) in enumerate(
        [("train", 0, "RGB"), ("train", 1, "RGB"), ("val", 0, "L"), ("val", 1, "RGBA")]
    ):
        rel_path = f"{split}/sample_{i}.png"
        path = root / rel_path
        path.parent.mkdir(exist_ok=True)
        image = Image.fromarray(np.roll(pattern, i * 13, axis=1)).convert(mode)
        image.save(path)
        raw = path.read_bytes()
        records.append(
            {
                "image_id": f"{split}-sample-{i}",
                "split": split,
                "label": label,
                "generator": "real" if label == 0 else "BigGAN",
                "family": "real" if label == 0 else "GAN",
                "class_id": i,
                "class_name": f"class_{i}",
                "filename": path.name,
                "rel_path": rel_path,
                "source_repo": "generated-test-data",
                "source_split": split,
                "source_image_path": rel_path,
                "source_md5": hashlib.md5(raw).hexdigest(),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "width": image.width,
                "height": image.height,
                "format": "PNG",
                "bytes_file": len(raw),
                "selection_seed": 5421,
            }
        )
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)
    return root, manifest, records


def test_manifest_drives_split_labels_and_ids(image_root):
    root, manifest, records = image_root
    dataset = GenImageDataset(manifest, split="train")
    assert len(dataset) == 2
    for index in range(2):
        sample = dataset[index]
        assert sample["image_id"] == records[index]["image_id"]
        assert sample["rel_path"] == records[index]["rel_path"]
        assert sample["label"].dtype == torch.float32
        assert sample["label"].ndim == 0
        assert sample["label"].item() == records[index]["label"]
        assert sample["image"].shape == (3, 224, 224)
        assert sample["image"].dtype == torch.float32
        assert torch.isfinite(sample["image"]).all()
        assert "view" not in sample


def test_default_collate_and_rgb_conversion(image_root):
    _, manifest, _ = image_root
    dataset = GenImageDataset(manifest, split="val")
    batch = next(iter(make_dataloader(dataset, batch_size=2)))
    assert batch["image"].shape == (2, 3, 224, 224)
    assert batch["label"].shape == (2,)
    assert batch["label"].tolist() == [0.0, 1.0]
    assert batch["split"] == ["val", "val"]
    gray = dataset[0]["image"]
    config = DataConfig()
    unnormalized = gray * torch.tensor(config.std)[:, None, None]
    unnormalized += torch.tensor(config.mean)[:, None, None]
    torch.testing.assert_close(unnormalized[0], unnormalized[1])
    torch.testing.assert_close(unnormalized[0], unnormalized[2])


def test_imagenet_normalization_has_expected_black_value():
    image = Image.new("RGB", (320, 260), (0, 0, 0))
    config = DataConfig()
    actual = EvalTransform(config, "clean")(image)
    expected = -torch.tensor(config.mean) / torch.tensor(config.std)
    torch.testing.assert_close(actual[:, 0, 0], expected)
    torch.testing.assert_close(actual, expected[:, None, None].expand_as(actual))


def test_b_and_c_share_views_and_do_not_consume_global_rng(image_root):
    _, manifest, _ = image_root
    b = GenImageDataset(manifest, split="train", condition="B", seed=17)
    c = GenImageDataset(manifest, split="train", condition="C", seed=17)
    b.set_epoch(3)
    c.set_epoch(3)
    initial_rng = torch.random.get_rng_state().clone()
    first = b[0]
    b[1]
    repeated = b[0]
    comparison = c[0]
    for item in (repeated, comparison):
        torch.testing.assert_close(first["image"], item["image"], rtol=0, atol=0)
        torch.testing.assert_close(first["view"], item["view"], rtol=0, atol=0)
        assert first["view_params"] == item["view_params"]
    assert torch.equal(initial_rng, torch.random.get_rng_state())
    assert isinstance(json.loads(first["view_params"]), dict)
    b.set_epoch(4)
    assert b[0]["view_params"] != first["view_params"]


def test_view_transform_repeats_for_fixed_seed_without_mutating_input():
    yy, xx = np.indices((256, 300))
    array = np.stack([(xx * 7) % 256, (yy * 11) % 256, (xx + yy) % 256], -1).astype(np.uint8)
    image = Image.fromarray(array)
    original = image.tobytes()
    transform = PairedViewTransform(DataConfig())
    first = transform(image, seed=83)
    second = transform(image, seed=83)
    assert image.tobytes() == original
    assert first["image"].shape == first["view"].shape == (3, 224, 224)
    torch.testing.assert_close(first["view"], second["view"], rtol=0, atol=0)
    assert first["view_params"] == second["view_params"]


@pytest.mark.parametrize(
    "suite",
    ["clean", "jpeg_q70", "jpeg_q50", "downsample_0.5", "gaussian_blur_1.0", "bias_control_jpeg95"],
)
def test_fixed_eval_suite_is_deterministic_and_preserves_identity(image_root, suite):
    _, manifest, _ = image_root
    assert suite in EVAL_SUITES
    dataset = GenImageDataset(manifest, split="val", suite=suite)
    first = dataset[1]
    second = dataset[1]
    assert first["image_id"] == second["image_id"] == "val-sample-3"
    assert first["label"].item() == 1
    assert first["transformation"] == suite
    assert first["image"].shape == (3, 224, 224)
    torch.testing.assert_close(first["image"], second["image"], rtol=0, atol=0)
    if suite != "clean":
        clean = GenImageDataset(manifest, split="val", suite="clean")[1]["image"]
        assert not torch.equal(first["image"], clean)


def test_order_and_views_match_across_b_c_loaders(image_root):
    _, manifest, _ = image_root
    b = GenImageDataset(manifest, split="train", condition="B", seed=92)
    c = GenImageDataset(manifest, split="train", condition="C", seed=92)
    b.set_epoch(2)
    c.set_epoch(2)
    batch_b = next(iter(make_dataloader(b, batch_size=2)))
    batch_c = next(iter(make_dataloader(c, batch_size=2)))
    assert batch_b["image_id"] == batch_c["image_id"]
    assert batch_b["view_params"] == batch_c["view_params"]
    torch.testing.assert_close(batch_b["view"], batch_c["view"], rtol=0, atol=0)


def test_persistent_worker_observes_epoch_and_matches_main_process(image_root):
    _, manifest, _ = image_root
    dataset = GenImageDataset(manifest, split="train", condition="B", seed=21)
    loader = make_dataloader(
        dataset, batch_size=2, shuffle=False, num_workers=1, persistent_workers=True
    )
    try:
        dataset.set_epoch(0)
        first = next(iter(loader))
        dataset.set_epoch(1)
        second = next(iter(loader))
        assert first["view_params"] != second["view_params"]
        for index in range(2):
            expected = dataset[index]
            assert expected["view_params"] == second["view_params"][index]
            torch.testing.assert_close(expected["view"], second["view"][index], rtol=0, atol=0)
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()


def test_fixture_requires_its_own_split_and_resolves_from_images_root(image_root):
    root, _, records = image_root
    fixture_folder = root / "fixture"
    fixture_folder.mkdir()
    fixture_manifest = fixture_folder / "manifest_fixture.csv"
    fixture_image = fixture_folder / "sample.png"
    fixture_image.write_bytes((root / records[0]["rel_path"]).read_bytes())
    record = dict(
        records[0], split="fixture", image_id="fixture-sample", rel_path="fixture/sample.png"
    )
    with fixture_manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=record)
        writer.writeheader()
        writer.writerow(record)
    dataset = GenImageDataset(fixture_manifest, split="fixture", root=root, condition="B")
    assert len(dataset) == 1
    assert dataset[0]["image_id"] == "fixture-sample"
    assert dataset[0]["view"].shape == (3, 224, 224)


def test_missing_image_fails_clearly_without_silent_substitution(image_root):
    root, manifest, records = image_root
    (root / records[0]["rel_path"]).unlink()
    with pytest.raises((FileNotFoundError, RuntimeError, OSError)):
        GenImageDataset(manifest, split="train")[0]


def test_unknown_suite_and_condition_fail_early(image_root):
    _, manifest, _ = image_root
    with pytest.raises(ValueError):
        GenImageDataset(manifest, split="train", condition="D")
    with pytest.raises(ValueError):
        GenImageDataset(manifest, split="val", suite="jpeg_random")


@pytest.mark.parametrize(
    "updates", [{"generator": "ADM"}, {"label": 0}, {"image_id": "train-sample-0"}]
)
def test_invalid_manifest_cannot_silently_enter_training(image_root, updates):
    root, manifest, records = image_root
    records[1].update(updates)
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)
    with pytest.raises(ValueError):
        GenImageDataset(manifest, split="train", root=root)


def test_paired_training_views_cannot_be_used_as_validation_inputs(image_root):
    _, manifest, _ = image_root
    with pytest.raises(ValueError):
        GenImageDataset(manifest, split="val", condition="C")


@pytest.mark.parametrize("bad_path", ["../outside.png", "/absolute.png", "C:/outside.png"])
def test_manifest_path_cannot_escape_images_root(image_root, bad_path):
    root, manifest, records = image_root
    records[0]["rel_path"] = bad_path
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)
    with pytest.raises((ValueError, RuntimeError)):
        GenImageDataset(manifest, split="train", root=root)[0]
