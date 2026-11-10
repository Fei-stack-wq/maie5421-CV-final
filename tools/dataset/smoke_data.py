"""Run data-interface checks on received images and export a perturbation preview."""
import argparse
import json
import platform
from pathlib import Path

import numpy as np
import PIL
from PIL import Image, ImageDraw
import torch

from src.datasets import DataConfig, EVAL_SUITES, GenImageDataset, make_dataloader
from src.datasets.transforms import prepare_image, perturb_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("Images/fixture/manifest_fixture.csv"))
    parser.add_argument("--root", type=Path, default=Path("Images"))
    parser.add_argument("--split", default="fixture")
    parser.add_argument("--config", type=Path, default=Path("configs/ouyang_data_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/ouyang/smoke"))
    parser.add_argument("--seed", type=int, default=5421)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    image_root = args.root.resolve()
    if output == image_root or image_root in output.parents:
        parser.error("smoke output must be outside the read-only image root")
    if args.split not in ("train", "fixture"):
        parser.error("paired-view smoke checks require train or fixture")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    config = DataConfig.from_json(args.config)
    common = dict(manifest=args.manifest, root=args.root, split=args.split,
                  config=config, seed=args.seed)
    baseline = GenImageDataset(**common, condition="A")
    paired_b = GenImageDataset(**common, condition="B")
    paired_c = GenImageDataset(**common, condition="C")
    checked = 0
    for index in range(len(baseline)):
        a, b, c = baseline[index], paired_b[index], paired_c[index]
        assert a["image"].shape == (3, config.image_size, config.image_size)
        assert a["image"].dtype == torch.float32
        assert torch.isfinite(a["image"]).all() and torch.isfinite(b["view"]).all()
        assert torch.equal(a["image"], b["image"])
        assert torch.equal(b["view"], c["view"])
        assert b["view_params"] == c["view_params"]
        checked += 1
    loader = make_dataloader(paired_b, batch_size=16, num_workers=args.num_workers)
    first = next(iter(loader))
    assert first["image"].ndim == 4 and first["label"].shape == (min(16, checked),)
    paired_b.set_epoch(1)
    paired_c.set_epoch(1)
    assert paired_b[0]["view_params"] == paired_c[0]["view_params"]
    # Pick one real and one image from each available fake generator.
    selected, seen = [], set()
    for index, row in enumerate(baseline.rows):
        if row["generator"] not in seen:
            seen.add(row["generator"])
            selected.append(index)
    tile, text_h = 224, 52
    sheet = Image.new("RGB", (len(EVAL_SUITES)*tile, len(selected)*(tile+text_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for ri, index in enumerate(selected):
        row = baseline.rows[index]
        with Image.open(args.root / row["rel_path"]) as original:
            for ci, suite in enumerate(EVAL_SUITES):
                image = prepare_image(original, config, suite == "bias_control_jpeg95")
                image = perturb_image(image, suite)
                x, y = ci * tile, ri * (tile + text_h)
                sheet.paste(image, (x, y + text_h))
                draw.text((x+5, y+4), suite, fill="black")
                draw.text((x+5, y+22), row["generator"], fill="black")
    suites = {}
    for suite in EVAL_SUITES:
        ds = GenImageDataset(**common, condition="A", suite=suite)
        for index in selected:
            sample = ds[index]
            assert sample["image_id"] == baseline[index]["image_id"]
            assert sample["image"].shape == (3, config.image_size, config.image_size)
            assert torch.isfinite(sample["image"]).all()
        suites[suite] = {"examples_checked": len(selected)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output_dir / "preview.png")
    result = {
        "status": "passed", "scope": "data interface only; not a model-training pilot",
        "split": args.split, "manifest": str(args.manifest.resolve()),
        "manifest_sha256": baseline.manifest_sha256,
        "data_config": config.to_dict(), "config_fingerprint": config.fingerprint,
        "seed": args.seed, "images_checked": checked,
        "batch_shape": list(first["image"].shape),
        "checks": ["RGB shape/dtype/finite", "A/B shared clean preprocessing",
                   "B/C identical views and parameters at epoch 0 and first item at epoch 1",
                   "default collate", "fixed suites retain image IDs"],
        "suites": suites,
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "Pillow": PIL.__version__, "numpy": np.__version__},
    }
    (args.output_dir / "smoke.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "images_checked": checked,
                      "output_dir": str(args.output_dir.resolve())}))


if __name__ == "__main__":
    main()
