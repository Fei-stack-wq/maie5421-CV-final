"""A deliberately untrained model for checking the local CLI pipeline.

The CLI retains this development mode for explicitly requested smoke checks.
Formal inference uses TONG's loader and OUYANG's EvalTransform in cli.predict.
"""

from pathlib import Path

import torch
import yaml
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode


SMOKE_CONFIG = {
    "schema_version": 1,
    "purpose": "pipeline_test_only",
    "model_status": "untrained",
    "model": "resnet18_single_logit",
    "condition": "LOCAL_SMOKE",
    "seed": 5421,
    "threshold": 0.5,
    "preprocess": {
        "rgb": True,
        "resize_shorter_side": 256,
        "center_crop": 224,
        "interpolation": "bilinear",
        "antialias": True,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
}
WARNING = "未训练模型，仅验证程序流程；输出不是有效的 AI 图片检测结果。"


def build_smoke_model():
    # weights=None also prevents network access / ImageNet-weight downloads.
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    return model


def create_smoke_checkpoint(output_dir: Path):
    checkpoint_path = output_dir / "best.pt"
    config_path = output_dir / "config.yaml"
    if checkpoint_path.exists() or config_path.exists():
        raise FileExistsError("目标目录已有权重或配置，请使用新的输出目录。")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SMOKE_CONFIG["seed"])
    model = build_smoke_model()
    torch.save(
        {"model_state_dict": model.state_dict(), "config": SMOKE_CONFIG},
        checkpoint_path,
    )
    config_path.write_text(
        yaml.safe_dump(SMOKE_CONFIG, sort_keys=False), encoding="utf-8"
    )
    return checkpoint_path, config_path


def load_smoke_model(checkpoint_path: Path, config_path: Path, allow_untrained: bool):
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config != SMOKE_CONFIG:
        raise ValueError("临时模型配置与固定的流程测试配置不匹配。")
    if not allow_untrained:
        raise ValueError("这是未训练权重。流程测试需显式添加 --allow-untrained。")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("config") != config:
        raise ValueError("checkpoint 与配置不匹配，停止预测。")
    model = build_smoke_model()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    prep = config["preprocess"]
    transform = transforms.Compose([
        transforms.Resize(
            prep["resize_shorter_side"],
            interpolation=InterpolationMode.BILINEAR,
            antialias=prep["antialias"],
        ),
        transforms.CenterCrop(prep["center_crop"]),
        transforms.ToTensor(),
        transforms.Normalize(prep["mean"], prep["std"]),
    ])
    return model, transform, config
