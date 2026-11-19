"""Single-image CPU inference with the team's checkpoint and preprocessing."""

import argparse
import hashlib
import json
import math
import pickle
import sys
import tempfile
from pathlib import Path

import torch
import yaml
from PIL import Image

from cli.smoke_model import WARNING, load_smoke_model
from src.datasets import DataConfig, EvalTransform
from src.training.checkpoint import load_model


def file_hash(path: Path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_prediction_model(checkpoint_path, config_path, allow_untrained):
    config = None
    if config_path is not None:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("配置文件必须是 YAML 键值映射。")
    if config is not None and config.get("purpose") == "pipeline_test_only":
        model, transform, config = load_smoke_model(
            checkpoint_path, config_path, allow_untrained
        )
        return model, transform, {
            "purpose": "pipeline_test_only", "model_status": "untrained",
            "warning": WARNING, "config": config,
            "threshold": config["threshold"], "preprocess": config["preprocess"],
        }

    model, metadata = load_model(checkpoint_path, device="cpu")
    if config is not None and config != metadata["config"]:
        raise ValueError("配置与 checkpoint 内保存的完整训练配置不匹配，请使用对应运行的 config.yaml。")
    preprocessing = metadata["preprocessing"]
    if preprocessing.get("provider") != "src.datasets" or "data_config" not in preprocessing:
        raise ValueError("checkpoint 缺少 OUYANG 的统一预处理参数。")
    data_config = DataConfig(**preprocessing["data_config"])
    if preprocessing.get("config_fingerprint") != data_config.fingerprint:
        raise ValueError("checkpoint 预处理参数与其指纹不匹配。")
    if metadata["threshold"] != 0.5:
        raise ValueError("checkpoint 阈值与小组约定的 0.5 不一致。")
    return model, EvalTransform(data_config, suite="clean"), {
        "purpose": "single_image_inference",
        "model_status": metadata["provenance"]["run_kind"],
        "config": metadata["config"], "threshold": metadata["threshold"],
        "preprocess": preprocessing, "training_provenance": metadata["provenance"],
    }


def predict(image_path: Path, checkpoint_path: Path, config_path=None, allow_untrained=False):
    inputs = [("图片", image_path), ("权重", checkpoint_path)]
    if config_path is not None:
        inputs.append(("配置", config_path))
    for description, path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"{description}文件不存在：{path}")
    # Validate the image before doing the more expensive model restoration.
    with Image.open(image_path) as image:
        if image.format not in {"JPEG", "PNG"}:
            raise ValueError("目前支持 JPEG/PNG 图片。")
        image = image.convert("RGB")
    model, transform, metadata = load_prediction_model(
        checkpoint_path, config_path, allow_untrained
    )
    with torch.inference_mode():
        logits = model(transform(image).unsqueeze(0))
        p_ai = torch.sigmoid(logits).item()
    if not math.isfinite(p_ai):
        raise ValueError("模型输出不是有限数值，停止预测。")
    predicted_id = int(p_ai >= metadata["threshold"])
    config = metadata.pop("config")
    return {
        **metadata,
        "image_path": str(image_path.resolve()),
        "image_sha256": file_hash(image_path),
        "predicted_label": "AI-Generated" if predicted_id else "Real",
        "predicted_label_id": predicted_id,
        "p_ai": p_ai,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": file_hash(checkpoint_path),
        "condition": config["condition"],
        "seed": config["seed"],
        "transformation": "clean",
        "device": "cpu",
    }


def write_result(output_path, result):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Replace only after a complete JSON write, so retries leave valid output.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output_path.parent,
                                     suffix=".tmp", delete=False) as stream:
        temporary_path = Path(stream.name)
        try:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.close()
            temporary_path.replace(output_path)
        finally:
            temporary_path.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="加载团队 checkpoint，在 CPU 上预测单张 JPEG/PNG 图片。")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", type=Path, help="可选训练 YAML；提供时检查其与 checkpoint 是否一致。")
    parser.add_argument("--output", type=Path, help="可选 JSON 路径；已存在的结果文件会覆盖。")
    parser.add_argument("--allow-untrained", action="store_true", help="明确允许未训练权重做流程测试。")
    args = parser.parse_args(argv)
    try:
        torch.set_num_threads(2)
        if args.output and args.output.resolve() in {
            path.resolve() for path in (args.image, args.checkpoint, args.config)
            if path is not None
        }:
            raise ValueError("输出路径不能覆盖输入图片、模型权重或配置。")
        result = predict(args.image, args.checkpoint, args.config, args.allow_untrained)
        if args.output:
            write_result(args.output, result)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError,
            EOFError, pickle.UnpicklingError, yaml.YAMLError, Image.DecompressionBombError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    if result["purpose"] == "pipeline_test_only":
        print(f"[仅流程测试] {WARNING}")
    print(f"模型：{result['condition']} | seed={result['seed']} | CPU | {result['model_status']}")
    print(f"图片：{args.image}")
    print(f"预测：{result['predicted_label']} | P(AI)={result['p_ai']:.6f}")
    if args.output:
        print(f"JSON：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
