import pytest

from src.datasets import DataConfig


@pytest.mark.parametrize("values", [
    {"image_size": 224.5}, {"resize_size": True},
    {"mean": (float("nan"), 0, 0)}, {"std": (1, float("inf"), 1)},
    {"jpeg_quality_min": 96}, {"downsample_min": 1},
])
def test_invalid_config_fails_before_image_loading(values):
    with pytest.raises(ValueError):
        DataConfig(**values)


def test_checked_in_config_matches_defaults():
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "configs/ouyang_data_v1.json"
    config = DataConfig.from_json(path)
    assert config.to_dict() == DataConfig().to_dict()
    assert config.fingerprint == DataConfig().fingerprint
