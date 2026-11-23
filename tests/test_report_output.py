"""Report writers must not overwrite the received data they validate."""
import sys

import pytest

from tools.dataset import qa_received, receive_images, smoke_data


@pytest.mark.parametrize("module", [qa_received, receive_images])
@pytest.mark.parametrize("name", ["manifest.csv", "original.png"])
def test_output_in_image_root_is_rejected_before_any_read_or_network(tmp_path, monkeypatch, module, name):
    root = tmp_path / "Images"
    root.mkdir()
    original = root / name
    original.write_bytes(b"preserve original data")
    monkeypatch.setattr(sys, "argv", [module.__name__, "--root", str(root), "--output", str(original)])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    assert original.read_bytes() == b"preserve original data"


def test_smoke_preview_cannot_overwrite_source_image(tmp_path, monkeypatch):
    root = tmp_path / "Images"
    root.mkdir()
    original = root / "preview.png"
    original.write_bytes(b"preserve original data")
    monkeypatch.setattr(sys, "argv", ["smoke", "--root", str(root), "--output-dir", str(root)])
    with pytest.raises(SystemExit) as error:
        smoke_data.main()
    assert error.value.code == 2
    assert original.read_bytes() == b"preserve original data"
