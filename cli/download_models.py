"""Download and verify the six released models and their validation records."""

import argparse
import json
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

from tools.download_c_models import sha256


ROOT = Path(__file__).resolve().parents[1]
MODELS = json.loads((Path(__file__).with_name("formal_models.json")).read_text())


def ensure_file(artifact, verify_only=False):
    path = ROOT / artifact["path"]
    if path.exists():
        if sha256(path) != artifact["sha256"]:
            raise ValueError(f"SHA256 不匹配，请检查已有文件：{path}")
        return path
    if verify_only:
        raise FileNotFoundError(f"缺少文件：{path}；请先运行 python -m cli.download_models")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as stream:
        temporary_path = Path(stream.name)
        try:
            with urlopen(artifact["url"], timeout=60) as response:
                shutil.copyfileobj(response, stream, length=1024 * 1024)
            stream.close()
            if sha256(temporary_path) != artifact["sha256"]:
                raise ValueError(f"下载内容的 SHA256 不匹配：{path.name}")
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="仅核对本地文件，离线可用。")
    args = parser.parse_args(argv)
    artifacts = [artifact for model in MODELS.values() for artifact in model["files"].values()]
    with ThreadPoolExecutor(max_workers=4) as executor:
        for path in executor.map(lambda artifact: ensure_file(artifact, args.verify_only), artifacts):
            print(f"SHA256 verified: {path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
