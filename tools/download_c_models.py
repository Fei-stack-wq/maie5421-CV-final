"""Download WANG's two final C models from GitHub Releases and verify SHA256."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/wang"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((Path(__file__).resolve().parents[1] /
                           "docs/wang-formal-results/checkpoint-manifest.json").read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("C_5421", "C_2026"):
        item = manifest[name]
        path = args.output_dir / (name + "_best.pt")
        if path.exists():
            if sha256(path) != item["best_sha256"]:
                raise ValueError(f"Existing file has wrong SHA256; left untouched: {path}")
        else:
            if args.verify_only:
                raise FileNotFoundError(path)
            url = item["download_url"]
            if not url.startswith("https://github.com/Fei-stack-wq/maie5421-CV-final/releases/download/"):
                raise ValueError("Unexpected model download origin")
            part = path.with_suffix(".pt.part")
            with part.open("xb") as stream, urlopen(url, timeout=120) as response:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    stream.write(chunk)
            if sha256(part) != item["best_sha256"]:
                raise ValueError(f"Downloaded file failed SHA256: {part}")
            part.replace(path)
        print(f"{name}: SHA256 verified; {path}")


if __name__ == "__main__":
    main()
