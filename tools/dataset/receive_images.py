"""Receive manifest-listed images from the pinned project revision, verifying SHA256.

This copies existing project images; it does not resample or rebuild a dataset.
Run as python -m tools.dataset.receive_images from the repository root.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import threading
from urllib.parse import quote

import requests

from .qa_received import load_manifest, safe_image_path

REVISION = "e7db18d0657865d5f344632534a3248cb49a6355"
_local = threading.local()


def receive(row, root, revision):
    target = safe_image_path(root, row["rel_path"])
    expected = row["sha256"].lower()
    if target.is_file():
        if hashlib.sha256(target.read_bytes()).hexdigest() == expected:
            return "already_verified"
        raise ValueError(f"Existing file differs from manifest; left untouched: {target}")
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
    url = ("https://raw.githubusercontent.com/Fei-stack-wq/maie5421-CV-final/"
           + revision + "/Images/" + quote(row["rel_path"], safe="/"))
    for attempt in range(3):
        try:
            response = _local.session.get(url, timeout=(15, 60))
            response.raise_for_status()
            data = response.content
            if len(data) != int(row["bytes_file"]):
                raise ValueError(f"File size mismatch: {row['image_id']}")
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"SHA256 mismatch: {row['image_id']}")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".receiving")
            temporary.write_bytes(data)
            temporary.replace(target)
            return "downloaded_verified"
        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("Images"))
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--fixture-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/ouyang/receipt_download.json"))
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision) or not 1 <= args.workers <= 64:
        parser.error("Use a full commit SHA and 1..64 workers")
    output = args.output.resolve()
    image_root = args.root.resolve()
    if output == image_root or image_root in output.parents:
        parser.error("report output must be outside the read-only image root")
    rows = load_manifest(args.root / "fixture/manifest_fixture.csv")
    if not args.fixture_only:
        rows += load_manifest(args.root / "manifest.csv")
    ids = [r["rel_path"] for r in rows]
    if len(set(ids)) != len(ids):
        parser.error("Duplicate destination paths in manifests")
    counts, errors = {}, []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(receive, row, args.root, args.revision): row for row in rows}
        for index, future in enumerate(as_completed(futures), 1):
            try:
                status = future.result()
                counts[status] = counts.get(status, 0) + 1
            except Exception as exc:
                errors.append({"image_id": futures[future]["image_id"], "error": str(exc)})
            if index % 250 == 0 or index == len(rows):
                print(json.dumps({"completed": index, "total": len(rows), "errors": len(errors)}), flush=True)
    report = {"revision": args.revision, "requested": len(rows), "counts": counts,
              "error_count": len(errors), "errors": errors,
              "status": "passed" if not errors else "failed"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
