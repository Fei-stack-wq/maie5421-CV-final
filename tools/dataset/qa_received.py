"""Read-only receipt QA and cross-split perceptual-duplicate screening.

Run from the repository root, for example::

    python tools/dataset/qa_received.py --metadata-only --output docs/qa_metadata.json
    python tools/dataset/qa_received.py --check-images --phash --output outputs/qa_full.json

An unchanged, complete manifest is not proof that local image files were received.
The report therefore keeps metadata, file integrity, and near-duplicate coverage
separate. This tool never deletes, rewrites, or reallocates source images.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import platform
from importlib.metadata import PackageNotFoundError, version
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_COUNTS = {
    "train/real": 3000,
    "train/BigGAN": 1500,
    "train/stable_diffusion_v_1_4": 1500,
    "val/real": 750,
    "val/BigGAN": 375,
    "val/stable_diffusion_v_1_4": 375,
    "test_seen/real": 750,
    "test_seen/BigGAN": 375,
    "test_seen/stable_diffusion_v_1_4": 375,
    "test_unseen_ADM/real": 750,
    "test_unseen_ADM/ADM": 750,
    "test_unseen_GLIDE/real": 750,
    "test_unseen_GLIDE/GLIDE": 750,
}
EXPECTED_FIXTURE = {
    "fixture/real": 50, "fixture/BigGAN": 25,
    "fixture/stable_diffusion_v_1_4": 25,
}
ALLOWED = {
    "train": {"real", "BigGAN", "stable_diffusion_v_1_4"},
    "val": {"real", "BigGAN", "stable_diffusion_v_1_4"},
    "test_seen": {"real", "BigGAN", "stable_diffusion_v_1_4"},
    "test_unseen_ADM": {"real", "ADM"},
    "test_unseen_GLIDE": {"real", "GLIDE"},
    "fixture": {"real", "BigGAN", "stable_diffusion_v_1_4"},
}
REQUIRED = {
    "image_id", "split", "label", "generator", "rel_path", "source_md5",
    "sha256", "width", "height", "format", "bytes_file", "class_id",
}


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        return list(reader)


def safe_image_path(root: Path, rel_path: str) -> Path:
    """Reject path traversal, Windows drive/UNC paths, and escaping symlinks."""
    if not rel_path or "\\" in rel_path:
        raise ValueError("rel_path must be a nonempty POSIX relative path")
    pure = PurePosixPath(rel_path)
    if pure.is_absolute() or PureWindowsPath(rel_path).drive or ".." in pure.parts:
        raise ValueError("absolute or parent-traversing rel_path")
    path = (root / pure).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError("rel_path escapes image root or points to root")
    return path


def _record(row: dict[str, str]) -> dict[str, str]:
    return {key: row[key] for key in ("image_id", "split", "label", "generator", "rel_path")}


def _duplicates(rows: list[dict[str, str]], field: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get(field):
            groups[row[field].lower() if field in {"source_md5", "sha256"} else row[field]].append(row)
    return [
        {"value": value, "rows": [_record(r) for r in group],
         "cross_split": len({r["split"] for r in group}) > 1}
        for value, group in groups.items() if len(group) > 1
    ]


def metadata_qa(
    rows: list[dict[str, str]], root: Path, *, project_counts: bool = True,
    fixture_rows: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Validate the actual manifests without opening any image files."""
    errors: list[str] = []
    fixture_rows = fixture_rows or []
    all_rows = rows + fixture_rows
    counts = Counter(f"{r['split']}/{r['generator']}" for r in rows)
    fixture_counts = Counter(f"{r['split']}/{r['generator']}" for r in fixture_rows)
    if not rows:
        errors.append("main manifest is empty")
    if project_counts:
        if dict(counts) != EXPECTED_COUNTS:
            errors.append("main split/generator counts do not match the frozen 12000-image protocol")
        if dict(fixture_counts) != EXPECTED_FIXTURE:
            errors.append("fixture counts do not match 50 real + 25 BigGAN + 25 SD1.4")
    for index, row in enumerate(all_rows):
        location = f"row {index + 1} ({row.get('image_id')})"
        try:
            if any(not row.get(key) for key in REQUIRED):
                raise ValueError("required field is empty")
            if row["label"] not in {"0", "1"}:
                raise ValueError("label must be 0 or 1")
            if int(row["label"]) != (0 if row["generator"] == "real" else 1):
                raise ValueError("label and generator disagree")
            if row["generator"] not in ALLOWED.get(row["split"], set()):
                raise ValueError("unknown split/generator or held-out generator leakage")
            if (index >= len(rows)) != (row["split"] == "fixture"):
                raise ValueError("fixture rows must be isolated from the main manifest")
            if any(int(row[key]) < 1 for key in ("width", "height", "bytes_file")):
                raise ValueError("dimensions/bytes_file must be positive")
            if not 0 <= int(row["class_id"]) < 1000:
                raise ValueError("class_id outside [0, 999]")
            if not re.fullmatch(r"[0-9a-fA-F]{32}", row["source_md5"]):
                raise ValueError("invalid source_md5")
            if not re.fullmatch(r"[0-9a-fA-F]{64}", row["sha256"]):
                raise ValueError("invalid sha256")
            safe_image_path(root, row["rel_path"])
            if PurePosixPath(row["rel_path"]).parts[0] != row["split"]:
                raise ValueError("rel_path does not start with its declared split")
        except (ValueError, TypeError) as exc:
            errors.append(f"{location}: {exc}")
    duplicates = {
        key: _duplicates(all_rows, key)
        for key in ("image_id", "rel_path", "source_md5", "sha256")
    }
    for key, groups in duplicates.items():
        if groups:
            errors.append(f"duplicate {key}: {len(groups)} groups (including fixture)")

    class_counts: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    formats: Counter[str] = Counter()
    resolutions: Counter[str] = Counter()
    for row in all_rows:
        try:
            if row["label"] in {"0", "1"} and 0 <= int(row["class_id"]) < 1000:
                class_counts[row["split"]][row["label"]][row["class_id"]] += 1
        except (ValueError, TypeError):
            pass
        formats[f"{row['generator']}/{row['format']}"] += 1
        resolutions[f"{row['generator']}/{row['width']}x{row['height']}"] += 1
    class_priors = {}
    for split, by_label in class_counts.items():
        real, fake = by_label.get("0", Counter()), by_label.get("1", Counter())
        classes = set(real) | set(fake)
        total_real, total_fake = sum(real.values()), sum(fake.values())
        tv = 0.5 * sum(
            abs(real[c] / max(total_real, 1) - fake[c] / max(total_fake, 1)) for c in classes
        )
        class_priors[split] = {
            "real_classes": len(real), "ai_classes": len(fake),
            "same_class_support": set(real) == set(fake),
            "label_class_distribution_total_variation": round(tv, 8),
            "max_absolute_real_ai_count_difference": max(
                (abs(real[c] - fake[c]) for c in classes), default=0
            ),
            "counts_by_label_and_class": {
                label: dict(sorted(values.items(), key=lambda item: int(item[0])))
                for label, values in by_label.items()
            },
        }
    if project_counts:
        expected_classes = {
            "train": 1000, "val": 375, "test_seen": 375,
            "test_unseen_ADM": 750, "test_unseen_GLIDE": 750,
        }
        for split, expected in expected_classes.items():
            prior = class_priors.get(split)
            if prior is None or prior["real_classes"] != expected or prior["ai_classes"] != expected:
                errors.append(f"{split}: real/AI class coverage differs from {expected}")
            elif not prior["same_class_support"] or prior["max_absolute_real_ai_count_difference"] > 1:
                errors.append(f"{split}: class-support or per-class label balance violates protocol")
    return {
        "ok": not errors, "errors": errors,
        "main_rows": len(rows), "fixture_rows": len(fixture_rows),
        "counts": dict(sorted(counts.items())),
        "fixture_counts": dict(sorted(fixture_counts.items())),
        "duplicate_groups": duplicates,
        "class_prior_audit": class_priors,
        "format_by_generator": dict(sorted(formats.items())),
        "resolution_by_generator": dict(sorted(resolutions.items())),
        "limitations": [
            "Metadata checks do not verify that image bytes exist or match their hashes.",
            "Class-prior audit compares real/AI in this subset; it does not estimate natural ImageNet priors.",
        ],
    }


def phash64(image: Any) -> int:
    """63 AC bits in a 64-bit integer: 32px grayscale DCT-II, top-left 8x8.

    Median excludes DC; DC bit is forced to zero. Pixels use Pillow LANCZOS,
    DCT uses scipy.fft.dctn(type=2, norm='ortho'). This is a screening hash,
    not a guarantee of semantic identity.
    """
    import numpy as np
    from PIL import Image
    from scipy.fft import dctn

    pixels = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    low = dctn(pixels, type=2, norm="ortho")[:8, :8].ravel()
    bits = low > np.median(low[1:])
    bits[0] = False
    result = 0
    for bit in bits:
        result = (result << 1) | int(bit)
    return result


def _hamming(left: int, right: int) -> int:
    value = left ^ right
    return value.bit_count() if hasattr(int, "bit_count") else bin(value).count("1")


class _BKTree:
    """Index unique hashes under Hamming distance; duplicate hashes are grouped."""

    def __init__(self) -> None:
        self.root: tuple[int, dict] | None = None

    def add(self, value: int) -> None:
        if self.root is None:
            self.root = (value, {})
            return
        node = self.root
        while True:
            distance = _hamming(value, node[0])
            if distance == 0:
                return
            if distance not in node[1]:
                node[1][distance] = (value, {})
                return
            node = node[1][distance]

    def query(self, value: int, radius: int):
        pending = [self.root] if self.root else []
        while pending:
            node = pending.pop()
            distance = _hamming(value, node[0])
            if distance <= radius:
                yield node[0], distance
            pending.extend(child for edge, child in node[1].items() if distance - radius <= edge <= distance + radius)


def screen_near_duplicates(
    hashed_rows: list[tuple[dict[str, str], int]], *, threshold: int = 4,
    max_candidates: int = 10000,
) -> dict[str, Any]:
    """Return all cross-split candidate counts and bounded review examples."""
    if not 0 <= threshold <= 63:
        raise ValueError("pHash Hamming threshold must be in [0, 63]")
    if max_candidates < 0:
        raise ValueError("max_candidates must be nonnegative")
    groups: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row, value in hashed_rows:
        groups[value].append(row)
    tree = _BKTree()
    candidates: list[dict[str, Any]] = []
    pair_counts: Counter[str] = Counter()
    count = 0

    def accumulate(left, right, distance, same_group=False):
        nonlocal count
        left_splits: dict[str, list] = defaultdict(list)
        right_splits: dict[str, list] = defaultdict(list)
        for row in left:
            left_splits[row["split"]].append(row)
        for row in right:
            right_splits[row["split"]].append(row)
        for split_a, group_a in sorted(left_splits.items()):
            for split_b, group_b in sorted(right_splits.items()):
                if split_a == split_b or (same_group and split_a > split_b):
                    continue
                number = len(group_a) * len(group_b)
                count += number
                pair_counts["|".join(sorted((split_a, split_b)))] += number
                remaining = max_candidates - len(candidates)
                if remaining > 0:
                    pairs = ((a, b) for a in group_a for b in group_b)
                    for a, b in islice(pairs, remaining):
                        candidates.append({
                            "left": _record(a), "right": _record(b),
                            "hamming_distance": distance,
                            "review_status": "unreviewed",
                        })

    for value in sorted(groups):
        group = groups[value]
        accumulate(group, group, 0, same_group=True)
        for neighbor, distance in tree.query(value, threshold):
            accumulate(group, groups[neighbor], distance)
        tree.add(value)
    return {
        "algorithm": "phash_ac63_dct32_lanczos_v1", "threshold_hamming": threshold,
        "scope": "all distinct split pairs including fixture versus formal sets",
        "hashed_images": len(hashed_rows), "unique_hashes": len(groups),
        "candidate_pair_count": count,
        "candidate_counts_by_split_pair": dict(sorted(pair_counts.items())),
        "candidates": candidates, "candidate_examples_truncated": count > len(candidates),
        "review_required": count > 0,
        "limitations": [
            "Candidates require human review; no image is automatically removed or resplit.",
            "A distance above threshold does not rule out a crop, rotation, or semantic duplicate.",
            "Within-split perceptual duplicates are outside this cross-split screening scope.",
        ],
    }


def image_qa(rows: list[dict[str, str]], root: Path, *, phash: bool = False):
    from PIL import Image

    errors, missing, hashed_rows = [], [], []
    present = decoded = verified = 0
    for row in rows:
        try:
            path = safe_image_path(root, row["rel_path"])
            if not path.is_file():
                missing.append(_record(row))
                continue
            data = path.read_bytes()
            present += 1
            problems = []
            if len(data) != int(row["bytes_file"]):
                problems.append("bytes_file mismatch")
            if hashlib.sha256(data).hexdigest() != row["sha256"].lower():
                problems.append("sha256 mismatch")
            if hashlib.md5(data).hexdigest() != row["source_md5"].lower():
                problems.append("source_md5 mismatch")
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                decoded += 1
                if image.size != (int(row["width"]), int(row["height"])):
                    problems.append("width/height mismatch")
                if image.format != row["format"]:
                    problems.append("format mismatch")
                if phash:
                    hashed_rows.append((row, phash64(image.convert("RGB"))))
            if problems:
                errors.append({**_record(row), "errors": problems})
            else:
                verified += 1
        except (OSError, ValueError, TypeError, SyntaxError) as exc:
            errors.append({**_record(row), "errors": [f"{type(exc).__name__}: {exc}"]})
    return {
        "requested": len(rows), "files_present": present, "images_decoded": decoded,
        "images_verified": verified, "missing_image_count": len(missing),
        "missing_images": missing, "error_count": len(errors), "errors": errors,
        "complete": verified == len(rows) and bool(rows),
    }, hashed_rows


def run_qa(
    root: Path, manifest: Path, fixture_manifest: Path | None, *,
    check_images: bool = False, phash: bool = False, project_counts: bool = True,
    phash_threshold: int = 4, max_candidates: int = 10000,
) -> dict[str, Any]:
    """Return a JSON-serializable receipt report; paths and manifests stay unchanged."""
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "image_root": str(root.resolve()), "manifest": str(manifest.resolve()),
        "manifest_sha256": None,
        "fixture_manifest": str(fixture_manifest.resolve()) if fixture_manifest else None,
        "fixture_manifest_sha256": None,
        "profile": "project_12000_plus_fixture100" if project_counts else "custom_counts",
        "mode": "images_and_phash" if phash else "images" if check_images else "metadata_only",
        "status": "failed", "full_qa_pass": False, "errors": [],
    }
    report["dependency_versions"] = {"python": platform.python_version()}
    for package in ("Pillow", "numpy", "scipy"):
        try:
            report["dependency_versions"][package] = version(package)
        except PackageNotFoundError:
            report["dependency_versions"][package] = None
    try:
        report["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        rows = load_manifest(manifest)
        fixture_rows = []
        if fixture_manifest is not None:
            report["fixture_manifest_sha256"] = hashlib.sha256(fixture_manifest.read_bytes()).hexdigest()
            fixture_rows = load_manifest(fixture_manifest)
        metadata = metadata_qa(rows, root, project_counts=project_counts, fixture_rows=fixture_rows)
        report["metadata"] = metadata
    except (OSError, ValueError) as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        report["exit_code"] = 1
        return report
    if not metadata["ok"]:
        report["errors"].append("metadata validation failed; image processing skipped")
        report["exit_code"] = 1
        return report
    if not check_images and not phash:
        report.update({"status": "metadata_pass_images_unchecked", "exit_code": 0})
        report["pending"] = ["receipt/decode/hash check", "cross-split perceptual duplicate screening"]
        return report
    try:
        integrity, hashed = image_qa(rows + fixture_rows, root, phash=phash)
        report["images"] = integrity
        if phash:
            report["near_duplicates"] = screen_near_duplicates(
                hashed, threshold=phash_threshold, max_candidates=max_candidates,
            )
            report["near_duplicates"]["coverage_complete"] = len(hashed) == len(rows) + len(fixture_rows)
    except ImportError as exc:
        report["errors"].append(f"missing image/pHash dependency: {exc}")
        report.update({"status": "incomplete", "exit_code": 2})
        return report
    if integrity["error_count"]:
        report.update({"status": "failed", "exit_code": 1})
    elif not integrity["complete"]:
        report.update({"status": "incomplete_missing_images", "exit_code": 2})
    elif not phash:
        report.update({"status": "image_integrity_pass_phash_pending", "exit_code": 2})
    elif report["near_duplicates"]["review_required"]:
        report.update({"status": "review_required", "exit_code": 3})
    else:
        report.update({"status": "passed", "exit_code": 0, "full_qa_pass": True})
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "Images")
    parser.add_argument("--manifest", type=Path, help="default: ROOT/manifest.csv")
    parser.add_argument("--fixture-manifest", type=Path, help="default: ROOT/fixture/manifest_fixture.csv")
    parser.add_argument("--skip-fixture", action="store_true", help="custom datasets only; omits fixture checks")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--metadata-only", action="store_true", help="default; never claims complete receipt")
    mode.add_argument("--check-images", action="store_true", help="verify received bytes, decode, size, and format")
    parser.add_argument("--phash", action="store_true", help="also check images and screen cross-split pHash candidates")
    parser.add_argument("--phash-threshold", type=int, default=4)
    parser.add_argument("--max-candidates", type=int, default=10000, help="maximum candidate examples in JSON; counts stay complete")
    parser.add_argument("--custom-counts", action="store_true", help="disable only the project 12000+100 expected-count checks")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "qa_received.json")
    args = parser.parse_args(argv)
    if args.metadata_only and args.phash:
        parser.error("--metadata-only cannot be combined with --phash")
    if args.skip_fixture and not args.custom_counts:
        parser.error("--skip-fixture requires --custom-counts")
    if not 0 <= args.phash_threshold <= 63 or args.max_candidates < 0:
        parser.error("pHash threshold must be 0..63 and max-candidates must be nonnegative")
    output = args.output.resolve()
    image_root = args.root.resolve()
    if output == image_root or image_root in output.parents:
        parser.error("report output must be outside the read-only image root")
    manifest = args.manifest or args.root / "manifest.csv"
    fixture = None if args.skip_fixture else args.fixture_manifest or args.root / "fixture" / "manifest_fixture.csv"
    report = run_qa(
        args.root, manifest, fixture, check_images=args.check_images, phash=args.phash,
        project_counts=not args.custom_counts,
        phash_threshold=args.phash_threshold, max_candidates=args.max_candidates,
    )
    protected = {manifest.resolve(), args.root.resolve()}
    if fixture is not None:
        protected.add(fixture.resolve())
    if args.output.resolve() in protected:
        parser.error("output must not overwrite an input manifest or image root")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "full_qa_pass": report["full_qa_pass"],
                      "report": str(args.output.resolve())}, ensure_ascii=False))
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
