"""Match a fresh full data audit to the versioned candidate review."""

import argparse
import hashlib
import json
from pathlib import Path


def candidate_keys(items):
    return {(*sorted((r["left"]["image_id"], r["right"]["image_id"])), r["hamming_distance"])
            for r in items}


def check_reports(qa, reference, review):
    if not qa["metadata"]["ok"] or qa.get("errors"):
        raise ValueError("data metadata audit failed")
    images, near = qa["images"], qa["near_duplicates"]
    if not images["complete"] or images["images_verified"] != 12100 or images["error_count"]:
        raise ValueError("full 12100-image audit required")
    if not near["coverage_complete"] or near["hashed_images"] != 12100 or near["candidate_examples_truncated"]:
        raise ValueError("full perceptual screening required")
    if (near["algorithm"], near["threshold_hamming"]) != (review["screening_algorithm"], review["threshold_hamming"]):
        raise ValueError("screening method differs from reviewed version")
    if review["unresolved_pairs"] or review["confirmed_duplicate_pairs"]:
        raise ValueError("candidate decisions remain unresolved")
    expected = candidate_keys(review["reviews"])
    if (candidate_keys(near["candidates"]) != expected
            or candidate_keys(reference["near_duplicates"]["candidates"]) != expected
            or near["candidate_pair_count"] != len(expected)):
        raise ValueError("new or changed candidates require review")
    if any(r["decision"] != "different_images_false_positive" for r in review["reviews"]):
        raise ValueError("unaccepted candidate decision")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=Path("Images"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.output.resolve() == root or root in args.output.resolve().parents or args.output.resolve() == args.qa.resolve():
        parser.error("readiness output must be outside Images/ and must not overwrite the raw QA")
    qa = json.loads(args.qa.read_text())
    ref_path = Path("artifacts/ouyang/qa_received.json")
    reference = json.loads(ref_path.read_text())
    review = json.loads(Path("artifacts/ouyang/qa_visual_review.json").read_text())
    if hashlib.sha256(ref_path.read_bytes()).hexdigest() != review["source_qa_sha256"]:
        raise ValueError("versioned source QA hash mismatch")
    manifest = (args.root / "manifest.csv").read_bytes()
    if hashlib.sha256(manifest.replace(b"\r\n", b"\n")).hexdigest() != review["manifest_sha256"]:
        raise ValueError("manifest differs from reviewed version")
    if hashlib.sha256(manifest).hexdigest() != qa["manifest_sha256"]:
        raise ValueError("fresh QA does not describe current manifest bytes")
    check_reports(qa, reference, review)
    from src.datasets.manifest import resolve_image_path
    for pair in review["reviews"]:
        for side in ("left", "right"):
            path = resolve_image_path(args.root, pair[side]["rel_path"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != pair[side + "_sha256"]:
                raise ValueError("reviewed image bytes changed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "ready_for_training": True, "fresh_qa_sha256": hashlib.sha256(args.qa.read_bytes()).hexdigest(),
        "manifest_sha256": qa["manifest_sha256"], "verified_images": 12100,
        "reviewed_candidate_pairs": len(review["reviews"]), "reviewer_type": review["reviewer_type"],
        "raw_qa_full_pass": qa["full_qa_pass"],
        "note": "Fresh byte audit and screening match the existing versioned review; not new human sign-off.",
    }, indent=2) + "\n")
    print("Full data audit matches the versioned candidate review")


if __name__ == "__main__":
    main()
