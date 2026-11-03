"""Independent verification + bias audit of the built GenImage subset."""
import os, sys, io, json, hashlib, csv
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, ".pydeps"))
from PIL import Image

OUT = os.path.join(ROOT, "Images")
EXPECT = {
    "train/real": 3000, "train/BigGAN": 1500, "train/stable_diffusion_v_1_4": 1500,
    "val/real": 750, "val/BigGAN": 375, "val/stable_diffusion_v_1_4": 375,
    "test_seen/real": 750, "test_seen/BigGAN": 375, "test_seen/stable_diffusion_v_1_4": 375,
    "test_unseen_ADM/real": 750, "test_unseen_ADM/ADM": 750,
    "test_unseen_GLIDE/real": 750, "test_unseen_GLIDE/GLIDE": 750,
}


def main():
    report = {"ok": True, "errors": [], "warnings": []}
    with io.open(os.path.join(OUT, "manifest.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    report["n_manifest_rows"] = len(rows)
    if len(rows) != 12000:
        report["errors"].append("manifest rows %d != 12000" % len(rows))

    counts = Counter(r["split"] + "/" + r["generator"] for r in rows)
    report["counts"] = dict(sorted(counts.items()))
    for k, v in EXPECT.items():
        if counts.get(k, 0) != v:
            report["errors"].append("count mismatch %s: %d != %d" % (k, counts.get(k, 0), v))

    # hashes / duplication
    md5s = Counter(r["source_md5"] for r in rows)
    dups = {k: v for k, v in md5s.items() if v > 1}
    report["duplicate_source_md5"] = len(dups)
    if dups:
        report["errors"].append("duplicate source md5: %d" % len(dups))
    shas = Counter(r["sha256"] for r in rows)
    report["duplicate_sha256"] = sum(1 for v in shas.values() if v > 1)
    if report["duplicate_sha256"]:
        report["errors"].append("duplicate sha256 across set")

    # cross-split leakage (same image bytes in different splits)
    by_split = defaultdict(set)
    for r in rows:
        by_split[r["source_md5"]].add(r["split"])
    cross = [k for k, v in by_split.items() if len(v) > 1]
    report["cross_split_shared_images"] = len(cross)
    if cross:
        report["errors"].append("images shared across splits: %d" % len(cross))

    # unseen generators must not leak into train/val/seen-test
    seen_split_gens = Counter(r["split"] + "/" + r["generator"] for r in rows)
    for bad in ("train/ADM", "train/GLIDE", "val/ADM", "val/GLIDE", "test_seen/ADM", "test_seen/GLIDE"):
        if seen_split_gens.get(bad):
            report["errors"].append("held-out generator leaked: " + bad)

    # file / image integrity
    bad_files = []
    size_bytes = 0
    fmts = Counter()
    res = Counter()
    for r in rows:
        p = os.path.join(OUT, r["rel_path"].replace("/", os.sep))
        if not os.path.exists(p):
            bad_files.append((r["rel_path"], "missing"))
            continue
        with open(p, "rb") as fh:
            b = fh.read()
        size_bytes += len(b)
        if hashlib.sha256(b).hexdigest() != r["sha256"]:
            bad_files.append((r["rel_path"], "sha256 mismatch"))
            continue
        try:
            with Image.open(io.BytesIO(b)) as im:
                im.load()
                w, h = im.size
                f = im.format
        except Exception as e:
            bad_files.append((r["rel_path"], "decode " + repr(e)))
            continue
        if (w, h) != (int(r["width"]), int(r["height"])):
            bad_files.append((r["rel_path"], "size %s != manifest %s" % ((w, h), (r["width"], r["height"]))))
        if f != r["format"]:
            bad_files.append((r["rel_path"], "format %s != manifest %s" % (f, r["format"])))
        fmts[(r["generator"], f)] += 1
        res[(r["generator"], w, h)] += 1
    report["files_checked"] = len(rows)
    report["bad_files"] = bad_files[:20]
    report["bad_file_count"] = len(bad_files)
    if bad_files:
        report["errors"].append("bad files: %d" % len(bad_files))
    report["total_size_mb"] = round(size_bytes / 1e6, 1)
    report["format_by_generator"] = {"%s|%s" % k: v for k, v in sorted(fmts.items())}
    report["resolution_by_generator"] = {"%s|%dx%d" % k: v for k, v in sorted(res.items())}

    # class balance
    cls = defaultdict(Counter)
    for r in rows:
        cls[r["split"]][r["class_id"]] += 1
    report["classes_per_split"] = {k: len(v) for k, v in cls.items()}
    report["class_counts_min_max"] = {k: [min(v.values()), max(v.values())] for k, v in cls.items()}

    # per-class composition per split (real vs fake)
    comp = defaultdict(lambda: defaultdict(Counter))
    for r in rows:
        comp[r["split"]][r["class_id"]][r["generator"]] += 1
    mism = {}
    for split, per_class in comp.items():
        gens = set()
        for c in per_class.values():
            gens.update(c.keys())
        for g in gens:
            vals = {}
            for c, cnt in per_class.items():
                vals.setdefault(cnt.get(g, 0), 0)
                vals[cnt.get(g, 0)] += 1
            mism[split + "/" + g] = dict(sorted(vals.items()))
    report["per_class_count_histogram"] = mism

    report["ok"] = not report["errors"]
    with io.open(os.path.join(OUT, "verification.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2)[:4000])
    print("\nOK" if report["ok"] else "\nFAILED: %d errors" % len(report["errors"]))


if __name__ == "__main__":
    main()
