"""Build the 12,000-image GenImage subset under Images/.

Steps
  1. scan local parquet shards (metadata columns only) -> row index
  2. deterministic class-stratified sampling (seed 5421), zero leakage
  3. extract exact source bytes from the image column, verify md5 per file
  4. write Images/<split>/<generator>/... + manifest.csv + dataset_summary.json
"""
import os, sys, io, json, hashlib, random, time, csv
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, ".pydeps"))
import pyarrow.parquet as pq

CACHE = os.path.join(ROOT, "tools", "dataset", "cache", "shards")
OUT = os.path.join(ROOT, "Images")
REPO = "naatee/genimage-subset-9"
SEED = 5421

META_COLS = ["image_path", "md5", "label", "generator", "family", "class_id", "class_name",
             "split", "width", "height", "format", "bytes"]

PLAN = {
    "train":             {"classes": 1000, "real": 3000, "BigGAN": 1500, "stable_diffusion_v_1_4": 1500},
    "val":               {"classes": 375,  "real": 750,  "BigGAN": 375,  "stable_diffusion_v_1_4": 375},
    "test_seen":         {"classes": 375,  "real": 750,  "BigGAN": 375,  "stable_diffusion_v_1_4": 375},
    "test_unseen_ADM":   {"classes": 750,  "real": 750,  "ADM": 750},
    "test_unseen_GLIDE": {"classes": 750,  "real": 750,  "GLIDE": 750},
}
SRC_SPLIT = {"train": "train", "val": "val", "test_seen": "test",
             "test_unseen_ADM": "test", "test_unseen_GLIDE": "test"}


def shard_files():
    names = sorted(os.listdir(CACHE))
    return [os.path.join(CACHE, n) for n in names if n.endswith(".parquet")]


def scan():
    rows = []
    for path in shard_files():
        pf = pq.ParquetFile(path)
        md = pf.metadata
        rg_start, acc = [], 0
        for i in range(md.num_row_groups):
            rg_start.append(acc)
            acc += md.row_group(i).num_rows
        tbl = pf.read(columns=META_COLS)
        shard = os.path.basename(path)
        cols = {c: tbl.column(c).to_pylist() for c in META_COLS}
        n = tbl.num_rows
        rg_of = np.searchsorted(np.array(rg_start), np.arange(n), side="right") - 1
        row_in_rg = np.arange(n) - np.array(rg_start)[rg_of]
        kept = 0
        for i in range(n):
            g = cols["generator"][i]
            if g not in ("nature", "BigGAN", "stable_diffusion_v_1_4", "ADM", "GLIDE"):
                continue
            kept += 1
            rows.append({
                "shard": shard, "rg": int(rg_of[i]), "row_in_rg": int(row_in_rg[i]),
                "image_path": cols["image_path"][i], "md5": cols["md5"][i],
                "label": int(cols["label"][i]), "generator": g,
                "family": cols["family"][i], "class_id": int(cols["class_id"][i]),
                "class_name": cols["class_name"][i], "split": cols["split"][i],
                "width": int(cols["width"][i]), "height": int(cols["height"][i]),
                "format": cols["format"][i], "bytes": int(cols["bytes"][i]),
            })
        print("  scanned %-34s rows=%5d kept=%5d" % (shard, n, kept), flush=True)
    return rows


def index_pool(rows, src_split, gen):
    idx = {}
    for r in rows:
        if r["split"] != src_split:
            continue
        if gen == "real":
            if r["generator"] != "nature":
                continue
        elif r["generator"] != gen:
            continue
        idx.setdefault(r["class_id"], []).append(r)
    return idx


def spread(target, classes, avail, rng):
    n = len(classes)
    base = target // n
    rem = target - base * n
    counts = {c: base for c in classes}
    if base > 0:
        for c in classes:
            if avail(c) < base:
                raise RuntimeError("class %d lacks base %d" % (c, base))
    if rem:
        cand = [c for c in classes if avail(c) >= base + 1]
        if len(cand) < rem:
            raise RuntimeError("cannot place %d extra counts" % rem)
        for c in rng.sample(cand, rem):
            counts[c] += 1
    return counts


def plan_split(rows, our_split, rng, used_md5):
    src = SRC_SPLIT[our_split]
    spec = PLAN[our_split]
    n_cls = spec["classes"]
    real_pool = index_pool(rows, src, "real")
    fake_gens = [g for g in spec if g not in ("classes", "real")]
    fake_pools = {g: index_pool(rows, src, g) for g in fake_gens}
    base_cand = [c for c in sorted(real_pool)
                 if all(len(fake_pools[g].get(c, [])) >= 1 for g in fake_gens)]
    chosen = None
    for n in range(n_cls, 0, -25):
        base = spec["real"] // n
        if base < 1:
            continue
        need = [c for c in base_cand if len(real_pool[c]) >= base]
        if len(need) >= n:
            chosen = (n, need)
            break
    if chosen is None:
        raise RuntimeError("cannot find feasible class set for " + our_split)
    n_use, pool_cand = chosen
    if n_use != n_cls:
        print("  [warn] %s: reduced class count %d -> %d" % (our_split, n_cls, n_use), flush=True)
    cls_set = sorted(rng.sample(pool_cand, n_use))

    picks = []
    for g in fake_gens:
        pool = fake_pools[g]
        counts = spread(spec[g], cls_set, lambda c, pool=pool: len(pool.get(c, [])), rng)
        for c in cls_set:
            k = counts[c]
            cand_rows = [r for r in pool[c] if r["md5"] not in used_md5]
            if len(cand_rows) < k:
                raise RuntimeError("insufficient %s class %d" % (g, c))
            for r in rng.sample(cand_rows, k):
                used_md5.add(r["md5"])
                picks.append((our_split, g, r))
    real_counts = spread(spec["real"], cls_set,
                        lambda c: len([r for r in real_pool.get(c, []) if r["md5"] not in used_md5]), rng)
    for c in cls_set:
        k = real_counts[c]
        cand_rows = [r for r in real_pool[c] if r["md5"] not in used_md5]
        if len(cand_rows) < k:
            raise RuntimeError("insufficient real class %d" % c)
        for r in rng.sample(cand_rows, k):
            used_md5.add(r["md5"])
            picks.append((our_split, "real", r))
    return picks, cls_set


def extract(picks):
    by_shard = {}
    for our_split, gen, r in picks:
        by_shard.setdefault(r["shard"], []).append((our_split, gen, r))
    records = []
    total = len(picks)
    done = 0
    for shard, items in sorted(by_shard.items()):
        pf = pq.ParquetFile(os.path.join(CACHE, shard))
        by_rg = {}
        for it in items:
            by_rg.setdefault(it[2]["rg"], []).append(it)
        for rg, group in sorted(by_rg.items()):
            tbl = pf.read_row_group(rg, columns=["image"])
            bytes_arr = tbl.column("image").combine_chunks().field("bytes")
            for our_split, gen, r in group:
                b = bytes_arr[r["row_in_rg"]].as_py()
                if hashlib.md5(b).hexdigest() != r["md5"]:
                    raise RuntimeError("md5 mismatch: " + r["image_path"])
                sub = "real" if gen == "real" else gen
                ext = ".JPEG" if r["format"] == "JPEG" else "." + r["format"]
                fname = "%03d_%s%s" % (r["class_id"], r["md5"][:10], ext)
                reldir = our_split + "/" + sub
                os.makedirs(os.path.join(OUT, reldir), exist_ok=True)
                path = os.path.join(OUT, reldir, fname)
                with open(path, "wb") as fh:
                    fh.write(b)
                records.append({
                    "image_id": "%s-%s" % (our_split, r["md5"][:12]),
                    "split": our_split, "label": 0 if gen == "real" else 1,
                    "generator": gen, "family": r["family"],
                    "class_id": r["class_id"], "class_name": r["class_name"],
                    "filename": fname, "rel_path": reldir + "/" + fname,
                    "source_repo": REPO, "source_split": r["split"],
                    "source_image_path": r["image_path"], "source_md5": r["md5"],
                    "sha256": hashlib.sha256(b).hexdigest(),
                    "width": r["width"], "height": r["height"], "format": r["format"],
                    "bytes_source": r["bytes"], "bytes_file": len(b),
                    "selection_seed": SEED,
                })
                done += 1
            print("  extract %-34s rg%-2d %3d  [%d/%d]" % (shard, rg, len(group), done, total), flush=True)
    return records


def write_outputs(records):
    fields = ["image_id", "split", "label", "generator", "family", "class_id", "class_name",
              "filename", "rel_path", "source_repo", "source_split", "source_image_path",
              "source_md5", "sha256", "width", "height", "format", "bytes_source",
              "bytes_file", "selection_seed"]
    records.sort(key=lambda r: (r["split"], r["generator"], r["class_id"], r["filename"]))
    with io.open(os.path.join(OUT, "manifest.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow(r)
    summary = {"source_repo": REPO, "seed": SEED, "n_images": len(records),
               "counts": {}, "classes": {}, "formats": {}, "resolutions": {}}
    for r in records:
        k = r["split"] + "/" + r["generator"]
        summary["counts"][k] = summary["counts"].get(k, 0) + 1
        summary["classes"].setdefault(r["split"], set()).add(r["class_id"])
        fk = r["generator"] + ":" + r["format"]
        summary["formats"][fk] = summary["formats"].get(fk, 0) + 1
        rk = r["generator"] + ":%dx%d" % (r["width"], r["height"])
        summary["resolutions"][rk] = summary["resolutions"].get(rk, 0) + 1
    summary["classes"] = {k: len(v) for k, v in summary["classes"].items()}
    with io.open(os.path.join(OUT, "dataset_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    return summary


def main():
    t0 = time.time()
    if not os.path.isdir(CACHE):
        raise SystemExit("shard cache missing: " + CACHE)
    print("scanning shards...", flush=True)
    rows = scan()
    print("candidate rows: %d (%.1fs)" % (len(rows), time.time() - t0), flush=True)
    rng = random.Random(SEED)
    used = set()
    picks = []
    for our_split in PLAN:
        p, cls_set = plan_split(rows, our_split, rng, used)
        picks.extend(p)
        print("  plan %-18s picks=%4d classes=%3d" % (our_split, len(p), len(cls_set)), flush=True)
    print("total picks: %d (unique md5: %d)" % (len(picks), len(used)), flush=True)
    if len(picks) != 12000:
        raise SystemExit("unexpected total %d" % len(picks))
    records = extract(picks)
    summary = write_outputs(records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("DONE %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
