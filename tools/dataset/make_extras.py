"""Create the 100-image development fixture and a visual contact sheet."""
import os, sys, io, json, hashlib, csv, random
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, ".pydeps"))
import pyarrow.parquet as pq
import pyarrow.compute as pc
from PIL import Image, ImageDraw

CACHE = os.path.join(ROOT, "tools", "dataset", "cache", "shards")
OUT = os.path.join(ROOT, "Images")
SEED = 5421


def load_used():
    with io.open(os.path.join(OUT, "manifest.csv"), encoding="utf-8") as fh:
        return {r["source_md5"] for r in csv.DictReader(fh)}


def scan_train():
    rows = []
    for name in sorted(os.listdir(CACHE)):
        if not name.startswith("train-"):
            continue
        pf = pq.ParquetFile(os.path.join(CACHE, name))
        tbl = pf.read(columns=["image_path", "md5", "generator", "split", "class_id", "class_name", "format"])
        for r in tbl.to_pylist():
            if r["split"] == "train" and r["generator"] in ("nature", "BigGAN", "stable_diffusion_v_1_4"):
                r["shard"] = name
                rows.append(r)
    return rows


def read_bytes(name, md5):
    pf = pq.ParquetFile(os.path.join(CACHE, name))
    md = pf.metadata
    for rg in range(md.num_row_groups):
        t = pf.read_row_group(rg, columns=["image", "md5"])
        hits = t.filter(pc.equal(t.column("md5"), md5))
        if hits.num_rows:
            return hits.column("image").combine_chunks().field("bytes")[0].as_py()
    raise KeyError(md5)


def build_fixture(rows, used):
    rng = random.Random(SEED + 7)
    by_gen = {"nature": [], "BigGAN": [], "stable_diffusion_v_1_4": []}
    for r in rows:
        if r["md5"] in used:
            continue
        by_gen[r["generator"]].append(r)
    for k in by_gen:
        rng.shuffle(by_gen[k])
    picks = []

    def take(gen, n, label):
        seen_cls, seen_md5 = set(), set()
        for r in by_gen[gen]:
            if len(seen_cls) >= n:
                break
            if r["class_id"] in seen_cls or r["md5"] in seen_md5:
                continue
            seen_cls.add(r["class_id"])
            seen_md5.add(r["md5"])
            picks.append((label, r))
        return len(seen_cls)

    n_real = take("nature", 50, "real")
    n_big = take("BigGAN", 25, "BigGAN")
    n_sd = take("stable_diffusion_v_1_4", 25, "stable_diffusion_v_1_4")
    picked_cls = {p[1]["class_id"] for p in picks}
    print("  fixture composition real=%d BigGAN=%d SD1.4=%d" % (n_real, n_big, n_sd))
    return picks, len(picked_cls)


def write_fixture(picks):
    fields = ["image_id", "split", "label", "generator", "family", "class_id", "class_name",
              "filename", "rel_path", "source_repo", "source_split", "source_image_path",
              "source_md5", "sha256", "width", "height", "format", "bytes_file", "selection_seed"]
    recs = []
    for gen, r in picks:
        b = read_bytes(r["shard"], r["md5"])
        if hashlib.md5(b).hexdigest() != r["md5"]:
            raise RuntimeError("md5 mismatch " + r["md5"])
        ext = ".JPEG" if r["format"] == "JPEG" else "." + r["format"]
        fname = "%03d_%s%s" % (r["class_id"], r["md5"][:10], ext)
        sub = "fixture/" + ("real" if gen == "real" else gen)
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
        with open(os.path.join(OUT, sub, fname), "wb") as fh:
            fh.write(b)
        with Image.open(io.BytesIO(b)) as im:
            w, h = im.size
        recs.append({"image_id": "fixture-" + r["md5"][:12], "split": "fixture",
                     "label": 0 if gen == "real" else 1, "generator": gen,
                     "family": r["generator"] and ("real" if gen == "real" else "GAN" if gen == "BigGAN" else "diffusion"),
                     "class_id": r["class_id"], "class_name": r["class_name"], "filename": fname,
                     "rel_path": sub + "/" + fname, "source_repo": "naatee/genimage-subset-9",
                     "source_split": "train", "source_image_path": r["image_path"],
                     "source_md5": r["md5"], "sha256": hashlib.sha256(b).hexdigest(),
                     "width": w, "height": h, "format": r["format"], "bytes_file": len(b),
                     "selection_seed": SEED + 7})
    with io.open(os.path.join(OUT, "fixture", "manifest_fixture.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in recs:
            w.writerow(r)
    return recs


def contact_sheet():
    with io.open(os.path.join(OUT, "manifest.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rng = random.Random(11)
    groups = [("real", [r for r in rows if r["split"] == "train" and r["generator"] == "real"]),
              ("BigGAN", [r for r in rows if r["split"] == "train" and r["generator"] == "BigGAN"]),
              ("SD1.4", [r for r in rows if r["split"] == "train" and r["generator"] == "stable_diffusion_v_1_4"]),
              ("ADM (unseen)", [r for r in rows if r["split"] == "test_unseen_ADM" and r["generator"] == "ADM"]),
              ("GLIDE (unseen)", [r for r in rows if r["split"] == "test_unseen_GLIDE" and r["generator"] == "GLIDE"])]
    tile = 220
    cols = 5
    rows_n = len(groups)
    W, H = 40 + cols * (tile + 10), 40 + rows_n * (tile + 34)
    sheet = Image.new("RGB", (W, H), (250, 250, 250))
    d = ImageDraw.Draw(sheet)
    for ri, (name, grp) in enumerate(groups):
        d.text((12, 20 + ri * (tile + 34)), name, fill=(20, 20, 20))
        for ci in range(cols):
            r = rng.choice(grp)
            p = os.path.join(OUT, r["rel_path"].replace("/", os.sep))
            with Image.open(p) as im:
                im = im.convert("RGB")
                im.thumbnail((tile, tile))
                x = 40 + ci * (tile + 10) + (tile - im.width) // 2
                y = 30 + ri * (tile + 34) + (tile - im.height) // 2
                sheet.paste(im, (x, y))
            d.rectangle([40 + ci * (tile + 10), 30 + ri * (tile + 34),
                         40 + ci * (tile + 10) + tile, 30 + ri * (tile + 34) + tile], outline=(200, 200, 200))
    sheet.save(os.path.join(OUT, "preview_contact_sheet.png"))
    print("contact sheet saved", sheet.size)


def main():
    used = load_used()
    rows = scan_train()
    print("train-split candidates:", len(rows))
    picks, ncls = build_fixture(rows, used)
    recs = write_fixture(picks)
    print("fixture images:", len(recs), "classes:", ncls)
    contact_sheet()


if __name__ == "__main__":
    main()
