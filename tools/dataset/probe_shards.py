"""Probe remote GenImage-subset parquet shards (footer only) and map generator blocks."""
import sys, os, io, json, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, ".pydeps"))
sys.path.insert(0, os.path.join(ROOT, "tools", "dataset"))
import pyarrow as pa, pyarrow.parquet as pq
from hfhttp import HttpRangeFile

REPO = "naatee/genimage-subset-9"
BASE = "https://huggingface.co/datasets/%s/resolve/main/" % REPO


def tree():
    req = urllib.request.Request("https://huggingface.co/api/datasets/%s/tree/main?recursive=true" % REPO,
                                 headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.loads(r.read())
    return [(x["path"], x.get("size", 0)) for x in d if x.get("type") == "file" and x["path"].startswith("data/") and x["path"].endswith(".parquet")]


def probe(path):
    url = BASE + path
    rec = {"path": path, "url": url, "rows": None, "row_groups": []}
    try:
        f = HttpRangeFile(url)
        pf = pq.ParquetFile(pa.PythonFile(f, mode="r"))
        rec["size"] = f.size
        rec["rows"] = pf.metadata.num_rows
        for i in range(pf.metadata.num_row_groups):
            rg = pf.metadata.row_group(i)
            st = {}
            for c in range(rg.num_columns):
                col = rg.column(c)
                if col.path_in_schema in ("image_path", "generator"):
                    s = col.statistics
                    st[col.path_in_schema] = [str(s.min), str(s.max)] if s else None
            rec["row_groups"].append({"i": i, "rows": rg.num_rows, "bytes": rg.total_byte_size, "stats": st})
        rec["ok"] = True
    except Exception as e:
        rec["ok"] = False
        rec["error"] = repr(e)
    return rec


def main():
    files = sorted(tree())
    print("shards:", len(files), flush=True)
    t0 = time.time()
    out = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for i, rec in enumerate(ex.map(probe, [p for p, _ in files])):
            out.append(rec)
            print("[%d/%d] %s rows=%s ok=%s (%.0fs)" % (i + 1, len(files), rec["path"], rec["rows"], rec.get("ok"), time.time() - t0), flush=True)
    os.makedirs(os.path.join(ROOT, "tools", "dataset", "cache"), exist_ok=True)
    with io.open(os.path.join(ROOT, "tools", "dataset", "cache", "shard_map.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("DONE %.1fs -> shard_map.json" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
