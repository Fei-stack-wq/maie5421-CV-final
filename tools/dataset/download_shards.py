"""Download the GenImage-subset parquet shards needed for the 12k dataset."""
import os, sys, time, json, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, "tools", "dataset", "cache", "shards")
REPO = "naatee/genimage-subset-9"
BASE = "https://huggingface.co/datasets/%s/resolve/main/data/" % REPO

SHARDS = (
    ["train-%05d-of-00054.parquet" % i for i in (0, 1, 2, 3, 4, 5)]  # ADM + BigGAN blocks (real pool)
    + ["train-%05d-of-00054.parquet" % i for i in range(26, 35)]      # SD1.4 block
    + ["val-%05d-of-00009.parquet" % i for i in (0, 1, 2, 3, 4, 5)]   # real pool + BigGAN + SD1.4
    + ["test-%05d-of-00027.parquet" % i for i in range(0, 5)]         # ADM / BigGAN / Glide
    + ["test-%05d-of-00027.parquet" % i for i in range(13, 18)]       # SD1.4
)


def fetch(name):
    dst = os.path.join(CACHE, name)
    if os.path.exists(dst):
        return "skip", os.path.getsize(dst), 0.0
    tmp = dst + ".part"
    url = BASE + name
    t0 = time.time()
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=180) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
            got += len(b)
    os.replace(tmp, dst)
    return "ok", got, time.time() - t0


def main():
    os.makedirs(CACHE, exist_ok=True)
    print("downloading %d shards -> %s" % (len(SHARDS), CACHE), flush=True)
    overall = time.time()
    for i, name in enumerate(SHARDS, 1):
        try:
            st, size, dt = fetch(name)
            print("[%2d/%2d] %-32s %-4s %7.1f MB %5.1fs" % (i, len(SHARDS), name, st, size / 1e6, dt), flush=True)
        except Exception as e:
            print("[%2d/%2d] %-32s ERROR %r" % (i, len(SHARDS), name, e), flush=True)
    print("DONE in %.1fs (%.1f GB)" % (time.time() - overall, sum(os.path.getsize(os.path.join(CACHE, n)) for n in SHARDS if os.path.exists(os.path.join(CACHE, n))) / 1e9), flush=True)


if __name__ == "__main__":
    main()
