"""Compute the bias-audit tables used in the dataset card."""
import os, sys, io, json, csv
from collections import Counter, defaultdict
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "Images")

with io.open(os.path.join(OUT, "manifest.csv"), encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh))

res = []
for g in ("real", "BigGAN", "stable_diffusion_v_1_4", "ADM", "GLIDE"):
    sub = [r for r in rows if r["generator"] == g]
    w = np.array([int(r["width"]) for r in sub])
    h = np.array([int(r["height"]) for r in sub])
    b = np.array([int(r["bytes_file"]) for r in sub])
    res.append({
        "generator": g, "n": len(sub),
        "format": sorted({r["format"] for r in sub}),
        "w_min": int(w.min()), "w_med": int(np.median(w)), "w_max": int(w.max()),
        "h_min": int(h.min()), "h_med": int(np.median(h)), "h_max": int(h.max()),
        "bytes_mean_kb": round(float(b.mean()) / 1024, 1),
    })

splits = []
for s in ("train", "val", "test_seen", "test_unseen_ADM", "test_unseen_GLIDE"):
    sub = [r for r in rows if r["split"] == s]
    cc = Counter(r["class_id"] for r in sub)
    splits.append({"split": s, "n": len(sub), "classes": len(cc),
                   "per_class_min": min(cc.values()), "per_class_max": max(cc.values()),
                   "real": sum(1 for r in sub if r["generator"] == "real"),
                   "fake": sum(1 for r in sub if r["generator"] != "real")})

audit = {"by_generator": res, "by_split": splits,
         "total_bytes_mb": round(sum(int(r["bytes_file"]) for r in rows) / 1e6, 1)}
with io.open(os.path.join(OUT, "bias_audit.json"), "w", encoding="utf-8") as fh:
    json.dump(audit, fh, ensure_ascii=False, indent=2)
print(json.dumps(audit, ensure_ascii=False, indent=2))
