"""Test metrics split by board orientation (pcb1 boards appear in two 180-degree orientations)."""
import csv, json, sys
from collections import Counter
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aiot.data import read_manifest
from aiot.evaluate import metrics
from aiot.models import OrientationAligner, load_rgb

rows = [r for r in read_manifest(Path("data/manifests/pcb1_v2.csv")) if r["split"] in ("train", "val", "test")]
imgs = {r["sample_id"]: load_rgb(Path("data/raw") / r["path"], 96) for r in rows}
al = OrientationAligner(); al.fit(np.stack([imgs[r["sample_id"]] for r in rows if r["split"] == "train"]))
# ref = majority orientation in train normals
orient = {sid: ("ref" if al(x) is x else "rot180") for sid, x in imgs.items()}
counts = Counter((r["split"], "anomaly" if r["label"] == "1" else "normal", orient[r["sample_id"]]) for r in rows)
out = {"counts": {"/".join(k): v for k, v in sorted(counts.items())}, "models": {}}
for m in ["template_z-pcb1-v2", "patch_knn-pcb1-v2"]:
    if not Path(f"runs/{m}/predictions.csv").exists():
        continue
    P = [p for p in csv.DictReader(open(f"runs/{m}/predictions.csv")) if p["split"] == "test"]
    thr = float(P[0]["threshold"]); res = {}
    for o in ("ref", "rot180"):
        sub = [p for p in P if orient[p["sample_id"]] == o]
        mm = metrics([int(p["label"]) for p in sub], [float(p["score"]) for p in sub], thr)
        res[o] = {k: mm[k] for k in ("n_normal", "n_anomaly", "recall", "fpr", "auroc")}
    out["models"][m] = res
Path("runs/orientation_analysis.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=1))
