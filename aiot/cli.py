"""CLI: build-visa, split, audit, apply-review, stress, fit, evaluate."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from . import __version__
from .data import audit, build_visa_manifest, group_split, make_stress, read_manifest, save_json, sha256_file, write_manifest
from .evaluate import latency, metrics
from .models import MODELS, environment, load_artifact, load_rgb, save_artifact


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def cmd_build_visa(a):
    rows = build_visa_manifest(Path(a.root), Path(a.split_csv), a.category, a.val_ratio, a.seed)
    write_manifest(rows, Path(a.out))
    print(f"{len(rows)} rows -> {a.out}")


def cmd_split(a):
    rows = read_manifest(Path(a.input))
    root = Path(a.root)
    for r in rows:
        r["sha256"] = r.get("sha256") or sha256_file(root / r["path"])
    rows = group_split(rows, a.val_ratio, a.test_ratio, a.seed)
    write_manifest(rows, Path(a.out))
    print(f"{len(rows)} rows -> {a.out}")


def cmd_audit(a):
    rep = audit(read_manifest(Path(a.manifest)), Path(a.root))
    rep["manifest"], rep["manifest_sha256"], rep["created_at"] = a.manifest, sha256_file(Path(a.manifest)), _now()
    out = Path(a.out or Path(a.manifest).with_suffix(".audit.json"))
    save_json(rep, out)
    print(json.dumps({k: rep[k] for k in ("n_rows", "counts", "n_errors", "n_warnings")}, ensure_ascii=False))
    print(f"-> {out}")
    sys.exit(1 if rep["n_errors"] else 0)


def cmd_apply_review(a):
    """Apply review decisions (split/group changes) to a new manifest version."""
    rows = read_manifest(Path(a.manifest))
    by_id = {r["sample_id"]: r for r in rows}
    with open(a.review, newline="", encoding="utf-8") as f:
        decisions = list(csv.DictReader(f))
    for d in decisions:
        r = by_id[d["sample_id"]]
        if d["action"] == "set_split":
            r["split"] = d["value"]
        elif d["action"] == "set_group":
            r["group_id"] = d["value"]
        else:
            raise ValueError(f"unsupported action: {d['action']}")
    write_manifest(rows, Path(a.out))
    print(f"applied {len(decisions)} decisions -> {a.out}")


def cmd_stress(a):
    rows = read_manifest(Path(a.manifest))
    stress = make_stress(rows, Path(a.root), Path(a.root), a.n_per_class, a.seed)
    write_manifest(rows + stress, Path(a.out))
    print(f"added {len(stress)} stress images -> {a.out}")


def _images(rows, root: Path, size: int) -> np.ndarray:
    return np.stack([load_rgb(root / r["path"], size) for r in rows])


def cmd_fit(a):
    root, rows = Path(a.root), read_manifest(Path(a.manifest))
    train = [r for r in rows if r["split"] == "train" and r["label"] == "0"]
    val = [r for r in rows if r["split"] == "val" and r["label"] == "0"]
    kw = {"grid": a.grid} if a.method == "patch_knn" and a.grid else {}
    model = MODELS[a.method](size=a.size, align=a.align, **kw)
    model.fit(_images(train, root, a.size))
    val_scores = [model.score(load_rgb(root / r["path"], a.size))[0] for r in val]
    thr = float(np.quantile(val_scores, a.percentile / 100, method="higher"))
    category = train[0]["category"]
    version = a.version or f"{a.method}-{category}-{dt.datetime.now():%Y%m%d-%H%M%S}"
    meta = {
        "model_version": version, "method": a.method, "category": category, "params": model.params(),
        "origin": sorted({r["origin"] for r in train}),
        "threshold": thr, "threshold_rule": f"percentile {a.percentile} of validation normal scores, method=higher; anomaly if score > threshold",
        "n_train_normal": len(train), "n_val_normal": len(val),
        "val_score_stats": {"min": float(min(val_scores)), "median": float(np.median(val_scores)), "max": float(max(val_scores))},
        "manifest": a.manifest, "manifest_sha256": sha256_file(Path(a.manifest)),
        "environment": environment(), "created_at": _now(), "aiot_version": __version__,
        "note": "anomaly_score is not a probability; scales differ between methods",
    }
    out = Path(a.out) / version
    save_artifact(model, meta, out)
    print(f"{a.method}: threshold={thr:.4f} -> {out}")


def _save_contact_sheet(items, root: Path, out: Path, title_fn, n_max: int = 16, thumb: int = 160):
    items = items[:n_max]
    if not items:
        return
    from PIL import ImageDraw
    cols = 4
    rows_n = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows_n * (thumb + 28)), "white")
    d = ImageDraw.Draw(sheet)
    for i, it in enumerate(items):
        im = Image.open(root / it["path"]).convert("RGB").resize((thumb, thumb))
        x, y = (i % cols) * thumb, (i // cols) * (thumb + 28)
        sheet.paste(im, (x, y))
        d.text((x + 3, y + thumb + 2), title_fn(it), fill="black")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def cmd_evaluate(a):
    root, rows = Path(a.root), read_manifest(Path(a.manifest))
    model, meta = load_artifact(Path(a.model))
    thr = meta["threshold"]
    out = Path(a.out) / meta["model_version"]
    result = {"model_version": meta["model_version"], "method": meta["method"], "category": meta["category"],
              "manifest_sha256": sha256_file(Path(a.manifest)), "environment": environment(), "created_at": _now(), "sets": {}}
    preds = []
    for split in a.splits:
        subset = [r for r in rows if r["split"] in split.split("+")]  # "test+stress" merges splits
        if not subset:
            continue
        arrs = [load_rgb(root / r["path"], model.size) for r in subset]
        scores = [model.score(x)[0] for x in arrs]
        y = [int(r["label"]) for r in subset]
        m = metrics(y, scores, thr)
        m["latency"] = latency(lambda p: model.score(load_rgb(root / p, model.size)), [r["path"] for r in subset][: a.latency_n], a.runs)
        result["sets"][split] = m
        errs = []
        for r, s in zip(subset, scores):
            pred = int(s > thr)
            preds.append({"sample_id": r["sample_id"], "split": split, "path": r["path"], "label": int(r["label"]),
                          "score": round(s, 6), "threshold": thr, "prediction": pred})
            if pred != int(r["label"]):
                errs.append({"sample_id": r["sample_id"], "path": r["path"], "label": int(r["label"]), "score": s,
                             "type": "FP" if pred == 1 else "FN"})
        errs.sort(key=lambda e: -abs(e["score"] - thr))
        save_json(errs, out / f"errors_{split}.json")
        for t in ("FP", "FN"):
            _save_contact_sheet([e for e in errs if e["type"] == t], root, out / f"{t}_{split}.png",
                                lambda e: f"{e['sample_id'][-14:]} s={e['score']:.2f}")
    with open(out / "predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(preds[0]))
        w.writeheader()
        w.writerows(preds)
    save_json(result, out / "metrics.json")
    for split, m in result["sets"].items():
        fmt = lambda v: "n/a" if v is None else f"{v:.3f}"
        print(f"{meta['method']:<11} {split:<6} TP/FP/FN/TN={m['TP']}/{m['FP']}/{m['FN']}/{m['TN']} "
              f"P={fmt(m['precision'])} R={fmt(m['recall'])} F1={fmt(m['f1'])} FPR={fmt(m['fpr'])} "
              f"AUROC={fmt(m['auroc'])} AP={fmt(m['ap'])} p95={m['latency']['p95_ms']:.2f}ms")


def main(argv=None):
    p = argparse.ArgumentParser(prog="aiot")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("build-visa", help="manifest from the VisA 1cls split")
    s.add_argument("--root", required=True); s.add_argument("--split-csv", required=True)
    s.add_argument("--category", default="pcb1"); s.add_argument("--val-ratio", type=float, default=0.1)
    s.add_argument("--seed", type=int, default=42); s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_build_visa)

    s = sub.add_parser("split", help="group split for our own data")
    s.add_argument("--input", required=True); s.add_argument("--root", required=True); s.add_argument("--out", required=True)
    s.add_argument("--val-ratio", type=float, default=0.15); s.add_argument("--test-ratio", type=float, default=0.25)
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(fn=cmd_split)

    s = sub.add_parser("audit"); s.add_argument("--manifest", required=True); s.add_argument("--root", required=True)
    s.add_argument("--out"); s.set_defaults(fn=cmd_audit)

    s = sub.add_parser("apply-review", help="apply a review table -> new manifest")
    s.add_argument("--manifest", required=True); s.add_argument("--review", required=True); s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_apply_review)

    s = sub.add_parser("stress"); s.add_argument("--manifest", required=True); s.add_argument("--root", required=True)
    s.add_argument("--out", required=True); s.add_argument("--n-per-class", type=int, default=20)
    s.add_argument("--seed", type=int, default=42); s.set_defaults(fn=cmd_stress)

    s = sub.add_parser("fit"); s.add_argument("--manifest", required=True); s.add_argument("--root", required=True)
    s.add_argument("--method", choices=list(MODELS), required=True); s.add_argument("--size", type=int, default=96)
    s.add_argument("--percentile", type=float, default=95); s.add_argument("--version")
    s.add_argument("--grid", type=int, help="patch_knn: grid cells per side (default 8)")
    s.add_argument("--align", action="store_true", help="flip to a common 180-degree orientation")
    s.add_argument("--out", default="artifacts"); s.set_defaults(fn=cmd_fit)

    s = sub.add_parser("evaluate"); s.add_argument("--manifest", required=True); s.add_argument("--root", required=True)
    s.add_argument("--model", required=True); s.add_argument("--splits", nargs="+", default=["test", "stress"])
    s.add_argument("--latency-n", type=int, default=50); s.add_argument("--runs", type=int, default=3)
    s.add_argument("--out", default="runs"); s.set_defaults(fn=cmd_evaluate)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
