"""Manifest, splits and data audit."""
from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

COLUMNS = ["sample_id", "path", "label", "split", "group_id", "category", "condition", "origin", "sha256"]
MIN_SIDE = 32


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    missing = [c for c in COLUMNS if rows and c not in rows[0]]
    if missing:
        raise ValueError(f"manifest is missing columns: {missing}")
    return rows


def write_manifest(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLUMNS})


def build_visa_manifest(root: Path, split_csv: Path, category: str, val_ratio: float, seed: int) -> list[dict]:
    """Official VisA 1cls split. Test is kept as is; val is taken from train normals."""
    rows = []
    with open(split_csv, newline="") as f:
        for r in csv.DictReader(f):
            if r["object"] != category:
                continue
            label = 0 if r["label"] == "normal" else 1
            stem = Path(r["image"]).stem
            rows.append({
                "sample_id": f"{category}_{'N' if label == 0 else 'A'}_{stem}",
                "path": r["image"],
                "label": label,
                "split": r["split"],
                # no batch/session info in VisA, so one group per image
                "group_id": f"{category}_{'N' if label == 0 else 'A'}_{stem}",
                "category": category,
                "condition": "nominal",
                "origin": "VisA_20220922 (CC BY 4.0)",
            })
    train = sorted([r for r in rows if r["split"] == "train"], key=lambda r: r["sample_id"])
    rng = random.Random(seed)
    rng.shuffle(train)
    n_val = int(round(len(train) * val_ratio))
    for r in train[:n_val]:
        r["split"] = "val"
    for r in rows:
        r["sha256"] = sha256_file(root / r["path"])
    return sorted(rows, key=lambda r: (r["split"], r["sample_id"]))


def group_split(rows: list[dict], val_ratio: float, test_ratio: float, seed: int) -> list[dict]:
    """Split by group_id for our own data. Anomalies never go to train."""
    groups = defaultdict(list)
    for r in rows:
        groups[r["group_id"]].append(r)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n = len(keys)
    n_test = max(1, int(round(n * test_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    for i, k in enumerate(keys):
        split = "test" if i < n_test else "val" if i < n_test + n_val else "train"
        has_anomaly = any(int(r["label"]) == 1 for r in groups[k])
        if split == "train" and has_anomaly:
            split = "test"
        for r in groups[k]:
            r["split"] = split
    return rows


def make_stress(rows: list[dict], root: Path, out_dir: Path, n_per_class: int, seed: int,
                brightness: float = 1.25, offset: int = 12, blur: float = 1.2) -> list[dict]:
    """Stress copies of test images: brighter (x1.25 + 12) and blurred (r=1.2). Labels unchanged."""
    rng = random.Random(seed)
    out = []
    for label in (0, 1):
        pool = sorted([r for r in rows if r["split"] == "test" and int(r["label"]) == label], key=lambda r: r["sample_id"])
        for r in rng.sample(pool, min(n_per_class, len(pool))):
            img = Image.open(root / r["path"]).convert("RGB")
            arr = np.asarray(img, dtype=np.float32) * brightness + offset
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(blur))
            rel = Path("stress") / f"{r['sample_id']}_bright_blur.png"
            (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            img.save(out_dir / rel)
            out.append({**r, "sample_id": r["sample_id"] + "_stress", "path": str(rel), "split": "stress",
                        "condition": f"brightness x{brightness}+{offset}, blur r={blur}",
                        "sha256": sha256_file(out_dir / rel)})
    return out


DERIVED_SPLITS = {"stress", "excluded"}  # skipped in leakage checks


def gray_signature(img: Image.Image, size: int = 64) -> np.ndarray:
    """Normalized 64x64 grayscale, used for near-duplicate correlation.

    Note: dHash gave thousands of false matches on aligned boards, so we use correlation instead.
    """
    a = np.asarray(img.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32).ravel()
    return (a - a.mean()) / (a.std() + 1e-6)


def audit(rows: list[dict], root: Path, near_dup_corr: float = 0.99) -> dict:
    """Checks: corrupt/small images, labels, SHA256, group leakage, exact and near duplicates."""
    errors, warnings = [], []
    group_splits = defaultdict(set)
    sha_splits = defaultdict(set)
    hashes = []
    counts = defaultdict(lambda: [0, 0])
    for r in rows:
        sid = r["sample_id"]
        if str(r["label"]) not in ("0", "1"):
            errors.append({"sample_id": sid, "check": "label", "detail": r["label"]})
            continue
        p = root / r["path"]
        if not p.exists():
            errors.append({"sample_id": sid, "check": "missing_file", "detail": r["path"]})
            continue
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                im.load()
                if min(im.size) < MIN_SIDE:
                    errors.append({"sample_id": sid, "check": "too_small", "detail": im.size})
                hashes.append((sid, r["split"], r["group_id"], gray_signature(im)))
        except Exception as e:  # noqa: BLE001
            errors.append({"sample_id": sid, "check": "corrupt", "detail": str(e)})
            continue
        if r.get("sha256") and sha256_file(p) != r["sha256"]:
            errors.append({"sample_id": sid, "check": "sha256_changed", "detail": r["path"]})
        counts[r["split"]][int(r["label"])] += 1
        if r["split"] == "train" and int(r["label"]) == 1:
            errors.append({"sample_id": sid, "check": "anomaly_in_train", "detail": ""})
        group_splits[r["group_id"]].add(r["split"])
        sha_splits[r.get("sha256") or sha256_file(p)].add((r["split"], sid))
    for g, s in group_splits.items():
        s = s - DERIVED_SPLITS
        if len(s) > 1:
            errors.append({"sample_id": g, "check": "group_leakage", "detail": sorted(s)})
    for h, items in sha_splits.items():
        splits = {sp for sp, _ in items}
        if len(items) > 1:
            entry = {"sample_id": [i for _, i in sorted(items)], "check": "exact_duplicate", "detail": sorted(splits)}
            (errors if len(splits) > 1 else warnings).append(entry)
    # near duplicates across different groups
    if hashes:
        sig = np.stack([h[3] for h in hashes])
        corr = sig @ sig.T / sig.shape[1]
        for i in range(len(hashes)):
            for j in range(i + 1, len(hashes)):
                if corr[i, j] < near_dup_corr:
                    continue
                (s1, sp1, g1, _), (s2, sp2, g2, _) = hashes[i], hashes[j]
                if g1 == g2 or DERIVED_SPLITS & {sp1, sp2}:
                    continue
                entry = {"sample_id": [s1, s2], "check": "near_duplicate", "detail": f"{sp1} vs {sp2}, corr={corr[i, j]:.4f}"}
                (errors if sp1 != sp2 else warnings).append(entry)
    return {
        "n_rows": len(rows),
        "counts": {k: {"normal": v[0], "anomaly": v[1]} for k, v in sorted(counts.items())},
        "n_errors": len(errors),
        "n_warnings": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "near_dup_rule": f"Pearson correlation of 64x64 grayscale >= {near_dup_corr}; different split -> error, same split -> warning; review by eye",
    }


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
