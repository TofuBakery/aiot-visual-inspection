"""Tiny synthetic boards so tests run without the VisA download."""
from pathlib import Path

import numpy as np
from PIL import Image

from aiot.data import sha256_file, write_manifest


def board(seed: int, defect: bool = False, size: int = 96) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.full((size, size, 3), (40, 110, 160), np.float32) + rng.normal(0, 20, (size, size, 3))
    y = int(rng.integers(10, 30))
    arr[y:y + 10, 10:86] = (200, 200, 200)
    if defect:
        arr[50:66, 40:56] = (240, 60, 30)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def make_dataset(root: Path, n_train=20, n_val=6, n_test=6) -> list[dict]:
    rows = []
    specs = [("train", 0, n_train), ("val", 0, n_val), ("test", 0, n_test), ("test", 1, n_test)]
    seed = 0
    for split, label, n in specs:
        for i in range(n):
            seed += 1
            sid = f"toy_{split}_{'A' if label else 'N'}_{i:02d}"
            rel = Path("toy") / f"{sid}.png"
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            board(seed, defect=bool(label)).save(root / rel)
            rows.append({"sample_id": sid, "path": str(rel), "label": label, "split": split, "group_id": sid,
                         "category": "toy", "condition": "nominal", "origin": "synthetic", "sha256": sha256_file(root / rel)})
    write_manifest(rows, root / "manifest.csv")
    return rows
