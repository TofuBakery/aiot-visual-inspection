"""CPU baselines B0 template_z and B1 patch_knn. Fit on normal images only."""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb(source, size: int) -> np.ndarray:
    """Path or PIL image -> float32 array in [0, 1], size x size x 3."""
    img = source if isinstance(source, Image.Image) else Image.open(source)
    img = img.convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def _gray_z(x: np.ndarray) -> np.ndarray:
    g = x.mean(axis=-1).ravel()
    return (g - g.mean()) / (g.std() + 1e-6)


class OrientationAligner:
    """Flips an image 180 degrees when it matches the flipped reference better.

    Note: pcb1 boards appear in two orientations; the reference is learned from train normals only.
    """

    def __init__(self):
        self.ref = None

    def fit(self, images: np.ndarray, iters: int = 3) -> None:
        ref = _gray_z(images[0])
        for _ in range(iters):
            aligned = [self._apply(x, ref) for x in images]
            ref = _gray_z(np.mean(aligned, axis=0))
        self.ref = ref

    @staticmethod
    def _apply(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
        g = _gray_z(x)
        g_rot = _gray_z(x[::-1, ::-1])
        return x if g @ ref >= g_rot @ ref else np.ascontiguousarray(x[::-1, ::-1])

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x if self.ref is None else self._apply(x, self.ref)


class TemplateZ:
    """B0: per-pixel mean/std of train normals; score = mean of the top 1% z values."""

    name = "template_z"

    def __init__(self, size: int = 96, top_frac: float = 0.01, sigma_floor: float = 0.02, align: bool = False):
        self.size, self.top_frac, self.sigma_floor, self.align = size, top_frac, sigma_floor, align
        self.mu = self.sigma = None
        self.aligner = OrientationAligner()

    def fit(self, images: np.ndarray) -> "TemplateZ":
        if self.align:
            self.aligner.fit(images)
            images = np.stack([self.aligner(x) for x in images])
        self.mu = images.mean(axis=0)
        self.sigma = images.std(axis=0)
        return self

    def zmap(self, x: np.ndarray) -> np.ndarray:
        return (np.abs(x - self.mu) / np.maximum(self.sigma, self.sigma_floor)).mean(axis=-1)

    def score(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        if self.align:
            x = self.aligner(x)
        zm = self.zmap(x)
        z = zm.ravel()
        k = max(1, int(round(z.size * self.top_frac)))
        return float(np.partition(z, -k)[-k:].mean()), zm

    def state(self) -> dict:
        st = {"mu": self.mu, "sigma": self.sigma}
        if self.align:
            st["orient_ref"] = self.aligner.ref
        return st

    def load_state(self, st: dict) -> None:
        self.mu, self.sigma = st["mu"], st["sigma"]
        self.aligner.ref = st.get("orient_ref")

    def params(self) -> dict:
        return {"size": self.size, "top_frac": self.top_frac, "sigma_floor": self.sigma_floor, "align": self.align}


class PatchKNN:
    """B1: 8x8 grid of patches, 10 stats per patch, nearest-neighbour distance to normal patches."""

    name = "patch_knn"

    def __init__(self, size: int = 96, grid: int = 8, top_k: int = 3, align: bool = False):
        self.size, self.grid, self.top_k, self.align = size, grid, top_k, align
        self.feat_mean = self.feat_std = self.memory = None
        self.aligner = OrientationAligner()

    def features(self, x: np.ndarray) -> np.ndarray:
        p = self.size // self.grid
        lum = x @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        gx = np.abs(np.diff(lum, axis=1, append=lum[:, -1:]))
        gy = np.abs(np.diff(lum, axis=0, append=lum[-1:, :]))
        feats = []
        for i in range(self.grid):
            for j in range(self.grid):
                sl = (slice(i * p, (i + 1) * p), slice(j * p, (j + 1) * p))
                patch, l = x[sl].reshape(-1, 3), lum[sl].ravel()
                feats.append(np.concatenate([
                    patch.mean(0), patch.std(0),
                    np.quantile(l, [0.1, 0.9]),
                    [gx[sl].mean(), gy[sl].mean()],
                ]))
        return np.asarray(feats, dtype=np.float32)

    def fit(self, images: np.ndarray) -> "PatchKNN":
        if self.align:
            self.aligner.fit(images)
            images = np.stack([self.aligner(x) for x in images])
        f = np.concatenate([self.features(x) for x in images])
        self.feat_mean, self.feat_std = f.mean(0), f.std(0) + 1e-6
        self.memory = (f - self.feat_mean) / self.feat_std
        self._mem_sq = (self.memory ** 2).sum(1)
        return self

    def score(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        if self.align:
            x = self.aligner(x)
        q = (self.features(x) - self.feat_mean) / self.feat_std
        d2 = (q ** 2).sum(1)[:, None] - 2 * q @ self.memory.T + self._mem_sq[None, :]
        d = np.sqrt(np.maximum(d2.min(1), 0))
        top = np.sort(d)[-self.top_k:]
        return float(top.mean()), d.reshape(self.grid, self.grid)

    def state(self) -> dict:
        st = {"feat_mean": self.feat_mean, "feat_std": self.feat_std, "memory": self.memory}
        if self.align:
            st["orient_ref"] = self.aligner.ref
        return st

    def load_state(self, st: dict) -> None:
        self.feat_mean, self.feat_std, self.memory = st["feat_mean"], st["feat_std"], st["memory"]
        self.aligner.ref = st.get("orient_ref")
        self._mem_sq = (self.memory ** 2).sum(1)

    def params(self) -> dict:
        return {"size": self.size, "grid": self.grid, "patch": self.size // self.grid, "top_k": self.top_k, "align": self.align,
                "n_features": 10, "memory_patches": None if self.memory is None else int(len(self.memory))}


MODELS = {"template_z": TemplateZ, "patch_knn": PatchKNN}


def environment() -> dict:
    import PIL
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
           "processor": platform.processor() or "unknown", "numpy": np.__version__, "pillow": PIL.__version__}
    return env


def save_artifact(model, meta: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "model.npz", **model.state())
    (out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_artifact(model_dir: Path):
    meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    cls = MODELS[meta["method"]]
    p = dict(meta["params"])
    for k in ("patch", "n_features", "memory_patches"):
        p.pop(k, None)
    model = cls(**p)
    with np.load(model_dir / "model.npz") as st:
        model.load_state({k: st[k] for k in st.files})
    return model, meta
