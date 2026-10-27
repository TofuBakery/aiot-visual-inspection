"""Metrics (positive class = anomaly) and latency."""
from __future__ import annotations

import time

import numpy as np


def confusion(y: np.ndarray, pred: np.ndarray) -> dict:
    return {"TP": int(((pred == 1) & (y == 1)).sum()), "FP": int(((pred == 1) & (y == 0)).sum()),
            "FN": int(((pred == 0) & (y == 1)).sum()), "TN": int(((pred == 0) & (y == 0)).sum())}


def _div(a: float, b: float):
    return None if b == 0 else a / b


def auroc(y: np.ndarray, s: np.ndarray):
    """Mann-Whitney U with average ranks for ties."""
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def average_precision(y: np.ndarray, s: np.ndarray):
    """Same definition as sklearn average_precision_score."""
    if y.sum() == 0 or (y == 0).sum() == 0:
        return None
    order = np.argsort(-s, kind="mergesort")
    ys, ss = y[order], s[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    last = np.r_[np.where(np.diff(ss))[0], len(ss) - 1]
    prec = tp[last] / (tp[last] + fp[last])
    rec = tp[last] / y.sum()
    rec_prev = np.r_[0, rec[:-1]]
    return float(((rec - rec_prev) * prec).sum())


def metrics(y, scores, threshold: float) -> dict:
    y, s = np.asarray(y, int), np.asarray(scores, float)
    pred = (s > threshold).astype(int)
    c = confusion(y, pred)
    p = _div(c["TP"], c["TP"] + c["FP"])
    r = _div(c["TP"], c["TP"] + c["FN"])
    f1 = None if p is None or r is None or (p + r) == 0 else 2 * p * r / (p + r)
    return {"n_normal": int((y == 0).sum()), "n_anomaly": int((y == 1).sum()), **c,
            "precision": p, "recall": r, "f1": f1, "fpr": _div(c["FP"], c["FP"] + c["TN"]),
            "auroc": auroc(y, s), "ap": average_precision(y, s), "threshold": threshold}


def latency(fn, items, runs: int = 3) -> dict:
    """Batch 1, one warmup, median of `runs` per image. Includes decode + resize."""
    fn(items[0])
    per_image = []
    for it in items:
        ts = []
        for _ in range(runs):
            t0 = time.perf_counter()
            fn(it)
            ts.append((time.perf_counter() - t0) * 1000)
        per_image.append(float(np.median(ts)))
    a = np.asarray(per_image)
    return {"runs_per_image": runs, "n_images": len(items), "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)), "mean_ms": float(a.mean())}
