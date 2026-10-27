# Model card: baselines, VisA pcb1

Run date: 2026-10-21. Machine: Linux x86_64, 4 CPU threads, Python 3.13, numpy 2.5.3, Pillow 12.3.0. No GPU.
Manifest: `data/manifests/pcb1_v2.csv` (SHA256 `b42c9535d51d6ab2411856746a4aea465bbf93376a96a906dcb8c7b26dcb0cc1`).

## 1. Models

| ID | Method | Parameters |
|---|---|---|
| B0 `template_z-pcb1-v2` | Per-pixel mean μ and std σ of train normals. z = \|x−μ\|/max(σ, 0.02), averaged over channels. Score = mean of the top 1% z values | RGB 96×96 |
| B1 `patch_knn-pcb1-v2` | 8×8 grid of 12×12 patches. 10 features per patch: RGB mean, RGB std, q10/q90 brightness, horizontal and vertical gradient. Euclidean distance to the nearest normal patch; score = mean of the 3 farthest patches | 96×96, memory of 52,096 patches from 814 images |

Threshold: 95th percentile of scores on the 89 validation normals, `method=higher`. Anomaly when `score > threshold`. The test set is not used to pick the threshold. `anomaly_score` is not a probability, and B0 and B1 use different scales.

Both models only need numpy and Pillow; no CNN or pretrained weights. They are the baselines for the main model in later weeks.

## 2. Official test results (100 normal, 100 anomaly)

| Model | TP/FP/FN/TN | Precision | Recall | F1 | FPR | AUROC | AP | Threshold | P95 ms* |
|---|---|---|---|---|---|---|---|---|---|
| B0 template_z | 44/8/56/92 | 0.846 | 0.440 | 0.579 | 8.0% | 0.832 | 0.802 | 4.033 | 63.9 |
| **B1 patch_knn** | **70/11/30/89** | **0.864** | **0.700** | **0.773** | 11.0% | **0.908** | **0.904** | 0.505 | 121.4 |

\* Batch of 1: one warmup, median of 3 runs per image, over 50 images. Includes reading and decoding the 1404×1070 JPEG, resizing and inference.

**Note:** validation only has normal images, so comparing B0 and B1 still relies on the test set. Next week we add a validation set with synthetic defects so models can be chosen without the test set.

## 3. Stress (20 normal, 20 anomaly; brightness ×1.25 +12 and blur r=1.2)

Both B0 and B1 have 100% FPR at the chosen threshold: all 20/20 normal images are flagged. AUROC is 0.735 and 0.695. A change in lighting and sharpness is enough to break the threshold. This is the biggest weakness so far.

## 4. Error analysis

**By defect type (test recall):**

| Defect | B0 | B1 |
|---|---|---|
| missing (sensor mesh missing) | 6/20 | **18/20** |
| scratch | 10/21 | 17/21 |
| melt | 20/54 | 36/54 |
| bent (bent pin) | **11/15** | 8/15 |

B1 catches defects that change surface texture (missing, scratch). Both models miss small defects such as light melt or slightly bent pins, because at 96×96 the defect covers only a few pixels. See `runs/patch_knn-pcb1-v2/FN_test.png` and `FP_test.png`.

**By orientation** (`runs/orientation_analysis.json`):

| Model | AUROC, majority orientation (68 N / 5 A) | AUROC, rotated 180° (32 N / 95 A) |
|---|---|---|
| B0 | 0.921 | **0.681** |
| B1 | 1.000 | 0.887 |

B0 compares each pixel with the mean image, so part of its score comes from detecting the rotation rather than the defect. In the rotated group, which holds 95% of the anomalies, B0 drops to 0.681 AUROC. B1 compares patches regardless of position and is less affected.

## 5. Next steps

1. Two-person label check on the 30 review images.
2. Validation with synthetic defects, so models are chosen without the test set.
3. A stronger model using pretrained CNN features (PatchCore), compared with B1.
4. Handle lighting changes (100% FPR on stress).
5. Package the model as an API for the web and workflow parts.

## 6. Artifacts

Each model lives in `artifacts/<model_version>/` with `model.npz` and `metadata.json` (method, parameters, threshold, threshold rule, manifest SHA256, environment). Each run writes `runs/<model_version>/` with `metrics.json`, `predictions.csv`, `errors_*.json` and FP/FN contact sheets.
