# Dataset card: VisA pcb1, manifest v2

Owner: Data Preparation (Group 1). Created: 2026-10-21.

## 1. Dataset shortlist

| Dataset | Task, labels | Size | License | Notes |
|---|---|---|---|---|
| **VisA** (Amazon Science, ECCV 2022) | Anomaly detection; image-level normal/anomaly labels and pixel masks | 10,821 images, 12 product types | CC BY 4.0 (checked `LICENSE-DATASET` in the download) | **Chosen.** Commercial use allowed with attribution. Official split `1cls.csv` |
| MVTec AD | Anomaly detection; normal-only train, test with masks | 5,354 images, 15 categories | CC BY-NC-SA 4.0 | Academic reference; no commercial use |
| MVTec AD 2 | Anomaly detection with lighting changes | 8,000+ images, 8 scenarios | CC BY-NC-SA 4.0 | Useful for lighting stress tests later |
| DeepPCB | PCB defect detection, bounding boxes (6 defect types) | 1,500 template/test pairs | **Not verified**, check the original repo | Detection task, not image classification |
| NEU surface defect | Steel surface defect classification, 6 classes | 1,800 grayscale images | **Not verified** | No normal class, so not suited to anomaly detection |

Why VisA pcb1: image labels and masks, a license that allows commercial use, and a fixed camera and background similar to a QA/QC inspection station. 30 images were reviewed before the choice was locked (`docs/figures/pcb1_review_30.png`).

## 2. About pcb1

* Product: HC-SR04 ultrasonic sensor module, shot top-down on grey fabric, JPEG 1404×1070.
* Original labels: 1,004 normal, 100 anomaly. Defect types in test: melt 54, scratch 21, missing 20, bent 15 (one image can have several). Details: `data/manifests/pcb1_annotations.csv`.
* Source: `https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar`, official split `split_csv/1cls.csv`. Raw files are kept unchanged in `data/raw/`.

## 3. Manifest and splits

Columns: `sample_id,path,label,split,group_id,category,condition,origin,sha256`. Label 0 is normal, 1 is anomaly.

| Split | Normal | Anomaly | Used for |
|---|---|---|---|
| train | 814 | 0 | Model fitting (normal only) |
| val | 89 | 0 | Threshold (95th percentile of scores) |
| test | 100 | 100 | Official VisA test, unchanged, evaluation only |
| stress | 20 | 20 | Copies of test images, brightness ×1.25 +12 and Gaussian blur r=1.2 |
| excluded | 1 | 0 | Near duplicate of a test image, removed from train (needs_review) |

* Validation is a random 10% of train normals (seed 42). The official test split is not changed.
* `group_id`: VisA has no batch or session info, so each image is its own group. The two near-duplicate pairs are merged into groups (section 4).
* Versions: `pcb1_v1.csv` (SHA256 `9aafec2d…`) is generated automatically; `pcb1_v2.csv` (SHA256 `b42c9535…`) is the reviewed version, built with `aiot apply-review` and `data/manifests/review_v2.csv`.

## 4. Audit

Command: `python -m aiot audit --manifest data/manifests/pcb1_v2.csv --root data/raw`. It checks corrupt or tiny images, labels outside 0/1, changed SHA256, anomalies in train, groups spread over several splits, byte-identical files across splits, and **near duplicates** (Pearson correlation of 64×64 grayscale ≥ 0.99).

* v1: 2 near-duplicate errors. `0187` (test) and `0188` (train), correlation 0.9942. `0042` (val) and `0043` (train), correlation 0.9947. Two different boards usually correlate at about 0.96–0.98. See `runs/audit/near_dup_pairs.png`.
* Fix (v2): each pair is merged into one group. 0042/0043 both go to train. For 0187/0188, 0187 stays in test and 0188 is removed from train.
* v2: **0 errors, 0 warnings** (`runs/audit/pcb1_v2.audit.json`).
* Note: a 64-bit dHash reported over 4,800 false "near duplicates", because aligned boards look the same at low resolution. Correlation is used instead.

## 5. Orientation bias

Boards are shot in two orientations, 180° apart (pins up or pins down). The split by label is very uneven (`scripts/orientation_analysis.py`):

| Set | Majority orientation (ref) | Rotated 180° |
|---|---|---|
| train normal | 651 | 163 |
| test normal | 68 | 32 |
| test anomaly | 5 | **95** |

A model can learn the orientation instead of the defect, so whole-test metrics are optimistic, especially for B0. The model card reports metrics per orientation.

## 6. Label check

30 images are selected for review (`data/manifests/pcb1_review_30.csv`: 12 normal, 18 anomaly). Plan: two members label them independently, record disagreements, then agree on rules. **Not done yet**; the reviewer columns are empty.

## 7. Limitations

* No batch or session info, so a batch-independent holdout cannot be shown.
* Fixed lighting, background and camera; different from a real factory line.
* The stress set is a synthetic transform and does not represent real lighting changes.
* Required attribution (CC BY 4.0): "VisA dataset, Zou et al., SPot-the-Difference, ECCV 2022, Amazon Science".
