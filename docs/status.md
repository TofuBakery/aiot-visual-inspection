# Status

Last updated 2026-10-27, version 0.1.0.

## Implemented

| Area | State | Code |
|---|---|---|
| Dataset shortlist and choice (VisA pcb1, CC BY 4.0) | done | `docs/Dataset_Card.md` |
| Manifest with SHA256, official test kept, val from train normals | done | `aiot/data.py` |
| Group split for own camera data | done, not used yet | `aiot/data.py` |
| Audit: corrupt files, labels, checksums, leakage, exact and near duplicates | done | `aiot/data.py` |
| Review table applied to a new manifest version (v2) | done | `data/manifests/review_v2.csv` |
| Stress set (brightness and blur) | done | `aiot/data.py` |
| Baselines B0 template_z and B1 patch_knn | done, CPU only | `aiot/models.py` |
| Evaluation: metrics, latency, predictions, FP/FN sheets | done | `aiot/evaluate.py`, `aiot/cli.py` |
| Orientation analysis | done | `scripts/orientation_analysis.py` |
| Common event format with the other blocks | draft, being agreed | `docs/Event_Contract.md` |
| Automated tests | 11 passing | `tests/` |

## Reference results

Official VisA pcb1 test set, 100 normal and 100 anomaly images. Raw files: `runs/`.

| Model | TP/FP/FN/TN | Precision | Recall | F1 | FPR | AUROC |
|---|---|---|---|---|---|---|
| B0 template_z | 44/8/56/92 | 0.846 | 0.44 | 0.579 | 8% | 0.832 |
| B1 patch_knn | 70/11/30/89 | 0.864 | 0.70 | 0.773 | 11% | 0.908 |

How to read these numbers:

- Validation has normal images only, so comparing B0 and B1 still uses the test set.
- On the stress set both models flag 20/20 normal images: lighting changes break the threshold.
- 95/100 test anomalies are rotated 180° while most normals are not, so B0 partly detects the rotation; its AUROC on rotated images is 0.681.
- Small defects (light melt, slightly bent pins) are often missed at 96×96.

## Open

- Two-person label check on the 30 review images.
- Validation set with synthetic defects for model selection.
- Stronger model with pretrained CNN features (PatchCore).
- Robustness to lighting changes.
- Prediction API for the web and workflow parts.
- Images from a real camera.
