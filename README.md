# aiot-visual-inspection

Data preparation and AI model block of an **industrial AI monitoring framework** (Computer Engineering Project, HCMUT, semester HK261).

It prepares a product image dataset, checks it for errors and leaks, and trains anomaly detection models that learn only from normal products, then flags each image as `normal` or `anomaly`. The current build runs on the public VisA pcb1 dataset in place of factory camera images (the brief's *immediate start* phase) and will be wired into the group's web dashboard and workflow tools later through an API.

```
VisA / own images ──► Manifest ──────────► Audit ──────────────► Model ───────────────► Evaluate
                      SHA256 per image,     corrupt files,         B0 template_z,          metrics, FP/FN sheets,
                      train/val/test,       labels, leakage,       B1 patch_knn,           stress set,
                      group split           near duplicates        fit on normals only     orientation analysis
                                                                   threshold from val
```

## Features

- **Manifest**: one CSV row per image with label, split, group and SHA256; the official VisA test split is kept as is and validation is taken from train normals.
- **Splits for own data**: group split by `group_id`, so images of the same board never land in two splits, and anomalies never go to train.
- **Audit**: corrupt or tiny images, invalid labels, changed checksums, anomalies in train, group leakage, byte-identical files and near duplicates (grayscale correlation), with errors and warnings in one JSON report.
- **Review**: dataset fixes are recorded as a review table and applied to produce a new manifest version, so every change is traceable.
- **Stress set**: brighter and blurred copies of test images to check robustness to lighting.
- **Baselines**: B0 template_z and B1 patch_knn, CPU only, numpy and Pillow only.
- **Thresholds**: chosen from validation normals only; the test set is used for evaluation only.
- **Evaluation**: precision, recall, F1, FPR, AUROC, AP, latency, per-image predictions and FP/FN contact sheets.
- **Reproducible**: every model stores its parameters, threshold rule, manifest SHA256 and environment.

## Quick start

Requires Python 3.11+. No GPU needed.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
./run_all.sh                           # download VisA pcb1, build, audit, fit, evaluate
```

Other options:

```bash
python -m aiot audit --manifest data/manifests/pcb1_v2.csv --root data/raw
python -m aiot fit --manifest data/manifests/pcb1_v2.csv --root data/raw --method patch_knn --version patch_knn-pcb1-v2
python -m aiot evaluate --manifest data/manifests/pcb1_v2.csv --root data/raw --model artifacts/patch_knn-pcb1-v2
python -m aiot split --input my_images.csv --root my_data --out data/manifests/my_v1.csv   # own camera data
```

On Windows, run the lines of `run_all.sh` one by one. The repo holds manifests only, not the VisA images; audit reports an error if a downloaded image does not match its checksum.

## Outputs

| Path | Content |
|---|---|
| `data/manifests/<name>.csv` | one row per image: label, split, group, SHA256 |
| `runs/audit/<manifest>.audit.json` | audit errors, warnings and counts per split |
| `artifacts/<version>/model.npz` | fitted model state |
| `artifacts/<version>/metadata.json` | method, parameters, threshold and rule, manifest SHA256, environment |
| `runs/<version>/metrics.json` | metrics and latency per split |
| `runs/<version>/predictions.csv` | score, threshold and prediction per image |
| `runs/<version>/errors_<split>.json`, `FP_<split>.png`, `FN_<split>.png` | misclassified images, worst first |
| `runs/orientation_analysis.json` | test metrics split by board orientation |

The dataset is described in [docs/Dataset_Card.md](docs/Dataset_Card.md), the models and results in [docs/Model_Card.md](docs/Model_Card.md), and the shared event format in [docs/Event_Contract.md](docs/Event_Contract.md).

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q          # 11 tests, no dataset needed
```

The tests use small synthetic board images and cover metrics, manifest reading, every audit check, group split, the stress set, both baselines (defect scores higher than normal, save and load) and a full fit and evaluate run through the CLI.

## Project layout

```
aiot/
  data.py         manifest, VisA split, group split, stress set, audit
  models.py       B0 template_z, B1 patch_knn, save/load
  evaluate.py     metrics and latency
  cli.py          command line
scripts/          orientation analysis
data/manifests/   manifests and review tables
artifacts/        fitted models
runs/             audit and evaluation results
tests/            unit and CLI tests
docs/             dataset card, model card, event contract, status
run_all.sh        full rebuild
```

## Status and limitations

What is implemented, measured and still open is tracked in [docs/status.md](docs/status.md). In short: the data pipeline and both baselines run end to end, and B1 reaches F1 0.773 and AUROC 0.908 on the official test set; both baselines flag every image in the lighting stress set, and the two-person label check is not done yet.

## Licensing

This repository has no license file yet. The VisA dataset is licensed under CC BY 4.0 and must be credited as "VisA dataset, Zou et al., SPot-the-Difference, ECCV 2022, Amazon Science". The code only depends on numpy and Pillow.
