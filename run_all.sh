#!/usr/bin/env bash
set -euo pipefail
PY=${PY:-python3}
if [ ! -d data/raw/pcb1 ]; then
  echo "Downloading VisA (1.9 GB), keeping pcb1 only..."
  mkdir -p data/raw && cd data/raw
  curl -sSL https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar | tar -x --wildcards 'pcb1/*' 'split_csv/*' 'LICENSE*'
  cd ../..
fi
$PY -m aiot build-visa --root data/raw --split-csv data/raw/split_csv/1cls.csv --category pcb1 --out data/manifests/pcb1_v1_base.csv
$PY -m aiot stress --manifest data/manifests/pcb1_v1_base.csv --root data/raw --out data/manifests/pcb1_v1.csv
$PY -m aiot audit --manifest data/manifests/pcb1_v1.csv --root data/raw --out runs/audit/pcb1_v1.audit.json || echo "v1 has near duplicates (expected), applying review"
$PY -m aiot apply-review --manifest data/manifests/pcb1_v1.csv --review data/manifests/review_v2.csv --out data/manifests/pcb1_v2.csv
$PY -m aiot audit --manifest data/manifests/pcb1_v2.csv --root data/raw --out runs/audit/pcb1_v2.audit.json
for m in template_z patch_knn; do
  $PY -m aiot fit --manifest data/manifests/pcb1_v2.csv --root data/raw --method $m --version $m-pcb1-v2
  $PY -m aiot evaluate --manifest data/manifests/pcb1_v2.csv --root data/raw --model artifacts/$m-pcb1-v2
done
$PY scripts/orientation_analysis.py > /dev/null
