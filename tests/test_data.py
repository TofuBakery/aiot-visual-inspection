import shutil

import pytest

from aiot.data import audit, group_split, make_stress, read_manifest, write_manifest
from helpers import make_dataset


def test_manifest_roundtrip_and_missing_columns(tmp_path):
    rows = make_dataset(tmp_path)
    assert len(read_manifest(tmp_path / "manifest.csv")) == len(rows)
    (tmp_path / "bad.csv").write_text("sample_id,path\na,b\n")
    with pytest.raises(ValueError):
        read_manifest(tmp_path / "bad.csv")


def test_audit_clean(tmp_path):
    rows = make_dataset(tmp_path)
    rep = audit(read_manifest(tmp_path / "manifest.csv"), tmp_path)
    assert rep["n_errors"] == 0, rep["errors"]
    assert rep["counts"]["test"] == {"normal": 6, "anomaly": 6}
    assert len(rows) == 38


def test_audit_finds_problems(tmp_path):
    rows = make_dataset(tmp_path)
    test_img = next(r for r in rows if r["split"] == "test" and r["label"] == 0)
    shutil.copyfile(tmp_path / test_img["path"], tmp_path / rows[0]["path"])  # train copy of a test image
    rows[0]["sha256"] = test_img["sha256"]
    rows[1]["label"] = 1                                                       # anomaly in train
    rows[2]["group_id"] = rows[-1]["group_id"]                                 # group in two splits
    rows[3]["sha256"] = "0" * 64                                               # file changed
    rows[4]["path"] = "toy/missing.png"
    write_manifest(rows, tmp_path / "m2.csv")
    checks = {e["check"] for e in audit(read_manifest(tmp_path / "m2.csv"), tmp_path)["errors"]}
    assert {"exact_duplicate", "near_duplicate", "anomaly_in_train", "group_leakage", "sha256_changed", "missing_file"} <= checks


def test_group_split_keeps_anomalies_out_of_train(tmp_path):
    rows = make_dataset(tmp_path)
    for r in rows:
        r["group_id"] = r["sample_id"][-2:]  # several images per group
    out = group_split(rows, val_ratio=0.2, test_ratio=0.2, seed=1)
    assert not [r for r in out if r["split"] == "train" and int(r["label"]) == 1]
    by_group = {}
    for r in out:
        by_group.setdefault(r["group_id"], set()).add(r["split"])
    assert all(len(s) == 1 for s in by_group.values())


def test_stress_keeps_labels(tmp_path):
    rows = make_dataset(tmp_path)
    stress = make_stress(rows, tmp_path, tmp_path, n_per_class=3, seed=0)
    assert len(stress) == 6 and {r["split"] for r in stress} == {"stress"}
    assert sorted(int(r["label"]) for r in stress) == [0, 0, 0, 1, 1, 1]
    assert all((tmp_path / r["path"]).exists() for r in stress)
