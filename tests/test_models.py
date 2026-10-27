import json

import numpy as np
import pytest

from aiot.cli import main
from aiot.models import MODELS, load_artifact, load_rgb, save_artifact
from helpers import board, make_dataset


@pytest.mark.parametrize("method", ["template_z", "patch_knn"])
def test_defect_scores_higher(method, tmp_path):
    train = np.stack([load_rgb(board(s), 96) for s in range(20)])
    model = MODELS[method]().fit(train)
    normal = model.score(load_rgb(board(100), 96))[0]
    defect = model.score(load_rgb(board(101, defect=True), 96))[0]
    assert defect > normal
    save_artifact(model, {"method": method, "params": model.params()}, tmp_path / method)
    loaded, _ = load_artifact(tmp_path / method)
    assert loaded.score(load_rgb(board(101, defect=True), 96))[0] == pytest.approx(defect)


def test_cli_fit_and_evaluate(tmp_path):
    make_dataset(tmp_path)
    m, r = str(tmp_path / "manifest.csv"), str(tmp_path)
    main(["fit", "--manifest", m, "--root", r, "--method", "patch_knn", "--version", "toy", "--out", str(tmp_path / "art")])
    main(["evaluate", "--manifest", m, "--root", r, "--model", str(tmp_path / "art/toy"), "--splits", "test",
          "--latency-n", "2", "--runs", "1", "--out", str(tmp_path / "runs")])
    res = json.loads((tmp_path / "runs/toy/metrics.json").read_text())["sets"]["test"]
    assert res["auroc"] == 1.0 and res["recall"] == 1.0
