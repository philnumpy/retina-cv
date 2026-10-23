from __future__ import annotations

import argparse
import json

import numpy as np
import pytest
import torch
import yaml

from src.train_cls import class_weights, outputs_to_grades
from src.utils.common import load_config, print_table, save_json, seed_everything


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "my_exp.yaml"
    path.write_text(yaml.safe_dump({"lr": 0.001, "seed": 3}))
    return path


def args(config, seed=None, smoke=False, set_=()):
    return argparse.Namespace(config=str(config), seed=seed, smoke=smoke, set=list(set_))


def test_run_name_from_file_and_seed(config):
    assert load_config(args(config))["run_name"] == "my_exp"
    assert load_config(args(config, seed=3))["run_name"] == "my_exp"  # the config's own seed keeps the plain name
    cfg = load_config(args(config, seed=1))
    assert cfg["run_name"] == "my_exp__seed1" and cfg["seed"] == 1


def test_smoke_prefix_and_overrides(config):
    cfg = load_config(args(config, smoke=True, set_=["lr=0.5", "workers=0", "arch=unet", "flag=true"]))
    assert cfg["run_name"] == "smoke_my_exp" and cfg["smoke"] is True
    assert cfg["lr"] == 0.5 and cfg["workers"] == 0 and cfg["arch"] == "unet" and cfg["flag"] is True


def test_seed_everything_is_reproducible():
    seed_everything(7)
    a = (np.random.rand(3), torch.rand(3))
    seed_everything(7)
    b = (np.random.rand(3), torch.rand(3))
    np.testing.assert_array_equal(a[0], b[0])
    assert torch.equal(a[1], b[1])


def test_save_json_handles_numpy(tmp_path):
    path = tmp_path / "m.json"
    save_json(path, {"f": np.float32(0.5), "i": np.int64(3), "a": np.arange(3), "p": tmp_path})
    data = json.loads(path.read_text())
    assert data["f"] == 0.5 and data["i"] == 3 and data["a"] == [0, 1, 2] and data["p"] == str(tmp_path)


def test_print_table(capsys):
    text = print_table("title", {"dice": 0.81234567, "n": 20})
    assert "0.8123" in text and "20" in text and "title" in capsys.readouterr().out


def test_class_weights_favour_rare_grades():
    w = class_weights(np.array([0] * 50 + [1] * 10 + [2] * 25 + [3] * 5 + [4] * 10))
    assert w.mean().item() == pytest.approx(1.0)
    assert w[3] > w[1] > w[0]


def test_outputs_to_grades():
    logits = torch.tensor([[0.1, 3.0, 0, 0, 0], [0, 0, 0, 0, 9.0]])
    np.testing.assert_array_equal(outputs_to_grades(logits, "classification"), [1, 4])
    scores = torch.tensor([[-0.7], [1.4], [2.6], [7.0]])
    np.testing.assert_array_equal(outputs_to_grades(scores, "regression"), [0, 1, 3, 4])
