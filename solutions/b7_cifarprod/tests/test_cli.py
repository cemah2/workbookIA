"""Tests of the configuration and of the command line: invalid settings refused, exit codes, a smoke training."""

import json

import pytest
from safetensors.torch import load_file

from cifarprod.cli import main
from cifarprod.config import load_config


@pytest.mark.parametrize("override", ["epochs=0", "lr=0", "lr=5", "lr=1e-3", "lerning_rate=0.01", "widths=[]",
                                      "arch=resnet", "batch_size=true", "seed"])
def test_invalid_config_refused(override):
    with pytest.raises(ValueError):
        load_config(None, [override])


def test_overrides_win_over_the_file(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("lr: 0.1\nepochs: 3\nwidths: [4, 8]\n", encoding="utf-8")
    config = load_config(path, ["lr=0.01", "epochs=2"])
    assert (config.lr, config.epochs, config.widths) == (0.01, 2, (4, 8)), f"expected (0.01, 2, (4, 8)), got {(config.lr, config.epochs, config.widths)}"


def test_exit_code_of_an_invalid_config(tmp_path):
    code = main(["train", "--set", "epochs=0", "--runs", str(tmp_path / "runs.jsonl")])
    assert code == 2, f"expected the exit code 2 for epochs=0, got {code}"


def test_exit_code_without_data(tmp_path):
    code = main(["train", "--set", f"data={tmp_path / 'absent.npz'}", "--runs", str(tmp_path / "runs.jsonl")])
    assert code == 1, f"expected the exit code 1 without data file, got {code}"


@pytest.mark.slow
def test_smoke_train(tmp_path, synthetic_file):
    runs = tmp_path / "runs.jsonl"
    code = main(["train", "--set", f"data={synthetic_file}", "--set", "n_train=96", "--set", "widths=[4, 8]",
                 "--out", str(tmp_path / "run"), "--runs", str(runs)])
    assert code == 0, f"expected the exit code 0, got {code}"
    weights = tmp_path / "run" / "model.safetensors"
    assert weights.is_file(), f"expected the weights in {weights}"
    assert "0.weight" in load_file(weights), f"expected the weights of the model, got {sorted(load_file(weights))[:4]}"
    record = json.loads(runs.read_text(encoding="utf-8").splitlines()[-1])
    assert record["status"] == "ok", f"expected a run logged with the status ok, got {record['status']!r}"
