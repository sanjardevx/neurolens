"""End-to-end run of the train CLI on the fake dataset (smoke settings, a few seconds)."""
import json
from pathlib import Path

from neurolens_ml.train import lr_factor, main

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "cls_b0.yaml"


def test_lr_warmup_then_cosine():
    assert lr_factor(0, warmup=1, total=10) == 0.5
    assert lr_factor(1, warmup=1, total=10) == 1.0
    assert lr_factor(10, warmup=1, total=10) < 0.05


def test_train_cli_writes_all_outputs(fake_data, tmp_path):
    data, splits = fake_data
    main(["--config", str(CONFIG), "--data", str(data), "--splits", str(splits), "--out", str(tmp_path), "--smoke"])
    name = "baseline_crop"
    for f in [f"classifier_{name}.pt", f"history_{name}.csv", f"curves_{name}.png", f"confusion_val_{name}.png"]:
        assert (tmp_path / f).exists(), f
    metrics = json.loads((tmp_path / f"metrics_{name}.json").read_text())
    assert metrics["test"] is None  # training never opens the test split
    assert metrics["n_val"] == 8 and 0 <= metrics["val"]["acc"] <= 1
