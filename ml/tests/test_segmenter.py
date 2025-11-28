import json

import pandas as pd
import pytest
import torch

pytest.importorskip("monai")
pytest.importorskip("nibabel")

from neurolens_ml import train_seg  # noqa: E402
from neurolens_ml.models.segmenter import DiceBCELoss, create_segmenter, predict_volume, region_dice  # noqa: E402

TINY = dict(init_filters=8, dropout=0.0)


def test_segmenter_output_shape():
    model = create_segmenter(TINY).eval()
    with torch.no_grad():
        assert model(torch.randn(1, 4, 32, 32, 32)).shape == (1, 3, 32, 32, 32)


def test_sliding_window_keeps_the_brain_size():
    model = create_segmenter(TINY).eval()
    with torch.no_grad():
        pred = predict_volume(model, torch.randn(1, 4, 40, 36, 28), patch=(32, 32, 32), sw_batch_size=2)
    assert pred.shape == (1, 3, 40, 36, 28) and pred.dtype == torch.bool  # not divisible by 32: padded, then cut back


def test_loss_rewards_the_right_answer():
    target = torch.zeros(1, 3, 8, 8, 8)
    target[:, :, 2:6, 2:6, 2:6] = 1
    loss = DiceBCELoss()
    good, bad = loss(target * 20 - 10, target), loss(10 - target * 20, target)
    assert 0 <= good < 0.05 < bad


def test_region_dice_with_empty_regions():
    target = torch.zeros(3, 4, 4, 4, dtype=torch.bool)
    target[0, :2] = True  # TC: 32 voxels; WT and ET empty
    pred = torch.zeros_like(target)
    pred[0, :1] = True  # TC: half found
    pred[2, 0, 0, 0] = True  # ET: one voxel where there is none
    dice = region_dice(pred, target)
    assert dice.tolist() == pytest.approx([2 * 16 / 48, 1.0, 0.0])  # WT: both empty → 1, ET: false alarm → 0


def test_train_seg_smoke_run(brats_root, tmp_path):
    splits = tmp_path / "splits.csv"
    pd.DataFrame({"id": ["BraTS2021_00000", "BraTS2021_00002"], "split": ["train", "val"]}).to_csv(splits, index=False)
    out = tmp_path / "out"
    metrics = train_seg.main(["--config", str(train_seg.DATA.parent / "configs" / "seg_segresnet.yaml"),
                              "--data", str(brats_root), "--splits", str(splits), "--out", str(out), "--smoke"])
    assert metrics["test"] is None and metrics["n_train"] == 1 and metrics["n_val"] == 1
    assert set(metrics["val"]) == {"tc", "wt", "et", "mean"}
    for f in ["segmenter_segresnet.pt", "history_seg_segresnet.csv", "curves_seg_segresnet.png", "val_cases_seg_segresnet.csv"]:
        assert (out / f).exists()
    assert json.loads((out / "metrics_seg_segresnet.json").read_text())["epochs_run"] == 2


def test_missing_cases_fail_early(brats_root, tmp_path):
    splits = tmp_path / "splits.csv"
    pd.DataFrame({"id": ["BraTS2021_00000", "BraTS2021_09999"], "split": ["train", "val"]}).to_csv(splits, index=False)
    with pytest.raises(FileNotFoundError, match="1 cases"):
        train_seg.split_cases(brats_root, splits)
