import json

import pandas as pd
import pytest
import torch

pytest.importorskip("monai")
pytest.importorskip("nibabel")

from neurolens_ml import train_seg  # noqa: E402
from neurolens_ml.models.segmenter import (  # noqa: E402
    DiceBCELoss, create_segmenter, drop_small_et, predict_volume, region_dice,
)

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
                              "--data", str(brats_root), "--splits", str(splits), "--out", str(out), "--smoke",
                              "--cache-dir", str(tmp_path / "cache")])
    assert metrics["test"] is None and metrics["n_train"] == 1 and metrics["n_val"] == 1
    assert set(metrics["val"]) == {"tc", "wt", "et", "mean"}
    for f in ["segmenter_segresnet.pt", "history_seg_segresnet.csv", "curves_seg_segresnet.png", "val_cases_seg_segresnet.csv"]:
        assert (out / f).exists()
    assert json.loads((out / "metrics_seg_segresnet.json").read_text())["epochs_run"] == 2
    assert len(list((tmp_path / "cache").glob("*.pt"))) == 2  # each scan prepared once: 1 train + 1 val


def test_missing_cases_fail_early(brats_root, tmp_path):
    splits = tmp_path / "splits.csv"
    pd.DataFrame({"id": ["BraTS2021_00000", "BraTS2021_09999"], "split": ["train", "val"]}).to_csv(splits, index=False)
    with pytest.raises(FileNotFoundError, match="1 cases"):
        train_seg.split_cases(brats_root, splits)


def test_cache_gives_the_same_data(brats_root, tmp_path):
    from neurolens_ml.data.brats import find_cases, val_transforms

    case = find_cases(brats_root)[:1]
    fresh = train_seg.make_dataset(case, val_transforms(), None)[0]
    cached = train_seg.make_dataset(case, val_transforms(), tmp_path / "cache")
    first, second = cached[0], cached[0]  # second read comes from disk
    for key in ["image", "label"]:
        assert torch.equal(train_seg.plain(fresh[key]), second[key]) and torch.equal(first[key], second[key])


def test_batch_can_mix_cached_and_fresh_items(brats_root, tmp_path):
    """Fine-tuning repeats the corrected cases, so one batch holds a scan read from the cache next to a new one."""
    from monai.data import DataLoader

    from neurolens_ml.data.brats import find_cases, train_transforms

    a, b = find_cases(brats_root)[:2]
    tf = train_transforms((32, 32, 32))
    train_seg.make_dataset([a], tf, tmp_path / "cache")[0]  # a is cached, b is not
    batch = next(iter(DataLoader(train_seg.make_dataset([a, b], tf, tmp_path / "cache"), batch_size=2)))
    assert batch["image"].shape[0] == 2


def test_drop_small_et_clears_only_small_et():
    pred = torch.zeros(2, 3, 4, 4, 4, dtype=torch.bool)
    pred[:, :2, :2] = True  # TC and WT in both cases
    pred[0, 2, 0, 0, :3] = True  # case 0: 3 ET voxels
    pred[1, 2, :2] = True  # case 1: 32 ET voxels
    out = drop_small_et(pred, min_voxels=10)
    assert out[0, 2].sum() == 0 and out[1, 2].sum() == 32
    assert torch.equal(out[:, :2], pred[:, :2])  # core and whole tumor untouched
    assert torch.equal(drop_small_et(pred[0], 3), pred[0])  # single case, limit not reached


def test_min_et_tuning_picks_the_best_limit_on_val():
    from neurolens_ml.evaluate_seg import tune_min_et, with_min_et

    cases = pd.DataFrame({"et": [0.0, 0.0, 0.9, 0.8], "pred_et_voxels": [40, 120, 5000, 90],
                          "true_et_voxels": [0, 0, 6000, 150]})
    assert with_min_et(cases, 100).tolist() == [1.0, 0.0, 0.9, 0.0]  # a real small ET is lost too
    best, table = tune_min_et(cases)
    assert best == 200 and table.val_et.max() == pytest.approx((1 + 1 + 0.9 + 0) / 4)  # both false alarms gone


def test_evaluate_seg_val_then_test(brats_root, tmp_path):
    from neurolens_ml import evaluate_seg

    cfg = tmp_path / "tiny.yaml"
    cfg.write_text("name: tiny\ninit_filters: 8\ndropout: 0.0\npatch: 32\nsw_batch_size: 1\nsw_overlap: 0.5\n"
                   "amp: false\nnum_workers: 0\n")
    ckpt = tmp_path / "segmenter_tiny.pt"
    torch.save(create_segmenter(dict(init_filters=8, dropout=0.0)).state_dict(), ckpt)
    splits = tmp_path / "splits.csv"
    pd.DataFrame({"id": ["BraTS2021_00000", "BraTS2021_00002"], "split": ["val", "test"]}).to_csv(splits, index=False)
    args = ["--config", str(cfg), "--checkpoint", str(ckpt), "--data", str(brats_root), "--splits", str(splits),
            "--out", str(tmp_path / "out")]
    with pytest.raises(FileNotFoundError, match="run --split val first"):
        evaluate_seg.main([*args, "--split", "test"])
    val = evaluate_seg.main(args)
    assert val["min_et_voxels"] in evaluate_seg.MIN_ET_GRID and set(val["val"]) == {"tc", "wt", "et", "mean"}
    test = evaluate_seg.main([*args, "--split", "test"])
    assert test["n_test"] == 1 and test["min_et_voxels"] == val["min_et_voxels"]
    per_case = pd.read_csv(tmp_path / "out" / "eval_seg_test_tiny.csv")
    assert per_case.true_wt_voxels.tolist() == [1536]  # true volume counted on the cropped brain


def test_select_feedback_takes_the_worst_val_cases():
    from neurolens_ml.finetune_seg import select_feedback

    val = pd.DataFrame({"id": ["a", "b", "c", "d"], "tc": [0.9, 0.1, 0.5, 0.95],
                        "wt": [0.9, 0.2, 0.6, 0.95], "et": [0.9, 0.0, 0.4, 0.95]})
    sel = select_feedback(val, k=2).set_index("id")
    assert sel.role.to_dict() == {"a": "val_rest", "b": "feedback", "c": "feedback", "d": "val_rest"}
    assert sel.loc["b", "v1_mean"] == pytest.approx(0.1)


def test_finetune_smoke_run_with_simulated_and_real_feedback(brats_root, tmp_path):
    from .helpers import write_case

    from neurolens_ml import finetune_seg
    from neurolens_ml.models.segmenter import create_segmenter

    for i in ("00004", "00005"):
        write_case(brats_root, f"BraTS2021_{i}")
    splits, selection, ckpt = tmp_path / "splits.csv", tmp_path / "selection.csv", tmp_path / "segmenter_segresnet.pt"
    pd.DataFrame({"id": ["BraTS2021_00000", "BraTS2021_00002", "BraTS2021_00004", "BraTS2021_00005"],
                  "split": ["train", "val", "val", "test"]}).to_csv(splits, index=False)
    pd.DataFrame({"id": ["BraTS2021_00002", "BraTS2021_00004"], "role": ["feedback", "val_rest"]}).to_csv(selection, index=False)
    config = finetune_seg.DATA.parent / "configs" / "seg_finetune.yaml"
    cfg = finetune_seg.load_config(config) | train_seg.SMOKE_OVERRIDES
    torch.save(create_segmenter(cfg).state_dict(), ckpt)  # stands in for v1
    real = tmp_path / "export" / "train"
    write_case(real, "NL-AB12CD")  # a doctor-corrected case in the API's export layout
    out = tmp_path / "out"
    m = finetune_seg.main(["--config", str(config), "--data", str(brats_root), "--splits", str(splits),
                           "--selection", str(selection), "--checkpoint", str(ckpt), "--feedback-dir", str(real),
                           "--out", str(out), "--smoke", "--cache-dir", str(tmp_path / "cache")])
    assert m["test"] is None and m["feedback"] | {"source": ""} == {"simulated": 1, "real": 1, "repeat": 5, "source": ""}
    assert m["v1"]["val_rest"]["n"] == 1 and m["v2"]["feedback"]["n"] == 1
    # The starting model is saved first, so v2 is never worse than v1 on the selection cases
    assert m["v2"]["val_rest"]["mean"] >= m["v1"]["val_rest"]["mean"]
    for f in ["segmenter_segresnet_fb1.pt", "history_seg_segresnet_fb1.csv", "compare_val_rest_segresnet_fb1.csv",
              "compare_feedback_segresnet_fb1.csv", "metrics_seg_segresnet_fb1.json"]:
        assert (out / f).exists()
