import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("monai")
nib = pytest.importorskip("nibabel")

from neurolens_ml import plot_brats, prep_brats  # noqa: E402
from neurolens_ml.data.brats import (  # noqa: E402
    case_stats, extract_tars, find_cases, load_transforms, make_splits, train_transforms, val_transforms,
)

from .helpers import SHAPE  # noqa: E402

SPLITS = Path(__file__).resolve().parents[1] / "data" / "brats_splits.csv"
CASES = Path(__file__).resolve().parents[1] / "data" / "brats_cases.csv"


def test_find_cases_keeps_only_complete_cases(brats_root):
    cases = find_cases(brats_root)
    assert [c["id"] for c in cases] == ["BraTS2021_00000", "BraTS2021_00002"]
    assert [p.split("_")[-1] for p in cases[0]["image"]] == ["flair.nii.gz", "t1.nii.gz", "t1ce.nii.gz", "t2.nii.gz"]


def test_monai_load_gives_nested_region_masks(brats_root):
    d = load_transforms()(find_cases(brats_root)[0])
    assert tuple(d["image"].shape) == (4, *SHAPE)
    assert tuple(d["label"].shape) == (3, *SHAPE)
    tc, wt, et = d["label"].bool()
    assert (int(tc.sum()), int(wt.sum()), int(et.sum())) == (128, 1536, 16)


def test_case_stats_volumes(brats_root):
    s = case_stats(find_cases(brats_root)[0])
    assert s["same_shape"] and s["spacing"] == "1x1x1" and s["labels"] == "0 1 2 4"
    assert (s["wt_cm3"], s["tc_cm3"], s["et_cm3"]) == (1.54, 0.13, 0.02)  # 1 voxel = 1 mm³


def test_extract_tars_once(brats_root, tmp_path):
    src = tmp_path / "input"
    src.mkdir()
    with tarfile.open(src / "BraTS2021_Training_Data.tar", "w") as t:
        t.add(brats_root, arcname=brats_root.name)
    dst = extract_tars(src, tmp_path / "unpacked")
    assert len(find_cases(dst)) == 2
    (src / "BraTS2021_Training_Data.tar").unlink()
    assert extract_tars(src, dst) == dst  # second call skips, even without the tar


def test_prep_cli_writes_report(brats_root, tmp_path):
    report = prep_brats.main(["--data", str(brats_root), "--out", str(tmp_path / "out")])
    assert report["n_cases"] == 2 and report["unknown_labels"] == []
    assert report["monai_check"]["regions_nest"]
    saved = json.loads((tmp_path / "out" / "brats_report.json").read_text())
    assert saved["label_values"] == [0, 1, 2, 4]
    assert (tmp_path / "out" / "brats_cases.csv").exists()


def test_val_transforms_crop_brain_and_normalize(brats_root):
    d = val_transforms()(find_cases(brats_root)[0])
    image, label = d["image"], d["label"]
    assert tuple(image.shape) == (4, 20, 16, 12)  # background cut, brain kept
    assert int(label[1].sum()) == 1536  # no tumor voxel lost
    for channel in image:
        assert abs(float(channel.mean())) < 1e-3 and abs(float(channel.std()) - 1) < 0.01


def test_train_transforms_give_random_patches(brats_root):
    case = find_cases(brats_root)[0]
    tf = train_transforms(patch=(16, 16, 16))
    tf.set_random_state(seed=0)
    d = tf(case)
    assert tuple(d["image"].shape) == (4, 16, 16, 16) and tuple(d["label"].shape) == (3, 16, 16, 16)
    tc, wt, et = d["label"].bool()
    assert (et <= tc).all() and (tc <= wt).all()  # flips move image and mask together
    big = train_transforms(patch=(32, 32, 32))(case)  # bigger than the brain: padded
    assert tuple(big["image"].shape) == (4, 32, 32, 32)


def test_make_splits_stratified_and_repeatable():
    rng = np.random.default_rng(0)
    cases = pd.DataFrame({"id": [f"c{i}" for i in range(200)], "wt_cm3": rng.uniform(3, 300, 200),
                          "tc_cm3": 10.0, "et_cm3": np.where(np.arange(200) < 20, 0.0, 5.0)})
    s = make_splits(cases)
    assert s.split.value_counts().to_dict() == {"train": 140, "val": 30, "test": 30}
    assert s[s.et_cm3 == 0].split.value_counts().to_dict() == {"train": 14, "val": 3, "test": 3}
    assert s.equals(make_splits(cases))


def test_committed_brats_splits_are_by_patient():
    splits, cases = pd.read_csv(SPLITS), pd.read_csv(CASES)
    assert not splits.id.duplicated().any()
    assert set(splits.id) == set(cases.id)
    share = splits.split.value_counts(normalize=True)
    assert share["train"] == pytest.approx(0.70, abs=0.01)
    assert share["val"] == pytest.approx(0.15, abs=0.01) and share["test"] == pytest.approx(0.15, abs=0.01)


def test_pick_cases_spans_tumor_sizes():
    cases = pd.DataFrame({"id": list("abcdefghi"), "wt_cm3": [5, 90, 10, 300, 50, 70, 120, 200, 30],
                          "et_cm3": [1, 1, 1, 1, 0, 1, 1, 1, 1]})
    assert plot_brats.pick_cases(cases) == ["a", "i", "f", "g", "d", "e"]  # 5, 30, 70, 120, 300 cm³ + the no-ET case


def test_plot_cli_writes_figures(brats_root, tmp_path):
    cases_csv = tmp_path / "cases.csv"
    pd.DataFrame([case_stats(c) for c in find_cases(brats_root)]).to_csv(cases_csv, index=False)
    out = tmp_path / "plots"
    checks = plot_brats.main(["--data", str(brats_root), "--cases", str(cases_csv), "--out", str(out)])
    assert (out / "brats_slices.png").exists() and (out / "brats_patch.png").exists()
    assert [c["tumor_inside_brain"] for c in checks["slices"]] == [1.0, 1.0]
    assert checks["patch"]["raw"] == checks["patch"]["training patch"] == 1.0
