import json
import tarfile

import numpy as np
import pytest

pytest.importorskip("monai")
nib = pytest.importorskip("nibabel")

from neurolens_ml import prep_brats  # noqa: E402
from neurolens_ml.data.brats import MODALITIES, case_stats, extract_tars, find_cases, load_transforms  # noqa: E402

SHAPE = (24, 20, 16)


def fake_mask() -> np.ndarray:
    """Edema around a core around an enhancing part, like a real BraTS mask."""
    m = np.zeros(SHAPE, np.uint8)
    m[4:20, 4:16, 4:12] = 2  # 16*12*8 = 1536 voxels
    m[8:16, 8:12, 6:10] = 1  # 8*4*4 = 128
    m[10:14, 9:11, 7:9] = 4  # 4*2*2 = 16
    return m


def write_case(root, case_id: str, skip: str | None = None) -> None:
    folder = root / case_id
    folder.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for m in MODALITIES:
        if m != skip:
            nib.save(nib.Nifti1Image(rng.random(SHAPE, np.float32) * 1000, np.eye(4)), folder / f"{case_id}_{m}.nii.gz")
    nib.save(nib.Nifti1Image(fake_mask(), np.eye(4)), folder / f"{case_id}_seg.nii.gz")


@pytest.fixture
def brats_root(tmp_path):
    root = tmp_path / "BraTS2021_Training_Data"
    write_case(root, "BraTS2021_00000")
    write_case(root, "BraTS2021_00002")
    write_case(root, "BraTS2021_00003", skip="t2")  # incomplete case is ignored
    return root


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
