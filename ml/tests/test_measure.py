import numpy as np
import pytest

from neurolens_ml.measure import lobe_estimate, measure, tumor_location, tumor_volumes, voxel_cm3

from .helpers import fake_mask

LPS = np.diag([-1.0, -1.0, 1.0, 1.0])  # how many scanners (and BraTS) store the axes: x to the left, y to the back


def regions(labels: np.ndarray) -> np.ndarray:
    """BraTS labels → TC / WT / ET channels, like the model output."""
    return np.stack([np.isin(labels, (1, 4)), labels > 0, labels == 4])


def test_voxel_volume_follows_the_affine():
    assert voxel_cm3(np.eye(4)) == pytest.approx(0.001)
    assert voxel_cm3(np.diag([1.0, 1.0, 3.0, 1.0])) == pytest.approx(0.003)  # thick slices
    assert voxel_cm3(LPS) == pytest.approx(0.001)  # flipped axes don't change the size
    c, s = np.cos(0.3), np.sin(0.3)
    tilted = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1.5, 0], [0, 0, 0, 1]])
    assert voxel_cm3(tilted) == pytest.approx(0.0015)


def test_tumor_volumes_split_into_parts():
    v = tumor_volumes(regions(fake_mask()), voxel=0.001)  # 1536 WT, 128 TC, 16 ET voxels
    assert v == {"whole_tumor": 1.54, "tumor_core": 0.13, "enhancing": 0.02,
                 "necrosis_non_enhancing": 0.11, "edema": 1.41}
    assert tumor_volumes(regions(fake_mask()), voxel=0.003)["whole_tumor"] == pytest.approx(4.61)


@pytest.mark.parametrize("front, top, lobe", [
    (0.8, 0.6, "frontal"), (0.1, 0.6, "occipital"), (0.45, 0.3, "temporal"),
    (0.45, 0.8, "parietal"), (0.2, 0.1, "cerebellum / brainstem"),
])
def test_lobe_estimate(front, top, lobe):
    assert lobe_estimate(front, top) == lobe


def test_side_and_lobe_from_position():
    brain = np.zeros((100, 120, 90), bool)
    brain[10:90, 10:110, 5:85] = True
    wt = np.zeros_like(brain)
    wt[65:80, 85:100, 50:60] = True  # right, front, upper half
    assert tumor_location(wt, brain) == {"side": "right", "right_share": 1.0, "lobe_estimate": "frontal"}
    wt[:] = False
    wt[40:60, 50:60, 50:60] = True  # across the midline
    assert tumor_location(wt, brain)["side"] == "both sides"
    assert tumor_location(np.zeros_like(brain), brain)["side"] is None  # no tumor found


def test_orientation_is_read_from_the_file(tmp_path):
    """Same voxels, two affines: with LPS the tumor is on the patient's left and at the back; with RAS right and front."""
    nib = pytest.importorskip("nibabel")
    pytest.importorskip("monai")
    from neurolens_ml.data.brats import MODALITIES, find_cases, load_transforms

    shape = (60, 70, 50)
    img = np.zeros(shape, np.float32)
    img[5:55, 5:65, 5:45] = 500
    seg = np.zeros(shape, np.uint8)
    seg[42:50, 50:60, 30:38] = 2  # high index on axes 0 and 1, upper half
    found = {}
    for name, affine in [("lps", LPS), ("ras", np.eye(4))]:
        folder = tmp_path / name / f"BraTS2021_{name}"
        folder.mkdir(parents=True)
        for m in MODALITIES:
            nib.save(nib.Nifti1Image(img, affine), folder / f"BraTS2021_{name}_{m}.nii.gz")
        nib.save(nib.Nifti1Image(seg, affine), folder / f"BraTS2021_{name}_seg.nii.gz")
        d = load_transforms()(find_cases(tmp_path / name)[0])
        found[name] = measure(np.asarray(d["image"]), np.asarray(d["label"]), np.asarray(d["label"].affine))
    assert (found["lps"]["side"], found["lps"]["lobe_estimate"]) == ("left", "occipital")
    assert (found["ras"]["side"], found["ras"]["lobe_estimate"]) == ("right", "frontal")
    assert found["lps"]["volume_cm3"] == found["ras"]["volume_cm3"] == {
        "whole_tumor": 0.64, "tumor_core": 0.0, "enhancing": 0.0, "necrosis_non_enhancing": 0.0, "edema": 0.64}


def test_volumes_are_nested_even_if_the_model_is_not():
    mask = np.zeros((3, 10, 10, 10), bool)
    mask[1, :5] = True  # WT: 500 voxels
    mask[0, :2] = True  # TC: 200
    mask[2, 4:7] = True  # ET: 300, partly outside TC and WT
    v = tumor_volumes(mask, voxel=0.001)
    assert v["enhancing"] <= v["tumor_core"] <= v["whole_tumor"]
    assert v == {"whole_tumor": 0.7, "tumor_core": 0.5, "enhancing": 0.3, "necrosis_non_enhancing": 0.2, "edema": 0.2}
