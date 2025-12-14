"""The torch-free CPU path (infer_seg) must give what the validated MONAI / PyTorch path gives."""
import numpy as np
import pytest
import torch

pytest.importorskip("monai")
nib = pytest.importorskip("nibabel")

from monai.inferers import sliding_window_inference  # noqa: E402

from neurolens_ml import infer_seg  # noqa: E402
from neurolens_ml.data.brats import find_cases, val_transforms  # noqa: E402
from neurolens_ml.models.segmenter import create_segmenter, drop_small_et, predict_volume  # noqa: E402
from neurolens_ml.train_seg import plain  # noqa: E402

from .helpers import BRAIN, SHAPE, fake_mask  # noqa: E402

PATCH = 16


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    model = create_segmenter(dict(init_filters=8, dropout=0.0)).eval()
    return model, lambda x: model(torch.from_numpy(x)).detach().numpy()


@pytest.fixture
def lps_case(tmp_path):
    """A fake case stored with flipped axes (LPS), like many real scans."""
    folder = tmp_path / "BraTS2021_00000"
    folder.mkdir()
    rng, affine = np.random.default_rng(1), np.diag([-1.0, -1.0, 1.0, 1.0])
    for m in ["flair", "t1", "t1ce", "t2"]:
        img = np.zeros(SHAPE, np.float32)
        img[BRAIN] = 100 + rng.random(img[BRAIN].shape, np.float32) * 1000
        nib.save(nib.Nifti1Image(img, affine), folder / f"BraTS2021_00000_{m}.nii.gz")
    nib.save(nib.Nifti1Image(fake_mask(), affine), folder / "BraTS2021_00000_seg.nii.gz")
    return find_cases(tmp_path)[0]


def test_numpy_preprocessing_matches_val_transforms(lps_case):
    image, affine = infer_seg.load_ras(lps_case["image"])
    expected = val_transforms()(lps_case)
    got = infer_seg.normalize(image[(slice(None), *infer_seg.brain_box(image))])
    assert got.shape == tuple(expected["image"].shape) == (4, 20, 16, 12)
    assert np.abs(got - np.asarray(expected["image"], np.float32)).max() < 1e-2  # float16 rounding at most


def test_window_starts_match_monai():
    from monai.data.utils import dense_patch_slices
    from monai.inferers.utils import _get_scan_interval

    for size in [16, 20, 33, 145, 160]:
        interval = _get_scan_interval((size,), (PATCH,), 1, (0.5,))
        monai_starts = [s[0].start for s in dense_patch_slices((size,), (PATCH,), interval)]
        assert infer_seg.window_starts(size, PATCH) == monai_starts


def test_gaussian_weights_match_monai():
    from monai.data.utils import compute_importance_map

    assert np.allclose(infer_seg.gaussian_weights(PATCH), compute_importance_map((PATCH,) * 3, "gaussian").numpy())


def test_sliding_window_matches_monai(tiny):
    model, run = tiny
    volume = np.random.default_rng(0).standard_normal((4, 40, 36, 12)).astype(np.float32)  # 12 < patch: padded
    with torch.no_grad():
        expected = sliding_window_inference(torch.from_numpy(volume)[None], (PATCH,) * 3, 1, model, overlap=0.5,
                                            mode="gaussian")[0].numpy()
    assert np.abs(infer_seg.sliding_window(volume, run, PATCH) - expected).max() < 1e-4


def test_segment_matches_the_torch_pipeline(tiny, lps_case):
    model, run = tiny
    image, _ = infer_seg.load_ras(lps_case["image"])
    d = val_transforms()(lps_case)
    with torch.no_grad():
        expected = predict_volume(model, plain(d["image"]).float()[None], (PATCH,) * 3, sw_batch_size=1)[0]
    expected = drop_small_et(expected, infer_seg.MIN_ET_VOXELS).numpy()
    box = infer_seg.brain_box(image)
    mask = infer_seg.segment(image, run, PATCH)
    assert np.array_equal(mask[(slice(None), *box)], expected)
    assert not mask.any() or mask[(slice(None), *box)].sum() == mask.sum()  # nothing outside the brain box


def test_drop_small_et_same_rule_as_torch():
    mask = np.zeros((3, 8, 8, 8), bool)
    mask[:, :4, :4, :2] = True  # 32 ET voxels
    for limit in [10, 32, 33]:
        expected = drop_small_et(torch.from_numpy(mask), limit).numpy()
        assert np.array_equal(infer_seg.drop_small_et(mask, limit), expected)
