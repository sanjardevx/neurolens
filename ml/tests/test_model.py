import numpy as np
import pytest
import torch

from neurolens_ml.config import CLASSES
from neurolens_ml.data import crop_brain, make_transforms, preprocess
from neurolens_ml.models import ClassifierWithCam, create_classifier

from .helpers import fake_slice, rgb


@pytest.fixture(scope="module")
def wrapper():
    torch.manual_seed(0)
    return ClassifierWithCam(create_classifier("efficientnet_b0", pretrained=False)).eval()


def test_output_shapes(wrapper):
    logits, cams = wrapper(torch.randn(2, 3, 64, 64))
    assert logits.shape == (2, len(CLASSES))
    assert cams.shape == (2, len(CLASSES), 2, 2)  # 64 px / 32 = 2


def test_cam_average_gives_the_logits(wrapper):
    """For a GAP + linear head, the spatial mean of each CAM plus the bias is that class's logit."""
    with torch.no_grad():
        logits, cams = wrapper(torch.randn(2, 3, 64, 64))
    bias = wrapper.model.get_classifier().bias
    assert torch.allclose(cams.mean((2, 3)) + bias, logits, atol=1e-4)


def test_numpy_preprocess_matches_training_pipeline():
    cfg = dict(img_size=64, strong_aug=False)
    img = rgb(fake_slice())
    _, eval_tf = make_transforms(cfg)
    expected = eval_tf(image=crop_brain(img))["image"].numpy()
    got = preprocess(img, 64)[0]
    assert got.shape == (3, 64, 64) and got.dtype == np.float32
    assert np.abs(got - expected).max() < 1e-4


def test_classify_matches_the_torch_model(wrapper):
    from neurolens_ml.infer_cls import classify

    img = rgb(fake_slice())
    run = lambda x: tuple(t.detach().numpy() for t in wrapper(torch.from_numpy(x)))  # noqa: E731
    out = classify(img, run, size=64)
    with torch.no_grad():
        logits, _ = wrapper(torch.from_numpy(preprocess(img, 64)))
    probs = torch.softmax(logits[0], 0).numpy()
    assert out["tumor_type"] == CLASSES[probs.argmax()] and abs(out["confidence"] - probs.max()) < 1e-4
    assert abs(sum(out["probabilities"].values()) - 1) < 1e-3
    assert out["gradcam"].shape == (64, 64, 3) and out["gradcam"].dtype == np.uint8
