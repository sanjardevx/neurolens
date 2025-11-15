import numpy as np

from neurolens_ml.data import crop_brain, head_mask

from .helpers import fake_slice, rgb


def test_crop_is_square_and_cuts_background():
    img = rgb(fake_slice(h=120, w=100))
    out = crop_brain(img)
    assert out.shape[0] == out.shape[1]
    assert out.shape[0] < 120  # the black border is gone
    assert out.dtype == img.dtype


def test_crop_keeps_the_head():
    img = rgb(fake_slice())
    out = crop_brain(img)
    assert out.max() == 220  # the bright blob is still inside


def test_black_image_is_returned_unchanged():
    img = np.zeros((64, 64, 3), np.uint8)
    assert crop_brain(img) is img


def test_head_mask_finds_the_head():
    img = rgb(fake_slice(h=120, w=100))
    mask = head_mask(img)
    assert mask.shape == (120, 100)
    assert mask[60, 50] and not mask[2, 2]
