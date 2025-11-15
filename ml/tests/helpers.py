import numpy as np


def fake_slice(h: int = 120, w: int = 100, k: int = 0) -> np.ndarray:
    """Grayscale 'MRI': black background, a gray head, a bright blob whose place depends on the class."""
    img = np.zeros((h, w), np.uint8)
    img[15:-15, 20:-20] = 70
    img[30 + 10 * k: 45 + 10 * k, 35:55] = 220
    return img


def rgb(gray: np.ndarray) -> np.ndarray:
    return np.repeat(gray[..., None], 3, axis=2)
