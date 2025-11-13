"""Brain crop: cut an MRI slice to the head so every source dataset has the same framing."""
import cv2
import numpy as np


def _head_blobs(img: np.ndarray):
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    # Low threshold: separates the head from the black background (Otsu would keep only a bright tumor)
    mask = (gray > max(10, 0.08 * np.percentile(gray, 99.5))).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))  # drop specks and thin text
    return cv2.connectedComponentsWithStats(mask)


def head_mask(img: np.ndarray) -> np.ndarray:
    """Boolean mask of the head (largest bright blob). All True if nothing is found."""
    n, labels, stats, _ = _head_blobs(img)
    if n < 2:
        return np.ones(img.shape[:2], bool)
    return labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])


def crop_brain(img: np.ndarray, margin: float = 0.03) -> np.ndarray:
    """Crop an RGB slice to the head and pad it to a black square. Returns the input if no head is found."""
    n, _, stats, _ = _head_blobs(img)
    if n < 2:
        return img
    x, y, w, h, area = stats[1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])]
    if area < 0.05 * img.shape[0] * img.shape[1]:
        return img
    m = int(margin * max(w, h))
    x0, y0 = max(0, x - m), max(0, y - m)
    x1, y1 = min(img.shape[1], x + w + m), min(img.shape[0], y + h + m)
    crop = img[y0:y1, x0:x1]
    side = max(crop.shape[:2])  # pad to a square, keep the aspect ratio
    out = np.zeros((side, side, 3), dtype=img.dtype)
    oy, ox = (side - crop.shape[0]) // 2, (side - crop.shape[1]) // 2
    out[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
    return out
