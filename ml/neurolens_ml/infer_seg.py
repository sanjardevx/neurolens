"""3D tumor segmentation without PyTorch: NIfTI → RAS → brain crop → normalize → ONNX sliding window → masks.

Mirrors val_transforms + predict_volume + drop_small_et (tests check it against them), so the backend
needs only numpy, nibabel and onnxruntime.
"""
from collections.abc import Callable, Sequence
from itertools import product

import nibabel as nib
import numpy as np

PATCH = 128
OVERLAP = 0.5
MIN_ET_VOXELS = 50  # chosen on val (ml/results/08_seg_eval)


def load_ras(paths: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """4 MRI files (FLAIR, T1, T1ce, T2) → image [4, X, Y, Z] float32 in RAS orientation and its affine."""
    imgs = [nib.as_closest_canonical(nib.load(str(p))) for p in paths]
    return np.stack([i.get_fdata(dtype=np.float32) for i in imgs]), imgs[0].affine


def brain_box(image: np.ndarray) -> tuple[slice, slice, slice]:
    """Bounding box of voxels above 0 in any MRI type (skull-stripped scans are 0 outside the brain)."""
    idx = np.argwhere((image > 0).any(0))
    lo, hi = idx.min(0), idx.max(0) + 1
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def normalize(image: np.ndarray) -> np.ndarray:
    """Each MRI type to mean 0 / std 1 over its brain voxels; background stays 0."""
    out = image.astype(np.float32).copy()
    for c in out:
        brain = c != 0
        if brain.any():
            std = c[brain].std()
            c[brain] = (c[brain] - c[brain].mean()) / (std if std > 0 else 1.0)
    # Training and val read float16 from the disk cache; round the same way
    return out.astype(np.float16).astype(np.float32)


def gaussian_weights(patch: int = PATCH, sigma_scale: float = 0.125) -> np.ndarray:
    """Window weights: high in the middle, low at the edges, so overlapping windows blend smoothly."""
    x = np.arange(patch, dtype=np.float32) - (patch - 1) / 2
    g = np.exp(-(x ** 2) / (2 * (patch * sigma_scale) ** 2))
    w = g[:, None, None] * g[None, :, None] * g[None, None, :]
    return np.maximum(w, max(w.min(), 1e-3)).astype(np.float32)


def window_starts(size: int, patch: int = PATCH, overlap: float = OVERLAP) -> list[int]:
    if size <= patch:
        return [0]
    step = max(int(patch * (1 - overlap)), 1)
    n = next(d for d in range(size) if d * step + patch >= size) + 1
    return [min(d * step, size - patch) for d in range(n)]


def sliding_window(volume: np.ndarray, run: Callable[[np.ndarray], np.ndarray], patch: int = PATCH,
                   overlap: float = OVERLAP, progress: Callable[[float], None] | None = None) -> np.ndarray:
    """volume [4, X, Y, Z] → blended logits [3, X, Y, Z]; run maps [1, 4, p, p, p] → [1, 3, p, p, p].
    One window at a time: a batch of 4 windows needs more than 8 GB of RAM on CPU."""
    pads = [(max(patch - s, 0) // 2, max(patch - s, 0) - max(patch - s, 0) // 2) for s in volume.shape[1:]]
    vol = np.pad(volume, [(0, 0), *pads])
    weights = gaussian_weights(patch)
    logits = np.zeros((3, *vol.shape[1:]), np.float32)
    total = np.zeros(vol.shape[1:], np.float32)
    windows = list(product(*(window_starts(s, patch, overlap) for s in vol.shape[1:])))
    for i, (x, y, z) in enumerate(windows, 1):
        win = (slice(x, x + patch), slice(y, y + patch), slice(z, z + patch))
        logits[(slice(None), *win)] += run(vol[(None, slice(None), *win)])[0] * weights
        total[win] += weights
        if progress:
            progress(i / len(windows))  # share of windows done, for the job status
    logits /= total
    return logits[(slice(None), *(slice(a, a + s) for (a, _), s in zip(pads, volume.shape[1:])))]


def drop_small_et(mask: np.ndarray, min_voxels: int = MIN_ET_VOXELS) -> np.ndarray:
    """Same rule as models.segmenter.drop_small_et: tiny enhancing-tumor blobs are mostly false alarms."""
    mask = mask.copy()
    if mask[2].sum() < min_voxels:
        mask[2] = False
    return mask


def segment(image: np.ndarray, run: Callable[[np.ndarray], np.ndarray], patch: int = PATCH,
            progress: Callable[[float], None] | None = None) -> np.ndarray:
    """RAS image [4, X, Y, Z] → region masks [3, X, Y, Z] (TC, WT, ET) on the same voxel grid."""
    box = brain_box(image)
    logits = sliding_window(normalize(image[(slice(None), *box)]), run, patch, progress=progress)
    mask = np.zeros((3, *image.shape[1:]), bool)
    mask[(slice(None), *box)] = drop_small_et(logits > 0)  # sigmoid > 0.5
    return mask


def onnx_runner(path: str, threads: int | None = None) -> Callable[[np.ndarray], np.ndarray]:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads = threads
    session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    return lambda x: session.run(None, {"image": x.astype(np.float32)})[0]


def label_map(mask: np.ndarray) -> np.ndarray:
    """TC / WT / ET masks → one BraTS label volume for viewers: 1 necrotic core, 2 edema, 4 enhancing."""
    tc, wt, et = mask.astype(bool)
    labels = np.zeros(wt.shape, np.uint8)
    labels[wt] = 2
    labels[tc] = 1
    labels[et] = 4
    return labels
