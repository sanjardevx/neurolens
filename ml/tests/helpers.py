from pathlib import Path

import numpy as np


def fake_slice(h: int = 120, w: int = 100, k: int = 0) -> np.ndarray:
    """Grayscale 'MRI': black background, a gray head, a bright blob whose place depends on the class."""
    img = np.zeros((h, w), np.uint8)
    img[15:-15, 20:-20] = 70
    img[30 + 10 * k: 45 + 10 * k, 35:55] = 220
    return img


def rgb(gray: np.ndarray) -> np.ndarray:
    return np.repeat(gray[..., None], 3, axis=2)


# --- fake BraTS case (NIfTI) ---
SHAPE = (24, 20, 16)
BRAIN = (slice(2, 22), slice(2, 18), slice(2, 14))  # skull-stripped: zeros outside, 20 x 16 x 12 brain


def fake_mask() -> np.ndarray:
    """Edema around a core around an enhancing part, like a real BraTS mask."""
    m = np.zeros(SHAPE, np.uint8)
    m[4:20, 4:16, 4:12] = 2  # 16*12*8 = 1536 voxels
    m[8:16, 8:12, 6:10] = 1  # 8*4*4 = 128
    m[10:14, 9:11, 7:9] = 4  # 4*2*2 = 16
    return m


def write_case(root: Path, case_id: str, skip: str | None = None) -> None:
    import nibabel as nib

    from neurolens_ml.data.brats import MODALITIES

    folder = root / case_id
    folder.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for m in MODALITIES:
        if m != skip:
            img = np.zeros(SHAPE, np.float32)
            img[BRAIN] = 100 + rng.random(img[BRAIN].shape, np.float32) * 1000
            nib.save(nib.Nifti1Image(img, np.eye(4)), folder / f"{case_id}_{m}.nii.gz")
    nib.save(nib.Nifti1Image(fake_mask(), np.eye(4)), folder / f"{case_id}_seg.nii.gz")
