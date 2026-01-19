"""Doctor-corrected 3D masks: check an edited mask from the viewer and measure it again."""
import gzip
from pathlib import Path

import nibabel as nib
import numpy as np
from neurolens_ml.measure import tumor_volumes, voxel_cm3

LABELS = {0, 1, 2, 4}  # BraTS: 1 necrotic core, 2 edema, 4 enhancing
CORRECTED = "mask_corrected.nii.gz"


class MaskError(ValueError):
    """The edited mask can't be used; the message is shown to the doctor."""


def read_corrected(data: bytes, ai_mask: Path) -> tuple[np.ndarray, np.ndarray]:
    """Edited mask (NIfTI from the viewer, any orientation) → labels on the AI mask's grid + that grid's affine."""
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    try:
        img = nib.Nifti1Image.from_bytes(data)
    except Exception as e:
        raise MaskError("the file is not a NIfTI mask") from e
    ref = nib.load(ai_mask)
    # The viewer saves in the scan's own orientation; the AI mask is stored in RAS, so compare in RAS
    img = nib.as_closest_canonical(img)
    if img.shape != ref.shape:
        raise MaskError(f"the mask is {img.shape}, the scan is {ref.shape}")
    if not np.allclose(img.affine, ref.affine, atol=0.01):
        raise MaskError("the mask does not line up with the scan")
    labels = np.rint(np.asarray(img.dataobj)).astype(np.int16)
    if extra := set(np.unique(labels).tolist()) - LABELS:
        raise MaskError(f"unknown mask labels {sorted(extra)}; only 1, 2 and 4 are allowed")
    return labels.astype(np.uint8), ref.affine


def save_corrected(labels: np.ndarray, affine: np.ndarray, out_dir: Path) -> dict[str, float]:
    """Store next to the AI mask (the original is kept for retraining) and return the new volumes."""
    nib.save(nib.Nifti1Image(labels, affine), out_dir / CORRECTED)
    et = labels == 4
    tc = (labels == 1) | et
    return tumor_volumes(np.stack([tc, labels > 0, et]), voxel_cm3(affine))


def changed_voxels(labels: np.ndarray, ai_mask: Path) -> int:
    return int((labels != np.asarray(nib.load(ai_mask).dataobj)).sum())
