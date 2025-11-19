"""BraTS 2021 (Task 1): per patient, 4 MRI types + an expert tumor mask, as NIfTI files.

Mask labels: 0 background, 1 necrotic / non-enhancing core, 2 edema, 4 enhancing tumor.
The model predicts 3 overlapping regions built from them (MONAI order): TC, WT, ET.
"""
import tarfile
from pathlib import Path

import nibabel as nib
import numpy as np
from monai import transforms as T

MODALITIES = ("flair", "t1", "t1ce", "t2")
REGIONS = {"tc": (1, 4), "wt": (1, 2, 4), "et": (4,)}  # tumor core, whole tumor, enhancing tumor
KNOWN_LABELS = {0, 1, 2, 4}


def extract_tars(src: str | Path, dst: str | Path) -> Path:
    """Kaggle ships BraTS 2021 as .tar files on a read-only disk; unpack them once to a writable folder."""
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for tar in sorted(Path(src).glob("**/*.tar")):
        done = dst / f".{tar.stem}.done"
        if done.exists():
            continue
        with tarfile.open(tar) as t:
            t.extractall(dst, filter="data")
        done.touch()
    return dst


def find_cases(root: str | Path) -> list[dict]:
    """One dict per patient that has all 4 MRI types and the mask, in the format MONAI dict transforms expect."""
    cases = {}
    for seg in sorted(Path(root).glob("**/*_seg.nii.gz")):
        case_id = seg.name.removesuffix("_seg.nii.gz")
        images = [seg.with_name(f"{case_id}_{m}.nii.gz") for m in MODALITIES]
        if case_id not in cases and all(p.exists() for p in images):
            cases[case_id] = {"id": case_id, "image": [str(p) for p in images], "label": str(seg)}
    return list(cases.values())


def load_transforms() -> T.Compose:
    """Load one case: image (4, H, W, D) float, label (3, H, W, D) region masks TC / WT / ET."""
    return T.Compose([
        T.LoadImaged(keys=["image", "label"]),
        T.EnsureChannelFirstd(keys="image"),
        T.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        # Same orientation for every scan, so uploads from other scanners line up with training data
        T.Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),
    ])


def case_stats(case: dict) -> dict:
    """Header checks and tumor volumes from the mask, without loading the 4 MRI volumes."""
    seg = nib.load(case["label"])
    mask = np.rint(np.asarray(seg.dataobj)).astype(np.uint8)
    shapes = {nib.load(p).shape for p in case["image"]} | {seg.shape}
    voxel_cm3 = float(np.prod(seg.header.get_zooms()[:3])) / 1000
    stats = {
        "id": case["id"],
        "shape": "x".join(map(str, seg.shape)),
        "same_shape": len(shapes) == 1,
        "spacing": "x".join(f"{z:g}" for z in seg.header.get_zooms()[:3]),
        "labels": " ".join(map(str, np.unique(mask))),
    }
    for region, labels in REGIONS.items():
        stats[f"{region}_cm3"] = round(float(np.isin(mask, labels).sum()) * voxel_cm3, 2)
    return stats
