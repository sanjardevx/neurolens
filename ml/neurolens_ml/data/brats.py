"""BraTS 2021 (Task 1): per patient, 4 MRI types + an expert tumor mask, as NIfTI files.

Mask labels: 0 background, 1 necrotic / non-enhancing core, 2 edema, 4 enhancing tumor.
The model predicts 3 overlapping regions built from them (MONAI order): TC, WT, ET.
"""
import tarfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from monai import transforms as T
from sklearn.model_selection import train_test_split

MODALITIES = ("flair", "t1", "t1ce", "t2")
REGIONS = {"tc": (1, 4), "wt": (1, 2, 4), "et": (4,)}  # tumor core, whole tumor, enhancing tumor
KNOWN_LABELS = {0, 1, 2, 4}
PATCH = (128, 128, 128)  # random training patch; fits a T4 with a 3D U-Net and still covers most of the brain


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


def train_transforms(patch: tuple[int, int, int] = PATCH) -> T.Compose:
    """Brain crop, per-modality normalization, a random patch, flips on all 3 axes and small intensity changes."""
    return T.Compose([
        *val_transforms().transforms,
        T.SpatialPadd(keys=["image", "label"], spatial_size=patch),  # a small brain can be under 128 voxels on an axis
        T.RandSpatialCropd(keys=["image", "label"], roi_size=patch, random_size=False),
        *[T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=axis) for axis in range(3)],
        # Scanners differ in contrast: scale and shift each modality by up to 10% of its spread
        T.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0, channel_wise=True),
        T.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0, channel_wise=True),
    ])


def val_transforms() -> T.Compose:
    """Whole brain, no randomness: val and test run sliding-window over it."""
    return T.Compose([
        *load_transforms().transforms,
        # Cut the black background around the skull-stripped brain (about half of the 240x240x155 volume)
        T.CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
        # MRI intensities have no fixed unit, so each modality is scaled by its own brain voxels (background stays 0)
        T.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        # Half the size in the disk cache; the training loop casts back to float32
        T.CastToTyped(keys=["image", "label"], dtype=(np.float16, np.uint8)),
    ])


def make_splits(cases: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """70 / 15 / 15 by patient, stratified by tumor size so each split has small and large tumors.

    Cases with no enhancing tumor get their own group: the enhancing-tumor Dice treats them differently.
    """
    strata = pd.qcut(cases.wt_cm3, 4, labels=False).astype(str)
    strata[cases.et_cm3 == 0] = "no_et"
    train, rest = train_test_split(cases.id, test_size=0.30, stratify=strata, random_state=seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=strata[rest.index], random_state=seed)
    split = pd.Series("train", index=cases.index)
    split[val.index], split[test.index] = "val", "test"
    return cases[["id", "wt_cm3", "tc_cm3", "et_cm3"]].assign(split=split)[["id", "split", "wt_cm3", "tc_cm3", "et_cm3"]]


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
