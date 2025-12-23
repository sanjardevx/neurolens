"""Read uploaded scans, keep only pixels + geometry, and check they look like a brain MRI the models can use.

Privacy by design: DICOM headers are never stored. A DICOM series is turned into a NIfTI volume
(pixels + position in space), NIfTI files are re-saved with a fresh header, and images are re-encoded
as PNG (no EXIF). Only a short allowlist of technical tags (no names, IDs, dates or free text) is kept.
"""
import gzip
import io
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
from PIL import Image

MODALITIES = ("flair", "t1", "t1ce", "t2")  # the 3D model needs all four, co-registered
# Technical DICOM tags we keep. Everything else (name, ID, birth date, hospital, dates, descriptions) is dropped.
DICOM_KEEP = ("Modality", "BodyPartExamined", "MagneticFieldStrength", "SliceThickness", "RepetitionTime", "EchoTime")
BRAIN_BODY_PARTS = {"BRAIN", "HEAD", "SKULL", "HEADNECK"}
LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])


class ScanError(ValueError):
    """The upload can't be used; the message is shown to the doctor."""


# ---------- reading ----------

def read_nifti(data: bytes, name: str) -> tuple[np.ndarray, np.ndarray]:
    """NIfTI bytes → (3D float32 volume in RAS orientation, affine)."""
    suffix = ".nii.gz" if name.lower().endswith(".gz") else ".nii"
    try:
        img = nib.Nifti1Image.from_bytes(gzip.decompress(data) if suffix == ".nii.gz" else data)
    except Exception as e:  # nibabel raises many types for broken files
        raise ScanError(f"{name}: not a readable NIfTI file") from e
    img = nib.as_closest_canonical(img)
    vol = np.asarray(img.dataobj, dtype=np.float32)
    if vol.ndim == 4 and vol.shape[3] == 1:
        vol = vol[..., 0]
    if vol.ndim != 3:
        raise ScanError(f"{name}: expected a 3D volume, got {vol.ndim}D")
    return vol, img.affine


def read_dicom_zip(data: bytes, name: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """A zip with one DICOM series → (3D float32 volume in RAS, affine, allowlisted technical tags)."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        raise ScanError(f"{name}: not a zip file") from e
    slices = []
    for info in zf.infolist():
        if info.is_dir():
            continue
        try:
            ds = pydicom.dcmread(io.BytesIO(zf.read(info)))
        except pydicom.errors.InvalidDicomError:
            continue  # skip readme files, DICOMDIR leftovers, etc.
        if "PixelData" in ds:
            slices.append(ds)
    if len(slices) < 2:
        raise ScanError(f"{name}: no DICOM series found (need 2+ slices)")
    if len({ds.SeriesInstanceUID for ds in slices}) > 1:
        raise ScanError(f"{name}: more than one series in the zip; upload one series per MRI type")
    first = slices[0]
    if first.get("Modality") != "MR":
        raise ScanError(f"{name}: modality is {first.get('Modality')}, not MR")
    part = str(first.get("BodyPartExamined", "")).upper()
    if part and part not in BRAIN_BODY_PARTS:
        raise ScanError(f"{name}: body part is {part}, not brain")
    vol, affine = _stack_slices(slices, name)
    meta = {k: _plain(first.get(k)) for k in DICOM_KEEP if first.get(k) is not None}
    return vol, affine, meta


def _stack_slices(slices: list, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Sort slices along the slice normal and build the voxel → patient-space affine (DICOM LPS → NIfTI RAS)."""
    iop = np.array(slices[0].ImageOrientationPatient, dtype=float)
    row_dir, col_dir = iop[:3], iop[3:]  # direction of increasing column index, of increasing row index
    normal = np.cross(row_dir, col_dir)
    slices = sorted(slices, key=lambda ds: float(np.dot(normal, np.array(ds.ImagePositionPatient, dtype=float))))
    try:
        # Pixel [row, col] → voxel [i = col, j = row], the NIfTI order
        vol = np.stack([(ds.pixel_array.astype(np.float32) * float(ds.get("RescaleSlope", 1))
                         + float(ds.get("RescaleIntercept", 0))).T for ds in slices], axis=-1)
    except Exception as e:  # compressed transfer syntaxes need extra decoders
        raise ScanError(f"{name}: can't read the pixel data (compressed DICOM is not supported)") from e
    dy, dx = (float(v) for v in slices[0].PixelSpacing)  # spacing between rows, between columns
    first = np.array(slices[0].ImagePositionPatient, dtype=float)
    step = (np.array(slices[-1].ImagePositionPatient, dtype=float) - first) / (len(slices) - 1)
    lps = np.eye(4)
    lps[:3, 0], lps[:3, 1], lps[:3, 2], lps[:3, 3] = row_dir * dx, col_dir * dy, step, first
    img = nib.as_closest_canonical(nib.Nifti1Image(vol, LPS_TO_RAS @ lps))
    return np.asarray(img.dataobj, dtype=np.float32), img.affine


def _plain(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def read_image(data: bytes, name: str) -> np.ndarray:
    """JPG / PNG → uint8 RGB array (EXIF and other metadata are not kept)."""
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as e:
        raise ScanError(f"{name}: not a readable image") from e
    return np.asarray(img.convert("RGB"))


def read_volume(data: bytes, name: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """One MRI type: a .nii / .nii.gz file or a .zip of DICOM slices."""
    lower = name.lower()
    if lower.endswith((".nii", ".nii.gz")):
        return (*read_nifti(data, name), {})
    if lower.endswith(".zip"):
        return read_dicom_zip(data, name)
    raise ScanError(f"{name}: use .nii, .nii.gz, or a .zip of DICOM slices")


# ---------- quality checks ----------

def check_volumes(volumes: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    """The 3D model was trained on BraTS-like input: 4 co-registered, skull-stripped MRI types."""
    shapes = {v.shape for v, _ in volumes.values()}
    if len(shapes) > 1:
        raise ScanError(f"the 4 MRI types have different sizes {sorted(shapes)}; they must be co-registered")
    affines = [a for _, a in volumes.values()]
    if any(not np.allclose(a, affines[0], atol=1e-2) for a in affines):
        raise ScanError("the 4 MRI types are not in the same position; they must be co-registered")
    shape = shapes.pop()
    spacing = np.sqrt((affines[0][:3, :3] ** 2).sum(0))
    if min(shape) < 32 or max(shape) > 512:
        raise ScanError(f"volume size {shape} is outside 32-512 voxels per axis")
    if spacing.min() < 0.3 or spacing.max() > 7:
        raise ScanError(f"voxel size {np.round(spacing, 2).tolist()} mm is outside 0.3-7 mm")
    for m, (v, _) in volumes.items():
        if not np.isfinite(v).all():
            raise ScanError(f"{m}: contains NaN or infinite values")
        if v.max() <= v.min():
            raise ScanError(f"{m}: the volume is empty (one constant value)")
        # Skull-stripped scans are exactly 0 outside the brain (BraTS: ~65% of voxels)
        if (v == 0).mean() < 0.3:
            raise ScanError(f"{m}: does not look skull-stripped (the 3D model needs brain-only, BraTS-style scans)")


def check_image(img: np.ndarray) -> None:
    """Basic check that a 2D upload looks like an MRI slice. Not a guarantee: the doctor reviews every case."""
    h, w = img.shape[:2]
    if min(h, w) < 128 or max(h, w) > 4096:
        raise ScanError(f"image size {w}x{h} is outside 128-4096 px")
    rgb = img.astype(np.int16)
    if np.abs(rgb[..., 0] - rgb[..., 1]).mean() > 8 or np.abs(rgb[..., 1] - rgb[..., 2]).mean() > 8:
        raise ScanError("the image is in color; MRI slices are grayscale")
    gray = rgb.mean(2)
    if (gray < 20).mean() < 0.1:
        raise ScanError("no dark background around the head; this does not look like an MRI slice")
    if gray.std() < 10:
        raise ScanError("the image is almost flat (no structure)")


# ---------- saving (fresh files, no source metadata) ----------

def save_volume(vol: np.ndarray, affine: np.ndarray, path: Path) -> None:
    """New header from pixels + affine only: no description, no extensions, nothing from the upload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(vol.astype(np.float32), affine), str(path))


def save_image(img: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path, format="PNG")  # built from the array: no EXIF
