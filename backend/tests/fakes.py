"""Fake uploads: skull-stripped MRI volumes as NIfTI or DICOM (with a made-up patient), and 2D slices."""
import gzip
import io
import random
import zipfile

import nibabel as nib
import numpy as np
from PIL import Image
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

PATIENT_NAME, PATIENT_ID = "Doe^John", "MRN-12345"
SHAPE = (48, 56, 40)
MARKER = (slice(36, 44), slice(24, 32), slice(18, 22))  # high column index = patient's LEFT in DICOM (LPS)


def brain(shape=SHAPE, seed=0) -> np.ndarray:
    """Ellipsoid 'brain' of noise, 0 outside (skull-stripped), with a bright marker block."""
    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    inside = ((x - shape[0] / 2) / (shape[0] * 0.45)) ** 2 + ((y - shape[1] / 2) / (shape[1] * 0.45)) ** 2 \
        + ((z - shape[2] / 2) / (shape[2] * 0.45)) ** 2 < 1
    vol = np.where(inside, np.random.default_rng(seed).uniform(100, 400, shape), 0).astype(np.float32)
    vol[MARKER] = 1000
    return vol


def nifti_bytes(vol: np.ndarray, affine=np.eye(4), note: str = PATIENT_NAME) -> bytes:
    img = nib.Nifti1Image(vol, affine)
    img.header["descrip"] = note.encode()  # free-text header field some tools fill with patient info
    return gzip.compress(img.to_bytes())


def dicom_zip(vol: np.ndarray, modality="MR", body_part="BRAIN", gap=2.0, two_series=False) -> bytes:
    """One axial DICOM series, slices shuffled in the zip. Pixel [row j, col i] of slice k = vol[i, j, k]."""
    series = [generate_uid(), generate_uid()]
    files = []
    for k in range(vol.shape[2]):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID = MRImageStorage, generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = Dataset()
        ds.file_meta = meta
        ds.PatientName, ds.PatientID, ds.PatientBirthDate = PATIENT_NAME, PATIENT_ID, "19700101"
        ds.InstitutionName, ds.SeriesDescription = "Fake Hospital", f"AX FLAIR {PATIENT_NAME}"
        ds.Modality, ds.BodyPartExamined, ds.MagneticFieldStrength = modality, body_part, 3.0
        ds.SOPClassUID, ds.SOPInstanceUID = MRImageStorage, meta.MediaStorageSOPInstanceUID
        ds.SeriesInstanceUID = series[k % 2] if two_series else series[0]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [0.0, 0.0, k * gap]
        ds.PixelSpacing, ds.SliceThickness = [1.0, 1.0], gap
        ds.Rows, ds.Columns = vol.shape[1], vol.shape[0]
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.PixelData = np.ascontiguousarray(vol[:, :, k].T).astype(np.uint16).tobytes()
        buf = io.BytesIO()
        ds.save_as(buf, enforce_file_format=True)
        files.append((f"IM{k:04d}.dcm", buf.getvalue()))
    random.Random(0).shuffle(files)
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as zf:
        for name, data in files:
            zf.writestr(name, data)
        zf.writestr("README.txt", "not a dicom file")
    return out.getvalue()


def slice_png(color=False, exif_artist: str | None = None, fmt="JPEG") -> bytes:
    """A 2D 'MRI slice': gray head on black. Optional EXIF that must not survive the upload."""
    img = np.zeros((256, 256, 3), np.uint8)
    img[40:216, 50:206] = np.random.default_rng(0).integers(60, 200, (176, 156, 1))
    if color:
        img[..., 0] = 220
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    if exif_artist:
        exif = Image.Exif()
        exif[0x013B] = exif_artist  # Artist
        pil.save(buf, format=fmt, exif=exif)
    else:
        pil.save(buf, format=fmt)
    return buf.getvalue()


def four(make) -> dict:
    """multipart files for the 4 MRI types; make(m) returns (filename, bytes)."""
    return {m: make(m) for m in ("flair", "t1", "t1ce", "t2")}


# ---------- stand-in ONNX models (same inputs / outputs as the real ones, simple fixed rules) ----------

def _save(graph, path) -> None:
    import onnx
    from onnx import helper

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def stub_classifier(path, winner: int = 0) -> None:
    """image [N, 3, H, W] → cams [N, 4, H/32, W/32] (one 32x32 conv) and logits = mean of cams. Class `winner` wins."""
    from onnx import TensorProto, helper, numpy_helper

    w = np.zeros((4, 3, 32, 32), np.float32)
    w[winner] = 1e-5  # tiny: the heatmap varies a little, the bias decides the class
    b = np.full(4, -3.0, np.float32)
    b[winner] = 3.0
    nodes = [helper.make_node("Conv", ["image", "w", "b"], ["cams"], strides=[32, 32]),
             helper.make_node("ReduceMean", ["cams", "axes"], ["logits"], keepdims=0)]
    graph = helper.make_graph(
        nodes, "stub_classifier",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["n", 3, "h", "w"])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["n", 4]),
         helper.make_tensor_value_info("cams", TensorProto.FLOAT, ["n", 4, "hc", "wc"])],
        [numpy_helper.from_array(w, "w"), numpy_helper.from_array(b, "b"),
         numpy_helper.from_array(np.array([2, 3], np.int64), "axes")])
    _save(graph, path)


def stub_segmenter(path) -> None:
    """image [N, 4, 128³] → logits [N, 3, 128³] = normalized FLAIR - 2 for all 3 regions: the bright marker is 'tumor'."""
    from onnx import TensorProto, helper, numpy_helper

    w = np.zeros((3, 4, 1, 1, 1), np.float32)
    w[:, 0] = 1.0
    graph = helper.make_graph(
        [helper.make_node("Conv", ["image", "w", "b"], ["logits"])], "stub_segmenter",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["n", 4, 128, 128, 128])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["n", 3, 128, 128, 128])],
        [numpy_helper.from_array(w, "w"), numpy_helper.from_array(np.full(3, -2.0, np.float32), "b")])
    _save(graph, path)
