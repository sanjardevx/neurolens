import gzip

import nibabel as nib
import numpy as np
import pytest
from PIL import Image

from .conftest import ADMIN, DOCTOR_A, DOCTOR_B
from .fakes import PATIENT_ID, SHAPE, brain, dicom_zip, four, nifti_bytes, slice_png


@pytest.fixture
def case(client):
    return client.post("/cases", headers=DOCTOR_A).json()


def upload(client, case, files, headers=DOCTOR_A):
    return client.post(f"/cases/{case['id']}/scans", headers=headers, files=files)


def stored(client, scan) -> list:
    return sorted((client.app.state.storage / "cases").glob(f"*/{scan['id']}/*"))


def all_bytes(paths) -> bytes:
    return b"".join(gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes() for p in paths)


def test_nifti_upload_gets_a_fresh_header(client, case):
    r = upload(client, case, four(lambda m: (f"{m}.nii.gz", nifti_bytes(brain()))))
    assert r.status_code == 201, r.text
    scan = r.json()
    assert scan["file_format"] == "nifti" and scan["qc_passed"] and scan["info"]["shape"] == list(SHAPE)
    paths = stored(client, scan)
    assert [p.name for p in paths] == ["flair.nii.gz", "t1.nii.gz", "t1ce.nii.gz", "t2.nii.gz"]
    assert b"Doe" not in all_bytes(paths)  # the free-text header field was not copied


def test_dicom_upload_keeps_pixels_and_geometry_only(client, case):
    r = upload(client, case, four(lambda m: (f"{m}.zip", dicom_zip(brain()))))
    assert r.status_code == 201, r.text
    scan = r.json()
    assert scan["file_format"] == "dicom" and scan["info"]["voxel_mm"] == [1.0, 1.0, 2.0]
    assert scan["info"]["dicom_tags"]["flair"] == {"Modality": "MR", "BodyPartExamined": "BRAIN",
                                                   "MagneticFieldStrength": 3.0, "SliceThickness": 2.0}
    everything = all_bytes(stored(client, scan)) + r.text.encode() + client.get("/audit", headers=ADMIN).text.encode()
    for secret in [b"Doe", PATIENT_ID.encode(), b"19700101", b"Fake Hospital"]:
        assert secret not in everything


def test_dicom_left_stays_left(client, case):
    """The marker sits at a high column index, which DICOM (LPS) puts on the patient's left.
    In the stored RAS volume the patient's left is the low x side."""
    scan = upload(client, case, four(lambda m: (f"{m}.zip", dicom_zip(brain())))).json()
    vol = nib.load(stored(client, scan)[0]).get_fdata()
    x = np.argwhere(vol == 1000)[:, 0]
    assert x.max() < SHAPE[0] / 2  # left half
    assert nib.aff2axcodes(nib.load(stored(client, scan)[0]).affine) == ("R", "A", "S")


@pytest.mark.parametrize("files, reason", [
    (four(lambda m: (f"{m}.zip", dicom_zip(brain(), modality="CT"))), "not MR"),
    (four(lambda m: (f"{m}.zip", dicom_zip(brain(), body_part="KNEE"))), "not brain"),
    (four(lambda m: (f"{m}.zip", dicom_zip(brain(), two_series=True))), "more than one series"),
    ({m: (f"{m}.nii.gz", nifti_bytes(brain())) for m in ("flair", "t1", "t1ce")}, "missing MRI types: t2"),
    (four(lambda m: (f"{m}.nii.gz", nifti_bytes(brain() + 5))), "skull-stripped"),
    (four(lambda m: (f"{m}.nii.gz", nifti_bytes(brain((48, 56, 40 if m != "t2" else 36))))), "different sizes"),
    (four(lambda m: (f"{m}.nii.gz", nifti_bytes(brain(), np.diag([1, 1, 1 if m != "t1" else 2, 1])))), "same position"),
    (four(lambda m: (f"{m}.txt", b"hello")), "use .nii"),
    (four(lambda m: (f"{m}.nii.gz", b"broken")), "not a readable NIfTI"),
    ({"image": ("photo.jpg", slice_png(color=True))}, "color"),
    ({"image": ("x.jpg", slice_png()), "flair": ("f.nii.gz", nifti_bytes(brain()))}, "not both"),
], ids=["ct", "knee", "two-series", "missing", "not-skull-stripped", "sizes", "positions", "wrong-type",
        "broken", "color", "both"])
def test_bad_uploads_are_rejected_with_a_reason(client, case, files, reason):
    r = upload(client, case, files)
    assert r.status_code == 422 and reason in r.json()["detail"]
    assert client.get(f"/cases/{case['id']}/scans", headers=DOCTOR_A).json() == []
    assert not (client.app.state.storage / "cases").exists() or not any((client.app.state.storage / "cases").rglob("*.*"))
    log = client.get("/audit", headers=ADMIN).json()
    assert log[0]["action"] == "scan.rejected" and reason in log[0]["detail"]["reason"]


def test_image_upload_drops_exif(client, case):
    r = upload(client, case, {"image": ("slice.jpg", slice_png(exif_artist=PATIENT_ID))})
    assert r.status_code == 201, r.text
    png = stored(client, r.json())[0]
    assert png.name == "image.png" and not Image.open(png).getexif()
    assert PATIENT_ID.encode() not in png.read_bytes()


def test_only_the_case_owner_uploads(client, case):
    files = {"image": ("slice.png", slice_png(fmt="PNG"))}
    assert upload(client, case, files, DOCTOR_B).status_code == 404
    assert upload(client, case, files, ADMIN).status_code == 403
    assert upload(client, case, files).status_code == 201


def test_too_large_upload(client, case):
    client.app.state.max_upload_bytes = 1000
    r = upload(client, case, {"image": ("slice.jpg", slice_png())})
    assert r.status_code == 413
