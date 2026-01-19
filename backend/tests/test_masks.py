import gzip

import nibabel as nib
import numpy as np
import pytest

from .conftest import ADMIN, DOCTOR_A, DOCTOR_B
from .fakes import brain, four, nifti_bytes, slice_png


@pytest.fixture
def done(client):
    """A finished 3D case: (case, result) with the stub AI mask (a small enhancing block)."""
    case = client.post("/cases", headers=DOCTOR_A).json()
    r = client.post(f"/cases/{case['id']}/scans", headers=DOCTOR_A,
                    files=four(lambda m: (f"{m}.nii.gz", nifti_bytes(brain()))))
    assert r.status_code == 201, r.text
    [result] = client.get(f"/cases/{case['id']}/results", headers=DOCTOR_A).json()
    return case, result


def ai_mask(client, case, result) -> nib.Nifti1Image:
    r = client.get(f"/cases/{case['id']}/results/{result['id']}/mask", headers=DOCTOR_A)
    return nib.Nifti1Image.from_bytes(gzip.decompress(r.content))


def send(client, case, result, img: nib.Nifti1Image, headers=DOCTOR_A):
    return client.post(f"/cases/{case['id']}/results/{result['id']}/mask", headers=headers,
                       files={"mask": ("corrected.nii", img.to_bytes())})


def with_edema(img: nib.Nifti1Image) -> nib.Nifti1Image:
    labels = np.asarray(img.dataobj).astype(np.uint8)
    labels[10:20, 10:20, 10:20] = 2  # the doctor paints 1,000 voxels of edema
    return nib.Nifti1Image(labels, img.affine)


def test_doctor_saves_a_corrected_mask_and_volumes_are_measured_again(client, done):
    case, result = done
    original = ai_mask(client, case, result)
    r = send(client, case, result, with_edema(original))
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["has_corrected_mask"] and out["has_mask"]
    vols = out["info"]["corrected_mask"]["volumes_cm3"]
    assert vols["edema"] == 1.0 and vols["whole_tumor"] == round(result["volumes_cm3"]["whole_tumor"] + 1.0, 2)
    saved = client.get(f"/cases/{case['id']}/results/{result['id']}/corrected", headers=DOCTOR_A)
    assert saved.status_code == 200 and saved.headers["cache-control"] == "no-store"
    labels = np.asarray(nib.Nifti1Image.from_bytes(gzip.decompress(saved.content)).dataobj)
    assert (labels == 2).sum() == 1000
    # The AI mask is kept as it was, for retraining on the difference
    assert np.array_equal(np.asarray(ai_mask(client, case, result).dataobj), np.asarray(original.dataobj))
    log = client.get("/audit", headers=ADMIN).json()
    entry = next(e for e in log if e["action"] == "mask.correct")
    assert entry["detail"]["changed_voxels"] == 1000


def test_a_mask_saved_in_another_orientation_lands_on_the_same_voxels(client, done):
    """The viewer saves in the scan's own orientation (LPS for DICOM); the API turns it back to RAS."""
    case, result = done
    ras = with_edema(ai_mask(client, case, result))
    to_lps = nib.orientations.ornt_transform(nib.orientations.axcodes2ornt("RAS"), nib.orientations.axcodes2ornt("LPS"))
    lps = ras.as_reoriented(to_lps)
    assert nib.aff2axcodes(lps.affine) == ("L", "P", "S")
    assert send(client, case, result, lps).json()["info"]["corrected_mask"]["volumes_cm3"]["edema"] == 1.0


@pytest.mark.parametrize("bad, why", [
    (lambda m: nib.Nifti1Image(np.zeros((8, 8, 8), np.uint8), m.affine), "the mask is"),
    (lambda m: nib.Nifti1Image(np.full(m.shape, 3, np.uint8), m.affine), "unknown mask labels"),
    (lambda m: nib.Nifti1Image(np.asarray(m.dataobj), m.affine @ np.diag([1, 1, 1.5, 1])), "does not line up"),
])
def test_unusable_masks_are_refused_with_a_reason(client, done, bad, why):
    case, result = done
    r = send(client, case, result, bad(ai_mask(client, case, result)))
    assert r.status_code == 422 and why in r.json()["detail"]
    assert not client.get(f"/cases/{case['id']}/results", headers=DOCTOR_A).json()[0]["has_corrected_mask"]


def test_only_the_cases_doctor_can_correct_and_not_after_review(client, done):
    case, result = done
    fixed = with_edema(ai_mask(client, case, result))
    assert send(client, case, result, fixed, headers=DOCTOR_B).status_code == 404
    assert send(client, case, result, fixed, headers=ADMIN).status_code == 403
    client.post(f"/cases/{case['id']}/reviews", headers=DOCTOR_A, json={"result_id": result["id"], "decision": "approved"})
    assert send(client, case, result, fixed).status_code == 409


def test_a_2d_result_has_no_mask_to_correct(client):
    case = client.post("/cases", headers=DOCTOR_A).json()
    client.post(f"/cases/{case['id']}/scans", headers=DOCTOR_A, files={"image": ("s.png", slice_png(fmt="PNG"))})
    [result] = client.get(f"/cases/{case['id']}/results", headers=DOCTOR_A).json()
    img = nib.Nifti1Image(np.zeros((4, 4, 4), np.uint8), np.eye(4))
    assert send(client, case, result, img).status_code == 409
