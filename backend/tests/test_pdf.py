import nibabel as nib
import numpy as np
import pytest

from app.pdf import Signed, build_pdf, fingerprint, key_slices_3d

from .conftest import ADMIN, DOCTOR_A, DOCTOR_B
from .fakes import PATIENT_NAME, brain, four, nifti_bytes, slice_png


def signed(**kw) -> Signed:
    base = dict(case_code="NL-TEST01", scan="3D MRI (FLAIR, T1, T1ce, T2)", uploaded_at="2026-01-30 10:00 UTC",
                decision="Approved with edits", reviewer="a@clinic.test", approved_at="2026-01-30 11:00 UTC",
                tumor_type="Glioma", type_note="AI suggestion, 94% from the 2D model, not calibrated",
                volumes_cm3={"whole_tumor": 89.3, "tumor_core": 34.2, "enhancing": 21.6,
                             "necrosis_non_enhancing": 12.6, "edema": 55.1},
                location="In the right hemisphere, centered in the parietal region (estimate)",
                report_text="Findings:\n- necrotic/non‑enhancing core → checked", models="seg-v1 + cls-v1")
    return Signed(**{**base, **kw})


def test_pdf_carries_the_signed_fields():
    s = signed()
    data = build_pdf(s, images=[], compress=False)
    assert data.startswith(b"%PDF")
    for text in (b"NL-TEST01", b"a@clinic.test", b"2026-01-30 11:00 UTC", b"Approved with edits", b"89.3 cm",
                 fingerprint(s).encode()):
        assert text in data
    assert b"non-enhancing core -> checked" in data  # LLM typography made safe for the PDF fonts


def test_fingerprint_changes_with_the_report():
    assert fingerprint(signed()) != fingerprint(signed(report_text="Findings: something else"))


@pytest.fixture
def approved(client):
    case = client.post("/cases", headers=DOCTOR_A).json()
    client.post(f"/cases/{case['id']}/scans", headers=DOCTOR_A, files=four(lambda m: (f"{m}.nii.gz", nifti_bytes(brain()))))
    [result] = client.get(f"/cases/{case['id']}/results", headers=DOCTOR_A).json()
    return case, result


def test_key_slices_show_the_slice_with_the_most_tumor(client, approved):
    case, result = approved
    storage = client.app.state.storage
    [scan] = client.get(f"/cases/{case['id']}/scans", headers=DOCTOR_A).json()
    mask = storage / "results" / result["id"] / "mask.nii.gz"
    images, z = key_slices_3d(storage / "cases" / case["id"] / scan["id"], mask)
    labels = np.asarray(nib.load(mask).dataobj)
    assert len(images) == 2 and z == int((labels > 0).sum((0, 1)).argmax())
    assert all(png.startswith(b"\x89PNG") for _, png in images)


def test_only_an_approved_case_downloads_a_pdf(client, approved):
    case, result = approved
    url = f"/cases/{case['id']}/report.pdf"
    assert client.get(url, headers=DOCTOR_A).status_code == 409  # not reviewed
    client.post(f"/cases/{case['id']}/reviews", headers=DOCTOR_A,
                json={"result_id": result["id"], "decision": "rejected", "reason": "artifact"})
    assert client.get(url, headers=DOCTOR_A).status_code == 409  # rejected: nothing to sign
    client.post(f"/cases/{case['id']}/reopen", headers=DOCTOR_A)
    client.post(f"/cases/{case['id']}/reviews", headers=DOCTOR_A,
                json={"result_id": result["id"], "decision": "approved", "report_text": "Findings:\n- reviewed"})
    r = client.get(url, headers=DOCTOR_A)
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
    assert r.content.startswith(b"%PDF") and r.headers["cache-control"] == "no-store"
    assert f'filename="{case["code"]}_report.pdf"' in r.headers["content-disposition"]
    assert PATIENT_NAME.encode() not in r.content  # the fake patient from the upload never reaches the PDF
    assert client.get(url, headers=ADMIN).status_code == 200
    assert client.get(url, headers=DOCTOR_B).status_code == 404
    assert "report.pdf" in [e["action"] for e in client.get("/audit", headers=ADMIN).json()]


def test_2d_case_pdf_uses_the_slice_and_the_heatmap(client):
    case = client.post("/cases", headers=DOCTOR_A).json()
    client.post(f"/cases/{case['id']}/scans", headers=DOCTOR_A, files={"image": ("s.png", slice_png(fmt="PNG"))})
    [result] = client.get(f"/cases/{case['id']}/results", headers=DOCTOR_A).json()
    client.post(f"/cases/{case['id']}/reviews", headers=DOCTOR_A,
                json={"result_id": result["id"], "decision": "edited", "corrected_type": "meningioma"})
    r = client.get(f"/cases/{case['id']}/report.pdf", headers=DOCTOR_A)
    assert r.status_code == 200 and r.content.count(b"/Subtype /Image") == 2
