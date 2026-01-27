import pytest

from .conftest import ADMIN, DOCTOR_A
from .fakes import brain, four, nifti_bytes

from app.report import Features, as_text, template_report

VOLS = {"whole_tumor": 89.3, "tumor_core": 34.2, "enhancing": 21.6, "necrosis_non_enhancing": 12.6, "edema": 55.1}


def feats(**kw) -> Features:
    base = dict(scan="3d", ai_type="glioma", confidence=0.94, doctor_type=None, volumes_cm3=VOLS,
                mask_corrected=False, side="right", lobe="parietal")
    return Features(**{**base, **kw})


def test_3d_tumor_reads_like_a_short_report():
    r = template_report(feats())
    assert r["findings"][0] == "A tumor is outlined in the right hemisphere, centered in the parietal region (estimate)."
    assert "Whole tumor volume 89.3 cm³ (from the 3D segmentation): enhancing tumor 21.6 cm³, " \
           "necrotic / non-enhancing core 12.6 cm³, surrounding edema 55.1 cm³." in r["findings"]
    assert "The 2D classifier suggests glioma (94%, not calibrated)." in r["findings"]
    assert r["impression"] == "Findings suggest a glioma of about 89 cm³, in the right hemisphere, " \
                              "centered in the parietal region (estimate)."
    assert r["notes"] == ["The region is a rough position estimate, not an atlas label."]
    assert "not a medical device" in r["disclaimer"]


def test_midline_crossing_is_said_plainly():
    assert "crossing the midline (both hemispheres)" in template_report(feats(side="both sides"))["findings"][0]


def test_doctor_corrections_win_and_are_named():
    r = template_report(feats(doctor_type="meningioma", mask_corrected=True))
    assert "(from the doctor-corrected mask)" in r["findings"][1]
    assert "Tumor type set by the reviewing doctor: meningioma." in r["findings"]
    assert r["impression"].startswith("Findings suggest a meningioma")


def test_no_tumor_in_3d():
    r = template_report(feats(ai_type="notumor", volumes_cm3={**VOLS, "whole_tumor": 0.0}, side=None, lobe=None))
    assert r["findings"][0] == "The 3D segmentation outlined no tumor."
    assert r["impression"] == "No tumor outlined by the AI." and r["notes"] == []


@pytest.mark.parametrize("kw, note", [
    (dict(ai_type="notumor"), "The classifier says no tumor, but the 3D segmentation outlined one"),
    (dict(volumes_cm3={**VOLS, "whole_tumor": 0.0}, side=None, lobe=None),
     "The classifier suggests glioma, but no tumor was outlined in 3D"),
    (dict(volumes_cm3={**VOLS, "whole_tumor": 3.2}), "Small lesion (under 6 cm³)"),
])
def test_disagreements_and_weak_spots_get_a_note(kw, note):
    assert any(n.startswith(note) for n in template_report(feats(**kw))["notes"])


def test_2d_slice_has_no_volume_or_location():
    r = template_report(feats(scan="2d", volumes_cm3=None, side=None, lobe=None, ai_type="pituitary", confidence=0.81))
    assert r["findings"][0] == "Single 2D slice: tumor type only, no volume or location measured."
    assert r["impression"] == "Findings suggest a pituitary tumor on the uploaded slice."
    assert "cm³" not in as_text(r)


def test_text_version_has_all_sections():
    text = as_text(template_report(feats(volumes_cm3={**VOLS, "whole_tumor": 3.2})))
    assert text.startswith("Findings:\n- A tumor is outlined")
    assert "\nImpression: " in text and "\nNotes:\n- Small lesion" in text and text.endswith("makes the diagnosis.")


def test_report_endpoint_uses_the_numbers_and_the_doctors_type(client):
    case = client.post("/cases", headers=DOCTOR_A).json()
    client.post(f"/cases/{case['id']}/scans", headers=DOCTOR_A, files=four(lambda m: (f"{m}.nii.gz", nifti_bytes(brain()))))
    [result] = client.get(f"/cases/{case['id']}/results", headers=DOCTOR_A).json()
    url = f"/cases/{case['id']}/results/{result['id']}/report"
    r = client.get(url, headers=DOCTOR_A).json()
    assert r["provider"] == "template" and r["features"]["scan"] == "3d"
    assert r["features"]["volumes_cm3"]["whole_tumor"] == result["volumes_cm3"]["whole_tumor"]
    assert "Small lesion" in r["text"] and "Doe" not in r["text"]  # tiny stub tumor; no patient data anywhere
    client.post(f"/cases/{case['id']}/reviews", headers=DOCTOR_A,
                json={"result_id": result["id"], "decision": "edited", "corrected_type": "meningioma"})
    assert "Tumor type set by the reviewing doctor: meningioma." in client.get(url, headers=ADMIN).json()["text"]
