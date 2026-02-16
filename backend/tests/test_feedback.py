import csv
import json

import nibabel as nib
import numpy as np
import pytest

from app.feedback import export_feedback

from .conftest import DOCTOR_A
from .fakes import brain, four, nifti_bytes, slice_png
from .test_masks import ai_mask, send, with_edema


def new_3d(client):
    case = client.post("/cases", headers=DOCTOR_A).json()
    client.post(f"/cases/{case['id']}/scans", headers=DOCTOR_A, files=four(lambda m: (f"{m}.nii.gz", nifti_bytes(brain()))))
    [result] = client.get(f"/cases/{case['id']}/results", headers=DOCTOR_A).json()
    return case, result


def decide(client, case, result, **body):
    r = client.post(f"/cases/{case['id']}/reviews", headers=DOCTOR_A, json={"result_id": result["id"], **body})
    assert r.status_code == 201, r.text


@pytest.fixture
def reviewed(client):
    fixed = new_3d(client)
    send(client, *fixed, with_edema(ai_mask(client, *fixed)))
    decide(client, *fixed, decision="edited", corrected_type="meningioma")
    bad = new_3d(client)
    decide(client, *bad, decision="rejected", reason="bad_mask")
    artifact = new_3d(client)
    decide(client, *artifact, decision="rejected", reason="artifact")
    fine = new_3d(client)
    decide(client, *fine, decision="approved")
    flat = client.post("/cases", headers=DOCTOR_A).json()  # a 2D case has no mask: never exported
    client.post(f"/cases/{flat['id']}/scans", headers=DOCTOR_A, files={"image": ("s.png", slice_png(fmt="PNG"))})
    [r2d] = client.get(f"/cases/{flat['id']}/results", headers=DOCTOR_A).json()
    decide(client, flat, r2d, decision="approved")
    new_3d(client)  # not reviewed yet: not part of the feedback
    return {"fixed": fixed[0]["code"], "bad": bad[0]["code"], "artifact": artifact[0]["code"], "fine": fine[0]["code"]}


def export(client, tmp_path, tag="fb-test"):
    with client.app.state.db.sessions() as s:
        return export_feedback(s, client.app.state.files, tmp_path / "feedback", tag)


def test_feedback_is_grouped_and_written_in_brats_layout(client, reviewed, tmp_path):
    summary = export(client, tmp_path)
    assert summary["counts"] == {"train": 1, "relabel": 1, "excluded": 1, "approved": 1}
    root = tmp_path / "feedback" / "fb-test"
    code = reviewed["fixed"]
    files = sorted(p.name for p in (root / "train" / code).iterdir())
    assert files == sorted(f"{code}_{m}.nii.gz" for m in ("flair", "t1", "t1ce", "t2", "ai", "seg"))
    # The training label is the doctor's mask, on the same RAS grid as the scans
    seg, flair = (nib.load(root / "train" / code / f"{code}_{m}.nii.gz") for m in ("seg", "flair"))
    assert seg.shape == flair.shape and np.allclose(seg.affine, flair.affine)
    assert nib.aff2axcodes(seg.affine) == ("R", "A", "S") and (np.asarray(seg.dataobj) == 2).sum() == 1000
    relabel = root / "relabel" / reviewed["bad"]
    assert relabel.is_dir() and not list(relabel.glob("*_seg.nii.gz"))  # needs a label before training
    assert not (root / "excluded").exists() and not (root / "approved").exists()


def test_manifest_and_summary_say_what_changed_without_personal_data(client, reviewed, tmp_path):
    summary = export(client, tmp_path)
    root = tmp_path / "feedback" / "fb-test"
    rows = {r["case_code"]: r for r in csv.DictReader(open(root / "manifest.csv"))}
    fixed = rows[reviewed["fixed"]]
    assert fixed["group"] == "train" and fixed["doctor_type"] == "meningioma" and float(fixed["dice_ai_vs_doctor"]) < 1
    assert rows[reviewed["artifact"]]["group"] == "excluded" and rows[reviewed["fine"]]["group"] == "approved"
    assert summary["mean_dice_ai_vs_doctor"] == float(fixed["dice_ai_vs_doctor"])
    text = (root / "manifest.csv").read_text() + json.dumps(summary)
    assert "doctor-a" not in text and "@" not in text and "Doe" not in text


def test_a_tag_is_written_once(client, reviewed, tmp_path):
    export(client, tmp_path)
    with pytest.raises(FileExistsError):
        export(client, tmp_path)
