import io

import nibabel as nib
import numpy as np
import pytest
from PIL import Image

from .conftest import ADMIN, DOCTOR_A, DOCTOR_B
from .fakes import brain, dicom_zip, four, nifti_bytes, slice_png, stub_segmenter

MARKER_VOXELS = 8 * 8 * 4  # the stub segmenter finds exactly the bright marker block


@pytest.fixture
def case(client):
    return client.post("/cases", headers=DOCTOR_A).json()


def upload(client, case, files):
    r = client.post(f"/cases/{case['id']}/scans", headers=DOCTOR_A, files=files)
    assert r.status_code == 201, r.text
    return r.json()


def results(client, case, headers=DOCTOR_A):
    return client.get(f"/cases/{case['id']}/results", headers=headers).json()


def test_3d_upload_runs_segmentation_and_measures(client, case):
    upload(client, case, four(lambda m: (f"{m}.nii.gz", nifti_bytes(brain()))))
    [r] = results(client, case)
    assert r["error"] is None and r["progress"] == 100
    assert r["model_version"] == "seg-stub-v1 + cls-stub-v1"  # read from the model files that ran
    assert r["info"]["models"]["segmenter"] == {"version": "seg-stub-v1", "about": "test stand-in"}
    assert r["volumes_cm3"] == {"whole_tumor": 0.26, "tumor_core": 0.26, "enhancing": 0.26,
                                "necrosis_non_enhancing": 0.0, "edema": 0.0}
    assert r["side"] == "right"  # NIfTI with an identity affine: high x index is the patient's right
    assert r["tumor_type"] == "glioma" and r["probabilities"]["glioma"] == pytest.approx(r["confidence"], abs=1e-4)
    assert "T1ce axial slice" in r["info"]["classifier_input"] and r["has_mask"] and r["has_gradcam"]
    assert client.get(f"/cases/{case['id']}", headers=DOCTOR_A).json()["status"] == "ready"


def test_3d_mask_file_has_brats_labels_on_the_scan_grid(client, case):
    upload(client, case, four(lambda m: (f"{m}.nii.gz", nifti_bytes(brain()))))
    [r] = results(client, case)
    resp = client.get(f"/cases/{case['id']}/results/{r['id']}/mask", headers=DOCTOR_A)
    assert resp.status_code == 200 and case["code"] in resp.headers["content-disposition"]
    assert resp.headers["cache-control"] == "no-store"
    mask = nib.Nifti1Image.from_bytes(__import__("gzip").decompress(resp.content))
    labels = np.asarray(mask.dataobj)
    assert labels.shape == brain().shape and set(np.unique(labels)) == {0, 4} and (labels == 4).sum() == MARKER_VOXELS


def test_dicom_tumor_on_the_left_is_reported_left(client, case):
    upload(client, case, four(lambda m: (f"{m}.zip", dicom_zip(brain()))))
    [r] = results(client, case)
    assert r["side"] == "left" and r["volumes_cm3"]["whole_tumor"] == round(MARKER_VOXELS * 2 / 1000, 2)  # 2 mm slices


def test_2d_upload_gets_type_and_gradcam(client, case):
    upload(client, case, {"image": ("slice.png", slice_png(fmt="PNG"))})
    [r] = results(client, case)
    assert r["model_version"] == "cls-stub-v1" and r["tumor_type"] == "glioma" and r["volumes_cm3"] is None
    assert set(r["info"]["models"]) == {"classifier"}
    png = client.get(f"/cases/{case['id']}/results/{r['id']}/gradcam", headers=DOCTOR_A)
    assert png.headers["content-type"] == "image/png" and Image.open(io.BytesIO(png.content)).size == (384, 384)
    assert client.get(f"/cases/{case['id']}/results/{r['id']}/mask", headers=DOCTOR_A).status_code == 404


def test_missing_model_fails_the_case_cleanly(client, case):
    (client.app.state.models.dir / "segmenter3d.onnx").unlink()
    upload(client, case, four(lambda m: (f"{m}.nii.gz", nifti_bytes(brain()))))
    [r] = results(client, case)
    assert "model file not found: segmenter3d.onnx" in r["error"]
    assert client.get(f"/cases/{case['id']}", headers=DOCTOR_A).json()["status"] == "failed"
    failed = [e for e in client.get("/audit", headers=ADMIN).json() if e["action"] == "analysis.failed"]
    assert len(failed) == 1 and "segmenter3d.onnx" in failed[0]["detail"]["error"]


def test_new_segmenter_version_is_used_with_its_own_et_limit(client, case):
    # Swapping the file is the whole release: the version and the small-ET limit come with it
    stub_segmenter(client.app.state.models.dir / "segmenter3d.onnx", version="seg-stub-v2", min_et=MARKER_VOXELS + 1)
    client.app.state.models._seg = None  # a restart loads the new file
    upload(client, case, four(lambda m: (f"{m}.nii.gz", nifti_bytes(brain()))))
    [r] = results(client, case)
    assert r["model_version"] == "seg-stub-v2 + cls-stub-v1"
    assert r["volumes_cm3"]["enhancing"] == 0.0 and r["volumes_cm3"]["whole_tumor"] == 0.26  # ET under the limit dropped
    row = next(c for c in client.get("/cases", headers=DOCTOR_A).json() if c["id"] == case["id"])
    assert row["model_version"] == "seg-stub-v2 + cls-stub-v1"  # the worklist shows which model made it


def test_model_without_version_fails_the_case(client, case):
    import onnx

    path = client.app.state.models.dir / "segmenter3d.onnx"
    model = onnx.load(str(path))
    del model.metadata_props[:]
    onnx.save(model, str(path))
    upload(client, case, four(lambda m: (f"{m}.nii.gz", nifti_bytes(brain()))))
    [r] = results(client, case)
    assert "model file has no version" in r["error"] and r["model_version"] == ""


def test_results_follow_case_access(client, case):
    upload(client, case, {"image": ("slice.png", slice_png(fmt="PNG"))})
    [r] = results(client, case)
    assert client.get(f"/cases/{case['id']}/results", headers=DOCTOR_B).status_code == 404
    assert client.get(f"/cases/{case['id']}/results/{r['id']}/gradcam", headers=DOCTOR_B).status_code == 404
    assert len(results(client, case, ADMIN)) == 1


def test_analysis_is_audited(client, case):
    upload(client, case, {"image": ("slice.png", slice_png(fmt="PNG"))})
    actions = [e["action"] for e in client.get("/audit", headers=ADMIN).json()]
    assert actions[:3] == ["analysis.done", "scan.upload", "case.create"]


def test_worklist_rows_summarize_the_latest_result_and_review(client, case):
    empty = client.post("/cases", headers=DOCTOR_A).json()
    upload(client, case, four(lambda m: (f"{m}.nii.gz", nifti_bytes(brain()))))
    rows = {c["id"]: c for c in client.get("/cases", headers=DOCTOR_A).json()}
    row = rows[case["id"]]
    assert row["status"] == "ready" and row["progress"] == 100 and row["error"] is None
    assert row["tumor_type"] == "glioma" and row["tumor_cm3"] == 0.26 and row["decision"] is None
    assert rows[empty["id"]]["tumor_type"] is None and rows[empty["id"]]["progress"] is None

    [r] = results(client, case)
    client.post(f"/cases/{case['id']}/reviews", headers=DOCTOR_A, json={"result_id": r["id"], "decision": "approved"})
    row = next(c for c in client.get("/cases", headers=DOCTOR_A).json() if c["id"] == case["id"])
    assert row["status"] == "reviewed" and row["decision"] == "approved"


def test_progress_reaches_100_only_with_the_saved_result(client, case, monkeypatch):
    import app.analysis as analysis
    seen, run_3d = [], analysis.run_3d
    monkeypatch.setattr(analysis, "run_3d", lambda scan_dir, out_dir, models, report:
                        run_3d(scan_dir, out_dir, models, lambda pct: (seen.append(pct), report(pct))))
    upload(client, case, four(lambda m: (f"{m}.nii.gz", nifti_bytes(brain()))))
    assert seen and max(seen) < 100  # a poller that sees 100% must also see the tumor type and volumes
    [r] = results(client, case)
    assert r["progress"] == 100 and r["tumor_type"] and r["volumes_cm3"]


def test_viewer_gets_the_stored_scan_files_only_for_own_cases(client, case):
    scan = upload(client, case, four(lambda m: (f"{m}.nii.gz", nifti_bytes(brain()))))
    base = f"/cases/{case['id']}/scans/{scan['id']}"
    r = client.get(f"{base}/t1ce", headers=DOCTOR_A)
    assert r.status_code == 200 and case["code"] in r.headers["content-disposition"]
    assert r.headers["cache-control"] == "no-store"  # scans never stay in the browser cache
    vol = nib.Nifti1Image.from_bytes(__import__("gzip").decompress(r.content))
    assert vol.shape == brain().shape
    assert client.get(f"{base}/t1ce", headers=DOCTOR_B).status_code == 404  # another doctor's case
    assert client.get(f"{base}/t1ce", headers=ADMIN).status_code == 200
    assert client.get(f"{base}/image", headers=DOCTOR_A).status_code == 404  # a 3D scan has no 2D image
    assert client.get(f"{base}/seg", headers=DOCTOR_A).status_code == 404
