import pytest

from .conftest import ADMIN, DOCTOR_A
from .fakes import slice_png


@pytest.fixture
def ready(client):
    """A case of doctor A with a finished 2D analysis."""
    case = client.post("/cases", headers=DOCTOR_A).json()
    client.post(f"/cases/{case['id']}/scans", headers=DOCTOR_A, files={"image": ("s.png", slice_png(fmt="PNG"))})
    [result] = client.get(f"/cases/{case['id']}/results", headers=DOCTOR_A).json()
    return case, result


def review(client, case, **body):
    return client.post(f"/cases/{case['id']}/reviews", headers=DOCTOR_A, json=body)


def test_approve(client, ready):
    case, result = ready
    r = review(client, case, result_id=result["id"], decision="approved")
    assert r.status_code == 201 and r.json()["decision"] == "approved"
    assert client.get(f"/cases/{case['id']}", headers=DOCTOR_A).json()["status"] == "reviewed"


def test_edit_with_corrected_type_and_text(client, ready):
    case, result = ready
    r = review(client, case, result_id=result["id"], decision="edited", corrected_type="meningioma",
               report_text="Extra-axial mass, likely meningioma.")
    assert r.status_code == 201 and r.json()["corrected_type"] == "meningioma"


def test_reject_with_reason(client, ready):
    case, result = ready
    r = review(client, case, result_id=result["id"], decision="rejected", reason="bad_mask", note="misses the edema")
    assert r.status_code == 201 and r.json()["reason"] == "bad_mask"


@pytest.mark.parametrize("body", [
    {"decision": "rejected"},  # no reason
    {"decision": "rejected", "reason": "other"},  # 'other' without a note
    {"decision": "rejected", "reason": "tired"},  # not a known reason
    {"decision": "approved", "reason": "artifact"},  # reason only for rejections
    {"decision": "approved", "corrected_type": "glioma"},  # that is an edit
    {"decision": "edited"},  # an edit with no change
    {"decision": "edited", "corrected_type": "lymphoma"},  # not one of the 4 classes
    {"decision": "maybe"},
], ids=["no-reason", "other-no-note", "bad-reason", "reason-on-approve", "type-on-approve", "empty-edit",
        "unknown-type", "bad-decision"])
def test_invalid_reviews_are_refused(client, ready, body):
    case, result = ready
    assert review(client, case, result_id=result["id"], **body).status_code == 422
    assert client.get(f"/cases/{case['id']}/reviews", headers=DOCTOR_A).json() == []


def test_unfinished_or_foreign_results_cannot_be_reviewed(client, ready, monkeypatch):
    case, result = ready
    assert review(client, case, result_id="no-such-result", decision="approved").status_code == 404
    other = client.post("/cases", headers=DOCTOR_A).json()  # result belongs to another case
    assert client.post(f"/cases/{other['id']}/reviews", headers=DOCTOR_A,
                       json={"result_id": result["id"], "decision": "approved"}).status_code == 404
    def broken():
        raise RuntimeError("model crashed")
    monkeypatch.setattr(client.app.state.models, "classifier", broken)  # the next analysis fails
    client.post(f"/cases/{other['id']}/scans", headers=DOCTOR_A, files={"image": ("s.png", slice_png(fmt="PNG"))})
    [failed] = client.get(f"/cases/{other['id']}/results", headers=DOCTOR_A).json()
    assert client.post(f"/cases/{other['id']}/reviews", headers=DOCTOR_A,
                       json={"result_id": failed["id"], "decision": "approved"}).status_code == 409


def test_a_new_review_replaces_the_old_one_and_history_is_kept(client, ready):
    case, result = ready
    review(client, case, result_id=result["id"], decision="approved")
    review(client, case, result_id=result["id"], decision="rejected", reason="artifact")
    history = client.get(f"/cases/{case['id']}/reviews", headers=DOCTOR_A).json()
    assert [r["decision"] for r in history] == ["rejected", "approved"]  # newest first


def test_every_decision_is_audited(client, ready):
    case, result = ready
    review(client, case, result_id=result["id"], decision="approved")
    review(client, case, result_id=result["id"], decision="edited", report_text="ok")
    review(client, case, result_id=result["id"], decision="rejected", reason="wrong_type")
    log = [e for e in client.get("/audit", headers=ADMIN).json() if e["action"].startswith("review.")]
    assert [e["action"] for e in log] == ["review.reject", "review.edit", "review.approve"]
    assert log[0]["detail"]["reason"] == "wrong_type" and log[0]["detail"]["result_id"] == result["id"]
    assert all(e["user_id"] == "doctor-a" and e["case_id"] == case["id"] for e in log)
