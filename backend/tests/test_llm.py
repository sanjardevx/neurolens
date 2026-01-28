import json

import httpx
import pytest

from app.llm import LLM, LLMError, check, parse
from app.report import template_report

from .conftest import ADMIN, DOCTOR_A
from .fakes import brain, four, nifti_bytes
from .test_report import feats


def fake_llm(answer, status=200, seen=None) -> LLM:
    """An LLM whose server answers with `answer` (a dict becomes the model's JSON reply)."""
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(json.loads(request.content))
        content = answer if isinstance(answer, str) else json.dumps(answer)
        return httpx.Response(status, json={"choices": [{"message": {"content": content}}]})
    return LLM("https://llm.test/v1", "test-key", "test-model", transport=httpx.MockTransport(handler))


def reworded(draft: dict) -> dict:
    """What a good model returns: same facts and numbers, smoother words."""
    return {"findings": [s.replace("A tumor is outlined", "A mass is seen") for s in draft["findings"]],
            "impression": draft["impression"].replace("Findings suggest", "Consistent with"),
            "notes": draft["notes"]}


def test_a_good_rewording_is_used_and_the_disclaimer_is_ours():
    f = feats()
    draft = template_report(f)
    seen = []
    out = fake_llm(reworded(draft), seen=seen).rewrite(f, draft)
    assert out["findings"][0].startswith("A mass is seen") and out["disclaimer"] == draft["disclaimer"]
    facts = json.loads(seen[0]["messages"][1]["content"])  # what the model is given
    assert set(facts) == {"measurements", "draft"} and facts["measurements"]["volumes_cm3"]["whole_tumor"] == 89.3
    assert set(facts["measurements"]) <= {"scan", "ai_type", "confidence", "doctor_type", "volumes_cm3",
                                          "mask_corrected", "side", "lobe"}  # numbers only, never pixels


@pytest.mark.parametrize("change, why", [
    (lambda d: {**d, "impression": d["impression"] + " Estimated grade 4."}, "numbers not in the draft"),
    (lambda d: {**d, "impression": "Findings suggest a meningioma."}, "tumor type not in the draft"),
    (lambda d: {**d, "notes": []}, "dropped or added sentences"),
])
def test_a_wrong_rewording_is_refused(change, why):
    draft = template_report(feats())
    with pytest.raises(LLMError, match=why):
        check(change(reworded(draft)), draft)


def test_broken_answers_and_server_errors_are_errors():
    f = feats()
    draft = template_report(f)
    for llm in (fake_llm("Sure! Here is the report."), fake_llm({"impression": 3}), fake_llm({}, status=500)):
        with pytest.raises(LLMError):
            llm.rewrite(f, draft)
    assert parse('```json\n{"findings": [], "impression": "x", "notes": []}\n```')["impression"] == "x"


@pytest.fixture
def case_result(client):
    case = client.post("/cases", headers=DOCTOR_A).json()
    client.post(f"/cases/{case['id']}/scans", headers=DOCTOR_A, files=four(lambda m: (f"{m}.nii.gz", nifti_bytes(brain()))))
    [result] = client.get(f"/cases/{case['id']}/results", headers=DOCTOR_A).json()
    return case, result, f"/cases/{case['id']}/results/{result['id']}/report"


def test_llm_draft_is_saved_until_the_numbers_change(client, case_result):
    case, result, url = case_result
    template = client.get(url, headers=DOCTOR_A).json()
    client.app.state.llm = fake_llm(reworded(template["report"]))
    made = client.post(f"{url}/draft", headers=DOCTOR_A).json()
    assert made["provider"] == "llm" and made["model"] == "test-model" and made["fallback"] is None
    client.app.state.llm = None  # reading the saved draft needs no LLM
    assert client.get(url, headers=ADMIN).json()["report"] == made["report"]
    # The doctor changes the type: the saved wording no longer matches the numbers, back to the template
    client.post(f"/cases/{case['id']}/reviews", headers=DOCTOR_A,
                json={"result_id": result["id"], "decision": "edited", "corrected_type": "meningioma"})
    assert client.get(url, headers=DOCTOR_A).json()["provider"] == "template"
    assert "report.draft" in [e["action"] for e in client.get("/audit", headers=ADMIN).json()]


def test_llm_failures_fall_back_to_the_template_with_a_reason(client, case_result):
    _, _, url = case_result
    out = client.post(f"{url}/draft", headers=DOCTOR_A).json()  # no LLM set up
    assert out["provider"] == "template" and "No LLM is set up" in out["fallback"]
    client.app.state.llm = fake_llm({}, status=503)
    out = client.post(f"{url}/draft", headers=DOCTOR_A).json()
    assert out["provider"] == "template" and out["fallback"].startswith("LLM call failed")
    assert client.get(url, headers=DOCTOR_A).json()["provider"] == "template"  # nothing saved
