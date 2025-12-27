"""Every endpoint × every kind of caller. Doctors only reach their own cases (others' look missing: 404);
admins read everything but don't create or review; no login → 401; a login without a role → 403."""
import pytest

from .conftest import ADMIN, DOCTOR_A, DOCTOR_B, auth
from .fakes import slice_png

CALLERS = {
    "owner": DOCTOR_A,
    "other_doctor": DOCTOR_B,
    "admin": ADMIN,
    "no_login": {},
    "no_role": auth(sub="someone", role=None),
}

# (method, path, expected status per caller: owner, other doctor, admin, no login, no role)
ENDPOINTS = [
    ("GET", "/health", (200, 200, 200, 200, 200)),
    ("GET", "/me", (200, 200, 200, 401, 403)),
    ("POST", "/cases", (201, 201, 403, 401, 403)),
    ("GET", "/cases", (200, 200, 200, 401, 403)),
    ("GET", "/cases/{case}", (200, 404, 200, 401, 403)),
    ("POST", "/cases/{case}/scans", (201, 404, 403, 401, 403)),
    ("GET", "/cases/{case}/scans", (200, 404, 200, 401, 403)),
    ("GET", "/cases/{case}/results", (200, 404, 200, 401, 403)),
    ("GET", "/cases/{case}/results/{result}/gradcam", (200, 404, 200, 401, 403)),
    ("POST", "/cases/{case}/reviews", (201, 404, 403, 401, 403)),
    ("GET", "/cases/{case}/reviews", (200, 404, 200, 401, 403)),
    ("GET", "/audit", (403, 403, 200, 401, 403)),
]


@pytest.fixture
def ids(client):
    case = client.post("/cases", headers=DOCTOR_A).json()
    client.post(f"/cases/{case['id']}/scans", headers=DOCTOR_A, files={"image": ("s.png", slice_png(fmt="PNG"))})
    [result] = client.get(f"/cases/{case['id']}/results", headers=DOCTOR_A).json()
    return {"case": case["id"], "result": result["id"]}


@pytest.mark.parametrize("caller", list(CALLERS))
@pytest.mark.parametrize("method, path, expected", ENDPOINTS, ids=[f"{m} {p}" for m, p, _ in ENDPOINTS])
def test_access(client, ids, caller, method, path, expected):
    kwargs = {"headers": CALLERS[caller]}
    if path.endswith("/scans") and method == "POST":
        kwargs["files"] = {"image": ("s.png", slice_png(fmt="PNG"))}
    if path.endswith("/reviews") and method == "POST":
        kwargs["json"] = {"result_id": ids["result"], "decision": "approved"}
    r = client.request(method, path.format(**ids), **kwargs)
    assert r.status_code == expected[list(CALLERS).index(caller)], r.text
