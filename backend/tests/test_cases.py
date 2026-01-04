import re

from sqlalchemy import inspect

from .conftest import ADMIN, DOCTOR_A, DOCTOR_B


def test_all_tables_exist(client):
    tables = set(inspect(client.app.state.db.engine).get_table_names())
    assert {"cases", "scans", "results", "reviews", "audit_log"} <= tables


def test_doctor_creates_a_case_with_an_anonymous_code(client):
    r = client.post("/cases", headers=DOCTOR_A)
    assert r.status_code == 201
    case = r.json()
    assert re.fullmatch(r"NL-[A-HJ-KM-NP-Z2-9]{6}", case["code"]) and case["status"] == "new"
    assert "doctor_id" not in case


def test_admin_cannot_create_cases(client):
    assert client.post("/cases", headers=ADMIN).status_code == 403


def test_doctors_only_see_their_own_cases(client):
    a = client.post("/cases", headers=DOCTOR_A).json()
    b = client.post("/cases", headers=DOCTOR_B).json()
    assert [c["id"] for c in client.get("/cases", headers=DOCTOR_A).json()] == [a["id"]]
    assert [c["id"] for c in client.get("/cases", headers=DOCTOR_B).json()] == [b["id"]]
    assert client.get(f"/cases/{a['id']}", headers=DOCTOR_B).status_code == 404  # looks like it doesn't exist
    assert client.get(f"/cases/{a['id']}", headers=DOCTOR_A).status_code == 200


def test_admin_sees_all_cases(client):
    a = client.post("/cases", headers=DOCTOR_A).json()
    client.post("/cases", headers=DOCTOR_B)
    assert len(client.get("/cases", headers=ADMIN).json()) == 2
    assert client.get(f"/cases/{a['id']}", headers=ADMIN).status_code == 200


def test_actions_are_written_to_the_audit_log(client):
    case = client.post("/cases", headers=DOCTOR_A).json()
    client.get(f"/cases/{case['id']}", headers=ADMIN)
    log = client.get("/audit", headers=ADMIN).json()
    assert [(e["action"], e["user_id"], e["role"]) for e in log] == [
        ("case.view", "admin-1", "admin"), ("case.create", "doctor-a", "doctor")]
    assert log[1]["case_id"] == case["id"] and log[1]["detail"] == {"code": case["code"]}


def test_only_admins_read_the_audit_log(client):
    assert client.get("/audit", headers=DOCTOR_A).status_code == 403


def test_dashboard_origin_may_call_the_api(client):
    preflight = {"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET",
                 "Access-Control-Request-Headers": "authorization"}
    r = client.options("/cases", headers=preflight)
    assert r.status_code == 200 and r.headers["access-control-allow-origin"] == "http://localhost:3000"
    other = client.options("/cases", headers={**preflight, "Origin": "https://evil.test"})
    assert "access-control-allow-origin" not in other.headers
