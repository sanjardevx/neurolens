"""Full doctor flow against a running API: me → new case → upload 4 MRI types → wait for the AI → results,
mask → approve → audit log (admin).

    python backend/scripts/smoke_flow.py --url http://localhost:8000 --doctor-token <jwt> --admin-token <jwt>

Local Docker without Supabase: start the API with a test SUPABASE_JWT_SECRET and let the script sign test tokens:
    python backend/scripts/smoke_flow.py --url http://localhost:8000 --supabase-url https://local.test --secret <same secret>

The scan is synthetic (an ellipsoid 'brain' with a bright block), so the numbers check the pipeline, not the model.
"""
import argparse
import gzip
import sys
import time

import httpx
import jwt
import nibabel as nib
import numpy as np


def test_token(secret: str, supabase_url: str, sub: str, role: str) -> str:
    """Only for a local API started with the same test secret. Real tokens come from Supabase login."""
    claims = {"sub": sub, "email": f"{sub}@local.test", "aud": "authenticated", "iss": f"{supabase_url}/auth/v1",
              "exp": int(time.time()) + 3600, "app_metadata": {"role": role}}
    return jwt.encode(claims, secret, algorithm="HS256")


def fake_mri(shape=(240, 240, 155)) -> dict[str, bytes]:
    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    inside = ((x - 120) / 70) ** 2 + ((y - 120) / 85) ** 2 + ((z - 75) / 68) ** 2 < 1
    files = {}
    for i, m in enumerate(["flair", "t1", "t1ce", "t2"]):
        vol = np.random.default_rng(i).uniform(200, 400, shape) * inside
        vol[150:170, 110:130, 70:85] = 900  # a bright block on the right
        files[m] = gzip.compress(nib.Nifti1Image(vol.astype(np.float32), np.diag([1.0, 1, 1, 1])).to_bytes())
    return files


def check(r: httpx.Response, expected: int, what: str) -> dict | list:
    if r.status_code != expected:
        sys.exit(f"FAIL {what}: {r.status_code} {r.text[:300]}")
    print(f"ok   {what}")
    return r.json() if r.headers.get("content-type", "").startswith("application/json") else {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--doctor-token")
    ap.add_argument("--admin-token")
    ap.add_argument("--secret", help="sign local test tokens with this HS256 secret instead")
    ap.add_argument("--supabase-url", default="https://local.test")
    ap.add_argument("--timeout", type=int, default=300, help="seconds to wait for the 3D analysis")
    args = ap.parse_args()
    doctor = args.doctor_token or test_token(args.secret, args.supabase_url, "smoke-doctor", "doctor")
    admin = args.admin_token or test_token(args.secret, args.supabase_url, "smoke-admin", "admin")
    api = httpx.Client(base_url=args.url, timeout=120)
    as_doctor, as_admin = {"Authorization": f"Bearer {doctor}"}, {"Authorization": f"Bearer {admin}"}

    check(api.get("/health"), 200, "health")
    me = check(api.get("/me", headers=as_doctor), 200, "login token accepted")
    print(f"     role: {me['role']}")
    case = check(api.post("/cases", headers=as_doctor), 201, "new case")
    print(f"     case code: {case['code']}")
    files = {m: (f"{m}.nii.gz", data) for m, data in fake_mri().items()}
    t0 = time.time()
    scan = check(api.post(f"/cases/{case['id']}/scans", headers=as_doctor, files=files), 201, "upload 4 MRI types")
    print(f"     stored as {scan['file_format']}, shape {scan['info']['shape']}, voxel {scan['info']['voxel_mm']} mm")

    while True:
        [result] = check(api.get(f"/cases/{case['id']}/results", headers=as_doctor), 200, "poll results")
        print(f"     progress {result['progress']}%")
        if result["progress"] == 100 or result["error"]:
            break
        if time.time() - t0 > args.timeout:
            sys.exit("FAIL analysis did not finish in time")
        time.sleep(2)
    if result["error"]:
        sys.exit(f"FAIL analysis: {result['error']}")
    print(f"     done in {time.time() - t0:.1f} s | {result['model_version']}")
    print(f"     type {result['tumor_type']} ({result['confidence']:.2f}) | side {result['side']} | "
          f"lobe {result['lobe_estimate']} | volumes {result['volumes_cm3']}")
    mask = api.get(f"/cases/{case['id']}/results/{result['id']}/mask", headers=as_doctor)
    check(mask, 200, f"download mask ({len(mask.content) / 1e3:.0f} kB)")
    check(api.get(f"/cases/{case['id']}/results/{result['id']}/gradcam", headers=as_doctor), 200, "download Grad-CAM")
    check(api.post(f"/cases/{case['id']}/reviews", headers=as_doctor,
                   json={"result_id": result["id"], "decision": "approved"}), 201, "approve")
    status = check(api.get(f"/cases/{case['id']}", headers=as_doctor), 200, "case status")["status"]
    log = check(api.get("/audit", headers=as_admin), 200, "audit log (admin)")
    actions = [e["action"] for e in log if e["case_id"] == case["id"]]
    print(f"     case status: {status} | audit: {' ← '.join(actions)}")
    print("PASS full flow")


if __name__ == "__main__":
    main()
