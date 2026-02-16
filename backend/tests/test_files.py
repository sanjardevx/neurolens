import shutil

import httpx
import pytest

from app.files import FileStore, FileStoreError

from .conftest import DOCTOR_A
from .fakes import FakeBucket, brain, four, nifti_bytes
from .test_analysis import results, upload


def store(tmp_path, bucket, key="sb_secret_test"):
    return FileStore(tmp_path / "disk", "https://proj.supabase.co", key, transport=httpx.MockTransport(bucket))


def test_push_then_pull_brings_files_back_after_the_disk_is_lost(tmp_path):
    bucket = FakeBucket()
    files = store(tmp_path, bucket)
    (files.root / "results/r1").mkdir(parents=True)
    (files.root / "results/r1/mask.nii.gz").write_bytes(b"mask")
    files.push("results/r1")
    assert bucket.objects == {"results/r1/mask.nii.gz": b"mask"}
    shutil.rmtree(files.root)  # the hosted server restarted with an empty disk
    assert (files.pull("results/r1") / "mask.nii.gz").read_bytes() == b"mask"
    assert not list((files.root / "results/r1").glob(".*"))  # no half-written leftovers


def test_secret_key_header_and_legacy_jwt(tmp_path):
    bucket = FakeBucket()
    store(tmp_path, bucket).pull("x")
    assert bucket.headers[-1]["apikey"] == "sb_secret_test" and "authorization" not in bucket.headers[-1]
    store(tmp_path, bucket, key="eyJ.legacy.jwt").pull("x")
    assert bucket.headers[-1]["authorization"] == "Bearer eyJ.legacy.jwt"


def test_without_a_key_the_disk_is_the_only_copy(tmp_path):
    files = FileStore(tmp_path / "disk")
    assert not files.mirrored and files.pull("results/r1") == tmp_path / "disk/results/r1"
    files.push("results/r1")  # nothing to do, no error


def test_storage_errors_are_reported(tmp_path):
    files = store(tmp_path, lambda request: httpx.Response(500))
    (files.root / "f").mkdir(parents=True)
    (files.root / "f/a.bin").write_bytes(b"1")
    with pytest.raises(FileStoreError, match="upload of a.bin failed"):
        files.push("f")


def test_case_still_opens_after_the_server_loses_its_disk(client, tmp_path):
    case = client.post("/cases", headers=DOCTOR_A).json()
    bucket = FakeBucket()
    client.app.state.files = FileStore(client.app.state.storage, "https://proj.supabase.co", "sb_secret_test",
                                       transport=httpx.MockTransport(bucket))
    scan = upload(client, case, four(lambda m: (f"{m}.nii.gz", nifti_bytes(brain()))))
    [r] = results(client, case)
    assert r["progress"] == 100 and any(k.endswith("mask.nii.gz") for k in bucket.objects)
    shutil.rmtree(client.app.state.storage)  # restart: empty disk, the case list is still in Postgres
    base = f"/cases/{case['id']}"
    assert client.get(f"{base}/scans/{scan['id']}/flair", headers=DOCTOR_A).status_code == 200
    assert client.get(f"{base}/results/{r['id']}/mask", headers=DOCTOR_A).status_code == 200
