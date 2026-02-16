"""Scan and result files: local disk for work, mirrored to a private Supabase Storage bucket when configured.

A hosted server can lose its disk on restart (a free Hugging Face Space does) while the case list lives in
Postgres; the mirror keeps every case openable. Folders are flat (cases/<case>/<scan>/, results/<result>/):
push() uploads a folder after it is written, pull() brings it back before it is read. Without a key
(local dev, tests) both do nothing and the disk is the only copy.
"""
from pathlib import Path

import httpx


class FileStoreError(Exception):
    pass


class FileStore:
    def __init__(self, root: str | Path, supabase_url: str = "", key: str = "", bucket: str = "neurolens",
                 transport: httpx.BaseTransport | None = None):
        self.root = Path(root)
        self.bucket = bucket
        self.http = None
        if supabase_url and key:
            # New secret keys (sb_secret_...) go in the apikey header only; legacy service_role JWTs also as Bearer
            headers = {"apikey": key} | ({} if key.startswith("sb_") else {"Authorization": f"Bearer {key}"})
            self.http = httpx.Client(base_url=f"{supabase_url.rstrip('/')}/storage/v1", headers=headers,
                                     timeout=120, transport=transport)

    @property
    def mirrored(self) -> bool:
        return self.http is not None

    def _check(self, r: httpx.Response, what: str) -> httpx.Response:
        if r.status_code >= 400:
            raise FileStoreError(f"file storage {what} failed ({r.status_code})")
        return r

    def push(self, folder: str) -> None:
        """Upload every file of a folder (relative to the root), replacing older copies."""
        if not self.mirrored:
            return
        for f in sorted((self.root / folder).iterdir()):
            if f.is_file():
                self._check(self.http.post(f"/object/{self.bucket}/{folder}/{f.name}", content=f.read_bytes(),
                                           headers={"x-upsert": "true", "Content-Type": "application/octet-stream"}),
                            f"upload of {f.name}")

    def pull(self, folder: str) -> Path:
        """The local folder, with any file only in the mirror downloaded first."""
        local = self.root / folder
        if not self.mirrored:
            return local
        listed = self._check(self.http.post(f"/object/list/{self.bucket}", json={"prefix": folder, "limit": 1000}),
                             "listing").json()
        names = [o["name"] for o in listed if o.get("id")]  # entries without an id are sub-folders
        for name in names:
            if not (local / name).is_file():
                local.mkdir(parents=True, exist_ok=True)
                data = self._check(self.http.get(f"/object/{self.bucket}/{folder}/{name}"), f"download of {name}").content
                tmp = local / f".{name}.part"  # a half-written file must never look complete
                tmp.write_bytes(data)
                tmp.replace(local / name)
        return local
