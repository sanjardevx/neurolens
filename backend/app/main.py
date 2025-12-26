"""NeuroLens API: doctor-only. Doctors create and review their own cases; admins see everything and the audit log.

    uvicorn app.main:app --reload
"""
import secrets
from collections.abc import Iterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse
import numpy as np
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.orm import Session

from .auth import User, current_user, require
from .config import get_settings
from .analysis import Models, analyze, model_version
from .db import AuditLog, Case, Database, Result, Scan, init_db
from .scans import MODALITIES, ScanError, check_image, check_volumes, read_image, read_volume, save_image, save_volume

CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"  # no 0/O, 1/I/L: easy to read out loud


def create_app(database_url: str | None = None, storage_dir: str | None = None, model_dir: str | None = None) -> FastAPI:
    settings = get_settings()
    db = Database(database_url or settings.database_url)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        init_db(db.engine)
        yield

    app = FastAPI(title="NeuroLens API", version="0.1.0", lifespan=lifespan,
                  description="Research and portfolio project, not a medical device.")
    app.state.db = db
    app.state.storage = Path(storage_dir or settings.storage_dir)
    app.state.models = Models(model_dir or settings.model_dir)
    app.state.max_upload_bytes = settings.max_upload_mb * 1024 * 1024
    register_routes(app)
    return app


def get_db(request: Request) -> Iterator[Session]:
    yield from request.app.state.db.session()


def audit(db: Session, user: User, action: str, case_id: str | None = None, **detail: Any) -> None:
    db.add(AuditLog(user_id=user.id, role=user.role, action=action, case_id=case_id, detail=detail or None))


class CaseOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    code: str
    status: str
    created_at: datetime
    updated_at: datetime


class ScanOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    file_format: str
    qc_passed: bool | None
    info: dict[str, Any] | None
    created_at: datetime


class ResultOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    scan_id: str
    model_version: str
    progress: int
    tumor_type: str | None
    confidence: float | None
    probabilities: dict[str, float] | None
    volumes_cm3: dict[str, float] | None
    side: str | None
    lobe_estimate: str | None
    has_mask: bool
    has_gradcam: bool
    info: dict[str, Any] | None
    error: str | None
    created_at: datetime

    @classmethod
    def of(cls, r: Result) -> "ResultOut":
        fields = {k: getattr(r, k) for k in cls.model_fields if k not in ("has_mask", "has_gradcam")}
        return cls(**fields, has_mask=r.mask_path is not None, has_gradcam=r.gradcam_path is not None)


class MeOut(BaseModel):
    id: str
    email: str
    role: str


class AuditOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    at: datetime
    user_id: str
    role: str
    action: str
    case_id: str | None
    detail: dict[str, Any] | None


def new_case_code(db: Session) -> str:
    while True:
        code = "NL-" + "".join(secrets.choice(CODE_ALPHABET) for _ in range(6))
        if db.scalar(select(Case.id).where(Case.code == code)) is None:
            return code


def get_case_for(db: Session, user: User, case_id: str) -> Case:
    """A doctor only sees their own cases; to them, someone else's case does not exist (404, not 403)."""
    case = db.get(Case, case_id)
    if case is None or (user.role == "doctor" and case.doctor_id != user.id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Case not found")
    return case


def read_upload(request: Request, upload: UploadFile) -> bytes:
    limit = request.app.state.max_upload_bytes
    data = upload.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"{upload.filename}: file is too large")
    return data


def store_upload(request: Request, scan_dir: Path, image: UploadFile | None,
                 volumes: dict[str, UploadFile | None]) -> tuple[str, dict]:
    """Read, check and save one upload: either one 2D image or the 4 MRI types. Returns (format, info)."""
    given = [m for m, f in volumes.items() if f is not None]
    if image is not None and given:
        raise ScanError("upload either one image or the 4 MRI types, not both")
    if image is not None:
        img = read_image(read_upload(request, image), image.filename or "image")
        check_image(img)
        save_image(img, scan_dir / "image.png")
        return "image", {"shape": list(img.shape[:2])}
    if missing := [m for m in MODALITIES if m not in given]:
        raise ScanError(f"missing MRI types: {', '.join(missing)} (the 3D model needs {', '.join(MODALITIES)})")
    read = {m: read_volume(read_upload(request, f), f.filename or m) for m, f in volumes.items()}
    check_volumes({m: (vol, affine) for m, (vol, affine, _) in read.items()})
    for m, (vol, affine, _) in read.items():
        save_volume(vol, affine, scan_dir / f"{m}.nii.gz")
    vol, affine, _ = read["flair"]
    formats = {"dicom" if (f.filename or "").lower().endswith(".zip") else "nifti" for f in volumes.values()}
    info = {"shape": list(vol.shape), "voxel_mm": [round(float(v), 3) for v in np.sqrt((affine[:3, :3] ** 2).sum(0))],
            "dicom_tags": {m: t for m, (_, _, t) in read.items() if t} or None}
    return formats.pop() if len(formats) == 1 else "mixed", info


def register_routes(app: FastAPI) -> None:
    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/me", response_model=MeOut)
    def me(user: User = Depends(current_user)) -> MeOut:
        return MeOut(id=user.id, email=user.email, role=user.role)

    @app.post("/cases", response_model=CaseOut, status_code=status.HTTP_201_CREATED)
    def create_case(user: User = Depends(require("doctor")), db: Session = Depends(get_db)) -> Case:
        case = Case(code=new_case_code(db), doctor_id=user.id)
        db.add(case)
        db.flush()
        audit(db, user, "case.create", case.id, code=case.code)
        db.commit()
        return case

    @app.get("/cases", response_model=list[CaseOut])
    def list_cases(user: User = Depends(current_user), db: Session = Depends(get_db)) -> list[Case]:
        query = select(Case).order_by(Case.created_at.desc())
        if user.role == "doctor":
            query = query.where(Case.doctor_id == user.id)
        return list(db.scalars(query))

    @app.get("/cases/{case_id}", response_model=CaseOut)
    def read_case(case_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> Case:
        case = get_case_for(db, user, case_id)
        audit(db, user, "case.view", case.id)  # reading a case is logged too: medical data access is audited
        db.commit()
        return case

    @app.post("/cases/{case_id}/scans", response_model=ScanOut, status_code=status.HTTP_201_CREATED)
    def upload_scan(case_id: str, request: Request, background: BackgroundTasks,
                    image: UploadFile | None = File(None, description="one 2D MRI slice, JPG or PNG"),
                    flair: UploadFile | None = File(None, description=".nii / .nii.gz or a .zip of DICOM slices"),
                    t1: UploadFile | None = File(None), t1ce: UploadFile | None = File(None),
                    t2: UploadFile | None = File(None),
                    user: User = Depends(require("doctor")), db: Session = Depends(get_db)) -> Scan:
        case = get_case_for(db, user, case_id)
        scan = Scan(case_id=case.id, file_format="", storage_path=f"cases/{case.id}/pending")
        scan_dir = request.app.state.storage / "cases" / case.id
        try:
            db.add(scan)
            db.flush()
            scan.storage_path = f"cases/{case.id}/{scan.id}"
            fmt, info = store_upload(request, scan_dir / scan.id, image, {"flair": flair, "t1": t1, "t1ce": t1ce, "t2": t2})
        except ScanError as e:
            db.rollback()
            audit(db, user, "scan.rejected", case.id, reason=str(e))  # nothing from the file is kept
            db.commit()
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(e)) from e
        scan.file_format, scan.info, scan.qc_passed = fmt, info, True
        audit(db, user, "scan.upload", case.id, scan_id=scan.id, format=fmt)
        # Start the AI right away; it runs after the response is sent, the case shows its progress
        result = Result(case_id=case.id, scan_id=scan.id, model_version=model_version(fmt))
        db.add(result)
        case.status = "processing"
        db.commit()
        app = request.app
        background.add_task(analyze, app.state.db, app.state.storage, app.state.models, result.id, user)
        return scan

    @app.get("/cases/{case_id}/scans", response_model=list[ScanOut])
    def list_scans(case_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> list[Scan]:
        return get_case_for(db, user, case_id).scans

    @app.get("/cases/{case_id}/results", response_model=list[ResultOut])
    def list_results(case_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> list:
        case = get_case_for(db, user, case_id)
        return [ResultOut.of(r) for r in sorted(case.results, key=lambda r: r.created_at, reverse=True)]

    @app.get("/cases/{case_id}/results/{result_id}/{kind}")
    def result_file(case_id: str, result_id: str, kind: str, request: Request,
                    user: User = Depends(current_user), db: Session = Depends(get_db)) -> FileResponse:
        """The 3D mask (BraTS labels 1 / 2 / 4, for the viewer) or the Grad-CAM image."""
        case = get_case_for(db, user, case_id)
        result = db.get(Result, result_id)
        name = {"mask": result and result.mask_path, "gradcam": result and result.gradcam_path}.get(kind)
        if result is None or result.case_id != case.id or not name:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "File not found")
        media = "image/png" if name.endswith(".png") else "application/gzip"
        return FileResponse(request.app.state.storage / "results" / result.id / name, media_type=media,
                            filename=f"{case.code}_{name}")

    @app.get("/audit", response_model=list[AuditOut])
    def audit_log(limit: int = 200, user: User = Depends(require("admin")), db: Session = Depends(get_db)) -> list:
        return list(db.scalars(select(AuditLog).order_by(AuditLog.id.desc()).limit(min(limit, 1000))))


app = create_app()
