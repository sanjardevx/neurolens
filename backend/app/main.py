"""NeuroLens API: doctor-only. Doctors create and review their own cases; admins see everything and the audit log.

    uvicorn app.main:app --reload
"""
import secrets
from collections.abc import Iterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from neurolens_ml.infer_cls import CLASSES
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from .analysis import Models, analyze, model_version
from .auth import User, current_user, require
from .config import get_settings
from .db import DECISIONS, REJECT_REASONS, AuditLog, Case, Database, Result, Review, Scan, init_db
from .masks import CORRECTED, MaskError, changed_voxels, read_corrected, save_corrected
from .report import as_text, features, template_report
from .scans import MODALITIES, ScanError, check_image, check_volumes, read_image, read_volume, save_image, save_volume

VIEW_LOG_GAP = timedelta(minutes=1)  # one "opened the case" entry per person and case per minute
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
    # The dashboard runs on another origin; the token goes in a header, so no cookies are allowed
    app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
                       allow_methods=["GET", "POST"], allow_headers=["Authorization", "Content-Type"])
    register_routes(app)
    return app


def get_db(request: Request) -> Iterator[Session]:
    yield from request.app.state.db.session()


def audit(db: Session, user: User, action: str, case_id: str | None = None, **detail: Any) -> None:
    db.add(AuditLog(user_id=user.id, user_email=user.email, role=user.role, action=action, case_id=case_id,
                    detail=detail or None))


class CaseOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    code: str
    status: str
    created_at: datetime
    updated_at: datetime


class CaseRow(CaseOut):
    """One worklist line: the case plus a short summary of its latest result and review."""
    tumor_type: str | None = None
    confidence: float | None = None
    tumor_cm3: float | None = None  # whole tumor volume; only the 3D model measures it
    progress: int | None = None
    error: str | None = None
    decision: str | None = None

    @classmethod
    def of(cls, case: Case) -> "CaseRow":
        result = max(case.results, key=lambda r: r.created_at, default=None)
        review = max(case.reviews, key=lambda r: r.created_at, default=None)
        # The doctor's corrections win over the AI: their tumor type and their mask's volume
        corrected = ((result and result.info) or {}).get("corrected_mask") or {}
        volumes = corrected.get("volumes_cm3") or (result and result.volumes_cm3) or {}
        tumor_type = (review and review.corrected_type) or (result and result.tumor_type)
        return cls.model_validate(case).model_copy(update={
            # the AI's confidence was for its own type, so it means nothing next to the doctor's type
            "tumor_type": tumor_type, "confidence": None if review and review.corrected_type else result and result.confidence,
            "tumor_cm3": volumes.get("whole_tumor"), "progress": result and result.progress,
            "error": result and result.error, "decision": review and review.decision})


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
    tumor_type: str | None = None
    confidence: float | None = None
    probabilities: dict[str, float] | None
    volumes_cm3: dict[str, float] | None
    side: str | None
    lobe_estimate: str | None
    has_mask: bool
    has_corrected_mask: bool
    has_gradcam: bool
    info: dict[str, Any] | None
    error: str | None = None
    created_at: datetime

    @classmethod
    def of(cls, r: Result) -> "ResultOut":
        computed = ("has_mask", "has_corrected_mask", "has_gradcam")
        fields = {k: getattr(r, k) for k in cls.model_fields if k not in computed}
        return cls(**fields, has_mask=r.mask_path is not None, has_gradcam=r.gradcam_path is not None,
                   has_corrected_mask=bool((r.info or {}).get("corrected_mask")))


class ReviewIn(BaseModel):
    """approved: the AI result is right. edited: right with changes (tumor type, report text and / or the mask).
    rejected: not usable, with a reason (Week 9 retraining learns from edits and rejections)."""
    result_id: str
    decision: Literal[DECISIONS]
    reason: Literal[REJECT_REASONS] | None = None
    note: str | None = Field(None, max_length=2000)
    corrected_type: Literal[tuple(CLASSES)] | None = None
    report_text: str | None = Field(None, max_length=10000)

    @model_validator(mode="after")
    def check_decision(self) -> "ReviewIn":
        if self.decision == "rejected" and self.reason is None:
            raise ValueError("a rejection needs a reason")
        if self.reason == "other" and not (self.note or "").strip():
            raise ValueError("reason 'other' needs a note")
        if self.decision != "rejected" and self.reason is not None:
            raise ValueError("a reason is only for rejections")
        if self.decision == "approved" and self.corrected_type:
            raise ValueError("changing the tumor type is an edit, not an approval")
        return self


class ReviewOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    result_id: str
    doctor_id: str
    decision: str
    reason: str | None
    note: str | None
    corrected_type: str | None
    report_text: str | None
    created_at: datetime


class MeOut(BaseModel):
    id: str
    email: str
    role: str


class AuditOut(BaseModel):
    at: datetime
    user_id: str
    user_email: str | None
    role: str
    action: str
    case_id: str | None
    case_code: str | None
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


def medical_file(path: Path, media: str, filename: str) -> FileResponse:
    """Scan and result files are never cached by the browser: no copies left on a shared clinic computer,
    and a corrected mask is never hidden behind an old cached one."""
    return FileResponse(path, media_type=media, filename=filename, headers={"Cache-Control": "no-store"})


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

    @app.get("/cases", response_model=list[CaseRow])
    def list_cases(user: User = Depends(current_user), db: Session = Depends(get_db)) -> list[CaseRow]:
        query = (select(Case).order_by(Case.created_at.desc())
                 .options(selectinload(Case.results), selectinload(Case.reviews)))  # 3 queries, not one per case
        if user.role == "doctor":
            query = query.where(Case.doctor_id == user.id)
        return [CaseRow.of(c) for c in db.scalars(query)]

    @app.get("/cases/{case_id}", response_model=CaseOut)
    def read_case(case_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> Case:
        case = get_case_for(db, user, case_id)
        # Reading a case is logged too (medical data access is audited), but the page re-reads it after
        # each action; one entry per minute keeps the log readable without hiding a real visit
        since = datetime.now(UTC) - VIEW_LOG_GAP
        seen = db.scalar(select(AuditLog.id).where(AuditLog.user_id == user.id, AuditLog.case_id == case.id,
                                                   AuditLog.action == "case.view", AuditLog.at > since).limit(1))
        if seen is None:
            audit(db, user, "case.view", case.id)
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

    @app.get("/cases/{case_id}/scans/{scan_id}/{kind}")
    def scan_file(case_id: str, scan_id: str, kind: str, request: Request,
                  user: User = Depends(current_user), db: Session = Depends(get_db)) -> FileResponse:
        """One stored, de-identified scan file for the viewer: an MRI type (flair, t1, t1ce, t2) or the 2D image.
        Opening the case is already audited (case.view), so file reads are not logged one by one."""
        case = get_case_for(db, user, case_id)
        scan = db.get(Scan, scan_id)
        name = {**{m: f"{m}.nii.gz" for m in MODALITIES}, "image": "image.png"}.get(kind)
        path = request.app.state.storage / scan.storage_path / name if scan and name else None
        if scan is None or scan.case_id != case.id or path is None or not path.is_file():
            raise HTTPException(status.HTTP_404_NOT_FOUND, "File not found")
        media = "image/png" if kind == "image" else "application/gzip"
        return medical_file(path, media, f"{case.code}_{name}")

    @app.get("/cases/{case_id}/results", response_model=list[ResultOut])
    def list_results(case_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> list:
        case = get_case_for(db, user, case_id)
        return [ResultOut.of(r) for r in sorted(case.results, key=lambda r: r.created_at, reverse=True)]

    # Registered before the {kind} file route below, which would otherwise catch ".../report"
    @app.get("/cases/{case_id}/results/{result_id}/report")
    def report_draft(case_id: str, result_id: str, user: User = Depends(current_user),
                     db: Session = Depends(get_db)) -> dict:
        """Report draft from the numbers only (type, confidence, volumes, location), with the doctor's
        corrections (type, mask) taking priority. The LLM rewording comes on top of this later."""
        case = get_case_for(db, user, case_id)
        result = db.get(Result, result_id)
        if result is None or result.case_id != case.id:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Result not found")
        if result.progress < 100 or result.error:
            raise HTTPException(status.HTTP_409_CONFLICT, "The analysis is not finished; no report yet")
        review = max((r for r in case.reviews if r.result_id == result.id), key=lambda r: r.created_at, default=None)
        f = features(result, review)
        report = template_report(f)
        return {"provider": "template", "report": report, "text": as_text(report), "features": asdict(f)}

    @app.get("/cases/{case_id}/results/{result_id}/{kind}")
    def result_file(case_id: str, result_id: str, kind: str, request: Request,
                    user: User = Depends(current_user), db: Session = Depends(get_db)) -> FileResponse:
        """The 3D mask (BraTS labels 1 / 2 / 4, for the viewer) or the Grad-CAM image."""
        case = get_case_for(db, user, case_id)
        result = db.get(Result, result_id)
        corrected = result and (result.info or {}).get("corrected_mask")
        name = {"mask": result and result.mask_path, "gradcam": result and result.gradcam_path,
                "corrected": corrected and CORRECTED}.get(kind)
        if result is None or result.case_id != case.id or not name:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "File not found")
        media = "image/png" if name.endswith(".png") else "application/gzip"
        return medical_file(request.app.state.storage / "results" / result.id / name, media, f"{case.code}_{name}")

    @app.post("/cases/{case_id}/results/{result_id}/mask", response_model=ResultOut)
    def correct_mask(case_id: str, result_id: str, request: Request, mask: UploadFile = File(...),
                     user: User = Depends(require("doctor")), db: Session = Depends(get_db)) -> ResultOut:
        """The doctor's fixed 3D mask from the viewer. The AI mask stays untouched (Week 9 learns from the
        difference); a new save replaces the previous correction. Volumes are measured again."""
        case = get_case_for(db, user, case_id)
        result = db.get(Result, result_id)
        if result is None or result.case_id != case.id:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Result not found")
        if result.progress < 100 or result.error or not result.mask_path:
            raise HTTPException(status.HTTP_409_CONFLICT, "This result has no finished 3D mask to correct")
        if case.status == "reviewed":
            raise HTTPException(status.HTTP_409_CONFLICT, "The case is already reviewed; change the decision first")
        out_dir = request.app.state.storage / "results" / result.id
        try:
            labels, affine = read_corrected(read_upload(request, mask), out_dir / result.mask_path)
        except MaskError as e:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(e)) from e
        volumes = save_corrected(labels, affine, out_dir)
        # A new dict, so SQLAlchemy sees the JSON change
        result.info = {**(result.info or {}), "corrected_mask": {
            "by": user.id, "at": datetime.now(UTC).isoformat(timespec="seconds"), "volumes_cm3": volumes}}
        audit(db, user, "mask.correct", case.id, result_id=result.id,
              changed_voxels=changed_voxels(labels, out_dir / result.mask_path))
        db.commit()
        return ResultOut.of(result)

    @app.post("/cases/{case_id}/reviews", response_model=ReviewOut, status_code=status.HTTP_201_CREATED)
    def review_case(case_id: str, body: ReviewIn, user: User = Depends(require("doctor")),
                    db: Session = Depends(get_db)) -> Review:
        """Only the case's doctor decides. A new review replaces the previous one; all of them stay in the history."""
        case = get_case_for(db, user, case_id)
        result = db.get(Result, body.result_id)
        if result is None or result.case_id != case.id:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Result not found")
        if result.progress < 100 or result.error:
            raise HTTPException(status.HTTP_409_CONFLICT, "The analysis is not finished; nothing to review yet")
        # A corrected mask is a change too: then "approved" would claim the AI was right as it was
        mask_fixed = bool((result.info or {}).get("corrected_mask"))
        if body.decision == "edited" and not (body.report_text or body.corrected_type or mask_fixed):
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT,
                                "An edit needs a corrected tumor type, report text or a corrected mask")
        if body.decision == "approved" and mask_fixed:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT,
                                "The mask was corrected, so this is an edit, not a plain approval")
        review = Review(case_id=case.id, doctor_id=user.id, **body.model_dump())
        db.add(review)
        case.status = "reviewed"
        db.flush()
        action = {"approved": "review.approve", "edited": "review.edit", "rejected": "review.reject"}[body.decision]
        audit(db, user, action, case.id, review_id=review.id, result_id=result.id, reason=body.reason,
              corrected_type=body.corrected_type)
        db.commit()
        return review

    @app.post("/cases/{case_id}/reopen", response_model=CaseOut)
    def reopen_case(case_id: str, user: User = Depends(require("doctor")), db: Session = Depends(get_db)) -> Case:
        """"Change decision": the case needs review again (e.g. to fix the mask). Past reviews stay in the history."""
        case = get_case_for(db, user, case_id)
        if case.status != "reviewed":
            raise HTTPException(status.HTTP_409_CONFLICT, "Only a reviewed case can be reopened")
        case.status = "ready"
        audit(db, user, "review.reopen", case.id)
        db.commit()
        return case

    @app.get("/cases/{case_id}/reviews", response_model=list[ReviewOut])
    def list_reviews(case_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> list:
        return sorted(get_case_for(db, user, case_id).reviews, key=lambda r: r.created_at, reverse=True)

    @app.get("/audit", response_model=list[AuditOut])
    def audit_log(limit: int = 200, action: str | None = None, case_code: str | None = None,
                  user: User = Depends(require("admin")), db: Session = Depends(get_db)) -> list[AuditOut]:
        """Newest first. action: a prefix such as "review." or "mask."; case_code: e.g. NL-7KQ2XD."""
        query = (select(AuditLog, Case.code).outerjoin(Case, Case.id == AuditLog.case_id)
                 .order_by(AuditLog.id.desc()).limit(min(limit, 1000)))
        if action:
            query = query.where(AuditLog.action.startswith(action))
        if case_code:
            query = query.where(Case.code == case_code.strip().upper())
        return [AuditOut(at=e.at, user_id=e.user_id, user_email=e.user_email, role=e.role, action=e.action,
                         case_id=e.case_id, case_code=code, detail=e.detail) for e, code in db.execute(query)]


app = create_app()
