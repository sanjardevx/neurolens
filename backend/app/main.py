"""NeuroLens API: doctor-only. Doctors create and review their own cases; admins see everything and the audit log.

    uvicorn app.main:app --reload
"""
import secrets
from collections.abc import Iterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.orm import Session

from .auth import User, current_user, require
from .config import get_settings
from .db import AuditLog, Case, Database, init_db

CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"  # no 0/O, 1/I/L: easy to read out loud


def create_app(database_url: str | None = None) -> FastAPI:
    db = Database(database_url or get_settings().database_url)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        init_db(db.engine)
        yield

    app = FastAPI(title="NeuroLens API", version="0.1.0", lifespan=lifespan,
                  description="Research and portfolio project, not a medical device.")
    app.state.db = db
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

    @app.get("/audit", response_model=list[AuditOut])
    def audit_log(limit: int = 200, user: User = Depends(require("admin")), db: Session = Depends(get_db)) -> list:
        return list(db.scalars(select(AuditLog).order_by(AuditLog.id.desc()).limit(min(limit, 1000))))


app = create_app()
