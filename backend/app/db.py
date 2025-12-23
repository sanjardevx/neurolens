"""Database tables and sessions. SQLAlchemy models are the one schema source for Supabase Postgres and SQLite."""
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, CheckConstraint, DateTime, Float, ForeignKey, Integer, String, Text, create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker
from sqlalchemy.pool import StaticPool

CASE_STATUSES = ("new", "processing", "ready", "reviewed", "failed")
DECISIONS = ("approved", "edited", "rejected")


def new_id() -> str:
    return str(uuid.uuid4())


def now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Case(Base):
    """One patient study. No name or hospital ID is ever stored: only the anonymous code."""
    __tablename__ = "cases"
    __table_args__ = (CheckConstraint(f"status IN {CASE_STATUSES}", name="case_status"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    code: Mapped[str] = mapped_column(String(16), unique=True, index=True)  # e.g. NL-7KQ2XD
    doctor_id: Mapped[str] = mapped_column(String(36), index=True)  # Supabase auth user id
    status: Mapped[str] = mapped_column(String(16), default="new")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, onupdate=now)

    scans: Mapped[list["Scan"]] = relationship(back_populates="case", cascade="all, delete-orphan")
    results: Mapped[list["Result"]] = relationship(back_populates="case", cascade="all, delete-orphan")
    reviews: Mapped[list["Review"]] = relationship(back_populates="case", cascade="all, delete-orphan")


class Scan(Base):
    """An uploaded study (DICOM series, NIfTI set or JPG), stored after de-identification."""
    __tablename__ = "scans"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id", ondelete="CASCADE"), index=True)
    file_format: Mapped[str] = mapped_column(String(16))  # dicom | nifti | image
    storage_path: Mapped[str] = mapped_column(Text)  # folder under the storage root
    qc_passed: Mapped[bool | None] = mapped_column(default=None)
    qc_reason: Mapped[str | None] = mapped_column(Text, default=None)
    info: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)  # shape, voxel size, technical tags
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)

    case: Mapped[Case] = relationship(back_populates="scans")


class Result(Base):
    """AI output for a case: tumor type, 3D masks, volumes and location, with the model version that made it."""
    __tablename__ = "results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id", ondelete="CASCADE"), index=True)
    model_version: Mapped[str] = mapped_column(String(64))
    progress: Mapped[int] = mapped_column(Integer, default=0)  # 0-100, for the background 3D job
    tumor_type: Mapped[str | None] = mapped_column(String(32), default=None)
    confidence: Mapped[float | None] = mapped_column(Float, default=None)
    probabilities: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    volumes_cm3: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    side: Mapped[str | None] = mapped_column(String(16), default=None)
    lobe_estimate: Mapped[str | None] = mapped_column(String(32), default=None)
    mask_path: Mapped[str | None] = mapped_column(Text, default=None)
    gradcam_path: Mapped[str | None] = mapped_column(Text, default=None)
    error: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)

    case: Mapped[Case] = relationship(back_populates="results")


class Review(Base):
    """The doctor's decision on the AI result. Only an approved or edited review makes a report final."""
    __tablename__ = "reviews"
    __table_args__ = (CheckConstraint(f"decision IN {DECISIONS}", name="review_decision"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id", ondelete="CASCADE"), index=True)
    doctor_id: Mapped[str] = mapped_column(String(36))
    decision: Mapped[str] = mapped_column(String(16))
    reason: Mapped[str | None] = mapped_column(Text, default=None)  # required for a rejection
    report_text: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)

    case: Mapped[Case] = relationship(back_populates="reviews")


class AuditLog(Base):
    """Append-only: who did what to which case, and when."""
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, index=True)
    user_id: Mapped[str] = mapped_column(String(36), index=True)
    role: Mapped[str] = mapped_column(String(16))
    action: Mapped[str] = mapped_column(String(32))  # e.g. case.create, case.view, review.approve
    case_id: Mapped[str | None] = mapped_column(String(36), index=True, default=None)
    detail: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)


def make_engine(url: str) -> Engine:
    if url.startswith("sqlite"):
        # One shared connection for in-memory test databases; FastAPI runs handlers in threads
        pool = {"poolclass": StaticPool} if ":memory:" in url or url == "sqlite://" else {}
        return create_engine(url, connect_args={"check_same_thread": False}, **pool)
    return create_engine(url, pool_pre_ping=True)


def init_db(engine: Engine) -> None:
    """Create missing tables. On Supabase, also switch on row-level security with no policies:
    the public REST API (anon / logged-in keys) then sees nothing, and only this API, which connects
    as the database owner, can read and write."""
    Base.metadata.create_all(engine)
    if engine.dialect.name == "postgresql":
        with engine.begin() as conn:
            for table in Base.metadata.tables:
                conn.execute(text(f'ALTER TABLE "{table}" ENABLE ROW LEVEL SECURITY'))


class Database:
    def __init__(self, url: str):
        self.engine = make_engine(url)
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)

    def session(self) -> Iterator[Session]:
        with self.sessions() as s:
            yield s
