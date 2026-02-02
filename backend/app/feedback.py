"""Doctors' feedback → a versioned training set for the segmenter (Week 9 retraining).

Each reviewed 3D case lands in one of three groups:
- train: the doctor fixed the mask, so the corrected mask is a label the model got wrong;
- relabel: rejected for a bad mask but not fixed; it needs a label before it can be used;
- excluded: rejected for an image artifact or a wrong type only; nothing to learn for the mask.
Approved cases without a fix are counted (they show where the model is already right) but not exported.
Files use the BraTS layout (<id>/<id>_flair.nii.gz ... <id>_seg.nii.gz) on one RAS grid, so the ML code reads them as is.
"""
import csv
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import nibabel as nib
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from .db import Case, Result
from .masks import CORRECTED

MRI = ("flair", "t1", "t1ce", "t2")


@dataclass
class Row:
    case_code: str
    group: str  # train | relabel | excluded | approved
    decision: str
    reason: str | None
    ai_type: str | None
    doctor_type: str | None
    model_version: str
    wt_ai_cm3: float | None
    wt_doctor_cm3: float | None
    dice_ai_vs_doctor: float | None  # whole tumor; how much the doctor changed


def dice(a: np.ndarray, b: np.ndarray) -> float:
    total = a.sum() + b.sum()
    return 1.0 if total == 0 else round(float(2 * (a & b).sum() / total), 4)


def save_ras(src: Path, dst: Path) -> None:
    nib.save(nib.as_closest_canonical(nib.load(src)), dst)


def export_feedback(db: Session, storage: Path, out_root: Path, tag: str | None = None) -> dict:
    """Write <out_root>/<tag>/ with the cases, manifest.csv and summary.json; returns the summary."""
    cases = db.scalars(select(Case).where(Case.status == "reviewed")
                       .options(selectinload(Case.reviews), selectinload(Case.scans)))
    rows: list[Row] = []
    exports: list[tuple[Row, Result]] = []
    for case in cases:
        review = max(case.reviews, key=lambda r: r.created_at)
        result = db.get(Result, review.result_id)
        if result is None or not result.mask_path:  # 2D results have no mask to learn from
            continue
        out_dir = storage / "results" / result.id
        corrected = (result.info or {}).get("corrected_mask")
        ai = np.asarray(nib.load(out_dir / result.mask_path).dataobj) > 0
        doc = np.asarray(nib.load(out_dir / CORRECTED).dataobj) > 0 if corrected else None
        if corrected:
            group = "train"
        elif review.decision == "rejected" and review.reason == "bad_mask":
            group = "relabel"
        elif review.decision == "rejected":
            group = "excluded"
        else:
            group = "approved"
        row = Row(case.code, group, review.decision, review.reason, result.tumor_type, review.corrected_type,
                  result.model_version, (result.volumes_cm3 or {}).get("whole_tumor"),
                  (corrected or {}).get("volumes_cm3", {}).get("whole_tumor"),
                  dice(ai, doc) if doc is not None else None)
        rows.append(row)
        if group in ("train", "relabel"):
            exports.append((row, result))

    counts = {g: sum(r.group == g for r in rows) for g in ("train", "relabel", "excluded", "approved")}
    tag = tag or f"fb-{datetime.now(UTC):%Y%m%d}-{counts['train']}c"
    root = out_root / tag
    root.mkdir(parents=True, exist_ok=False)  # a tag is written once; a new export gets a new tag

    for row, result in exports:
        scan = next(s for s in db.get(Case, result.case_id).scans if s.id == result.scan_id)
        case_dir = root / row.group / row.case_code
        case_dir.mkdir(parents=True)
        for m in MRI:
            save_ras(storage / scan.storage_path / f"{m}.nii.gz", case_dir / f"{row.case_code}_{m}.nii.gz")
        results = storage / "results" / result.id
        save_ras(results / result.mask_path, case_dir / f"{row.case_code}_ai.nii.gz")
        if row.group == "train":  # the doctor's mask is the training label, in BraTS label values 1 / 2 / 4
            save_ras(results / CORRECTED, case_dir / f"{row.case_code}_seg.nii.gz")

    with open(root / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(Row.__dataclass_fields__))
        w.writeheader()
        w.writerows(asdict(r) for r in sorted(rows, key=lambda r: (r.group, r.case_code)))
    summary = {"tag": tag, "created_at": datetime.now(UTC).isoformat(timespec="seconds"), "counts": counts,
               "model_versions": sorted({r.model_version for r in rows}),
               "mean_dice_ai_vs_doctor": round(float(np.mean([r.dice_ai_vs_doctor for r in rows if r.group == "train"])), 4)
               if counts["train"] else None,
               "layout": "<group>/<code>/<code>_{flair,t1,t1ce,t2,ai,seg}.nii.gz (seg only in train), RAS"}
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
