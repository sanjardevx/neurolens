"""Report draft from structured numbers only (never the image). The template is always available;
an LLM may rephrase it later, and the template stays the fallback."""
from dataclasses import dataclass

from .db import Result, Review

SMALL_CM3 = 6.0  # the 3D model's known weak spot (Week 4 eval): tiny tumors are the failures
TYPES = {"glioma": "glioma", "meningioma": "meningioma", "pituitary": "pituitary tumor", "notumor": "no tumor"}
WITH_ARTICLE = {"glioma": "a glioma", "meningioma": "a meningioma", "pituitary": "a pituitary tumor", "notumor": "no tumor"}
DISCLAIMER = "AI-assisted draft from a research prototype, not a medical device. The reviewing doctor makes the diagnosis."


@dataclass
class Features:
    """Everything a report may say, and nothing else: no image, no patient data."""
    scan: str  # "3d" or "2d"
    ai_type: str | None
    confidence: float | None  # the 2D classifier's softmax, not calibrated
    doctor_type: str | None  # set when the doctor changed the type
    volumes_cm3: dict[str, float] | None  # the doctor's corrected mask wins over the AI's
    mask_corrected: bool
    side: str | None  # right | left | both sides
    lobe: str | None  # rough position rule


def features(result: Result, review: Review | None = None) -> Features:
    corrected = (result.info or {}).get("corrected_mask")
    return Features(
        scan="3d" if result.mask_path else "2d",
        ai_type=result.tumor_type,
        confidence=result.confidence,
        doctor_type=review.corrected_type if review else None,
        volumes_cm3=(corrected or {}).get("volumes_cm3") or result.volumes_cm3,
        mask_corrected=bool(corrected),
        side=result.side,
        lobe=result.lobe_estimate,
    )


def cm3(v: float) -> str:
    return f"{v:.1f} cm³"


def where(f: Features) -> str:
    if f.side is None:
        return ""
    if f.side == "both sides":
        place = "crossing the midline (both hemispheres)"
    else:
        place = f"in the {f.side} hemisphere"
    return f"{place}, centered in the {f.lobe} region (estimate)" if f.lobe else place


def template_report(f: Features) -> dict[str, list[str] | str]:
    """Findings, impression and notes as short plain sentences a doctor can edit."""
    findings: list[str] = []
    notes: list[str] = []
    v = f.volumes_cm3 or {}
    wt = v.get("whole_tumor", 0.0)
    found = f.scan == "3d" and wt > 0

    if f.scan == "2d":
        findings.append("Single 2D slice: tumor type only, no volume or location measured.")
    elif found:
        source = "the doctor-corrected mask" if f.mask_corrected else "the 3D segmentation"
        findings.append(f"A tumor is outlined {where(f)}." if f.side else "A tumor is outlined.")
        findings.append(f"Whole tumor volume {cm3(wt)} (from {source}): enhancing tumor {cm3(v.get('enhancing', 0))}, "
                        f"necrotic / non-enhancing core {cm3(v.get('necrosis_non_enhancing', 0))}, "
                        f"surrounding edema {cm3(v.get('edema', 0))}.")
    else:
        findings.append("The 3D segmentation outlined no tumor.")

    if f.ai_type:
        pct = f" ({round(f.confidence * 100)}%, not calibrated)" if f.confidence is not None else ""
        findings.append(f"The 2D classifier suggests {TYPES.get(f.ai_type, f.ai_type)}{pct}.")
    final_type = f.doctor_type or f.ai_type
    if f.doctor_type and f.doctor_type != f.ai_type:
        findings.append(f"Tumor type set by the reviewing doctor: {TYPES.get(f.doctor_type, f.doctor_type)}.")

    # The two models look at different things (one slice vs the whole volume); say so when they disagree
    if f.scan == "3d" and final_type == "notumor" and found:
        notes.append("The classifier says no tumor, but the 3D segmentation outlined one: check the scan closely.")
    if f.scan == "3d" and final_type not in (None, "notumor") and not found:
        notes.append(f"The classifier suggests {TYPES.get(final_type, final_type)}, but no tumor was outlined in 3D: "
                     "check the scan closely.")
    if found and wt < SMALL_CM3:
        notes.append(f"Small lesion (under {SMALL_CM3:.0f} cm³): the 3D model is least reliable at this size.")
    if f.lobe:
        notes.append("The region is a rough position estimate, not an atlas label.")

    if f.scan == "2d":
        impression = f"Findings suggest {WITH_ARTICLE.get(final_type, 'an unclear result')} on the uploaded slice."
    elif found:
        kind = WITH_ARTICLE.get(final_type, "a tumor") if final_type not in (None, "notumor") else "a tumor"
        impression = f"Findings suggest {kind} of about {wt:.0f} cm³" + (f", {where(f)}." if f.side else ".")
    else:
        impression = "No tumor outlined by the AI."
    return {"findings": findings, "impression": impression, "notes": notes, "disclaimer": DISCLAIMER}


def as_text(report: dict) -> str:
    lines = ["Findings:", *[f"- {s}" for s in report["findings"]], "", f"Impression: {report['impression']}"]
    if report["notes"]:
        lines += ["", "Notes:", *[f"- {s}" for s in report["notes"]]]
    return "\n".join([*lines, "", report["disclaimer"]])
