"""PDF of an approved case: key slices with the mask, the numbers and the doctor's final report,
signed with the reviewer, the approval time and a document fingerprint."""
import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import nibabel as nib
import numpy as np
from fpdf import FPDF
from PIL import Image

# Same colors as the viewer and the summary: edema green, necrotic core red, enhancing yellow
MASK_RGB = {2: (34, 197, 94), 1: (239, 68, 68), 4: (250, 204, 21)}
PARTS = [("edema", "Edema"), ("necrosis_non_enhancing", "Necrotic / non-enhancing core"),
         ("enhancing", "Enhancing tumor"), ("tumor_core", "Tumor core"), ("whole_tumor", "Whole tumor")]
DISCLAIMER = "Research prototype, not a medical device. AI-assisted; the reviewing doctor made the decision."
# The built-in PDF fonts are Latin-1; LLM text can carry typographic dashes and arrows
PLAIN = {"‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "→": "->", "‘": "'",
         "’": "'", "“": '"', "”": '"', "•": "-", "…": "...", " ": " ", " ": " ",
         " ": " "}


@dataclass
class Signed:
    case_code: str
    scan: str  # "3D MRI (FLAIR, T1, T1ce, T2)" or "2D slice"
    uploaded_at: str
    decision: str  # "Approved" or "Approved with edits"
    reviewer: str
    approved_at: str
    tumor_type: str
    type_note: str
    volumes_cm3: dict[str, float] | None
    location: str | None
    report_text: str
    models: str


def plain(text: str) -> str:
    return "".join(PLAIN.get(c, c) for c in text).encode("latin-1", "replace").decode("latin-1")


def fingerprint(s: Signed) -> str:
    """Changes if any signed field changes; printed on every page so a copy can be checked against the API."""
    fields = [s.case_code, s.decision, s.reviewer, s.approved_at, s.tumor_type, s.report_text]
    return hashlib.sha256("\n".join(fields).encode()).hexdigest()[:16]


def gray(slice2d: np.ndarray) -> np.ndarray:
    brain = slice2d[slice2d > 0]
    lo, hi = np.percentile(brain, [1, 99]) if brain.size else (0.0, 1.0)
    return (np.clip((slice2d - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def axial(a: np.ndarray, z: int) -> np.ndarray:
    """Axial slice of a RAS array as radiologists read it: front of the head up, patient's right on the left."""
    return np.fliplr(np.flipud(a[:, :, z].T))


def png(rgb: np.ndarray) -> bytes:
    buf = BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def key_slices_3d(scan_dir: Path, mask_file: Path) -> tuple[list[tuple[str, bytes]], int]:
    """Axial FLAIR with the mask and T1ce at the slice with the most tumor (both on the mask's RAS grid)."""
    labels = np.asarray(nib.load(mask_file).dataobj).astype(np.uint8)
    flair, t1ce = (np.asarray(nib.as_closest_canonical(nib.load(scan_dir / f"{m}.nii.gz")).dataobj, dtype=np.float32)
                   for m in ("flair", "t1ce"))
    tumor = (labels > 0).sum((0, 1))
    z = int(tumor.argmax()) if tumor.any() else labels.shape[2] // 2
    base = np.repeat(gray(axial(flair, z))[..., None], 3, axis=2).astype(np.float32)
    lab = axial(labels, z)
    for value, rgb in MASK_RGB.items():
        base[lab == value] = 0.55 * base[lab == value] + 0.45 * np.array(rgb, np.float32)
    t1 = np.repeat(gray(axial(t1ce, z))[..., None], 3, axis=2)
    return [(f"Axial FLAIR + mask, slice {z + 1}", png(base.astype(np.uint8))),
            (f"Axial T1ce, slice {z + 1}", png(t1))], z


class ReportPDF(FPDF):
    def __init__(self, s: Signed):
        super().__init__(format="A4")
        self.signed = s
        self.fp = fingerprint(s)
        self.set_auto_page_break(True, margin=18)
        self.set_title(f"NeuroLens report {s.case_code}")
        self.set_creator("NeuroLens")

    def footer(self) -> None:
        self.set_y(-14)
        self.set_font("Helvetica", size=7)
        self.set_text_color(110)
        self.cell(0, 4, plain(DISCLAIMER), new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 4, f"Case {self.signed.case_code} | document {self.fp} | page {self.page_no()}/{{nb}}")


def build_pdf(s: Signed, images: list[tuple[str, bytes]], compress: bool = True) -> bytes:
    pdf = ReportPDF(s)
    pdf.set_compression(compress)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 9, "NeuroLens | AI-assisted brain MRI report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=9)
    pdf.set_text_color(110)
    pdf.cell(0, 5, plain(DISCLAIMER), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0)
    pdf.ln(3)

    rows = [("Case", f"{s.case_code} (anonymous code; no patient name is stored)"), ("Scan", s.scan),
            ("Uploaded", s.uploaded_at), ("Tumor type", f"{s.tumor_type} ({s.type_note})")]
    if s.location:
        rows.append(("Location", s.location))
    rows += [("Decision", s.decision), ("Reviewed by", s.reviewer), ("Approved at", s.approved_at), ("Models", s.models)]
    for k, v in rows:
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(32, 5.5, k)
        pdf.set_font("Helvetica", size=9)
        pdf.multi_cell(0, 5.5, plain(v), align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    if images:
        w, gap = 58.0, 4.0
        top = pdf.get_y()
        for i, (caption, data) in enumerate(images):
            x = pdf.l_margin + i * (w + gap)
            pdf.image(BytesIO(data), x=x, y=top, w=w, h=w)
            pdf.set_xy(x, top + w + 1)
            pdf.set_font("Helvetica", size=7.5)
            pdf.multi_cell(w, 3.8, plain(caption), align="L")
        pdf.set_y(top + w + 10)
        pdf.set_font("Helvetica", size=7.5)
        pdf.set_text_color(110)
        pdf.cell(0, 4, "Radiological view: the patient's right is on the left. Mask: edema green, necrotic core red, "
                       "enhancing yellow.", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0)
        pdf.ln(2)

    if s.volumes_cm3 and s.volumes_cm3.get("whole_tumor"):
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Volumes", new_x="LMARGIN", new_y="NEXT")
        for key, label in PARTS:
            pdf.set_font("Helvetica", "B" if key == "whole_tumor" else "", 9)
            pdf.cell(70, 5.5, label, border="B")
            pdf.cell(30, 5.5, f"{s.volumes_cm3.get(key, 0):.1f} cm³", border="B", align="R",
                     new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Final report (approved by the doctor)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=9.5)
    pdf.multi_cell(0, 5, plain(s.report_text), align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 9)
    pdf.multi_cell(0, 5, plain(f"Signed: {s.reviewer}, {s.approved_at}. Document fingerprint {pdf.fp}."),
                   align="L", new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
