"""Run the AI on an uploaded scan, in the background, on ONNX Runtime (no PyTorch, no GPU).

- 2D image  → tumor type, confidence, Grad-CAM.
- 4 MRI types → 3D masks (edema / core / enhancing), volumes in cm³, side, rough lobe, and the tumor type
  from the 2D classifier on the T1ce slice through the tumor (a 2D model on one slice: shown as such).
"""
import threading
from collections.abc import Callable
from pathlib import Path

import nibabel as nib
import numpy as np
from neurolens_ml.infer_cls import classify, onnx_classifier
from neurolens_ml.infer_seg import label_map, load_ras, onnx_runner, segment
from neurolens_ml.measure import measure
from neurolens_ml.onnx_meta import read as read_meta
from PIL import Image

from .auth import User
from .db import AuditLog, Case, Database, Result, Scan
from .files import FileStore

# The served files; each carries its own version (and the segmenter its small-ET limit), see neurolens_ml.onnx_meta
CLASSIFIER, SEGMENTER = "classifier.onnx", "segmenter3d.onnx"
MRI_TYPES = ("flair", "t1", "t1ce", "t2")


class Models:
    """ONNX sessions, loaded once on first use and shared by all jobs."""

    def __init__(self, model_dir: str | Path):
        self.dir = Path(model_dir)
        self._lock = threading.Lock()
        self._cls = self._seg = None

    def _path(self, name: str) -> Path:
        path = self.dir / name
        if not path.exists():
            raise FileNotFoundError(f"model file not found: {name}")
        return path

    def classifier(self):
        with self._lock:
            if self._cls is None:
                self._cls = onnx_classifier(self._path(CLASSIFIER))
            return self._cls

    def segmenter(self):
        with self._lock:
            if self._seg is None:
                self._seg = onnx_runner(self._path(SEGMENTER))
            return self._seg


def card(run) -> dict:
    """What the result stores about a model that ran: its version and one line about it."""
    meta = read_meta(run.session)
    return {"version": meta["version"], "about": meta.get("about", "")}


def tumor_slice_rgb(t1ce: np.ndarray, wt: np.ndarray) -> tuple[np.ndarray, int]:
    """Axial T1ce slice with the most tumor (the middle one if none) as an 8-bit RGB image, front of the head up."""
    z = int(wt.sum((0, 1)).argmax()) if wt.any() else t1ce.shape[2] // 2
    s = t1ce[:, :, z]
    brain = s[s > 0]
    lo, hi = (np.percentile(brain, [1, 99]) if brain.size else (0.0, 1.0))
    gray = (np.clip((s - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    gray[s <= 0] = 0
    return np.repeat(np.rot90(gray)[..., None], 3, axis=2), z


def run_2d(scan_dir: Path, out_dir: Path, models: Models, report: Callable[[int], None]) -> dict:
    img = np.asarray(Image.open(scan_dir / "image.png").convert("RGB"))
    cls = models.classifier()
    out = classify(img, cls)
    Image.fromarray(out.pop("gradcam")).save(out_dir / "gradcam.png")
    return out | {"gradcam_path": "gradcam.png", "model_version": card(cls)["version"],
                  "info": {"classifier_input": "uploaded 2D slice", "models": {"classifier": card(cls)}}}


def run_3d(scan_dir: Path, out_dir: Path, models: Models, report: Callable[[int], None]) -> dict:
    image, affine = load_ras([scan_dir / f"{m}.nii.gz" for m in MRI_TYPES])
    seg, cls = models.segmenter(), models.classifier()
    cards = {"segmenter": card(seg), "classifier": card(cls)}  # before the long run: a file without a version fails fast
    min_et = int(read_meta(seg.session).get("min_et_voxels", 0))
    mask = segment(image, seg, progress=lambda f: report(int(5 + 85 * f)), min_et=min_et)
    nib.save(nib.Nifti1Image(label_map(mask), affine), str(out_dir / "mask.nii.gz"))
    m = measure(image, mask, affine)

    slice_rgb, z = tumor_slice_rgb(image[MRI_TYPES.index("t1ce")], mask[1])
    out = classify(slice_rgb, cls)
    Image.fromarray(out.pop("gradcam")).save(out_dir / "gradcam.png")
    report(95)
    return out | {"volumes_cm3": m["volume_cm3"], "side": m["side"], "lobe_estimate": m["lobe_estimate"],
                  "mask_path": "mask.nii.gz", "gradcam_path": "gradcam.png",
                  "model_version": f"{cards['segmenter']['version']} + {cards['classifier']['version']}",
                  "info": {"classifier_input": f"T1ce axial slice {z} (the 2D model sees one slice)",
                           "right_share": m["right_share"], "models": cards}}


def analyze(db: Database, files: FileStore, models: Models, result_id: str, user: User) -> None:
    """Background job. Progress and the outcome are written to the result row; the case ends ready or failed."""
    with db.sessions() as s:
        result = s.get(Result, result_id)
        scan, case = s.get(Scan, result.scan_id), s.get(Case, result.case_id)
        out_dir = files.root / "results" / result.id
        out_dir.mkdir(parents=True, exist_ok=True)

        def report(pct: int) -> None:
            result.progress = pct
            s.commit()

        try:
            run = run_2d if scan.file_format == "image" else run_3d
            out = run(files.pull(scan.storage_path), out_dir, models, report)
            files.push(f"results/{result.id}")  # before "ready": a restart must not lose the outputs
            for key, value in out.items():
                setattr(result, key, value)
            # 100% only in the same commit as the outputs: a client that sees 100 always sees the result
            result.progress = 100
            case.status = "ready"
            action, detail = "analysis.done", {"result_id": result.id}
        except Exception as e:  # any failure ends the job cleanly; the doctor sees the reason
            result.error = f"{type(e).__name__}: {e}"[:500]
            case.status = "failed"
            action, detail = "analysis.failed", {"result_id": result.id, "error": result.error}
        s.add(AuditLog(user_id=user.id, user_email=user.email, role=user.role, action=action, case_id=case.id, detail=detail))
        s.commit()
