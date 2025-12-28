"""Numbers for the doctor's report from a 3D tumor mask: volume per part and a rough location.

Masks are [3, X, Y, Z] in RAS orientation (what load_transforms gives): channels TC, WT, ET;
axis 0 runs left → right, axis 1 back → front, axis 2 bottom → top of the head.
"""
import numpy as np


def voxel_cm3(affine: np.ndarray) -> float:
    """Volume of one voxel in cm³ from the scan's affine; works for non-1 mm and tilted scans."""
    return abs(float(np.linalg.det(np.asarray(affine, dtype=float)[:3, :3]))) / 1000


def tumor_volumes(mask: np.ndarray, voxel: float) -> dict[str, float]:
    """cm³ of each part a radiologist reports, plus the two BraTS totals."""
    tc, wt, et = np.asarray(mask).astype(bool)
    # The model predicts the 3 regions separately, so a few voxels can break the nesting; report them nested
    tc = tc | et
    wt = wt | tc
    parts = {
        "whole_tumor": wt,
        "tumor_core": tc,
        "enhancing": et,
        "necrosis_non_enhancing": tc & ~et,
        "edema": wt & ~tc,
    }
    return {name: round(float(m.sum()) * voxel, 2) for name, m in parts.items()}


def lobe_estimate(front: float, top: float) -> str:
    """Rough lobe from where the tumor center sits in the brain box (0 = back / bottom, 1 = front / top).
    A position rule, not an atlas: the doctor confirms it on the scan."""
    if top < 0.3 and front < 0.4:
        return "cerebellum / brainstem"
    if front > 0.6:
        return "frontal"
    if front < 0.2:
        return "occipital"
    return "temporal" if top < 0.5 else "parietal"


def tumor_location(wt: np.ndarray, brain: np.ndarray) -> dict:
    """Side of the brain (by the share of tumor voxels right of the midline) and a rough lobe."""
    wt, brain = np.asarray(wt).astype(bool), np.asarray(brain).astype(bool)
    if not wt.any() or not brain.any():
        return {"side": None, "right_share": None, "lobe_estimate": None}
    lo, hi = (idx := np.argwhere(brain)).min(0), idx.max(0)
    midline = (lo[0] + hi[0]) / 2
    right_share = float((np.nonzero(wt)[0] > midline).mean())
    side = "right" if right_share >= 0.8 else "left" if right_share <= 0.2 else "both sides"
    center = (np.argwhere(wt).mean(0) - lo) / np.maximum(hi - lo, 1)
    return {"side": side, "right_share": round(right_share, 2), "lobe_estimate": lobe_estimate(center[1], center[2])}


def measure(image: np.ndarray, mask: np.ndarray, affine: np.ndarray) -> dict:
    """Report numbers for one scan: image [4, X, Y, Z] (background 0), mask [3, X, Y, Z], RAS affine."""
    brain = np.abs(np.asarray(image, dtype=np.float32)).sum(0) > 0
    return {"volume_cm3": tumor_volumes(mask, voxel_cm3(affine)), **tumor_location(np.asarray(mask)[1], brain)}
