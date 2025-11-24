"""Plot BraTS slices with the expert mask on top, to check that masks line up with the scans.

    python -m neurolens_ml.plot_brats [--data <BraTS folder>]

Writes brats_slices.png (small to large tumors), brats_patch.png (one patient through the
training transforms) and brats_plot_checks.json (share of tumor voxels that sit inside the brain).
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import out_dir  # noqa: E402
from .data.brats import MODALITIES, PATCH, find_cases, load_transforms, train_transforms, val_transforms  # noqa: E402
from .prep_brats import EXTRACT_TO, find_root  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
COLORS = {"edema": (0.2, 0.8, 0.2), "core": (0.9, 0.2, 0.2), "enhancing": (1.0, 0.85, 0.1)}


def pick_cases(cases: pd.DataFrame) -> list[str]:
    """Smallest, quartiles and largest whole tumor, plus one case with no enhancing part."""
    s = cases.sort_values("wt_cm3").reset_index(drop=True)
    ids = [s.id[round(q * (len(s) - 1))] for q in (0, 0.25, 0.5, 0.75, 1)]
    ids += list(s[s.et_cm3 == 0].id[:1])
    return list(dict.fromkeys(ids))


def regions(label: np.ndarray) -> dict[str, np.ndarray]:
    """TC / WT / ET channels back to the 3 disjoint parts a doctor sees."""
    tc, wt, et = np.asarray(label).astype(bool)
    return {"edema": wt & ~tc, "core": tc & ~et, "enhancing": et}


def tumor_slice(label: np.ndarray) -> int:
    """Axial slice with the most whole-tumor voxels."""
    return int(np.asarray(label[1]).sum((0, 1)).argmax())


def show(ax, gray: np.ndarray, label: np.ndarray | None = None, title: str = "") -> None:
    ax.imshow(np.rot90(gray), cmap="gray")  # rot90: front of the head at the top
    if label is not None:
        rgba = np.zeros((*gray.shape, 4))
        for name, mask in regions(label).items():
            rgba[mask] = (*COLORS[name], 0.45)
        ax.imshow(np.rot90(rgba))
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def inside_brain(image: np.ndarray, label: np.ndarray) -> float:
    """Share of whole-tumor voxels where the skull-stripped scan is not background. Should be ~1."""
    wt = np.asarray(label[1]).astype(bool)
    return round(float((np.abs(np.asarray(image)).max(0) > 0)[wt].mean()), 4) if wt.any() else 1.0


def plot_slices(by_id: dict, volumes: dict, ids: list[str], path: Path) -> list[dict]:
    height = 3.2 * len(ids) + 0.5  # + a strip at the bottom for the color legend
    fig, axes = plt.subplots(len(ids), 5, figsize=(15, height), squeeze=False)
    checks, tf = [], load_transforms()
    for r, cid in enumerate(ids):
        d = tf(by_id[cid])
        image, label = np.asarray(d["image"]), np.asarray(d["label"])
        z = tumor_slice(label)
        for k, m in enumerate(MODALITIES):
            show(axes[r, k], image[k, ..., z], title=f"{cid} · {m.upper()}" if k == 0 else m.upper())
        show(axes[r, 4], image[0, ..., z], label[..., z], f"FLAIR + mask · tumor {volumes[cid]:.0f} cm³")
        checks.append({"id": cid, "slice": z, "tumor_inside_brain": inside_brain(image, label)})
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS.values()], labels=list(COLORS),
               loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.5 / height, 1, 1))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return checks


def plot_patch(case: dict, path: Path, patch: tuple[int, int, int] = PATCH) -> dict:
    """The same patient raw, after the brain crop, and as a random flipped training patch."""
    steps = {"raw": load_transforms(), "brain crop + normalize": val_transforms(), "training patch": train_transforms(patch)}
    steps["training patch"].set_random_state(seed=0)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    check = {"id": case["id"]}
    for ax, (name, tf) in zip(axes, steps.items()):
        d = tf(case)
        image, label = np.asarray(d["image"]), np.asarray(d["label"])
        z = tumor_slice(label)
        show(ax, image[0, ..., z], label[..., z], f"{name}\n{'x'.join(map(str, image.shape[1:]))}")
        check[name] = inside_brain(image, label)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return check


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="folder with BraTS .tar files or unpacked case folders (default: /kaggle/input)")
    ap.add_argument("--extract-to", type=Path, default=EXTRACT_TO)
    ap.add_argument("--cases", type=Path, default=DATA / "brats_cases.csv", help="made by prep_brats")
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    by_id = {c["id"]: c for c in find_cases(find_root(args.data, args.extract_to))}
    cases = pd.read_csv(args.cases)
    cases = cases[cases.id.isin(by_id)]
    ids = pick_cases(cases)
    volumes = dict(zip(cases.id, cases.wt_cm3))
    out = out_dir(args.out)

    checks = {
        "slices": plot_slices(by_id, volumes, ids, out / "brats_slices.png"),
        "patch": plot_patch(by_id[ids[min(2, len(ids) - 1)]], out / "brats_patch.png"),  # median tumor
    }
    (out / "brats_plot_checks.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))
    return checks


if __name__ == "__main__":
    main()
