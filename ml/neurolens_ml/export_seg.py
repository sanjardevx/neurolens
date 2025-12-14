"""Export the 3D segmenter to ONNX, check it against PyTorch, time it on CPU, check measurements on real cases.

    python -m neurolens_ml.export_seg --config ml/configs/seg_segresnet.yaml [--cases 6 --data <BraTS folder>]

Writes segmenter3d.onnx, onnx_seg_report.json and, with --cases, onnx_seg_cases.csv + onnx_seg_cases.png.
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import onnx  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from .config import get_device, load_config, out_dir  # noqa: E402
from .evaluate_seg import find_checkpoint  # noqa: E402
from .export import OPSET  # noqa: E402
from .infer_seg import PATCH, load_ras, onnx_runner, segment  # noqa: E402
from .measure import measure  # noqa: E402
from .models.segmenter import create_segmenter, region_dice  # noqa: E402
from .prep_brats import EXTRACT_TO, find_root  # noqa: E402
from .train_seg import DATA, REGION_NAMES, split_cases  # noqa: E402


def export_onnx(model: torch.nn.Module, path: Path) -> None:
    # Batch of 2 in the example: with 1 the exporter would fix the batch size to 1
    example = torch.randn(2, 4, PATCH, PATCH, PATCH)
    torch.onnx.export(model, (example,), path, input_names=["image"], output_names=["logits"], opset_version=OPSET,
                      dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=16)}}, external_data=False)
    onnx.checker.check_model(onnx.load(path))


def torch_runner(model: torch.nn.Module, device: torch.device):
    @torch.no_grad()
    def run(x: np.ndarray) -> np.ndarray:
        return model(torch.from_numpy(x).to(device)).float().cpu().numpy()
    return run


def fake_brain(shape=(240, 240, 155), seed: int = 0) -> np.ndarray:
    """Brain-sized test input for timing: an ellipsoid of noise in a black volume, like a skull-stripped scan."""
    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    inside = ((x - 120) / 70) ** 2 + ((y - 120) / 85) ** 2 + ((z - 75) / 68) ** 2 < 1
    noise = np.random.default_rng(seed).uniform(100, 2000, (4, *shape)).astype(np.float32)
    return noise * inside


def dice(a: np.ndarray, b: np.ndarray) -> list[float]:
    return [round(v, 4) for v in region_dice(torch.from_numpy(a), torch.from_numpy(b)).tolist()]


def true_regions(seg_path: str) -> np.ndarray:
    labels = np.rint(nib.as_closest_canonical(nib.load(seg_path)).get_fdata()).astype(np.uint8)
    return np.stack([np.isin(labels, (1, 4)), labels > 0, labels == 4])


def plot_cases(rows: list[dict], images: list[np.ndarray], masks: list[np.ndarray], path: Path) -> None:
    """Axial and sagittal slice through each tumor center, left/right and front/back marked, to eyeball side and lobe."""
    fig, axes = plt.subplots(len(rows), 2, figsize=(8, 4 * len(rows)), squeeze=False)
    for ax, row, image, mask in zip(axes, rows, images, masks):
        found = np.argwhere(mask[1])
        cx, cy, cz = (found.mean(0) if len(found) else np.array(mask.shape[1:]) / 2).astype(int)  # no tumor: middle
        for a, flair, wt, marks in [(ax[0], image[0, :, :, cz], mask[1, :, :, cz], ("L", "R", "front")),
                                    (ax[1], image[0, cx], mask[1, cx], ("back", "front", "top"))]:
            a.imshow(np.rot90(flair), cmap="gray")  # rot90: RAS x / y → screen right / up
            a.contour(np.rot90(wt), levels=[0.5], colors="lime", linewidths=1)
            h, w = np.rot90(flair).shape
            a.text(2, h / 2, marks[0], color="yellow", va="center")
            a.text(w - 2, h / 2, marks[1], color="yellow", va="center", ha="right")
            a.text(w / 2, 4, marks[2], color="yellow", ha="center", va="top")
            a.axis("off")
        ax[0].set_title(f"{row['id']} · {row['side']} · {row['lobe_estimate']}", fontsize=9)
        ax[1].set_title(f"whole tumor {row['whole_tumor_cm3']} cm³", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", help="default: a Kaggle input or ml/models/segmenter_<name>.pt")
    ap.add_argument("--cases", type=int, default=0, help="real val cases to check (needs the BraTS data)")
    ap.add_argument("--data", help="folder with BraTS .tar files or case folders (default: /kaggle/input)")
    ap.add_argument("--extract-to", type=Path, default=EXTRACT_TO)
    ap.add_argument("--splits", type=Path, default=DATA / "brats_splits.csv")
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    out, device = out_dir(args.out), get_device()
    model = create_segmenter(cfg).eval()
    model.load_state_dict(torch.load(find_checkpoint(args.checkpoint, cfg["name"]), map_location="cpu"))
    onnx_path = out / "segmenter3d.onnx"
    export_onnx(model, onnx_path)
    run_onnx = onnx_runner(onnx_path)

    # 1) One window: ONNX vs PyTorch on CPU
    window = np.random.default_rng(0).standard_normal((1, 4, PATCH, PATCH, PATCH)).astype(np.float32)
    pt, ox = torch_runner(model, torch.device("cpu"))(window), run_onnx(window)
    report = {"onnx_mb": round(onnx_path.stat().st_size / 1e6, 1),
              "window_max_logit_diff": float(np.abs(pt - ox).max()),
              "window_same_mask": float(((pt > 0) == (ox > 0)).mean())}

    # 2) CPU time for a full-size scan (240 x 240 x 155, brain about 140 x 170 x 136 → 8 windows)
    image = fake_brain()
    t = time.perf_counter()
    segment(image, run_onnx)
    report["cpu_seconds_per_scan"] = round(time.perf_counter() - t, 1)
    print(json.dumps(report, indent=2), flush=True)

    # 3) Real val cases: same masks as PyTorch? same Dice? side and lobe sensible?
    if args.cases:
        model.to(device)
        run_torch = torch_runner(model, device)
        rows, images, masks = [], [], []
        for case in split_cases(find_root(args.data, args.extract_to), args.splits)["val"][:args.cases]:
            image, affine = load_ras(case["image"])
            true = true_regions(case["label"])
            t = time.perf_counter()
            mask = segment(image, run_onnx)
            seconds = time.perf_counter() - t
            m = measure(image, mask, affine)
            rows.append({"id": case["id"], "onnx_seconds": round(seconds, 1),
                         **dict(zip([f"agree_{r}" for r in REGION_NAMES], dice(mask, segment(image, run_torch)))),
                         **dict(zip([f"onnx_dice_{r}" for r in REGION_NAMES], dice(mask, true))),
                         "side": m["side"], "lobe_estimate": m["lobe_estimate"],
                         "whole_tumor_cm3": m["volume_cm3"]["whole_tumor"],
                         "true_whole_tumor_cm3": round(float(true[1].sum()) * abs(np.linalg.det(affine[:3, :3])) / 1000, 2)})
            images.append(image)
            masks.append(mask)
            print(rows[-1], flush=True)
        cases = pd.DataFrame(rows)
        cases.to_csv(out / "onnx_seg_cases.csv", index=False)
        plot_cases(rows, images, masks, out / "onnx_seg_cases.png")
        report["cases"] = len(cases)
        report["min_agreement_dice"] = float(cases[[f"agree_{r}" for r in REGION_NAMES]].min().min())
        report["onnx_val_dice"] = {r: round(float(cases[f"onnx_dice_{r}"].mean()), 4) for r in REGION_NAMES}

    (out / "onnx_seg_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
