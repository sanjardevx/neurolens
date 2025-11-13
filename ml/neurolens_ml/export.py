"""Export a checkpoint to ONNX (logits + per-class CAMs), check it against PyTorch on val, time it on CPU.

    python -m neurolens_ml.export --config ml/configs/cls_b3.yaml --checkpoint ml/models/classifier_strong_crop.pt
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from .config import CLASSES, find_data_dir, find_splits, load_config, out_dir
from .data import load_rgb, load_splits, preprocess
from .models import ClassifierWithCam, load_classifier

OPSET = 18


def export_onnx(cfg: dict, checkpoint: Path, onnx_path: Path, example: np.ndarray) -> torch.nn.Module:
    wrapper = ClassifierWithCam(load_classifier(cfg["model"], checkpoint)).eval()
    # The example needs 2+ images: with a batch of 1 the exporter would fix the batch size to 1
    torch.onnx.export(
        wrapper, (torch.from_numpy(example),), onnx_path,
        input_names=["image"], output_names=["logits", "cams"], opset_version=OPSET,
        dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=64)}},
        external_data=False,  # one self-contained file
    )
    onnx.checker.check_model(onnx.load(onnx_path))
    return wrapper


def compare(wrapper, sess, images: list[np.ndarray], ys: np.ndarray, size: int, crop: bool, batch: int = 16) -> dict:
    """ONNX vs PyTorch on the same inputs: predictions, logits and heatmaps."""
    pt, ox, cam_corr = [], [], []
    for s in range(0, len(images), batch):
        x = np.concatenate([preprocess(img, size, crop) for img in images[s:s + batch]])
        with torch.no_grad():
            lt, ct = (t.numpy() for t in wrapper(torch.from_numpy(x)))
        lo, co = sess.run(None, {"image": x})
        pt.append(lt)
        ox.append(lo)
        k = lt.argmax(1)
        cam_corr += [np.corrcoef(ct[b, k[b]].ravel(), co[b, k[b]].ravel())[0, 1] for b in range(len(x))]
    pt, ox = np.concatenate(pt), np.concatenate(ox)
    return dict(n=len(images), max_logit_diff=float(np.abs(pt - ox).max()),
                same_prediction=int((pt.argmax(1) == ox.argmax(1)).sum()),
                pytorch_errors=int((pt.argmax(1) != ys).sum()), onnx_errors=int((ox.argmax(1) != ys).sum()),
                min_cam_correlation=float(np.min(cam_corr)))


def median_ms(fn, runs: int = 30, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    return round(float(np.median(times)), 1)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Export the classifier to ONNX and check it.")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--onnx", help="output file (default: <out>/classifier.onnx)")
    p.add_argument("--check", type=int, default=0, help="val images to compare (0 = all)")
    p.add_argument("--data")
    p.add_argument("--splits")
    p.add_argument("--out")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    data_dir, out = find_data_dir(args.data), out_dir(args.out)
    onnx_path = Path(args.onnx) if args.onnx else out / "classifier.onnx"
    size, crop = cfg["img_size"], cfg["crop"]

    val = load_splits(find_splits(args.splits))["val"]
    if args.check:
        val = val.head(args.check)
    images = [load_rgb(data_dir / p) for p in val.path]
    wrapper = export_onnx(cfg, Path(args.checkpoint), onnx_path, np.concatenate([preprocess(i, size, crop) for i in images[:2]]))
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    check = compare(wrapper, sess, images, val.y.to_numpy(), size, crop)

    x1 = preprocess(images[0], size, crop)
    with torch.no_grad():
        speed = dict(preprocess_ms=median_ms(lambda: preprocess(images[0], size, crop)),
                     onnx_cpu_ms=median_ms(lambda: sess.run(None, {"image": x1})),
                     pytorch_cpu_ms=median_ms(lambda: wrapper(torch.from_numpy(x1))), cpu_threads=os.cpu_count())

    report = dict(model=cfg["model"], checkpoint=Path(args.checkpoint).name, onnx_file=onnx_path.name,
                  onnx_size_mb=round(onnx_path.stat().st_size / 1e6, 1), opset=OPSET,
                  input=f"image [batch, 3, {size}, {size}], RGB, brain crop + ImageNet normalize",
                  outputs={"logits": f"[batch, {len(CLASSES)}]", "cams": f"[batch, {len(CLASSES)}, h, w]"},
                  classes=CLASSES, check=check, speed=speed)
    (out / "onnx_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    assert check["same_prediction"] == check["n"], "ONNX and PyTorch disagree"


if __name__ == "__main__":
    main()
