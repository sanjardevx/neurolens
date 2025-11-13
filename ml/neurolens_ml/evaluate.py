"""Evaluate a trained checkpoint on val (default) or, once, on test.

    python -m neurolens_ml.evaluate --config ml/configs/cls_b3.yaml --checkpoint ml/models/classifier_strong_crop.pt
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .config import find_data_dir, find_splits, get_device, load_config, out_dir
from .data import load_splits, make_loader, make_transforms
from .engine import save_confusion, save_errors, score
from .models import load_classifier


@torch.no_grad()
def predict(model, loader, device, tta: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Labels and class probabilities. TTA averages the image and its left-right flip."""
    ys, probs = [], []
    for x, y in loader:
        x = x.to(device)
        p = model(x).float().softmax(1)
        if tta:
            p = (p + model(x.flip(3)).float().softmax(1)) / 2
        ys += y.tolist()
        probs.append(p.cpu())
    return np.array(ys), torch.cat(probs).numpy()


def evaluate(cfg: dict, checkpoint: Path, split: str, tta: bool, data_dir, splits_csv, out) -> dict:
    device = get_device()
    frame = load_splits(splits_csv)[split]
    _, eval_tf = make_transforms(cfg)
    loader = make_loader(frame, data_dir, eval_tf, {**cfg, "num_workers": 0}, shuffle=False)
    model = load_classifier(cfg["model"], checkpoint).to(device)
    ys, probs = predict(model, loader, device, tta)
    preds = probs.argmax(1)

    tag = cfg["name"] + ("_tta" if tta else "")
    result = dict(checkpoint=Path(checkpoint).name, split=split, tta=tta, n=len(frame), **score(ys, preds))
    save_confusion(ys, preds, f"{split} confusion matrix · {tag}", out / f"confusion_{split}_{tag}.png")
    save_errors(frame, data_dir, ys, preds, probs, cfg["crop"], out / f"errors_{split}_{tag}.png")
    (out / f"eval_{split}_{tag}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Evaluate a classifier checkpoint.")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="val", choices=["val", "test"],
                   help="tune on val; open test only once, for the final model")
    p.add_argument("--tta", action="store_true", help="average with the flipped image")
    p.add_argument("--data")
    p.add_argument("--splits")
    p.add_argument("--out")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    if args.split == "test":
        print("Opening the test split. Do this once, for the model already picked on val.")
    evaluate(cfg, Path(args.checkpoint), args.split, args.tta, find_data_dir(args.data), find_splits(args.splits),
             out_dir(args.out))


if __name__ == "__main__":
    main()
