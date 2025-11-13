"""Shared loop and reporting for training and evaluation."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # CLI: save figures, never open windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score

from .config import CLASSES
from .data import crop_brain, load_rgb


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    """One pass over the loader (trains if an optimizer is given). Returns loss, labels, predictions, probabilities."""
    train = optimizer is not None
    use_amp = device.type == "cuda"
    model.train(train)
    total_loss, ys, probs = 0.0, [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
            loss = criterion(logits.float(), y)  # loss in float32: fp16 logits can overflow to inf/NaN
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * len(y)
            ys += y.tolist()
            probs.append(logits.float().softmax(1).detach().cpu())
    probs = torch.cat(probs).numpy()
    return total_loss / len(ys), np.array(ys), probs.argmax(1), probs


def score(ys: np.ndarray, preds: np.ndarray, digits: int | None = 4) -> dict:
    """Accuracy, macro-F1 and error count. digits=None keeps full precision (used for early stopping)."""
    acc, f1 = float(accuracy_score(ys, preds)), float(f1_score(ys, preds, average="macro"))
    if digits is not None:
        acc, f1 = round(acc, digits), round(f1, digits)
    return dict(acc=acc, macro_f1=f1, errors=int((preds != ys).sum()))


def save_curves(hist: pd.DataFrame, tag: str, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(hist.epoch, hist.train_loss, label="train")
    ax1.plot(hist.epoch, hist.val_loss, label="val")
    ax1.set_title(f"loss · {tag}")
    ax1.legend()
    ax2.plot(hist.epoch, hist.train_acc, label="train acc")
    ax2.plot(hist.epoch, hist.val_acc, label="val acc")
    ax2.plot(hist.epoch, hist.val_f1, label="val macro-F1")
    ax2.set_title(f"metrics · {tag}")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=80)
    plt.close(fig)


def save_confusion(ys: np.ndarray, preds: np.ndarray, title: str, path: Path) -> None:
    disp = ConfusionMatrixDisplay.from_predictions(ys, preds, labels=range(len(CLASSES)), display_labels=CLASSES,
                                                   cmap="Blues", colorbar=False)
    disp.ax_.set_title(title)
    disp.figure_.savefig(path, dpi=80, bbox_inches="tight")
    plt.close(disp.figure_)


def save_errors(frame: pd.DataFrame, data_dir: Path, ys, preds, probs, crop: bool, path: Path) -> None:
    """Wrong predictions, most confident first: they show what the model still gets wrong."""
    wrong = np.nonzero(preds != ys)[0]
    wrong = wrong[np.argsort(-probs[wrong].max(1))][:12]
    if not len(wrong):
        return
    fig, axes = plt.subplots(2, 6, figsize=(18, 6), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, wrong):
        img = load_rgb(data_dir / frame.path[i])
        ax.imshow(crop_brain(img) if crop else img)
        ax.set_title(f"true {CLASSES[ys[i]]}\npred {CLASSES[preds[i]]} ({probs[i].max():.0%})", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=70)
    plt.close(fig)
