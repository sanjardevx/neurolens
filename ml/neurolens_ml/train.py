"""Train the classifier from a YAML config.

    python -m neurolens_ml.train --config ml/configs/cls_b3.yaml
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from .config import find_data_dir, find_splits, get_device, load_config, out_dir, seed_everything
from .data import load_splits, make_loader, make_transforms
from .engine import run_epoch, save_confusion, save_curves, save_errors, score
from .models import create_classifier


def lr_factor(epoch: int, warmup: int, total: int) -> float:
    """Linear warmup, then cosine decay to zero."""
    if epoch < warmup:
        return (epoch + 1) / (warmup + 1)
    return 0.5 * (1 + np.cos(np.pi * (epoch - warmup) / max(1, total - warmup)))


def train(cfg: dict, data_dir, splits_csv, out) -> dict:
    """Train with early stopping on val macro-F1. Saves the best checkpoint, history, curves and val report."""
    seed_everything(cfg["seed"])
    device = get_device()
    name = cfg["name"]
    frames = load_splits(splits_csv)
    train_df, val_df = frames["train"], frames["val"]
    train_tf, eval_tf = make_transforms(cfg)
    pin = device.type == "cuda"
    train_loader = make_loader(train_df, data_dir, train_tf, cfg, shuffle=True, pin=pin)
    val_loader = make_loader(val_df, data_dir, eval_tf, cfg, shuffle=False, pin=pin)

    model = create_classifier(cfg["model"], cfg["pretrained"]).to(device)
    # Class weights so a bigger class doesn't dominate; label smoothing keeps the model from being over-confident
    counts = train_df.y.value_counts().sort_index().to_numpy()
    weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: lr_factor(e, cfg["warmup_epochs"], cfg["epochs"]))
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    ckpt = out / f"classifier_{name}.pt"
    print(f"--- {name}: {cfg['model']} {cfg['img_size']}px, {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, "
          f"device {device}, train {len(train_df)} / val {len(val_df)}")

    best_f1, best_epoch, bad_epochs, history = -1.0, 0, 0, []
    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_ys, tr_preds, _ = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        va_loss, va_ys, va_preds, _ = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        tr, va = score(tr_ys, tr_preds, digits=None), score(va_ys, va_preds, digits=None)
        history.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr["acc"], val_loss=va_loss,
                            val_acc=va["acc"], val_f1=va["macro_f1"]))
        print(f"epoch {epoch:2d} | train loss {tr_loss:.3f} acc {tr['acc']:.3f} | "
              f"val loss {va_loss:.3f} acc {va['acc']:.3f} f1 {va['macro_f1']:.3f} | {time.time() - t0:.0f}s")
        if va["macro_f1"] > best_f1:
            best_f1, best_epoch, bad_epochs = va["macro_f1"], epoch, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["patience"]:
                print(f"early stop: no val F1 gain for {cfg['patience']} epochs")
                break

    hist = pd.DataFrame(history)
    hist.to_csv(out / f"history_{name}.csv", index=False)
    save_curves(hist, name, out / f"curves_{name}.png")

    # Val report of the best checkpoint
    model.load_state_dict(torch.load(ckpt, map_location=device))
    _, ys, preds, probs = run_epoch(model, val_loader, criterion, device)
    val = score(ys, preds)
    save_confusion(ys, preds, f"val confusion matrix · {name}", out / f"confusion_val_{name}.png")
    save_errors(val_df, data_dir, ys, preds, probs, cfg["crop"], out / f"errors_val_{name}.png")
    metrics = dict(config=cfg, best_epoch=best_epoch, val=val, test=None,
                   n_train=len(train_df), n_val=len(val_df), n_test=len(frames["test"]))
    (out / f"metrics_{name}.json").write_text(json.dumps(metrics, indent=2))
    print(f"best epoch {best_epoch} | val {val} | saved {ckpt}")
    return metrics


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Train the brain tumor classifier.")
    p.add_argument("--config", required=True, help="YAML run config, e.g. ml/configs/cls_b3.yaml")
    p.add_argument("--data", help="dataset folder with Training/ (auto-found on Kaggle)")
    p.add_argument("--splits", help="splits.csv (default: ml/data/splits.csv or a Kaggle input)")
    p.add_argument("--out", help="output folder (default: /kaggle/working or ./outputs)")
    p.add_argument("--smoke", action="store_true", help="tiny settings for a quick code check on fake data")
    args = p.parse_args(argv)
    cfg = load_config(args.config, smoke=args.smoke)
    train(cfg, find_data_dir(args.data), find_splits(args.splits), out_dir(args.out))


if __name__ == "__main__":
    main()
