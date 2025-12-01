"""Train the 3D tumor segmenter from a YAML config.

    python -m neurolens_ml.train_seg --config ml/configs/seg_segresnet.yaml [--data <BraTS folder>]

Saves the best checkpoint by mean val Dice (TC, WT, ET), the history, curves and per-case val Dice.
The test split is never touched here.
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from monai.data import DataLoader, Dataset, PersistentDataset  # noqa: E402
from monai.data.utils import json_hashing  # noqa: E402

from .config import get_device, load_config, out_dir, seed_everything  # noqa: E402
from .data.brats import REGIONS, find_cases, train_transforms, val_transforms  # noqa: E402
from .models.segmenter import DiceBCELoss, create_segmenter, predict_volume, region_dice  # noqa: E402
from .prep_brats import EXTRACT_TO, find_root  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
CACHE_DIR = Path("/tmp/brats_cache")  # ~40 GB of prepared scans on Kaggle's local disk, not in the outputs
REGION_NAMES = list(REGIONS)  # tc, wt, et: the model's output channel order
# Tiny run on fake volumes: checks the code path, not the model
SMOKE_OVERRIDES = dict(patch=32, batch_size=1, epochs=2, val_every=1, init_filters=8, num_workers=0, sw_batch_size=1)


def split_cases(root: Path, splits_csv: Path) -> dict[str, list[dict]]:
    by_id = {c["id"]: c for c in find_cases(root)}
    splits = pd.read_csv(splits_csv)
    missing = sorted(set(splits.id) - set(by_id))
    if missing:
        raise FileNotFoundError(f"{len(missing)} cases from {splits_csv.name} are missing under {root}, e.g. {missing[:3]}")
    return {s: [by_id[i] for i in splits[splits.split == s].id] for s in ("train", "val", "test")}


def make_dataset(cases: list[dict], transform, cache_dir: Path | None) -> Dataset:
    """With a cache dir, each scan is loaded, cropped and normalized once and saved to disk;
    later epochs only read it back and run the random part. Without the cache, Kaggle's 4 CPUs
    spend ~10 min per epoch unzipping and cropping scans while the GPU waits."""
    if cache_dir is None:
        return Dataset(cases, transform)
    return PersistentDataset(cases, transform, cache_dir=cache_dir, hash_transform=json_hashing)


def plain(x: torch.Tensor) -> torch.Tensor:
    """MONAI MetaTensor → plain tensor (cached items are already plain)."""
    return x.as_tensor() if hasattr(x, "as_tensor") else x


@torch.no_grad()
def evaluate(model: torch.nn.Module, cases: list[dict], cfg: dict, device: torch.device,
             cache_dir: Path | None = None) -> pd.DataFrame:
    """Sliding-window prediction on each whole brain; one row of region Dice per case."""
    model.eval()
    loader = DataLoader(make_dataset(cases, val_transforms(), cache_dir), batch_size=1, num_workers=cfg["num_workers"])
    rows = []
    for batch in loader:
        with torch.autocast(device.type, enabled=cfg["amp"] and device.type == "cuda"):
            pred = predict_volume(model, plain(batch["image"]).to(device).float(), (cfg["patch"],) * 3,
                                  cfg["sw_batch_size"], cfg["sw_overlap"])
        dice = region_dice(pred[0], plain(batch["label"][0]).to(device))
        rows.append({"id": batch["id"][0], **dict(zip(REGION_NAMES, dice.tolist()))})
    return pd.DataFrame(rows)


def save_curves(hist: pd.DataFrame, tag: str, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(hist.epoch, hist.train_loss)
    ax1.set_title(f"train loss (Dice + BCE) · {tag}")
    val = hist.dropna(subset=["val_mean"])
    for col in [*(f"val_{r}" for r in REGION_NAMES), "val_mean"]:
        ax2.plot(val.epoch, val[col], marker=".", label=col.removeprefix("val_").upper())
    ax2.set_title(f"val Dice · {tag}")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=80)
    plt.close(fig)


def train(cfg: dict, root: Path, splits_csv: Path, out: Path, cache_dir: Path | None = None) -> dict:
    seed_everything(cfg["seed"])
    device = get_device()
    torch.backends.cudnn.benchmark = True  # fixed 128³ inputs: let cuDNN pick the fastest 3D conv once
    name, patch = cfg["name"], (cfg["patch"],) * 3
    cases = split_cases(root, splits_csv)
    train_tf = train_transforms(patch)
    train_tf.set_random_state(seed=cfg["seed"])
    train_loader = DataLoader(make_dataset(cases["train"], train_tf, cache_dir), batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg["num_workers"], pin_memory=device.type == "cuda")

    model = create_segmenter(cfg).to(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    use_amp = cfg["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    ckpt = out / f"segmenter_{name}.pt"
    print(f"--- {name}: SegResNet {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, patch {patch}, "
          f"device {device}, train {len(cases['train'])} / val {len(cases['val'])} cases", flush=True)

    start, best, best_epoch, bad_checks, history = time.time(), -1.0, 0, 0, []
    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        model.train()
        losses = []
        for batch in train_loader:
            x, y = plain(batch["image"]).to(device).float(), plain(batch["label"]).to(device).float()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=use_amp):
                logits = model(x)
            loss = criterion(logits.float(), y)  # loss in fp32 for stable Dice sums
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        scheduler.step()

        row = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        msg = f"epoch {epoch:3d} | loss {row['train_loss']:.4f}"
        if epoch % cfg["val_every"] == 0 or epoch == cfg["epochs"]:
            dice = evaluate(model, cases["val"], cfg, device, cache_dir)[REGION_NAMES].mean()
            row |= {f"val_{r}": float(dice[r]) for r in REGION_NAMES} | {"val_mean": float(dice.mean())}
            msg += " | val Dice " + " ".join(f"{r.upper()} {dice[r]:.3f}" for r in REGION_NAMES) + f" mean {row['val_mean']:.3f}"
            if row["val_mean"] > best:
                best, best_epoch, bad_checks = row["val_mean"], epoch, 0
                torch.save(model.state_dict(), ckpt)
                msg += " *"
            else:
                bad_checks += 1
        history.append(row)
        hours, epoch_hours = (time.time() - start) / 3600, (time.time() - t0) / 3600
        print(f"{msg} | {epoch_hours * 60:.1f} min", flush=True)
        if bad_checks >= cfg["patience"]:
            print(f"early stop: no val Dice gain for {cfg['patience']} checks")
            break
        if hours + 2 * epoch_hours > cfg["max_hours"]:  # room for one more epoch and the final val report
            print(f"time limit: stopping after {hours:.1f} h")
            break

    hist = pd.DataFrame(history)
    hist.to_csv(out / f"history_seg_{name}.csv", index=False)
    save_curves(hist, name, out / f"curves_seg_{name}.png")

    # Val report of the best checkpoint, per case, to find where the model fails
    model.load_state_dict(torch.load(ckpt, map_location=device))
    per_case = evaluate(model, cases["val"], cfg, device, cache_dir)
    per_case.to_csv(out / f"val_cases_seg_{name}.csv", index=False)
    val = {r: round(float(per_case[r].mean()), 4) for r in REGION_NAMES}
    val["mean"] = round(float(np.mean(list(val.values()))), 4)
    metrics = dict(config=cfg, best_epoch=best_epoch, epochs_run=len(hist), hours=round((time.time() - start) / 3600, 2),
                   val=val, test=None, n_train=len(cases["train"]), n_val=len(cases["val"]), n_test=len(cases["test"]))
    (out / f"metrics_seg_{name}.json").write_text(json.dumps(metrics, indent=2))
    print(f"best epoch {best_epoch} | val Dice {val} | saved {ckpt}")
    return metrics


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="YAML run config, e.g. ml/configs/seg_segresnet.yaml")
    p.add_argument("--data", help="folder with BraTS .tar files or case folders (default: /kaggle/input)")
    p.add_argument("--extract-to", type=Path, default=EXTRACT_TO)
    p.add_argument("--splits", type=Path, default=DATA / "brats_splits.csv")
    p.add_argument("--out", help="output folder (default: /kaggle/working or ./outputs)")
    p.add_argument("--cache-dir", type=Path, default=CACHE_DIR, help="disk cache of prepared scans")
    p.add_argument("--no-cache", action="store_true", help="read and prepare every scan every epoch")
    p.add_argument("--smoke", action="store_true", help="tiny settings for a quick code check on fake data")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    if args.smoke:
        cfg.update(SMOKE_OVERRIDES)
    cache_dir = None if args.no_cache else args.cache_dir
    return train(cfg, find_root(args.data, args.extract_to), args.splits, out_dir(args.out), cache_dir)


if __name__ == "__main__":
    main()
