"""Fine-tune the 3D segmenter on BraTS train + corrected cases (the Week 9 feedback loop).

    python -m neurolens_ml.finetune_seg --config ml/configs/seg_finetune.yaml [--checkpoint segmenter_segresnet.pt]
                                        [--data <BraTS folder>] [--feedback-dir <export>/train]

Corrections come from ml/data/feedback_sim.csv: the k val cases where v1 was worst, with their expert BraTS masks
standing in for a doctor's fix (simulated feedback). Real corrections exported by the API
(backend/scripts/export_feedback.py) can be added with --feedback-dir. The other val cases pick the best epoch and
the starting model is saved first, so the result is never worse on them. The test split is not touched here (Day 3).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import get_device, load_config, out_dir, seed_everything
from .data.brats import find_cases
from .evaluate_seg import find_checkpoint
from .models.segmenter import create_segmenter
from .prep_brats import EXTRACT_TO, find_root
from .train_seg import CACHE_DIR, DATA, REGION_NAMES, SMOKE_OVERRIDES, evaluate, fit, split_cases

SELECTION = DATA / "feedback_sim.csv"


def select_feedback(val_cases: pd.DataFrame, k: int) -> pd.DataFrame:
    """The k val cases with the lowest mean region Dice become the corrected cases; ties broken by id."""
    df = val_cases.assign(mean=val_cases[REGION_NAMES].mean(axis=1)).sort_values(["mean", "id"]).reset_index(drop=True)
    df["role"] = np.where(df.index < k, "feedback", "val_rest")
    df = df.rename(columns={c: f"v1_{c}" for c in [*REGION_NAMES, "mean"]})
    return df[["id", "role", *(f"v1_{c}" for c in [*REGION_NAMES, "mean"])]].sort_values("id").round(4)


def summarize(per_case: pd.DataFrame) -> dict:
    out = {r: round(float(per_case[r].mean()), 4) for r in REGION_NAMES}
    return out | {"mean": round(float(np.mean(list(out.values()))), 4), "n": len(per_case)}


def finetune(cfg: dict, root: Path, splits_csv: Path, selection_csv: Path, checkpoint: Path, out: Path,
             cache_dir: Path | None = None, feedback_dir: Path | None = None) -> dict:
    seed_everything(cfg["seed"])
    device = get_device()
    torch.backends.cudnn.benchmark = True
    cases = split_cases(root, splits_csv)
    val_by_id = {c["id"]: c for c in cases["val"]}
    sel = pd.read_csv(selection_csv)
    feedback = [val_by_id[i] for i in sel[sel.role == "feedback"].id]
    val_rest = [val_by_id[i] for i in sel[sel.role == "val_rest"].id]
    real = find_cases(feedback_dir) if feedback_dir else []
    # Corrected cases are few next to 875 training scans; repeating them makes the model actually see them
    train_cases = cases["train"] + (feedback + real) * cfg["feedback_repeat"]
    print(f"--- {cfg['name']}: from {checkpoint.name}, train {len(cases['train'])} + {len(feedback)} simulated "
          f"+ {len(real)} real corrections x{cfg['feedback_repeat']}, select on {len(val_rest)} other val cases", flush=True)

    model = create_segmenter(cfg).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    start = time.time()
    before = {"val_rest": evaluate(model, val_rest, cfg, device, cache_dir),
              "feedback": evaluate(model, feedback, cfg, device, cache_dir)}
    v1 = {k: summarize(v) for k, v in before.items()}
    print(f"v1 before fine-tuning: {v1}", flush=True)

    hist, best_epoch, ckpt = fit(model, train_cases, val_rest, cfg, device, out, cache_dir,
                                 start_best=v1["val_rest"]["mean"])
    model.load_state_dict(torch.load(ckpt, map_location=device))
    after = {"val_rest": evaluate(model, val_rest, cfg, device, cache_dir),
             "feedback": evaluate(model, feedback, cfg, device, cache_dir)}
    v2 = {k: summarize(v) for k, v in after.items()}
    for part in ("val_rest", "feedback"):
        both = before[part].merge(after[part], on="id", suffixes=("_v1", "_v2"))
        both.to_csv(out / f"compare_{part}_{cfg['name']}.csv", index=False)

    metrics = dict(config=cfg, checkpoint_from=checkpoint.name, best_epoch=best_epoch, epochs_run=len(hist),
                   kept_v1=best_epoch == 0, hours=round((time.time() - start) / 3600, 2), v1=v1, v2=v2, test=None,
                   feedback={"simulated": len(feedback), "real": len(real), "repeat": cfg["feedback_repeat"],
                             "source": "expert BraTS masks of the k worst v1 val cases, standing in for doctors' fixes"})
    (out / f"metrics_seg_{cfg['name']}.json").write_text(json.dumps(metrics, indent=2))
    print(f"best epoch {best_epoch} (0 = v1 kept) | v1 {v1} | v2 {v2} | saved {ckpt}")
    return metrics


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", help="v1 weights (default: segmenter_segresnet.pt in Kaggle inputs or ml/models)")
    p.add_argument("--data", help="folder with BraTS .tar files or case folders (default: /kaggle/input)")
    p.add_argument("--extract-to", type=Path, default=EXTRACT_TO)
    p.add_argument("--splits", type=Path, default=DATA / "brats_splits.csv")
    p.add_argument("--selection", type=Path, default=SELECTION)
    p.add_argument("--feedback-dir", type=Path, help="real corrections exported by the API (the train/ folder)")
    p.add_argument("--out")
    p.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    if args.smoke:
        cfg.update(SMOKE_OVERRIDES)
    return finetune(cfg, find_root(args.data, args.extract_to), args.splits, args.selection,
                    find_checkpoint(args.checkpoint, cfg["base_name"]), out_dir(args.out),
                    None if args.no_cache else args.cache_dir, args.feedback_dir)


if __name__ == "__main__":
    main()
