"""Old vs new segmenter on the same test split (Week 9 Day 3). Test is opened once, for both models together.

    python -m neurolens_ml.compare_seg --old-config ml/configs/seg_segresnet.yaml --new-config ml/configs/seg_finetune.yaml

Each model gets its small-ET rule tuned on the 148 val cases that neither model trained on (the 40 corrected
cases are the new model's training data). The rules below are fixed before the run:
- keep the new model if its mean test Dice is at most MARGIN below the old one (the same margin as fine-tuning);
- report the paired per-case difference with a bootstrap 95% interval;
- report the small-ET subgroup (true ET under 5 cm³): the kind of case the corrections were about
  (median true ET 5.4 cm³ in the corrected set vs 23 cm³ in the rest), chosen by the label, not by either model.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import get_device, load_config, out_dir, seed_everything
from .evaluate_seg import find_checkpoint, summary, tune_min_et, with_min_et
from .models.segmenter import create_segmenter
from .prep_brats import EXTRACT_TO, find_root
from .train_seg import CACHE_DIR, DATA, evaluate, split_cases

MARGIN = 0.005
SMALL_ET_VOXELS = 5000  # 5 cm³ (1 voxel = 1 mm³)
BOOTSTRAP = 2000


def case_mean(cases: pd.DataFrame, min_et: int) -> pd.Series:
    """Per-case mean of TC, WT and ET Dice, with the model's small-ET rule applied."""
    return (cases.tc + cases.wt + with_min_et(cases, min_et)) / 3


def paired(diff: np.ndarray, seed: int) -> dict:
    """Mean new-minus-old difference, a bootstrap 95% interval and the better / same / worse counts (±0.01)."""
    rng = np.random.default_rng(seed)
    means = rng.choice(diff, (BOOTSTRAP, len(diff))).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean_diff": round(float(diff.mean()), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "better": int((diff > 0.01).sum()), "same": int((np.abs(diff) <= 0.01).sum()),
            "worse": int((diff < -0.01).sum()), "n": len(diff)}


def compare(cfgs: dict[str, dict], ckpts: dict[str, Path], root: Path, splits_csv: Path, selection_csv: Path,
            out: Path, cache_dir: Path | None = None) -> dict:
    seed_everything(42)
    device = get_device()
    cases = split_cases(root, splits_csv)
    sel = pd.read_csv(selection_csv)
    by_id = {c["id"]: c for c in cases["val"]}
    val_rest = [by_id[i] for i in sel[sel.role == "val_rest"].id]
    print(f"--- old {ckpts['old'].name} vs new {ckpts['new'].name}: ET rule from {len(val_rest)} val cases, "
          f"test {len(cases['test'])} cases", flush=True)

    res, per_case = {}, {}
    for tag in ("old", "new"):
        cfg = cfgs[tag]
        model = create_segmenter(cfg).to(device)
        model.load_state_dict(torch.load(ckpts[tag], map_location=device))
        min_et, _ = tune_min_et(evaluate(model, val_rest, cfg, device, cache_dir))
        test = evaluate(model, cases["test"], cfg, device, cache_dir)
        per_case[tag] = test.assign(mean=case_mean(test, min_et), et_rule=with_min_et(test, min_et))
        res[tag] = {"checkpoint": ckpts[tag].name, "min_et_voxels": min_et, "test": summary(test, min_et)}
        print(f"{tag}: ET limit {min_et} voxels | test {res[tag]['test']}", flush=True)

    both = per_case["old"].merge(per_case["new"], on="id", suffixes=("_old", "_new"))
    both.to_csv(out / "compare_test_seg.csv", index=False)
    small = (both.true_et_voxels_old < SMALL_ET_VOXELS).to_numpy()
    diff = (both.mean_new - both.mean_old).to_numpy()
    cols = {"tc": "tc", "wt": "wt", "et": "et_rule", "mean": "mean"}  # ET with each model's own small-ET rule
    sub = {tag: {k: round(float(both.loc[small, f"{c}_{tag}"].mean()), 4) for k, c in cols.items()}
           for tag in ("old", "new")}
    keep_new = res["new"]["test"]["mean"] >= res["old"]["test"]["mean"] - MARGIN
    result = {**res, "paired_all": paired(diff, 42), "small_et": {**sub, "paired": paired(diff[small], 42),
              "rule": f"true ET < {SMALL_ET_VOXELS / 1000:g} cm3"},
              "rule": f"keep new if test mean >= old - {MARGIN}", "keep": "new" if keep_new else "old"}
    (out / "metrics_seg_compare.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--old-config", required=True)
    p.add_argument("--new-config", required=True)
    p.add_argument("--old-checkpoint", help="default: segmenter_<name>.pt in Kaggle inputs or ml/models")
    p.add_argument("--new-checkpoint")
    p.add_argument("--data", help="folder with BraTS .tar files or case folders (default: /kaggle/input)")
    p.add_argument("--extract-to", type=Path, default=EXTRACT_TO)
    p.add_argument("--splits", type=Path, default=DATA / "brats_splits.csv")
    p.add_argument("--selection", type=Path, default=DATA / "feedback_sim.csv")
    p.add_argument("--out")
    p.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    p.add_argument("--smoke", action="store_true", help="tiny model settings for a code check on fake data")
    args = p.parse_args(argv)
    cfgs = {"old": load_config(args.old_config), "new": load_config(args.new_config)}
    if args.smoke:
        for cfg in cfgs.values():
            cfg.update(init_filters=8, num_workers=0, sw_batch_size=1, patch=32)
    ckpts = {"old": find_checkpoint(args.old_checkpoint, cfgs["old"]["name"]),
             "new": find_checkpoint(args.new_checkpoint, cfgs["new"]["name"])}
    return compare(cfgs, ckpts, find_root(args.data, args.extract_to), args.splits, args.selection,
                   out_dir(args.out), args.cache_dir)


if __name__ == "__main__":
    main()
