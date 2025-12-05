"""Evaluate the 3D segmenter. The small-ET rule is tuned on val; test runs once with the val choice.

    python -m neurolens_ml.evaluate_seg --config ml/configs/seg_segresnet.yaml               # val: pick the ET size limit
    python -m neurolens_ml.evaluate_seg --config ml/configs/seg_segresnet.yaml --split test  # test, once
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from .config import KAGGLE_INPUT, get_device, load_config, out_dir
from .models.segmenter import create_segmenter
from .prep_brats import EXTRACT_TO, find_root
from .train_seg import DATA, REGION_NAMES, evaluate, split_cases

MIN_ET_GRID = (0, 50, 100, 200, 300, 500, 750, 1000)  # voxels = mm³


def with_min_et(cases: pd.DataFrame, min_voxels: int) -> pd.Series:
    """ET Dice per case after dropping predicted ET smaller than min_voxels (same rule as drop_small_et).
    A dropped case scores 1 if it has no true ET, else 0."""
    dropped = cases.pred_et_voxels < min_voxels
    return cases.et.where(~dropped, (cases.true_et_voxels == 0).astype(float))


def summary(cases: pd.DataFrame, min_et: int) -> dict:
    dice = {r: float(cases[r].mean()) for r in REGION_NAMES}
    dice["et"] = float(with_min_et(cases, min_et).mean())
    dice["mean"] = sum(dice.values()) / 3
    return {k: round(v, 4) for k, v in dice.items()}


def tune_min_et(cases: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    """Best mean ET Dice on val; on a tie the smaller limit wins."""
    table = pd.DataFrame({"min_et_voxels": MIN_ET_GRID,
                          "val_et": [round(float(with_min_et(cases, t).mean()), 4) for t in MIN_ET_GRID]})
    return int(table.min_et_voxels[table.val_et.idxmax()]), table


def find_checkpoint(arg: str | None, name: str) -> Path:
    if arg:
        return Path(arg)
    local = Path(__file__).resolve().parents[1] / "models" / f"segmenter_{name}.pt"
    for p in [*KAGGLE_INPUT.glob(f"**/segmenter_{name}.pt"), local]:
        if p.exists():
            return p
    raise FileNotFoundError(f"segmenter_{name}.pt not found. Pass --checkpoint or attach the 07_segmentation output.")


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--checkpoint", help="default: a Kaggle input or ml/models/segmenter_<name>.pt")
    ap.add_argument("--data", help="folder with BraTS .tar files or case folders (default: /kaggle/input)")
    ap.add_argument("--extract-to", type=Path, default=EXTRACT_TO)
    ap.add_argument("--splits", type=Path, default=DATA / "brats_splits.csv")
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    name, out = cfg["name"], out_dir(args.out)
    rule_file = out / f"seg_min_et_{name}.json"
    if args.split == "test" and not rule_file.exists():
        raise FileNotFoundError(f"{rule_file.name} missing: run --split val first, the ET size limit comes from val only.")

    device = get_device()
    ckpt = find_checkpoint(args.checkpoint, name)
    model = create_segmenter(cfg).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    cases = split_cases(find_root(args.data, args.extract_to), args.splits)[args.split]
    print(f"--- {name}: {ckpt.name} on {len(cases)} {args.split} cases, device {device}", flush=True)
    per_case = evaluate(model, cases, cfg, device)
    per_case.to_csv(out / f"eval_seg_{args.split}_{name}.csv", index=False)

    if args.split == "val":
        min_et, table = tune_min_et(per_case)
        print(table.to_string(index=False))
        result = {"min_et_voxels": min_et, "tuning": table.to_dict("records"),
                  "val_raw": summary(per_case, 0), "val": summary(per_case, min_et)}
        rule_file.write_text(json.dumps(result, indent=2))
    else:
        min_et = json.loads(rule_file.read_text())["min_et_voxels"]
        result = {"min_et_voxels": min_et, "test_raw": summary(per_case, 0), "test": summary(per_case, min_et),
                  "n_test": len(per_case), "checkpoint": ckpt.name}
        (out / f"metrics_seg_test_{name}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "tuning"}, indent=2))
    return result


if __name__ == "__main__":
    main()
