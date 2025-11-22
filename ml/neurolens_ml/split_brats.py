"""Split BraTS by patient into train / val / test from brats_cases.csv (no scans needed).

    python -m neurolens_ml.split_brats
"""
import argparse
from pathlib import Path

import pandas as pd

from .data.brats import make_splits

DATA = Path(__file__).resolve().parents[1] / "data"


def main(argv: list[str] | None = None) -> pd.DataFrame:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", type=Path, default=DATA / "brats_cases.csv", help="made by prep_brats")
    ap.add_argument("--out", type=Path, default=DATA / "brats_splits.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    splits = make_splits(pd.read_csv(args.cases), args.seed)
    splits.to_csv(args.out, index=False)
    print(splits.groupby("split").agg(cases=("id", "size"), wt_median=("wt_cm3", "median"),
                                      no_et=("et_cm3", lambda v: int((v == 0).sum()))))
    return splits


if __name__ == "__main__":
    main()
