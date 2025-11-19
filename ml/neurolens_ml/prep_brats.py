"""Check the BraTS 2021 data: find every case, read headers and masks, load one case with MONAI.

    python -m neurolens_ml.prep_brats [--data <folder with .tar files or case folders>]

Writes brats_cases.csv (one row per patient) and brats_report.json.
"""
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd

from .config import KAGGLE_INPUT, out_dir
from .data.brats import KNOWN_LABELS, REGIONS, case_stats, extract_tars, find_cases, load_transforms

EXTRACT_TO = Path("/tmp/brats")  # outside /kaggle/working, so 13 GB of scans don't end up in the outputs


def find_root(arg: str | None, extract_to: Path) -> Path:
    src = Path(arg) if arg else KAGGLE_INPUT
    if next(src.glob("**/*_seg.nii.gz"), None):
        return src
    if next(src.glob("**/*.tar"), None):
        free_gb = shutil.disk_usage(extract_to.parent).free / 1e9
        print(f"unpacking .tar files to {extract_to} ({free_gb:.0f} GB free) ...", flush=True)
        return extract_tars(src, extract_to)
    raise FileNotFoundError("BraTS not found. Pass --data or attach dschettler8845/brats-2021-task1.")


def monai_check(case: dict) -> dict:
    """Load one case end to end and check the region masks nest: ET inside TC inside WT."""
    d = load_transforms()(case)
    image, label = d["image"], d["label"].bool()
    tc, wt, et = label
    return {
        "id": case["id"],
        "image_shape": list(image.shape),
        "label_shape": list(label.shape),
        "image_dtype": str(image.dtype),
        "intensity_max": [round(float(c.max()), 1) for c in image],
        "regions_nest": bool((et <= tc).all() and (tc <= wt).all()),
    }


def summary(df: pd.DataFrame) -> dict:
    seen = {int(v) for s in df.labels for v in s.split()}
    return {
        "n_cases": len(df),
        "shapes": dict(Counter(df["shape"])),
        "spacings_mm": dict(Counter(df.spacing)),
        "cases_with_mismatched_shapes": int((~df.same_shape).sum()),
        "label_values": sorted(seen),
        "unknown_labels": sorted(seen - KNOWN_LABELS),
        "cases_without_tumor": int((df.wt_cm3 == 0).sum()),
        "cases_without_enhancing": int((df.et_cm3 == 0).sum()),
        "volume_cm3": {r: df[f"{r}_cm3"].describe().round(1).to_dict() for r in REGIONS},
    }


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="folder with BraTS .tar files or unpacked case folders (default: /kaggle/input)")
    ap.add_argument("--extract-to", type=Path, default=EXTRACT_TO)
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    root = find_root(args.data, args.extract_to)
    cases = find_cases(root)
    print(f"{len(cases)} complete cases under {root}", flush=True)
    rows = []
    for i, case in enumerate(cases, 1):
        rows.append(case_stats(case))
        if i % 250 == 0:
            print(f"  read {i}/{len(cases)} masks", flush=True)
    df = pd.DataFrame(rows)

    out = out_dir(args.out)
    df.to_csv(out / "brats_cases.csv", index=False)
    report = summary(df) | {"monai_check": monai_check(cases[0])}
    (out / "brats_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
