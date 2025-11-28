"""Tiny fake datasets shared by the tests: 2D slices in the Kaggle layout, and BraTS NIfTI cases."""
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from neurolens_ml.config import CLASSES

from .helpers import fake_slice, write_case


@pytest.fixture(scope="session")
def fake_data(tmp_path_factory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("fake")
    rows = []
    for split, n in [("train", 6), ("val", 2), ("test", 2)]:
        for k, label in enumerate(CLASSES):
            folder = root / "data" / "Training" / label
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                name = f"{split}_{i}.jpg"
                Image.fromarray(fake_slice(k=k)).save(folder / name)
                rows.append(dict(path=f"Training/{label}/{name}", label=label, split=split))
    pd.DataFrame(rows).to_csv(root / "splits.csv", index=False)
    return root / "data", root / "splits.csv"


@pytest.fixture
def brats_root(tmp_path) -> Path:
    """2 complete fake BraTS cases + 1 missing its T2."""
    root = tmp_path / "BraTS2021_Training_Data"
    write_case(root, "BraTS2021_00000")
    write_case(root, "BraTS2021_00002")
    write_case(root, "BraTS2021_00003", skip="t2")  # incomplete case is ignored
    return root
