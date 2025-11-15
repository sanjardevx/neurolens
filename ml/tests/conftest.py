"""Tiny fake dataset in the Kaggle layout, shared by the tests."""
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from neurolens_ml.config import CLASSES

from .helpers import fake_slice


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
