"""The committed splits.csv must stay leak-free: these guard the test score."""
from pathlib import Path

import pandas as pd
import pytest

from neurolens_ml.config import CLASSES

SPLITS = Path(__file__).resolve().parents[1] / "data" / "splits.csv"


@pytest.fixture(scope="module")
def splits() -> pd.DataFrame:
    return pd.read_csv(SPLITS)


def test_no_image_in_two_splits(splits):
    assert not splits.path.duplicated().any()
    assert not splits.md5.duplicated().any()  # no exact copy anywhere


def test_duplicate_groups_stay_in_one_split(splits):
    assert splits.groupby("group").split.nunique().max() == 1


def test_every_split_has_every_class(splits):
    for split in ["train", "val", "test"]:
        assert set(splits[splits.split == split].label) == set(CLASSES)


def test_split_sizes_are_70_15_15(splits):
    share = splits.split.value_counts(normalize=True)
    assert share["train"] == pytest.approx(0.70, abs=0.01)
    assert share["val"] == pytest.approx(0.15, abs=0.01)
    assert share["test"] == pytest.approx(0.15, abs=0.01)
