"""Run configs (YAML) and where the data lives (local or Kaggle)."""
import random
from pathlib import Path

import numpy as np
import torch
import yaml

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
MEAN = (0.485, 0.456, 0.406)  # ImageNet stats, to match the pretrained backbones
STD = (0.229, 0.224, 0.225)
KAGGLE_INPUT, KAGGLE_WORKING = Path("/kaggle/input"), Path("/kaggle/working")

# Tiny run on fake data: checks the code path, not the model
SMOKE_OVERRIDES = dict(img_size=64, batch_size=8, epochs=2, num_workers=0, pretrained=False)


def load_config(path: str | Path, smoke: bool = False) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if smoke:
        cfg.update(SMOKE_OVERRIDES)
    return cfg


def find_data_dir(arg: str | None = None) -> Path:
    """Folder that holds Training/ and Testing/ of the Kaggle brain tumor dataset."""
    if arg:
        return Path(arg)
    for p in KAGGLE_INPUT.glob("**/Training"):
        return p.parent
    raise FileNotFoundError("Dataset not found. Pass --data or attach masoudnickparvar/brain-tumor-mri-dataset.")


def find_splits(arg: str | None = None) -> Path:
    if arg:
        return Path(arg)
    repo_splits = Path(__file__).resolve().parents[1] / "data" / "splits.csv"
    for p in [*KAGGLE_INPUT.glob("**/splits.csv"), repo_splits]:
        if p.exists():
            return p
    raise FileNotFoundError("splits.csv not found. Pass --splits.")


def out_dir(arg: str | None = None) -> Path:
    path = Path(arg) if arg else KAGGLE_WORKING if KAGGLE_WORKING.exists() else Path("outputs")
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
