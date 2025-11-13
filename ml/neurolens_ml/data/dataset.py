"""Splits, augmentations and the PyTorch dataset for the 2D classifier."""
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..config import CLASSES, MEAN, STD
from .crop import crop_brain


def load_rgb(path: str | Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))  # grayscale MRI -> 3 channels for the pretrained net


def load_splits(csv: str | Path) -> dict[str, pd.DataFrame]:
    """train / val / test frames from splits.csv, with an integer label column `y`."""
    df = pd.read_csv(csv)
    df["y"] = df.label.map({c: i for i, c in enumerate(CLASSES)})
    return {s: df[df.split == s].reset_index(drop=True) for s in ["train", "val", "test"]}


def make_transforms(cfg: dict) -> tuple[A.Compose, A.Compose]:
    """Train and eval transforms. Eval is only resize + normalize."""
    size = cfg["img_size"]
    if cfg["strong_aug"]:
        aug = [
            A.HorizontalFlip(p=0.5),  # brains are roughly left-right symmetric
            A.Affine(scale=(0.85, 1.15), translate_percent=(-0.08, 0.08), rotate=(-20, 20), p=0.8),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
            A.RandomGamma(p=0.3),  # the dataset has the same scans with other contrast curves
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),  # sources differ in sharpness: don't let that be a clue
            A.GaussNoise(p=0.2),
            A.CoarseDropout(p=0.3),
        ]
    else:
        aug = [
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-15, 15), p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        ]
    train_tf = A.Compose([A.Resize(size, size), *aug, A.Normalize(mean=MEAN, std=STD), ToTensorV2()])
    eval_tf = A.Compose([A.Resize(size, size), A.Normalize(mean=MEAN, std=STD), ToTensorV2()])
    return train_tf, eval_tf


class MRIDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, data_dir: Path, tf: A.Compose, crop: bool):
        self.paths = [data_dir / p for p in frame.path]
        self.ys = frame.y.tolist()
        self.tf, self.crop = tf, crop

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, i):
        img = load_rgb(self.paths[i])
        if self.crop:
            img = crop_brain(img)
        return self.tf(image=img)["image"], self.ys[i]


def make_loader(frame: pd.DataFrame, data_dir: Path, tf: A.Compose, cfg: dict, shuffle: bool, pin: bool = False) -> DataLoader:
    return DataLoader(MRIDataset(frame, data_dir, tf, cfg["crop"]), batch_size=cfg["batch_size"], shuffle=shuffle,
                      num_workers=cfg["num_workers"], pin_memory=pin, drop_last=shuffle)


def preprocess(img: np.ndarray, size: int, crop: bool = True) -> np.ndarray:
    """Numpy-only version of the eval transform (for ONNX serving): RGB uint8 -> float32 [1, 3, size, size]."""
    if crop:
        img = crop_brain(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    x = (img.astype(np.float32) / 255.0 - np.array(MEAN, np.float32)) / np.array(STD, np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])
