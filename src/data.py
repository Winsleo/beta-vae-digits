"""Binarised sklearn digits with a fixed, stratified 60/20/20 split."""
from types import SimpleNamespace

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from config import SPLIT_SEED


def load_binary_digits():
    """1797 images of 8x8, pixels 0-16 -> /16 -> threshold 0.5 (>=) -> 64-d float32."""
    digits = load_digits()
    x = (digits.data / 16.0 >= 0.5).astype("float32")
    return x, digits.target


def split_indices(y, seed: int = SPLIT_SEED):
    """Two-stage stratified split: 60% train, then the remaining 40% halved into val/test."""
    idx = np.arange(len(y))
    train, rest = train_test_split(idx, test_size=0.4, stratify=y, random_state=seed)
    val, test = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=seed)
    return train, val, test


def load_splits(seed: int = SPLIT_SEED) -> SimpleNamespace:
    """Returns an object with x_/y_/idx_ for train, val and test."""
    x, y = load_binary_digits()
    out = {}
    for name, idx in zip(("train", "val", "test"), split_indices(y, seed)):
        out[f"x_{name}"], out[f"y_{name}"], out[f"idx_{name}"] = x[idx], y[idx], idx
    return SimpleNamespace(**out)
