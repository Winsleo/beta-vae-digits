import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import os

def prepare_data(save_dir="data"):
    """
    Load digits dataset, binarize, and split into train/val/test.
    Save split indices for reproducibility.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    digits = load_digits()
    x = (digits.data / 16.0 >= 0.5).astype("float32")
    y = digits.target

    # Two-stage stratified split: 60% train, 20% val, 20% test
    indices = np.arange(len(y))
    train_idx, rest_idx = train_test_split(
        indices, test_size=0.4, stratify=y, random_state=42
    )
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=0.5, stratify=y[rest_idx], random_state=42
    )

    # Save indices
    np.save(os.path.join(save_dir, "train_idx.npy"), train_idx)
    np.save(os.path.join(save_dir, "val_idx.npy"), val_idx)
    np.save(os.path.join(save_dir, "test_idx.npy"), test_idx)

    # Save full data and labels
    np.save(os.path.join(save_dir, "x_all.npy"), x)
    np.save(os.path.join(save_dir, "y_all.npy"), y)

    print(f"✓ Data prepared: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    print(f"  Indices saved to {save_dir}/")

    return x, y, train_idx, val_idx, test_idx

if __name__ == "__main__":
    prepare_data()
