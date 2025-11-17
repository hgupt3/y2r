"""
Utility functions for splitting H5 dataset into train and validation sets.
"""

import numpy as np
from pathlib import Path


def create_train_val_split(h5_dir, val_ratio=0.1, seed=42):
    """
    Split H5 dataset files into train and validation sets.
    
    Args:
        h5_dir: Directory containing .hdf5 files
        val_ratio: Fraction of files to use for validation (default: 0.1)
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        train_files: List of .hdf5 file paths for training
        val_files: List of .hdf5 file paths for validation
    """
    all_files = sorted(Path(h5_dir).glob("*.hdf5"))
    
    if len(all_files) == 0:
        raise ValueError(f"No .hdf5 files found in {h5_dir}")
    
    np.random.seed(seed)
    indices = np.random.permutation(len(all_files))
    
    n_val = int(len(all_files) * val_ratio)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    val_files = [str(all_files[i]) for i in val_indices]
    train_files = [str(all_files[i]) for i in train_indices]
    
    print(f"Dataset split: {len(train_files)} train, {len(val_files)} val")
    
    return train_files, val_files

