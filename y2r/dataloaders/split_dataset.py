"""
Utility functions for splitting H5 dataset into train and validation sets.

Supports two modes:
- "episode": Split by H5 files (each file = one video/episode)
- "sample": Split individual samples across all files
"""

import numpy as np
from pathlib import Path
from torch.utils.data import Subset


def create_train_val_split(h5_dir, val_ratio=0.1, seed=42, split_mode="episode"):
    """
    Split H5 dataset files into train and validation sets.
    
    Args:
        h5_dir: Directory containing .hdf5 files
        val_ratio: Fraction of data to use for validation (default: 0.1)
        seed: Random seed for reproducibility (default: 42)
        split_mode: "episode" or "sample"
            - "episode": Split by H5 files (entire videos go to train or val)
            - "sample": Split individual samples (requires dataset to be created first)
    
    Returns:
        For "episode" mode:
            train_files: List of .hdf5 file paths for training
            val_files: List of .hdf5 file paths for validation
        
        For "sample" mode:
            train_indices: List of sample indices for training
            val_indices: List of sample indices for validation
    """
    if split_mode == "episode":
        return _split_by_episode(h5_dir, val_ratio, seed)
    elif split_mode == "sample":
        # For sample mode, we return indices - caller needs to handle this differently
        # This is a placeholder that will be used with total_samples from dataset
        return _get_sample_split_fn(val_ratio, seed)
    else:
        raise ValueError(f"Invalid split_mode: {split_mode}. Must be 'episode' or 'sample'")


def _split_by_episode(h5_dir, val_ratio, seed):
    """Split by H5 files (episode-wise)."""
    all_files = sorted(Path(h5_dir).glob("*.hdf5"))
    
    if len(all_files) == 0:
        raise ValueError(f"No .hdf5 files found in {h5_dir}")
    
    np.random.seed(seed)
    indices = np.random.permutation(len(all_files))
    
    n_val = max(1, int(len(all_files) * val_ratio))  # At least 1 val file
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    val_files = [str(all_files[i]) for i in val_indices]
    train_files = [str(all_files[i]) for i in train_indices]
    
    print(f"Dataset split (episode mode): {len(train_files)} train, {len(val_files)} val")
    
    return train_files, val_files


def _get_sample_split_fn(val_ratio, seed):
    """
    Return a function that splits samples given total count.
    This allows lazy splitting after dataset is created.
    """
    def split_samples(total_samples):
        np.random.seed(seed)
        indices = np.random.permutation(total_samples)
        
        n_val = max(1, int(total_samples * val_ratio))
        val_indices = indices[:n_val].tolist()
        train_indices = indices[n_val:].tolist()
        
        print(f"Dataset split (sample mode): {len(train_indices)} train, {len(val_indices)} val")
        
        return train_indices, val_indices
    
    return split_samples


def create_sample_split(dataset, val_ratio=0.1, seed=42):
    """
    Create train/val split by individual samples.
    
    Args:
        dataset: Full dataset to split
        val_ratio: Fraction of samples for validation
        seed: Random seed
    
    Returns:
        train_subset: Subset of dataset for training
        val_subset: Subset of dataset for validation
    """
    total_samples = len(dataset)
    
    np.random.seed(seed)
    indices = np.random.permutation(total_samples)
    
    n_val = max(1, int(total_samples * val_ratio))
    val_indices = indices[:n_val].tolist()
    train_indices = indices[n_val:].tolist()
    
    print(f"Dataset split (sample mode): {len(train_indices)} train, {len(val_indices)} val")
    
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    
    return train_subset, val_subset
