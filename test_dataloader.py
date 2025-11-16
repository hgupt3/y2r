#!/usr/bin/env python3
"""
Test script to verify HDF5 dataset works with dataloader.
"""
import sys
from pathlib import Path
import torch
import numpy as np
import cv2
import random

# Add dataloaders to path
sys.path.insert(0, str(Path(__file__).parent))

from dataloaders.track_dataloader import TrackDataset
from dataloaders.utils import get_dataloader


def visualize_sample(imgs, tracks, sample_idx, output_dir, img_size=128):
    """
    Visualize a sample with trajectory overlay.
    
    Args:
        imgs: (frame_stack, C, H, W) - image tensor [0-255]
        tracks: (num_track_ts, num_points, 2) - normalized track coordinates [0-1]
        sample_idx: Sample index for filename
        output_dir: Directory to save visualization
        img_size: Image size
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get the last frame (most recent observation)
    frame = imgs[-1].numpy()  # (C, H, W)
    frame = np.transpose(frame, (1, 2, 0))  # (H, W, C)
    frame = frame.astype(np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # For cv2 saving
    
    # Create visualization canvas
    vis_frame = frame.copy()
    
    num_track_ts, num_points, _ = tracks.shape
    
    # Generate colors for each track point
    np.random.seed(42)  # Consistent colors
    colors = []
    for _ in range(num_points):
        color = (
            int(np.random.randint(50, 255)),
            int(np.random.randint(50, 255)),
            int(np.random.randint(50, 255))
        )
        colors.append(color)
    
    # Draw trajectories
    for point_idx in range(num_points):
        trajectory = tracks[:, point_idx, :]  # (num_track_ts, 2)
        
        # Convert normalized coordinates to pixel coordinates
        trajectory_px = trajectory.numpy() * img_size
        trajectory_px = trajectory_px.astype(np.int32)
        
        # Draw lines connecting trajectory points
        for t in range(len(trajectory_px) - 1):
            pt1 = tuple(trajectory_px[t])
            pt2 = tuple(trajectory_px[t + 1])
            cv2.line(vis_frame, pt1, pt2, colors[point_idx], thickness=1, lineType=cv2.LINE_AA)
        
        # Draw start point (larger circle)
        start_pt = tuple(trajectory_px[0])
        cv2.circle(vis_frame, start_pt, radius=3, color=colors[point_idx], thickness=-1)
    
    # Save visualization
    output_path = output_dir / f"sample_{sample_idx:03d}_trajectories.png"
    cv2.imwrite(str(output_path), vis_frame)
    
    return output_path


def test_dataloader(h5_dir, img_size=128, frame_stack=1, num_track_ts=16, num_track_ids=32, 
                   visualize=False, vis_dir="test_visualizations", num_vis_samples=5):
    """
    Test loading data from HDF5 files.
    
    Args:
        h5_dir: Directory containing .hdf5 files
        img_size: Image size (default 128)
        frame_stack: Number of frames to stack (default 1)
        num_track_ts: Number of future track timesteps (default 16)
        num_track_ids: Minimum number of tracks required per frame (default 32)
        visualize: Whether to create visualizations (default False)
        vis_dir: Directory to save visualizations (default "test_visualizations")
        num_vis_samples: Number of samples to visualize (default 5)
    """
    print(f"\n{'='*70}")
    print("TESTING DATALOADER WITH H5 DATASET")
    print(f"{'='*70}")
    print(f"Dataset directory: {h5_dir}")
    print(f"Image size: {img_size}x{img_size}")
    print(f"Frame stack: {frame_stack}")
    print(f"Future track timesteps: {num_track_ts}")
    print(f"Minimum tracks per frame: {num_track_ids}")
    print(f"{'='*70}\n")
    
    # Create dataset
    print("Creating dataset...")
    try:
        dataset = TrackDataset(
            dataset_dir=h5_dir,
            img_size=img_size,
            num_track_ts=num_track_ts,
            num_track_ids=num_track_ids,
            frame_stack=frame_stack,
            cache_all=True,
            cache_image=False,
            num_demos=None,
            aug_prob=0.0  # No augmentation for testing
        )
        print(f"✓ Dataset created successfully")
        print(f"  Total samples: {len(dataset)}")
    except Exception as e:
        print(f"✗ Failed to create dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    if len(dataset) == 0:
        print("✗ Dataset is empty (no valid samples)")
        return False
    
    # Test loading a few samples
    print(f"\nTesting sample loading...")
    num_samples_to_test = min(5, len(dataset))
    
    for i in range(num_samples_to_test):
        try:
            imgs, tracks = dataset[i]
            
            print(f"\nSample {i}:")
            print(f"  Images shape: {imgs.shape}")
            print(f"  Tracks shape: {tracks.shape}")
            print(f"  Images dtype: {imgs.dtype}, range: [{imgs.min():.2f}, {imgs.max():.2f}]")
            print(f"  Tracks dtype: {tracks.dtype}, range: [{tracks.min():.4f}, {tracks.max():.4f}]")
            
            # Validate shapes
            assert imgs.shape == (frame_stack, 3, img_size, img_size), \
                f"Expected images shape ({frame_stack}, 3, {img_size}, {img_size}), got {imgs.shape}"
            assert tracks.shape[0] == num_track_ts, \
                f"Expected tracks timesteps {num_track_ts}, got {tracks.shape[0]}"
            assert tracks.shape[1] <= num_track_ids, \
                f"Expected tracks to have at most {num_track_ids} points, got {tracks.shape[1]}"
            assert tracks.shape[2] == 2, \
                f"Expected tracks to have 2 coordinates, got {tracks.shape[2]}"
            
            # Validate ranges
            assert 0 <= imgs.min() <= 255 and 0 <= imgs.max() <= 255, \
                f"Images should be in range [0, 255], got [{imgs.min()}, {imgs.max()}]"
            assert 0 <= tracks.min() <= 1 and 0 <= tracks.max() <= 1, \
                f"Tracks should be normalized to [0, 1], got [{tracks.min()}, {tracks.max()}]"
            
            print(f"  ✓ Sample {i} validated")
            
        except Exception as e:
            print(f"  ✗ Failed to load sample {i}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Test dataloader with batching
    print(f"\nTesting dataloader with batching...")
    try:
        dataloader = get_dataloader(
            dataset,
            mode="train",
            num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
            batch_size=2
        )
        
        batch = next(iter(dataloader))
        imgs_batch, tracks_batch = batch
        
        print(f"  Batch images shape: {imgs_batch.shape}")
        print(f"  Batch tracks shape: {tracks_batch.shape}")
        
        assert imgs_batch.shape[0] == 2, f"Expected batch size 2, got {imgs_batch.shape[0]}"
        assert imgs_batch.shape[1:] == (frame_stack, 3, img_size, img_size), \
            f"Expected batch shape (2, {frame_stack}, 3, {img_size}, {img_size}), got {imgs_batch.shape}"
        
        print(f"  ✓ Dataloader batching works correctly")
        
    except Exception as e:
        print(f"  ✗ Failed to test dataloader: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Visualize random samples if requested
    if visualize:
        print(f"\nCreating visualizations...")
        vis_output_dir = Path(vis_dir)
        
        # Select random samples to visualize
        num_to_vis = min(num_vis_samples, len(dataset))
        sample_indices = random.sample(range(len(dataset)), num_to_vis)
        
        for i, idx in enumerate(sample_indices):
            try:
                imgs, tracks = dataset[idx]
                output_path = visualize_sample(imgs, tracks, idx, vis_output_dir, img_size)
                print(f"  ✓ Saved visualization {i+1}/{num_to_vis}: {output_path}")
            except Exception as e:
                print(f"  ✗ Failed to visualize sample {idx}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n  Visualizations saved to: {vis_output_dir}")
    
    print(f"\n{'='*70}")
    print("✅ ALL TESTS PASSED!")
    print(f"{'='*70}\n")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test HDF5 dataset with dataloader")
    parser.add_argument("--h5_dir", type=str, default="/home/harsh/sam/data/h5_dataset",
                        help="Directory containing .hdf5 files")
    parser.add_argument("--img_size", type=int, default=128,
                        help="Image size")
    parser.add_argument("--frame_stack", type=int, default=1,
                        help="Number of frames to stack")
    parser.add_argument("--num_track_ts", type=int, default=16,
                        help="Number of future track timesteps")
    parser.add_argument("--num_track_ids", type=int, default=32,
                        help="Minimum number of tracks per frame")
    parser.add_argument("--visualize", action="store_true",
                        help="Create visualizations of samples")
    parser.add_argument("--vis_dir", type=str, default="test_visualizations",
                        help="Directory to save visualizations")
    parser.add_argument("--num_vis_samples", type=int, default=5,
                        help="Number of samples to visualize")
    
    args = parser.parse_args()
    
    success = test_dataloader(
        h5_dir=args.h5_dir,
        img_size=args.img_size,
        frame_stack=args.frame_stack,
        num_track_ts=args.num_track_ts,
        num_track_ids=args.num_track_ids,
        visualize=args.visualize,
        vis_dir=args.vis_dir,
        num_vis_samples=args.num_vis_samples
    )
    
    sys.exit(0 if success else 1)
