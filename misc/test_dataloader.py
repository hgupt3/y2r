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

# Add parent directory to path to import y2r
sys.path.insert(0, str(Path(__file__).parent.parent))

from y2r.dataloaders.track_dataloader import TrackDataset
from y2r.dataloaders.utils import get_dataloader


def visualize_sample(imgs, tracks, sample_idx, output_dir, img_size=128, vis_scale=4):
    """
    Visualize a sample with trajectory overlay using CoTracker-style visualization.
    Upscales the image for better visualization quality.
    
    Args:
        imgs: (frame_stack, C, H, W) - ImageNet normalized image tensor
        tracks: (num_track_ts, num_points, 2) - normalized track coordinates [0-1]
        sample_idx: Sample index for filename
        output_dir: Directory to save visualization
        img_size: Image size
        vis_scale: Scale factor for visualization (default 4x = 512x512 from 128x128)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Import CoTracker visualizer
    import sys
    from pathlib import Path as P
    sys.path.insert(0, str(P(__file__).parent.parent / "thirdparty"))
    from cotracker3.cotracker.utils.visualizer import Visualizer
    import torch.nn.functional as F
    
    # Get the last frame (most recent observation)
    frame = imgs[-1]  # (C, H, W), ImageNet normalized
    
    # Denormalize from ImageNet normalization to [0, 1]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    frame = frame * std + mean
    frame = torch.clamp(frame, 0, 1) * 255  # Convert to [0, 255]
    
    # Upscale frame for better visualization
    vis_size = img_size * vis_scale
    frame_upscaled = F.interpolate(
        frame.unsqueeze(0),  # Add batch: (1, C, H, W)
        size=(vis_size, vis_size),
        mode='bilinear',
        align_corners=False
    ).squeeze(0)  # Remove batch: (C, H, W)
    
    # Darken frame for better trajectory visibility (like CoTracker does)
    frame_darkened = frame_upscaled * 0.5
    
    # Convert tracks to pixel coordinates at upscaled resolution
    tracks_px = tracks.clone()
    tracks_px = tracks_px * vis_size  # Denormalize to upscaled pixel space
    
    # Add batch dimension: (num_track_ts, num_points, 2) -> (1, num_track_ts, num_points, 2)
    tracks_batch = tracks_px.unsqueeze(0)
    
    # Create visualizer with thin lines (circle radius = linewidth * 2)
    # At 4x upscale, linewidth=1 gives nice thin lines and small circles
    vis = Visualizer(save_dir=str(output_dir), pad_value=0, linewidth=1)
    
    # Render trajectory on frame
    rendered_frame = vis.visualize_trajectory_on_frame(
        frame=frame_darkened,
        tracks=tracks_batch,  # (1, num_track_ts, num_points, 2)
        visibility=None,
        segm_mask=None,
        query_frame=0,
        opacity=1.0,
    )
    
    # Save visualization
    output_path = output_dir / f"sample_{sample_idx:03d}_trajectories.png"
    cv2.imwrite(str(output_path), cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2BGR))
    
    return output_path


def test_dataloader(h5_dir, img_size=128, frame_stack=1, num_track_ts=16, num_track_ids=32, 
                   downsample_factor=1, visualize=False, vis_dir="test_visualizations", 
                   num_vis_samples=5, vis_scale=4, test_augmentations=False, aug_prob=0.9, aug_config=None):
    """
    Test loading data from HDF5 files.
    
    Args:
        h5_dir: Directory containing .hdf5 files
        img_size: Image size (default 128)
        frame_stack: Number of frames to stack (default 1)
        num_track_ts: Number of future track timesteps (default 16)
        num_track_ids: Minimum number of tracks required per frame (default 32)
        downsample_factor: Temporal downsampling factor (default 1)
        visualize: Whether to create visualizations (default False)
        vis_dir: Directory to save visualizations (default "test_visualizations")
        num_vis_samples: Number of samples to visualize (default 5)
        vis_scale: Upscaling factor for visualization quality (default 4x)
        test_augmentations: Whether to enable augmentations (default False)
        aug_prob: Augmentation probability when test_augmentations=True (default 0.9)
        aug_config: Dictionary with augmentation config from train_cfg.yaml (optional)
    """
    print(f"\n{'='*70}")
    print("TESTING DATALOADER WITH H5 DATASET")
    print(f"{'='*70}")
    print(f"Dataset directory: {h5_dir}")
    print(f"Image size: {img_size}x{img_size}")
    print(f"Frame stack: {frame_stack}")
    print(f"Downsample factor: {downsample_factor}x")
    print(f"Future track timesteps: {num_track_ts}")
    print(f"Minimum tracks per frame: {num_track_ids}")
    if test_augmentations:
        print(f"Augmentations: ENABLED (prob={aug_prob})")
        if aug_config:
            enabled_augs = []
            if aug_config.get('aug_color_jitter'):
                enabled_augs.append('color_jitter')
            if aug_config.get('aug_translation_px', 0) > 0:
                enabled_augs.append(f"translation({aug_config.get('aug_translation_px')}px)")
            if aug_config.get('aug_rotation_deg', 0) > 0:
                enabled_augs.append(f"rotate(±{aug_config.get('aug_rotation_deg')}°)")
            if aug_config.get('aug_hflip_prob', 0) > 0:
                enabled_augs.append(f"hflip({aug_config.get('aug_hflip_prob')})")
            if aug_config.get('aug_vflip_prob', 0) > 0:
                enabled_augs.append(f"vflip({aug_config.get('aug_vflip_prob')})")
            if aug_config.get('aug_noise_std', 0) > 0:
                enabled_augs.append(f"noise(std={aug_config.get('aug_noise_std')})")
            print(f"  Enabled: {', '.join(enabled_augs) if enabled_augs else 'none'}")
    else:
        print(f"Augmentations: DISABLED")
    print(f"{'='*70}\n")
    
    # Create dataset
    print("Creating dataset...")
    try:
        # Prepare dataset kwargs
        dataset_kwargs = {
            'dataset_dir': h5_dir,
            'img_size': img_size,
            'num_track_ts': num_track_ts,
            'num_track_ids': num_track_ids,
            'frame_stack': frame_stack,
            'downsample_factor': downsample_factor,
            'cache_all': True,
            'cache_image': True,  # Images are stored in HDF5, keep them in cache
            'num_demos': None,
            'aug_prob': aug_prob if test_augmentations else 0.0
        }
        
        # Add augmentation config if provided (compact format)
        if aug_config and test_augmentations:
            dataset_kwargs.update({
                'aug_color_jitter': aug_config.get('aug_color_jitter'),
                'aug_translation_px': aug_config.get('aug_translation_px', 0),
                'aug_rotation_deg': aug_config.get('aug_rotation_deg', 0),
                'aug_hflip_prob': aug_config.get('aug_hflip_prob', 0.0),
                'aug_vflip_prob': aug_config.get('aug_vflip_prob', 0.0),
                'aug_noise_std': aug_config.get('aug_noise_std', 0.0)
            })
        
        dataset = TrackDataset(**dataset_kwargs)
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
            # Images are ImageNet normalized, so range is roughly [-2, 2]
            assert -5 <= imgs.min() <= 5 and -5 <= imgs.max() <= 5, \
                f"Images should be normalized (roughly [-2, 2]), got [{imgs.min()}, {imgs.max()}]"
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
        if test_augmentations:
            print(f"  NOTE: Augmentations are ENABLED - you'll see augmented samples")
        vis_output_dir = Path(vis_dir)
        
        # Delete previous visualizations
        if vis_output_dir.exists():
            import shutil
            print(f"  Deleting previous visualizations in {vis_output_dir}...")
            shutil.rmtree(vis_output_dir)
        
        # Create fresh output directory
        vis_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Select random samples to visualize
        num_to_vis = min(num_vis_samples, len(dataset))
        sample_indices = random.sample(range(len(dataset)), num_to_vis)
        
        for i, idx in enumerate(sample_indices):
            try:
                imgs, tracks = dataset[idx]
                output_path = visualize_sample(imgs, tracks, idx, vis_output_dir, img_size, vis_scale=vis_scale)
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
    import yaml
    
    # Load defaults from train_cfg.yaml
    train_cfg_path = Path(__file__).parent.parent / "train_cfg.yaml"
    if train_cfg_path.exists():
        with open(train_cfg_path, 'r') as f:
            train_cfg = yaml.safe_load(f)
        default_h5_dir = train_cfg.get('dataset_dir', '/home/harsh/sam/data/h5_dataset')
        dataset_cfg = train_cfg.get('dataset_cfg', {})
        default_img_size = dataset_cfg.get('img_size', 128)
        default_frame_stack = dataset_cfg.get('frame_stack', 1)
        default_num_track_ts = dataset_cfg.get('num_track_ts', 16)
        default_num_track_ids = dataset_cfg.get('num_track_ids', 32)
        default_downsample_factor = dataset_cfg.get('downsample_factor', 1)
        default_aug_prob = train_cfg.get('training', {}).get('aug_prob', 0.9)
        
        # Extract augmentation config (compact format)
        aug_config = {
            'aug_color_jitter': dataset_cfg.get('aug_color_jitter'),
            'aug_translation_px': dataset_cfg.get('aug_translation_px', 0),
            'aug_rotation_deg': dataset_cfg.get('aug_rotation_deg', 0),
            'aug_hflip_prob': dataset_cfg.get('aug_hflip_prob', 0.0),
            'aug_vflip_prob': dataset_cfg.get('aug_vflip_prob', 0.0),
            'aug_noise_std': dataset_cfg.get('aug_noise_std', 0.0)
        }
    else:
        raise ValueError(f"train_cfg.yaml not found at {train_cfg_path}")
    
    parser = argparse.ArgumentParser(description="Test HDF5 dataset with dataloader")
    parser.add_argument("--h5_dir", type=str, default=default_h5_dir,
                        help=f"Directory containing .hdf5 files (default: from train_cfg.yaml)")
    parser.add_argument("--img_size", type=int, default=default_img_size,
                        help=f"Image size (default: {default_img_size} from train_cfg.yaml)")
    parser.add_argument("--frame_stack", type=int, default=default_frame_stack,
                        help=f"Number of frames to stack (default: {default_frame_stack} from train_cfg.yaml)")
    parser.add_argument("--num_track_ts", type=int, default=default_num_track_ts,
                        help=f"Number of future track timesteps (default: {default_num_track_ts} from train_cfg.yaml)")
    parser.add_argument("--num_track_ids", type=int, default=default_num_track_ids,
                        help=f"Minimum number of tracks per frame (default: {default_num_track_ids} from train_cfg.yaml)")
    parser.add_argument("--downsample_factor", type=int, default=default_downsample_factor,
                        help=f"Temporal downsampling factor (default: {default_downsample_factor} from train_cfg.yaml)")
    parser.add_argument("--visualize", action="store_true",
                        help="Create visualizations of samples")
    parser.add_argument("--vis_dir", type=str, default="test_visualizations",
                        help="Directory to save visualizations")
    parser.add_argument("--num_vis_samples", type=int, default=20,
                        help="Number of samples to visualize")
    parser.add_argument("--vis_scale", type=int, default=2,
                        help="Upscaling factor for visualization (e.g., 4 = 512x512 from 128x128)")
    parser.add_argument("--test_augmentations", action="store_true",
                        help="Enable data augmentations for testing (see augmented samples)")
    parser.add_argument("--aug_prob", type=float, default=default_aug_prob,
                        help=f"Augmentation probability when test_augmentations is enabled (default: {default_aug_prob} from train_cfg.yaml)")
    
    args = parser.parse_args()
    
    success = test_dataloader(
        h5_dir=args.h5_dir,
        img_size=args.img_size,
        frame_stack=args.frame_stack,
        num_track_ts=args.num_track_ts,
        num_track_ids=args.num_track_ids,
        downsample_factor=args.downsample_factor,
        visualize=args.visualize,
        vis_dir=args.vis_dir,
        num_vis_samples=args.num_vis_samples,
        vis_scale=args.vis_scale,
        test_augmentations=args.test_augmentations,
        aug_prob=args.aug_prob,
        aug_config=aug_config
    )
    
    sys.exit(0 if success else 1)
