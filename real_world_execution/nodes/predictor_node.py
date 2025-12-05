#!/usr/bin/env python3
"""
Predictor Node - IntentTracker Trajectory Prediction

Subscribes to raw camera images (color + depth) and query points.
Internally crops/resizes to model resolution, runs inference,
and publishes predicted trajectories in camera resolution.
"""

import os
import sys
from collections import deque
from pathlib import Path
import time
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.amp import autocast
import cv2

# Ensure tracker_interfaces messages are discoverable
REAL_WORLD_ROOT = Path(__file__).resolve().parents[1]
CUSTOM_MSG_PATH = REAL_WORLD_ROOT / 'install' / 'tracker_interfaces' / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages'
if CUSTOM_MSG_PATH.exists():
    sys.path.insert(0, str(CUSTOM_MSG_PATH))

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from tracker_interfaces.msg import QueryPoints, PredictedTracks

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from y2r.models.factory import create_model
from y2r.dataloaders.utils import NormalizationStats


class PredictorNode(Node):
    def __init__(self):
        super().__init__('predictor_node')
        
        # Declare parameters
        self.declare_parameter('enabled', True)
        self.declare_parameter('train_config_path', '')
        self.declare_parameter('checkpoint_path', '')
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('depth_min', 0.1)  # meters
        self.declare_parameter('depth_max', 2.5)  # meters
        
        # Get parameters
        self.enabled = self.get_parameter('enabled').value
        train_config_path = self.get_parameter('train_config_path').value
        checkpoint_path = self.get_parameter('checkpoint_path').value
        self.device = self.get_parameter('device').value
        self.depth_min = self.get_parameter('depth_min').value
        self.depth_max = self.get_parameter('depth_max').value
        
        if not self.enabled:
            self.get_logger().info('Predictor Node disabled in config')
            return
        
        self.get_logger().info('Predictor Node initializing...')
        self.get_logger().info(f'  Device: {self.device}')
        self.get_logger().info(f'  Train config: {train_config_path}')
        self.get_logger().info(f'  Checkpoint: {checkpoint_path}')
        
        # Validate paths
        if not train_config_path or not checkpoint_path:
            self.get_logger().error('train_config_path and checkpoint_path must be set!')
            raise ValueError('Missing required configuration')
        
        # Load training config to get model/dataset parameters
        self.train_cfg = self.load_config(train_config_path)
        self.frame_stack = self.train_cfg.model.frame_stack
        self.num_future_steps = self.train_cfg.model.num_future_steps
        self.model_type = getattr(self.train_cfg.model, 'model_type', 'direct')
        self.is_diffusion = (self.model_type == 'diffusion')
        
        # Get track type (2d or 3d)
        self.track_type = getattr(self.train_cfg.model, 'track_type', '2d')
        self.is_3d = (self.track_type == '3d')
        
        # Get model resolution from config
        model_res = getattr(self.train_cfg.model, 'model_resolution', [224, 224])
        self.model_size = model_res[0]  # Assume square
        
        # Get dataset FPS for frame sampling
        self.dataset_fps = getattr(self.train_cfg.dataset_cfg, 'target_fps', 12.0)
        self.frame_dt = 1.0 / self.dataset_fps
        
        self.get_logger().info(f'  Model type: {self.model_type}')
        self.get_logger().info(f'  Track type: {self.track_type}')
        self.get_logger().info(f'  Model resolution: {self.model_size}x{self.model_size}')
        self.get_logger().info(f'  Frame stack: {self.frame_stack}')
        self.get_logger().info(f'  Future steps: {self.num_future_steps}')
        self.get_logger().info(f'  Dataset FPS: {self.dataset_fps}')
        
        # Load normalization stats using NormalizationStats class
        stats_path = Path(self.train_cfg.dataset_dir) / 'normalization_stats.yaml'
        self.norm_stats = NormalizationStats(str(stats_path))
        self.get_logger().info(f'  Loaded normalization stats from {stats_path}')
        self.get_logger().info(f'  Disp mean: {self.norm_stats.disp_mean}')
        self.get_logger().info(f'  Disp std: {self.norm_stats.disp_std}')
        if self.is_3d:
            self.get_logger().info(f'  Depth mean: {self.norm_stats.depth_mean}')
            self.get_logger().info(f'  Depth std: {self.norm_stats.depth_std}')
            self.get_logger().info(f'  Depth clipping: [{self.depth_min:.2f}, {self.depth_max:.2f}] meters')
        
        # Load model
        self.model = self.load_model(checkpoint_path)
        
        # Frame buffer: store (timestamp, color_frame, depth_frame) tuples
        self.buffer_duration = (self.frame_stack - 1) * self.frame_dt
        self.frame_buffer = deque(maxlen=100)
        self.get_logger().info(f'  Frame buffer duration: {self.buffer_duration:.3f}s')
        
        # Camera resolution (will be set on first frame)
        self.cam_width = None
        self.cam_height = None
        
        # Crop parameters (computed on first frame)
        self.crop_size = None
        self.crop_x = None
        self.crop_y = None
        self.scale = None
        
        # Latest query points (in normalized [0,1] camera space)
        self.latest_query_points = None
        self.query_lock = False
        
        # FPS tracking
        self.fps_start_time = time.time()
        self.process_count = 0
        
        # Pre-compute Gaussian kernel for GPU-based depth fill
        self.blur_kernel_size = 51
        self.blur_kernel = self._create_gaussian_kernel(self.blur_kernel_size).to(self.device)
        
        # Pre-cache ImageNet normalization tensors on GPU (must be float32!)
        self.imagenet_mean = torch.as_tensor(self.norm_stats.imagenet_mean, dtype=torch.float32).view(1, 3, 1, 1).to(self.device)
        self.imagenet_std = torch.as_tensor(self.norm_stats.imagenet_std, dtype=torch.float32).view(1, 3, 1, 1).to(self.device)
        
        # Setup subscriptions based on track type
        # Latest depth frame (for 3D mode)
        self.latest_depth = None
        
        # Subscribe to color
        self.color_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.color_callback,
            1
        )
        
        # Subscribe to depth if 3D
        if self.is_3d:
            self.depth_sub = self.create_subscription(
                Image,
                '/camera/camera/aligned_depth_to_color/image_raw',
                self.depth_callback,
                1
            )
            self.get_logger().info('  Subscribed to color + depth')
        else:
            self.get_logger().info('  Subscribed to color only (2D mode)')
        
        # Subscribe to query points from perception
        self.query_subscription = self.create_subscription(
            QueryPoints,
            '/query_points',
            self.query_callback,
            1
        )
        
        # Publisher for predicted tracks
        self.track_publisher = self.create_publisher(
            PredictedTracks,
            '/predicted_tracks',
            1
        )
        
        self.get_logger().info('Predictor Node ready!')
    
    def load_config(self, config_path):
        """Load training configuration (same as train.py)"""
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        # Load dataset config if referenced
        if 'dataset_config' in cfg:
            dataset_config_path = cfg['dataset_config']
            if not os.path.isabs(dataset_config_path):
                config_dir = os.path.dirname(config_path)
                dataset_config_path = os.path.join(config_dir, dataset_config_path)
            
            with open(dataset_config_path, 'r') as f:
                dataset_cfg = yaml.safe_load(f)
            
            cfg['dataset_cfg'] = dataset_cfg
            cfg['dataset_dir'] = dataset_cfg['dataset_dir']
            
            # Derive model parameters from dataset config
            if 'model' in cfg:
                cfg['model']['num_future_steps'] = dataset_cfg['num_track_ts']
                cfg['model']['frame_stack'] = dataset_cfg['frame_stack']
        
        # Convert to namespace
        class Namespace:
            def __init__(self, d):
                for k, v in d.items():
                    if isinstance(v, dict):
                        setattr(self, k, Namespace(v))
                    else:
                        setattr(self, k, v)
        
        return Namespace(cfg)
    
    def load_model(self, checkpoint_path):
        """Load trained model from checkpoint"""
        self.get_logger().info(f'Loading model from: {checkpoint_path}')
        
        # Create model
        device = torch.device(self.device)
        # Extract disp_stats for model creation (only needed for diffusion model)
        disp_stats = {
            'displacement_mean': self.norm_stats.disp_mean.tolist(),
            'displacement_std': self.norm_stats.disp_std.tolist(),
        }
        model = create_model(self.train_cfg, disp_stats=disp_stats, device=device, from_pretrained=False)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Prefer EMA model for better quality
        if 'ema_model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['ema_model_state_dict'])
            self.get_logger().info('  Loaded EMA model state')
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            self.get_logger().info('  Loaded regular model state')
        else:
            model.load_state_dict(checkpoint)
            self.get_logger().info('  Loaded model state directly')
        
        model.eval()
        
        epoch = checkpoint.get('epoch', 'unknown')
        self.get_logger().info(f'  Checkpoint epoch: {epoch}')
        
        return model
    
    def compute_crop_params(self, width, height):
        """Compute center crop parameters for given camera resolution."""
        self.cam_width = width
        self.cam_height = height
        
        # Center crop to square
        self.crop_size = min(width, height)
        self.crop_x = (width - self.crop_size) // 2
        self.crop_y = (height - self.crop_size) // 2
        
        # Scale from crop to model resolution
        self.scale = self.model_size / self.crop_size
        
        self.get_logger().info(f'  Camera resolution: {width}x{height}')
        self.get_logger().info(f'  Center crop: {self.crop_size}x{self.crop_size} at ({self.crop_x}, {self.crop_y})')
        self.get_logger().info(f'  Scale factor: {self.scale:.4f}')
    
    def center_crop_resize(self, img):
        """Apply center crop and resize to model resolution."""
        # Center crop to square
        cropped = img[self.crop_y:self.crop_y+self.crop_size, 
                      self.crop_x:self.crop_x+self.crop_size]
        # Resize to model resolution
        resized = cv2.resize(cropped, (self.model_size, self.model_size))
        return resized
    
    def _create_gaussian_kernel(self, kernel_size, sigma=None):
        """Create a Gaussian kernel for GPU convolution."""
        if sigma is None:
            sigma = kernel_size / 6.0
        
        # Create 1D Gaussian
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        
        # Create 2D kernel via outer product
        gauss_2d = gauss_1d.unsqueeze(1) @ gauss_1d.unsqueeze(0)
        
        # Shape for conv2d: (out_channels, in_channels, H, W)
        return gauss_2d.unsqueeze(0).unsqueeze(0)
    
    def _gpu_blur_fill(self, depth_tensor):
        """
        Fill invalid depth using GPU-based Gaussian blur.
        
        Args:
            depth_tensor: (T, H, W) tensor on GPU, in meters
            
        Returns:
            Filled depth tensor (T, H, W)
        """
        # Mark invalid pixels
        invalid = (depth_tensor <= 0) | (depth_tensor < self.depth_min)
        
        if not invalid.any():
            return depth_tensor
        
        # Set invalid to 0 for blur
        depth_for_blur = depth_tensor.clone()
        depth_for_blur[invalid] = 0
        
        # Create valid mask (1 where valid, 0 where invalid)
        valid_mask = (~invalid).float()
        
        # Add batch and channel dims for conv2d: (T, H, W) -> (T, 1, H, W)
        depth_4d = depth_for_blur.unsqueeze(1)
        mask_4d = valid_mask.unsqueeze(1)
        
        # Apply Gaussian blur (padding to keep same size)
        pad = self.blur_kernel_size // 2
        blurred = F.conv2d(depth_4d, self.blur_kernel, padding=pad)
        blurred_mask = F.conv2d(mask_4d, self.blur_kernel, padding=pad)
        
        # Normalize by valid neighbor count
        blurred_mask = blurred_mask.clamp(min=1e-6)
        filled = (blurred / blurred_mask).squeeze(1)  # (T, H, W)
        
        # Replace only invalid pixels
        result = depth_tensor.clone()
        result[invalid] = filled[invalid]
        
        return result
    
    def cam_to_model_coords(self, points_norm):
        """
        Transform normalized [0,1] camera coords to normalized [0,1] model coords.
        
        Args:
            points_norm: (N, 2) array of normalized [0,1] coords in camera space
            
        Returns:
            (N, 2) array of normalized [0,1] coords in model space
        """
        # Convert to camera pixels
        cam_x = points_norm[:, 0] * self.cam_width
        cam_y = points_norm[:, 1] * self.cam_height
        
        # Apply crop offset and scale to get model pixels
        model_px_x = (cam_x - self.crop_x) * self.scale
        model_px_y = (cam_y - self.crop_y) * self.scale
        
        # Normalize to [0, 1] in model space
        model_x = model_px_x / self.model_size
        model_y = model_px_y / self.model_size
        
        # Clip to valid [0, 1] range
        model_x = np.clip(model_x, 0, 1)
        model_y = np.clip(model_y, 0, 1)
        
        return np.stack([model_x, model_y], axis=1)
    
    def model_to_cam_coords(self, points_model_norm):
        """
        Transform normalized [0,1] model x,y coords back to camera pixel coords.
        
        Args:
            points_model_norm: (N, T, 2) tensor of normalized model coords (x,y only)
            
        Returns:
            (N, T, 2) tensor in camera pixel coords
        """
        # Convert normalized to model pixels, then inverse transform to camera pixels
        model_px_x = points_model_norm[..., 0] * self.model_size
        model_px_y = points_model_norm[..., 1] * self.model_size
        
        # Inverse scale and add crop offset
        cam_x = (model_px_x / self.scale) + self.crop_x
        cam_y = (model_px_y / self.scale) + self.crop_y
        
        return torch.stack([cam_x, cam_y], dim=-1)
    
    def depth_callback(self, msg):
        """Store latest depth frame."""
        height, width = msg.height, msg.width
        depth_array = np.frombuffer(msg.data, dtype=np.uint16)
        self.latest_depth = depth_array.reshape((height, width))
    
    def color_callback(self, msg):
        """Handle color frames - main processing loop."""
        current_time = time.time()
        
        height, width = msg.height, msg.width
        img_array = np.frombuffer(msg.data, dtype=np.uint8)
        color_frame = img_array.reshape((height, width, 3))
        
        # Compute crop params on first frame
        if self.cam_width is None:
            self.compute_crop_params(width, height)
        
        # Get depth (use latest if 3D mode)
        depth_frame = self.latest_depth if self.is_3d else None
        
        # Add to buffer
        self.frame_buffer.append((current_time, color_frame, depth_frame))
        
        # Remove old frames
        cutoff_time = current_time - self.buffer_duration - 0.1
        while self.frame_buffer and self.frame_buffer[0][0] < cutoff_time:
            self.frame_buffer.popleft()
        
        # Try to predict
        if len(self.frame_buffer) >= self.frame_stack and self.latest_query_points is not None:
            self.predict_trajectories()
    
    def query_callback(self, msg):
        """Update latest query points from QueryPoints message."""
        if not self.query_lock:
            self.query_lock = True
            try:
                query_points = []
                for point in msg.points:
                    query_points.append([point.x, point.y])
                
                if len(query_points) > 0:
                    self.latest_query_points = np.array(query_points, dtype=np.float32)
            except Exception as e:
                self.get_logger().error(f'Error parsing query points: {e}')
            finally:
                self.query_lock = False
    
    def sample_frames_at_intervals(self):
        """
        Sample frame_stack frames from buffer at correct temporal intervals.
        
        Returns:
            (color_frames, depth_frames) or None if not enough frames
            depth_frames is None for 2D mode
        """
        if len(self.frame_buffer) < self.frame_stack:
            return None, None
        
        current_time = self.frame_buffer[-1][0]
        target_times = [current_time - i * self.frame_dt for i in range(self.frame_stack)]
        target_times = target_times[::-1]  # oldest to newest
        
        color_frames = []
        depth_frames = [] if self.is_3d else None
        
        for target_t in target_times:
            closest = min(self.frame_buffer, key=lambda x: abs(x[0] - target_t))
            color_frames.append(closest[1])
            if self.is_3d:
                depth_frames.append(closest[2])
        
        return color_frames, depth_frames
    
    def preprocess_frames(self, color_frames, depth_frames):
        """
        Preprocess frames for model input.
        
        Args:
            color_frames: list of numpy arrays (H, W, 3) uint8
            depth_frames: list of numpy arrays (H, W) uint16, or None for 2D
            
        Returns:
            frames: (1, T, 3, model_size, model_size) ImageNet normalized
            depth: (1, T, 1, model_size, model_size) normalized, or None
        """
        t0 = time.time()
        
        # Process color frames - crop/resize on CPU, then move to GPU for normalization
        processed_colors = []
        for frame in color_frames:
            cropped = self.center_crop_resize(frame)
            processed_colors.append(cropped)
        
        # Stack and move to GPU, then normalize there (much faster)
        frames = np.stack(processed_colors).astype(np.float32) / 255.0  # (T, H, W, 3) float32
        frames = torch.from_numpy(frames).to(self.device)
        frames = frames.permute(0, 3, 1, 2)  # (T, 3, H, W)
        
        # ImageNet normalization on GPU using cached tensors
        frames = (frames - self.imagenet_mean) / self.imagenet_std
        
        frames = frames.unsqueeze(0)  # (1, T, 3, H, W)
        
        # Process depth if 3D
        depth = None
        if self.is_3d and depth_frames is not None:
            processed_depths = []
            for d in depth_frames:
                cropped = self.center_crop_resize(d)
                processed_depths.append(cropped)
            
            # Stack and convert to meters (RealSense gives mm)
            depth_np = np.stack(processed_depths).astype(np.float32) / 1000.0  # (T, H, W) in meters
            depth = torch.from_numpy(depth_np).float().to(self.device)
            
            # Fill invalid depth using GPU-based blur
            depth = self._gpu_blur_fill(depth)
            
            # Clip depth to valid range before normalization
            depth = depth.clamp(self.depth_min, self.depth_max)
            
            # Normalize depth
            depth = self.norm_stats.normalize_depth_torch(depth)
            
            depth = depth.unsqueeze(0).unsqueeze(2)  # (1, T, 1, H, W)
        
        return frames, depth
    
    def sample_depth_at_points(self, depth_tensor, query_points_norm):
        """
        Sample depth values at query point locations from processed depth.
        
        Args:
            depth_tensor: (1, T, 1, H, W) normalized depth tensor (use latest frame)
            query_points_norm: (N, 2) normalized [0,1] model coords
            
        Returns:
            (N,) tensor of normalized depth values
        """
        # Use the latest depth frame: (1, T, 1, H, W) -> (H, W)
        depth_map = depth_tensor[0, -1, 0]  # Latest frame
        
        # Convert normalized coords to pixel coords for sampling
        x = (query_points_norm[:, 0] * self.model_size).long().clamp(0, self.model_size - 1)
        y = (query_points_norm[:, 1] * self.model_size).long().clamp(0, self.model_size - 1)
        
        sampled_depth = depth_map[y, x]
        return sampled_depth
    
    def predict_trajectories(self):
        """Run model inference and publish predictions."""
        try:
            total_start = time.time()
            
            # Sample frames
            color_frames, depth_frames = self.sample_frames_at_intervals()
            if color_frames is None:
                return
            
            # Preprocess frames
            frames, depth = self.preprocess_frames(color_frames, depth_frames)
            frames = frames.to(self.device)
            if depth is not None:
                depth = depth.to(self.device)
            
            # Get query coordinates and transform to model space
            if self.latest_query_points is None:
                return
            
            query_model = self.cam_to_model_coords(self.latest_query_points)  # (N, 2)
            
            # For 3D, add depth as z coordinate (already normalized)
            query_model_t = torch.tensor(query_model, dtype=torch.float32).to(self.device)
            if self.is_3d and depth is not None:
                query_depth = self.sample_depth_at_points(depth, query_model_t)  # Already normalized
                query_coords = torch.cat([query_model_t, query_depth.unsqueeze(1)], dim=1)  # (N, 3)
            else:
                query_coords = query_model_t  # (N, 2)
            
            query_coords = query_coords.unsqueeze(0)  # (1, N, coord_dim)
            
            # Run inference
            infer_start = time.time()
            with torch.no_grad(), autocast(device_type='cuda', enabled=True):
                if self.is_diffusion:
                    num_inference_steps = getattr(self.train_cfg.model, 'num_inference_steps', 10)
                    pred_disp = self.model.predict(frames, query_coords, depth=depth, num_inference_steps=num_inference_steps)
                else:
                    pred_disp = self.model.predict(frames, query_coords, depth=depth)
            infer_time = (time.time() - infer_start) * 1000
            
            # Denormalize displacements to get real changes
            pred_disp_denorm = self.norm_stats.denormalize_displacement_torch(pred_disp[0])
            
            # Denormalize query coords z if 3D (x,y are already in [0,1])
            query_xy = query_coords[0, :, :2]  # (N, 2) in [0,1]
            if self.is_3d and query_coords.shape[-1] == 3:
                query_z_denorm = self.norm_stats.denormalize_depth_torch(query_coords[0, :, 2])  # (N,) in meters
            
            # Compute absolute positions in real space
            # x,y: [0,1] + small_diff ≈ [0,1]
            # z: meters + meters = meters
            pred_xy = query_xy.unsqueeze(1) + pred_disp_denorm[..., :2]  # (N, T, 2)
            
            pred_z = None
            if self.is_3d and pred_disp_denorm.shape[-1] == 3:
                pred_z = query_z_denorm.unsqueeze(1) + pred_disp_denorm[..., 2]  # (N, T) in meters
            
            # Transform x,y back to camera pixel coords
            pred_tracks_cam = self.model_to_cam_coords(pred_xy)
            
            total_time = (time.time() - total_start) * 1000
            
            # Publish predictions
            self.publish_predictions(pred_tracks_cam, pred_z)
            
            # Store timing
            self._last_infer_time = infer_time
            self._last_total_time = total_time
            
            # Track FPS
            self.process_count += 1
            elapsed = time.time() - self.fps_start_time
            if elapsed > 2.0:
                fps = self.process_count / elapsed
                self.get_logger().info(f'FPS: {fps:.1f} | infer={self._last_infer_time:.1f}ms total={self._last_total_time:.1f}ms')
                self.process_count = 0
                self.fps_start_time = time.time()
        
        except Exception as e:
            self.get_logger().error(f'Error predicting trajectories: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def publish_predictions(self, pred_tracks, pred_z=None):
        """Publish predicted trajectories in camera pixel coords."""
        msg = PredictedTracks()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        
        # pred_tracks: (N, T, 2) tensor in camera pixels
        # pred_z: (N, T) tensor of denormalized depth values, or None
        N, T, _ = pred_tracks.shape
        
        for i in range(N):
            for t in range(T):
                msg.trajectory_x.append(float(pred_tracks[i, t, 0]))
                msg.trajectory_y.append(float(pred_tracks[i, t, 1]))
                if pred_z is not None:
                    msg.trajectory_z.append(float(pred_z[i, t]))
        
        msg.num_points = N
        msg.num_timesteps = T
        
        self.track_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PredictorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
