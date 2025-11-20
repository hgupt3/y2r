#!/usr/bin/env python3
"""
Predictor Node - IntentTracker Trajectory Prediction

Subscribes to preprocessed images and query points, maintains frame buffer
with temporal sampling, runs IntentTracker model inference, and publishes
predicted trajectories.
"""

import os
import sys
from collections import deque
from pathlib import Path
import time
import yaml
import numpy as np
import torch
import torch.nn.functional as TF
from torchvision import transforms

# Ensure tracker_interfaces messages are discoverable
REAL_WORLD_ROOT = Path(__file__).resolve().parents[1]
CUSTOM_MSG_PATH = REAL_WORLD_ROOT / 'install' / 'tracker_interfaces' / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages'
if CUSTOM_MSG_PATH.exists():
    sys.path.insert(0, str(CUSTOM_MSG_PATH))

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import Header
from tracker_interfaces.msg import QueryPoints, PredictedTracks

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from y2r.models.factory import create_model
from y2r.visualization import denormalize_displacements


class PredictorNode(Node):
    def __init__(self):
        super().__init__('predictor_node')
        
        # Declare parameters
        self.declare_parameter('enabled', True)
        self.declare_parameter('train_config_path', '')
        self.declare_parameter('checkpoint_path', '')
        self.declare_parameter('device', 'cuda')
        
        # Get parameters
        self.enabled = self.get_parameter('enabled').value
        train_config_path = self.get_parameter('train_config_path').value
        checkpoint_path = self.get_parameter('checkpoint_path').value
        self.device = self.get_parameter('device').value
        
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
        
        # Get dataset FPS for frame sampling
        # Note: This should match the FPS used during dataset creation (preprocess step)
        # For your dataset, this was 12 FPS (from dataset_scripts/config.yaml)
        self.dataset_fps = getattr(self.train_cfg.dataset_cfg, 'target_fps', 12.0)  # Default to 12 FPS if not in config
        self.frame_dt = 1.0 / self.dataset_fps  # Time between frames in training data
        
        self.get_logger().info(f'  Model type: {self.model_type}')
        self.get_logger().info(f'  Frame stack: {self.frame_stack}')
        self.get_logger().info(f'  Future steps: {self.num_future_steps}')
        self.get_logger().info(f'  Dataset FPS: {self.dataset_fps}')
        
        # Load normalization stats
        stats_path = Path(self.train_cfg.dataset_dir) / 'normalization_stats.yaml'
        self.disp_stats = self.load_normalization_stats(str(stats_path))
        self.get_logger().info(f'  Disp mean: {self.disp_stats["displacement_mean"]}')
        self.get_logger().info(f'  Disp std: {self.disp_stats["displacement_std"]}')
        
        # Load model
        self.model = self.load_model(checkpoint_path)
        
        # Frame buffer: store (timestamp, frame_array) tuples
        self.buffer_duration = (self.frame_stack - 1) * self.frame_dt
        # Worst case: 60 FPS * 0.083s buffer = ~5 frames needed, use 100 as safety cap
        self.frame_buffer = deque(maxlen=100)
        self.get_logger().info(f'  Frame buffer duration: {self.buffer_duration:.3f}s')
        
        # Latest query points
        self.latest_query_points = None
        self.query_lock = False
        
        # ImageNet normalization transform
        self.normalize_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # FPS tracking
        self.fps_start_time = time.time()
        self.process_count = 0
        
        # Subscribe to preprocessed images
        self.image_subscription = self.create_subscription(
            Image,
            '/preprocessed_image',
            self.image_callback,
            1  # Small queue for freshest frames
        )
        
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
    
    def load_normalization_stats(self, stats_path):
        """Load displacement normalization statistics"""
        with open(stats_path, 'r') as f:
            stats = yaml.safe_load(f)
        return stats
    
    def load_model(self, checkpoint_path):
        """Load trained model from checkpoint"""
        self.get_logger().info(f'Loading model from: {checkpoint_path}')
        
        # Create model (from_pretrained=False skips torch.hub.load for ViT)
        device = torch.device(self.device)
        model = create_model(self.train_cfg, disp_stats=self.disp_stats, device=device, from_pretrained=False)
        
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
    
    def image_callback(self, msg):
        """Store incoming frames in buffer with timestamps"""
        current_time = time.time()
        
        # Convert ROS Image to numpy
        height, width = msg.height, msg.width
        img_array = np.frombuffer(msg.data, dtype=np.uint8)
        frame = img_array.reshape((height, width, 3))
        
        # Add to buffer
        self.frame_buffer.append((current_time, frame))
        
        # Remove old frames outside buffer duration
        cutoff_time = current_time - self.buffer_duration - 0.1  # Small margin
        while self.frame_buffer and self.frame_buffer[0][0] < cutoff_time:
            self.frame_buffer.popleft()
        
        # Try to predict if we have enough frames and query points
        if len(self.frame_buffer) >= self.frame_stack and self.latest_query_points is not None:
            self.predict_trajectories()
    
    def query_callback(self, msg):
        """Update latest query points from QueryPoints message"""
        if not self.query_lock:
            self.query_lock = True
            try:
                # Parse QueryPoints custom message: geometry_msgs/Point[] points
                # Each point has x, y, z (we only use x, y which are normalized [0, 1])
                query_points = []
                for point in msg.points:
                    query_points.append([point.x, point.y])
                
                if len(query_points) > 0:
                    self.latest_query_points = query_points
            except Exception as e:
                self.get_logger().error(f'Error parsing query points: {e}')
            finally:
                self.query_lock = False
    
    def publish_predictions(self, query_coords, pred_tracks):
        """Publish predicted trajectories"""
        msg = PredictedTracks()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        
        # query_coords: (N, 2) tensor
        # pred_tracks: (N, T, 2) tensor
        N, T, _ = pred_tracks.shape
        
        # Convert to lists
        for i in range(N):
            p = Point()
            p.x = float(query_coords[i, 0])
            p.y = float(query_coords[i, 1])
            p.z = 0.0
            msg.query_points.append(p)
            
            for t in range(T):
                msg.trajectory_x.append(float(pred_tracks[i, t, 0]))
                msg.trajectory_y.append(float(pred_tracks[i, t, 1]))
        
        msg.num_points = N
        msg.num_timesteps = T
        
        self.track_publisher.publish(msg)
    
    def sample_frames_at_intervals(self):
        """
        Sample frame_stack frames from buffer at correct temporal intervals.
        Uses closest-time matching to get evenly spaced samples.
        
        Returns:
            list of numpy arrays or None if not enough frames
        """
        if len(self.frame_buffer) < self.frame_stack:
            return None
        
        # Target times: current, current - dt, current - 2*dt, ..., current - (frame_stack-1)*dt
        current_time = self.frame_buffer[-1][0]
        target_times = [current_time - i * self.frame_dt for i in range(self.frame_stack)]
        target_times = target_times[::-1]  # Reverse to get [oldest, ..., newest]
        
        # Find closest frame for each target time
        sampled_frames = []
        for target_t in target_times:
            # Find frame with minimum time difference
            closest_frame = min(self.frame_buffer, key=lambda x: abs(x[0] - target_t))
            sampled_frames.append(closest_frame[1])
        
        return sampled_frames
    
    def preprocess_frames(self, frames_list):
        """
        Preprocess frames for model input.
        
        Args:
            frames_list: list of numpy arrays (224, 224, 3) uint8 [0-255]
            
        Returns:
            torch tensor (1, frame_stack, 3, 224, 224) float32, ImageNet normalized
        """
        # Stack frames and normalize to [0, 1]
        frames = np.stack(frames_list) / 255.0  # (T, H, W, 3)
        frames = torch.from_numpy(frames).float()
        frames = frames.permute(0, 3, 1, 2)  # (T, 3, H, W)
        
        # Apply ImageNet normalization
        frames = self.normalize_transform(frames)
        
        return frames.unsqueeze(0)  # (1, T, 3, H, W)
    
    def predict_trajectories(self):
        """Run model inference and publish predictions"""
        try:
            total_start = time.time()
            
            # Sample frames at correct temporal spacing
            sample_start = time.time()
            frames_list = self.sample_frames_at_intervals()
            if frames_list is None:
                return  # Not enough frames yet
            sample_time = (time.time() - sample_start) * 1000
            
            # Preprocess frames
            preproc_start = time.time()
            frames = self.preprocess_frames(frames_list)
            frames = frames.to(self.device)
            preproc_time = (time.time() - preproc_start) * 1000
            
            # Get query coordinates
            if self.latest_query_points is None:
                return
            
            query_coords = torch.tensor(self.latest_query_points, dtype=torch.float32).unsqueeze(0)
            query_coords = query_coords.to(self.device)
            
            # Run inference
            infer_start = time.time()
            with torch.no_grad():
                if self.is_diffusion:
                    num_inference_steps = getattr(self.train_cfg.model, 'num_inference_steps', 10)
                    pred_disp = self.model.predict(frames, query_coords, num_inference_steps=num_inference_steps)
                else:
                    pred_disp = self.model.predict(frames, query_coords)
            infer_time = (time.time() - infer_start) * 1000
            
            # Denormalize displacements
            denorm_start = time.time()
            mean = np.array(self.disp_stats['displacement_mean'])
            std = np.array(self.disp_stats['displacement_std'])
            pred_disp_denorm = denormalize_displacements(pred_disp[0], mean, std)
            
            # Convert displacements to absolute positions
            pred_tracks = query_coords[0].unsqueeze(1) + pred_disp_denorm  # (N, T, 2)
            denorm_time = (time.time() - denorm_start) * 1000
            
            # Publish predictions
            pub_start = time.time()
            self.publish_predictions(query_coords[0], pred_tracks)
            pub_time = (time.time() - pub_start) * 1000
            
            total_time = (time.time() - total_start) * 1000
            
            # Store timing for FPS display (no per-frame logging spam)
            self._last_infer_time = infer_time
            self._last_total_time = total_time
            
            # Track FPS
            self.process_count += 1
            elapsed = time.time() - self.fps_start_time
            if elapsed > 2.0:
                fps = self.process_count / elapsed
                # Combined FPS + timing stats on one line (like perception node)
                avg_infer = getattr(self, '_last_infer_time', 0)
                avg_total = getattr(self, '_last_total_time', 0)
                self.get_logger().info(f'FPS: {fps:.1f} | Avg: infer={avg_infer:.1f}ms total={avg_total:.1f}ms')
                self.process_count = 0
                self.fps_start_time = time.time()
        
        except Exception as e:
            self.get_logger().error(f'Error predicting trajectories: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())


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
