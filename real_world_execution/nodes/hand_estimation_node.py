#!/usr/bin/env python3
"""
Hand Estimation Node - Real-Time Hand Pose Estimation using WiLoR

Detects hands using YOLO and estimates wrist pose using WiLoR.
Outputs wrist position (u, v, depth) and orientation for visualization.
"""

import os
import sys
from pathlib import Path

# Ensure custom tracker_interfaces messages are on sys.path
REAL_WORLD_ROOT = Path(__file__).resolve().parents[1]
CUSTOM_MSG_PATH = REAL_WORLD_ROOT / 'install' / 'tracker_interfaces' / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages'
if CUSTOM_MSG_PATH.exists():
    sys.path.insert(0, str(CUSTOM_MSG_PATH))

# Suppress torch.compile spam during warmup
os.environ['TORCHINDUCTOR_MAX_AUTOTUNE'] = '0'
os.environ['TORCH_LOGS'] = '-all'
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')

# Set PyOpenGL platform before importing pyrender
os.environ["PYOPENGL_PLATFORM"] = "egl"

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from tracker_interfaces.msg import HandPoses
import numpy as np
import cv2
import torch
import time

# Add thirdparty to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "thirdparty" / "WiLoR"))


def backproject_point(u, v, depth, fx, fy, cx, cy):
    """
    Backproject a 2D point to 3D using depth and intrinsics.
    
    Args:
        u, v: pixel coordinates
        depth: depth value in meters
        fx, fy, cx, cy: camera intrinsics
    
    Returns:
        (3,) 3D point in camera frame
    """
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    return np.array([x, y, z], dtype=np.float32)


class HandEstimationNode(Node):
    def __init__(self):
        super().__init__('hand_estimation_node')
        
        # Declare parameters
        self.declare_parameter('enabled', True)
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('detection_confidence', 0.75)
        self.declare_parameter('wrist_depth_offset', 0.02)
        self.declare_parameter('train_config_path', '')  # Path to model training config (for hand_mode)
        
        # Camera intrinsics
        self.declare_parameter('fx', 615.0)
        self.declare_parameter('fy', 615.0)
        self.declare_parameter('cx', 320.0)
        self.declare_parameter('cy', 240.0)
        
        # Get parameters
        self.enabled = self.get_parameter('enabled').value
        self.device = self.get_parameter('device').value
        self.detection_confidence = self.get_parameter('detection_confidence').value
        self.wrist_depth_offset = self.get_parameter('wrist_depth_offset').value
        
        # Load track_hands from model config (hand_mode)
        train_config_path = self.get_parameter('train_config_path').value
        if not train_config_path:
            raise ValueError('train_config_path must be set in config!')
        
        import yaml
        with open(train_config_path, 'r') as f:
            train_cfg = yaml.safe_load(f)
        self.track_hands = train_cfg['model']['hand_mode']
        if not self.track_hands:
            raise ValueError('model.hand_mode must be set in train config (left/right/both)!')
        
        # Camera intrinsics
        self.fx = self.get_parameter('fx').value
        self.fy = self.get_parameter('fy').value
        self.cx = self.get_parameter('cx').value
        self.cy = self.get_parameter('cy').value
        
        if not self.enabled:
            self.get_logger().info('Hand Estimation Node disabled in config')
            return
        
        self.get_logger().info('Hand Estimation Node initializing...')
        self.get_logger().info(f'  Device: {self.device}')
        self.get_logger().info(f'  Detection confidence: {self.detection_confidence}')
        self.get_logger().info(f'  Wrist depth offset: {self.wrist_depth_offset}m')
        self.get_logger().info(f'  Track hands: {self.track_hands}')
        
        # Camera resolution (set on first frame)
        self.cam_width = None
        self.cam_height = None
        
        # Latest depth frame
        self.latest_depth = None
        
        # Models (loaded at startup)
        self.wilor_model = None
        self.wilor_cfg = None
        self.detector = None
        
        # FPS tracking
        self.process_count = 0
        self.fps_start_time = time.time()
        
        # Load models
        self.load_models()
        
        # Subscribe to raw camera images
        self.color_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.color_callback,
            1
        )
        
        # Subscribe to aligned depth
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw',
            self.depth_callback,
            1
        )
        
        # Publisher for hand poses
        self.publisher = self.create_publisher(
            HandPoses,
            '/hand_poses',
            1
        )
        
        self.get_logger().info('Hand Estimation Node ready!')
    
    def load_models(self):
        """Load WiLoR model and YOLO hand detector."""
        self.get_logger().info('Loading models...')
        
        # Enable autocast for better performance
        if self.device.startswith("cuda"):
            torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        
        # Load WiLoR
        self.get_logger().info('Loading WiLoR model...')
        from wilor.models import load_wilor
        self.wilor_model, self.wilor_cfg = load_wilor(
            checkpoint_path=str(PROJECT_ROOT / "thirdparty/WiLoR/pretrained_models/wilor_final.ckpt"),
            cfg_path=str(PROJECT_ROOT / "thirdparty/WiLoR/pretrained_models/model_config.yaml")
        )
        self.wilor_model = self.wilor_model.to(self.device)
        self.wilor_model.eval()
        self.get_logger().info('✓ WiLoR loaded')
        
        # Load YOLO hand detector
        self.get_logger().info('Loading YOLO hand detector...')
        from ultralytics import YOLO
        self.detector = YOLO(str(PROJECT_ROOT / "thirdparty/WiLoR/pretrained_models/detector.pt"))
        self.detector = self.detector.to(self.device)
        self.get_logger().info('✓ YOLO detector loaded')
        
        self.get_logger().info('All models loaded successfully!')
    
    def depth_callback(self, msg):
        """Store latest depth frame."""
        height, width = msg.height, msg.width
        depth_array = np.frombuffer(msg.data, dtype=np.uint16)
        self.latest_depth = depth_array.reshape((height, width))
    
    def color_callback(self, msg):
        """Process incoming color frame for hand estimation."""
        if not self.enabled:
            return
        
        try:
            height, width = msg.height, msg.width
            
            # Set camera resolution on first frame
            if self.cam_width is None:
                self.cam_width = width
                self.cam_height = height
                self.get_logger().info(f'  Camera resolution: {width}x{height}')
            
            # Convert ROS Image to numpy (RGB)
            img_array = np.frombuffer(msg.data, dtype=np.uint8).reshape((height, width, 3))
            
            # Convert to BGR for YOLO
            frame_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Process frame
            self.process_frame(frame_bgr)
            
            # Track FPS
            self.process_count += 1
            elapsed = time.time() - self.fps_start_time
            if elapsed > 2.0:
                fps = self.process_count / elapsed
                self.get_logger().info(f'FPS: {fps:.1f}')
                self.process_count = 0
                self.fps_start_time = time.time()
                
        except Exception as e:
            self.get_logger().error(f'Error processing frame: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def get_best_detection(self, detections, is_right_target):
        """
        Get the best detection for a given hand side (left or right).
        
        Args:
            detections: YOLO detection results
            is_right_target: True for right hand, False for left hand
        
        Returns:
            Best detection dict or None if no matching detection
        """
        best_det = None
        best_conf = -1
        
        for det in detections:
            bbox = det.boxes.data.cpu().detach().squeeze().numpy()
            conf = det.boxes.conf.cpu().detach().squeeze().item()
            is_right = det.boxes.cls.cpu().detach().squeeze().item()
            
            # Check if this detection matches the target side
            if bool(is_right) == is_right_target:
                if conf > best_conf:
                    best_conf = conf
                    best_det = {
                        'bbox': bbox[:4],
                        'confidence': conf,
                        'is_right': is_right
                    }
        
        return best_det
    
    def process_frame(self, frame_bgr):
        """
        Process a single frame for hand pose estimation.
        Batches all detected hands together for efficient inference (following process_wilor.py pattern).
        
        Args:
            frame_bgr: BGR image (H, W, 3)
        """
        from wilor.utils import recursive_to
        from wilor.datasets.vitdet_dataset import ViTDetDataset
        from wilor.utils.renderer import cam_crop_to_full
        
        H, W = frame_bgr.shape[:2]
        
        # Initialize output message
        msg = HandPoses()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.left_valid = False
        msg.right_valid = False
        msg.left_u = float('nan')
        msg.left_v = float('nan')
        msg.left_depth = float('nan')
        msg.left_rotation = [float('nan')] * 9
        msg.right_u = float('nan')
        msg.right_v = float('nan')
        msg.right_depth = float('nan')
        msg.right_rotation = [float('nan')] * 9
        
        # Run YOLO detection
        detections = self.detector(frame_bgr, conf=self.detection_confidence, verbose=False)
        
        if len(detections) == 0 or len(detections[0]) == 0:
            self.publisher.publish(msg)
            return
        
        dets = detections[0]
        
        # Determine which hands to track
        track_left = self.track_hands in ['left', 'both']
        track_right = self.track_hands in ['right', 'both']
        
        # Get best detections for each hand
        left_det = self.get_best_detection(dets, is_right_target=False) if track_left else None
        right_det = self.get_best_detection(dets, is_right_target=True) if track_right else None
        
        # Collect detections to process
        hands_to_process = []
        if left_det is not None:
            hands_to_process.append(('left', left_det))
        if right_det is not None:
            hands_to_process.append(('right', right_det))
        
        if len(hands_to_process) == 0:
            self.publisher.publish(msg)
            return
        
        # ========== PHASE 1: Prepare all crops (following process_wilor.py pattern) ==========
        rescale_factor = 2.0  # Default WiLoR value
        all_batch_items = []
        all_metadata = []  # (side, is_right)
        
        for side, det in hands_to_process:
            bbox = det['bbox']
            is_right = det['is_right']
            
            boxes = np.array([bbox])
            right_arr = np.array([is_right])
            
            dataset = ViTDetDataset(
                self.wilor_cfg,
                frame_bgr,
                boxes,
                right_arr,
                rescale_factor=rescale_factor
            )
            
            # Get the single item from this dataset
            item = dataset[0]
            all_batch_items.append(item)
            all_metadata.append((side, is_right))
        
        # ========== PHASE 2: Batch WiLoR inference (following process_wilor.py pattern) ==========
        # Collate batch items by stacking
        batch = {}
        for key in all_batch_items[0].keys():
            if isinstance(all_batch_items[0][key], np.ndarray):
                batch[key] = torch.from_numpy(np.stack([item[key] for item in all_batch_items]))
            else:
                batch[key] = torch.tensor([item[key] for item in all_batch_items])
        
        batch = recursive_to(batch, self.device)
        
        # Run WiLoR inference (single forward pass for all hands)
        with torch.no_grad():
            out = self.wilor_model(batch)
        
        # ========== PHASE 3: Process outputs (following process_wilor.py pattern) ==========
        num_dets = len(all_batch_items)
        
        # Extract all outputs at once
        pred_cam_all = out['pred_cam'].cpu().numpy()
        global_orient_all = out['pred_mano_params']['global_orient'].cpu().numpy()
        pred_keypoints_3d_all = out['pred_keypoints_3d'].cpu().numpy()
        box_center_all = batch['box_center'].cpu().numpy()
        box_size_all = batch['box_size'].cpu().numpy()
        img_size_all = batch['img_size'].cpu().numpy()
        is_right_all = batch['right'].cpu().numpy()
        
        # Adjust pred_cam for handedness
        multiplier_all = (2 * is_right_all - 1)
        pred_cam_adjusted_all = pred_cam_all.copy()
        pred_cam_adjusted_all[:, 1] = multiplier_all * pred_cam_adjusted_all[:, 1]
        
        # Compute scaled focal lengths
        scaled_focal_lengths = self.wilor_cfg.EXTRA.FOCAL_LENGTH / self.wilor_cfg.MODEL.IMAGE_SIZE * img_size_all.max(axis=1)
        
        # Batch call to cam_crop_to_full
        pred_cam_t_full_all = cam_crop_to_full(
            torch.from_numpy(pred_cam_adjusted_all).float(),
            torch.from_numpy(box_center_all).float(),
            torch.from_numpy(box_size_all).float(),
            torch.from_numpy(img_size_all).float(),
            torch.from_numpy(scaled_focal_lengths).float(),
        ).numpy()
        
        # Process each detection
        for det_idx, (side, is_right) in enumerate(all_metadata):
            global_orient = global_orient_all[det_idx]
            pred_keypoints_3d = pred_keypoints_3d_all[det_idx]
            pred_cam_t_full = pred_cam_t_full_all[det_idx]
            scaled_focal_length = scaled_focal_lengths[det_idx]
            is_right_val = is_right_all[det_idx]
            
            # Get wrist position in WiLoR space
            joints_3d_wilor = pred_keypoints_3d.copy()
            joints_3d_wilor[:, 0] = (2 * is_right_val - 1) * joints_3d_wilor[:, 0]
            
            # Project wrist to 2D
            joints_3d_cam_wilor = joints_3d_wilor + pred_cam_t_full
            
            # Project to 2D using scaled focal length
            wrist_3d = joints_3d_cam_wilor[0]  # Wrist is joint 0
            wrist_u = scaled_focal_length * wrist_3d[0] / wrist_3d[2] + W / 2
            wrist_v = scaled_focal_length * wrist_3d[1] / wrist_3d[2] + H / 2
            
            # Clamp to image bounds
            wrist_u = float(np.clip(wrist_u, 0, W - 1))
            wrist_v = float(np.clip(wrist_v, 0, H - 1))
            
            # Sample depth at wrist location
            wrist_depth = -1.0  # Invalid marker (NaN breaks JSON serialization)
            if self.latest_depth is not None:
                u_int = int(np.clip(wrist_u, 0, W - 1))
                v_int = int(np.clip(wrist_v, 0, H - 1))
                
                # Get depth in meters (RealSense gives mm)
                depth_mm = self.latest_depth[v_int, u_int]
                if depth_mm > 0:
                    wrist_depth = float(depth_mm) / 1000.0 + self.wrist_depth_offset
                else:
                    # Try to sample from nearby pixels
                    for offset in range(1, 10):
                        for dv, du in [(-offset, 0), (offset, 0), (0, -offset), (0, offset)]:
                            nv, nu = v_int + dv, u_int + du
                            if 0 <= nv < H and 0 <= nu < W:
                                d = self.latest_depth[nv, nu]
                                if d > 0:
                                    wrist_depth = float(d) / 1000.0 + self.wrist_depth_offset
                                    break
                        if wrist_depth > 0:
                            break
            
            # Get wrist rotation (global_orient is already a rotation matrix)
            wrist_rot = global_orient.squeeze()  # (3, 3)
            
            # Store in message
            if side == 'left':
                msg.left_valid = True
                msg.left_u = wrist_u
                msg.left_v = wrist_v
                msg.left_depth = wrist_depth
                msg.left_rotation = wrist_rot.flatten().tolist()
            else:
                msg.right_valid = True
                msg.right_u = wrist_u
                msg.right_v = wrist_v
                msg.right_depth = wrist_depth
                msg.right_rotation = wrist_rot.flatten().tolist()
        
        # Publish
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = HandEstimationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

