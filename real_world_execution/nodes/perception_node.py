#!/usr/bin/env python3
"""
Perception Node - SAM2-based Object Tracking

Detects object on first frame using Florence-2 or Grounding-DINO,
then tracks it across subsequent frames using SAM2 video predictor.
Samples query points from the tracked mask and publishes them.
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
os.environ['TORCHINDUCTOR_MAX_AUTOTUNE'] = '0'  # Disable autotuning spam
os.environ['TORCH_LOGS'] = '-all'  # Suppress torch logs
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point, Polygon, Point32
from std_msgs.msg import Header
from std_srvs.srv import Trigger
from tracker_interfaces.msg import QueryPoints
import numpy as np
import cv2
import torch
from PIL import Image as PILImage
import time
import shutil
import tempfile
from scipy.spatial.distance import cdist  # For efficient FPS

# Add thirdparty to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "thirdparty"))
sys.path.insert(0, str(PROJECT_ROOT / "thirdparty" / "sam2-realtime"))

from sam2.build_sam import build_sam2_camera_predictor


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')
        
        # Declare parameters
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('detection_model', 'florence-2')
        self.declare_parameter('text_prompt', 'block')
        self.declare_parameter('num_query_points', 10)
        self.declare_parameter('mask_erosion_pixels', 5)
        self.declare_parameter('sampling_strategy', 'fps')  # 'random' or 'fps'
        self.declare_parameter('target_fps', 0.0)  # 0.0 = full throughput
        
        # Get parameters
        self.device = self.get_parameter('device').value
        self.detection_model = self.get_parameter('detection_model').value
        self.text_prompt = self.get_parameter('text_prompt').value
        self.num_query_points = self.get_parameter('num_query_points').value
        self.mask_erosion_pixels = self.get_parameter('mask_erosion_pixels').value
        self.sampling_strategy = self.get_parameter('sampling_strategy').value
        target_fps_param = self.get_parameter('target_fps').value
        # Handle None/null/0 as full throughput
        self.target_fps = target_fps_param if target_fps_param and target_fps_param > 0 else None
        
        # Hardcoded settings (from gsam_video defaults)
        self.box_threshold = 0.5
        self.text_threshold = 0.5
        self.sam2_config = 'configs/sam2.1/sam2.1_hiera_t_512'  # Tiny model at 512 res (optimized for speed)
        self.sam2_checkpoint = 'thirdparty/sam2/checkpoints/sam2.1_hiera_tiny.pt'
        self.sampling_method = 'random'  # Always use random sampling
        
        # Tracking state
        self.tracking_initialized = False
        self.current_frame_idx = 0
        self.frame_lock = False
        
        # For throttled mode only
        if self.target_fps is not None:
            self.latest_frame = None
            self.latest_frame_array = None
        
        # Models (loaded at startup)
        self.detection_processor = None
        self.detection_detection_model = None
        self.sam2_predictor = None
        
        # Object detection results
        self.detected_objects = {}  # {obj_id: label}
        self.initial_boxes = {}  # {obj_id: box} - store for re-initialization
        
        # FPS tracking
        self.process_count = 0
        self.fps_start_time = time.time()
        
        self.get_logger().info('Perception Node initializing...')
        self.get_logger().info(f'  Device: {self.device}')
        self.get_logger().info(f'  Detection model: {self.detection_model}')
        self.get_logger().info(f'  Text prompt: {self.text_prompt}')
        self.get_logger().info(f'  Query points: {self.num_query_points}')
        self.get_logger().info(f'  Mask erosion: {self.mask_erosion_pixels} pixels')
        self.get_logger().info(f'  Sampling strategy: {self.sampling_strategy}')
        if self.target_fps is not None:
            self.get_logger().info(f'  Target FPS: {self.target_fps:.1f} (throttled)')
        else:
            self.get_logger().info(f'  Target FPS: unlimited (full throughput)')
        
        # Load models
        self.load_models()
        
        # Subscribe to preprocessed images
        self.subscription = self.create_subscription(
            Image,
            '/preprocessed_image',
            self.image_callback,
            1  # Small queue - always process latest frame
        )
        
        # Publisher for query points
        self.publisher = self.create_publisher(
            QueryPoints,
            '/query_points',
            1
        )
        
        # Publisher for tracked mask contours (for visualization)
        self.contour_publisher = self.create_publisher(
            Polygon,
            '/tracked_contours',
            1  # Small queue - drop old contours
        )
        
        # Service for resetting tracking
        self.reset_service = self.create_service(
            Trigger,
            '/reset_tracking',
            self.reset_tracking_callback
        )
        
        # Timer for processing at target FPS (only if throttling enabled)
        if self.target_fps is not None:
            timer_period = 1.0 / self.target_fps
            self.timer = self.create_timer(timer_period, self.process_frame)
        else:
            self.timer = None
        
        self.get_logger().info('Perception Node ready!')
    
    def load_models(self):
        """Load detection model and SAM2 predictor"""
        self.get_logger().info('Loading models...')
        
        # Enable autocast for better performance
        if self.device.startswith("cuda"):
            torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        
        # Load detection model
        if self.detection_model == "grounding-dino":
            self.get_logger().info('Loading Grounding DINO...')
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            self.detection_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
            self.detection_detection_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                "IDEA-Research/grounding-dino-base"
            ).to(self.device)
            self.get_logger().info('✓ Grounding DINO loaded')
            
        elif self.detection_model == "florence-2":
            self.get_logger().info('Loading Florence-2...')
            from transformers import AutoProcessor, AutoModelForCausalLM
            self.detection_processor = AutoProcessor.from_pretrained(
                "microsoft/Florence-2-large",
                trust_remote_code=True
            )
            self.detection_detection_model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Florence-2-large",
                trust_remote_code=True,
                attn_implementation="eager"
            ).eval().to(self.device)
            self.get_logger().info('✓ Florence-2 loaded')
        else:
            raise ValueError(f"Unknown detection model: {self.detection_model}")
        
        # Load SAM2 camera predictor (real-time version)
        self.get_logger().info('Loading SAM2 camera predictor (tiny model @ 512 resolution)...')
        sam2_checkpoint_path = str(PROJECT_ROOT / self.sam2_checkpoint)
        
        # Simple setup: just use vos_optimized=True like the benchmark
        self.sam2_predictor = build_sam2_camera_predictor(
            self.sam2_config,
            sam2_checkpoint_path,
            device=self.device,
            vos_optimized=True  # Use optimized VOS version
        )
        self.get_logger().info('✓ SAM2 camera predictor loaded')
        self.get_logger().info('All models loaded successfully!')
    
    def image_callback(self, msg):
        """
        Callback for preprocessed images.
        If throttling: store latest frame for timer processing.
        If full throughput: process immediately.
        """
        if self.target_fps is None:
            # Full throughput mode - process immediately
            # Convert to numpy array
            height, width = msg.height, msg.width
            img_array = np.frombuffer(msg.data, dtype=np.uint8)
            frame_array = img_array.reshape((height, width, 3))
            self.process_frame_data(frame_array)
        else:
            # Throttled mode - store for timer processing
            if not self.frame_lock:
                self.latest_frame = msg
                # Convert to numpy array for processing
                height, width = msg.height, msg.width
                img_array = np.frombuffer(msg.data, dtype=np.uint8)
                self.latest_frame_array = img_array.reshape((height, width, 3))
    
    def process_frame(self):
        """Process frame at target FPS (called by timer when throttling)"""
        if self.latest_frame_array is None:
            return  # No frame received yet
        
        try:
            self.frame_lock = True
            frame_array = self.latest_frame_array.copy()
            self.frame_lock = False
            
            self.process_frame_data(frame_array)
                
        except Exception as e:
            self.get_logger().error(f'Error in timer processing: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            self.frame_lock = False
    
    def process_frame_data(self, frame_array):
        """Core frame processing logic."""
        try:
            if not self.tracking_initialized:
                # First frame: run detection and initialize tracking
                self.get_logger().info('Initializing tracking on first frame...')
                success = self.initialize_tracking(frame_array)
                if success:
                    self.get_logger().info('✓ Tracking initialized successfully')
                else:
                    self.get_logger().warn('Failed to initialize tracking, will retry next frame...')
                    return
            else:
                # Subsequent frames: propagate and sample
                self.get_logger().debug(f'Propagating frame {self.current_frame_idx + 1}...')
                self.propagate_and_sample(frame_array)
            
            # Track FPS
            self.process_count += 1
            elapsed = time.time() - self.fps_start_time
            if elapsed > 2.0:
                fps = self.process_count / elapsed
                # Combined FPS + timing stats on one line
                avg_track = getattr(self, '_last_track_time', 0)
                avg_total = getattr(self, '_last_total_time', 0)
                self.get_logger().info(f'FPS: {fps:.1f} | Avg: track={avg_track:.1f}ms total={avg_total:.1f}ms')
                self.process_count = 0
                self.fps_start_time = time.time()
                
        except Exception as e:
            self.get_logger().error(f'Error processing frame: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def initialize_tracking(self, frame_array):
        """
        Run detection on first frame and initialize SAM2 tracking (real-time version)
        Uses load_first_frame + add_new_prompt from camera predictor API
        """
        try:
            # No temp directory needed for real-time version!
            
            # Convert to PIL for detection
            frame_pil = PILImage.fromarray(frame_array)
            
            # Run detection
            self.get_logger().info(f'Running {self.detection_model} detection...')
            input_boxes, text_labels = self.run_detection(frame_pil)
            
            if input_boxes is None or len(input_boxes) == 0:
                self.get_logger().warn('No objects detected!')
                return False
            
            self.get_logger().info(f'Detected {len(input_boxes)} object(s): {text_labels}')
            
            # Build object dictionary
            self.detected_objects = {}
            self.initial_boxes = {}
            for i, label in enumerate(text_labels):
                obj_id = i + 1
                self.detected_objects[obj_id] = label
                self.initial_boxes[obj_id] = input_boxes[i]
            
            # Initialize SAM2 camera predictor with first frame
            self.get_logger().info('Loading first frame into SAM2...')
            self.sam2_predictor.load_first_frame(frame_array)
            
            # Add bounding boxes as prompts
            for obj_id, box in self.initial_boxes.items():
                _, out_obj_ids, out_mask_logits = self.sam2_predictor.add_new_prompt(
                    frame_idx=0,  # First frame
                    obj_id=obj_id,
                    bbox=box  # Note: parameter is 'bbox' not 'box'
                )
                self.get_logger().info(f'  Added box for object {obj_id} ({self.detected_objects[obj_id]})')
            
            self.tracking_initialized = True
            self.current_frame_idx = 0
            return True
            
        except Exception as e:
            self.get_logger().error(f'Error initializing tracking: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            return False
    
    def run_detection(self, frame_pil):
        """
        Run object detection using Florence-2 or Grounding-DINO
        Reference: grounded_sam2.py lines 109-208
        
        Returns:
            input_boxes: numpy array of shape (N, 4)
            text_labels: list of N strings
        """
        if self.detection_model == "grounding-dino":
            # Prepare input
            inputs = self.detection_processor(
                images=frame_pil,
                text=self.text_prompt,
                return_tensors="pt"
            ).to(self.device)
            
            # Run detection
            with torch.no_grad():
                outputs = self.detection_detection_model(**inputs)
            
            # Post-process results
            results = self.detection_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                target_sizes=[frame_pil.size[::-1]],  # (height, width)
                threshold=self.box_threshold
            )
            
            if len(results[0]["boxes"]) == 0:
                return None, None
            
            input_boxes = results[0]["boxes"].cpu().numpy()
            text_labels = results[0]["labels"]
            
        elif self.detection_model == "florence-2":
            # Prepare prompt
            task_prompt = "<OPEN_VOCABULARY_DETECTION>"
            prompt = task_prompt + self.text_prompt
            
            # Prepare input
            inputs = self.detection_processor(
                text=prompt,
                images=frame_pil,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate predictions
            with torch.no_grad():
                generated_ids = self.detection_detection_model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    early_stopping=False,
                    do_sample=False,
                    num_beams=3,
                    use_cache=False
                )
            
            # Decode and parse
            generated_text = self.detection_processor.batch_decode(
                generated_ids, skip_special_tokens=False
            )[0]
            parsed_answer = self.detection_processor.post_process_generation(
                generated_text,
                task=task_prompt,
                image_size=(frame_pil.width, frame_pil.height)
            )
            
            # Extract results
            results = parsed_answer[task_prompt]
            
            if len(results.get("bboxes", [])) == 0:
                return None, None
            
            input_boxes = np.array(results["bboxes"])
            text_labels = results["bboxes_labels"]
        
        else:
            raise ValueError(f"Unknown detection model: {self.detection_model}")
        
        return input_boxes, text_labels
    
    def propagate_and_sample(self, frame_array):
        """
        Propagate mask to current frame and sample query points (real-time version)
        Uses predictor.track() for incremental tracking
        """
        try:
            self.current_frame_idx += 1
            
            # Detailed timing measurements
            total_start = time.time()
            
            # Real-time tracking: just call track() with new frame!
            track_start = time.time()
            out_obj_ids, out_mask_logits = self.sam2_predictor.track(frame_array)
            track_time = (time.time() - track_start) * 1000
            
            mask_found = False
            mask_processing_time = 0
            publish_time = 0
            sample_time = 0
            
            # Process masks for current frame
            for j, out_obj_id in enumerate(out_obj_ids):
                if out_obj_id in self.detected_objects:
                    # Get mask
                    mask_start = time.time()
                    mask = (out_mask_logits[j] > 0.0).cpu().numpy().squeeze()
                    mask_processing_time += (time.time() - mask_start) * 1000
                    
                    self.get_logger().debug(f'Mask shape: {mask.shape}, sum: {mask.sum()}')
                    
                    # Publish contours for visualization (much smaller than full mask)
                    pub_start = time.time()
                    self.publish_contours(mask)
                    publish_time += (time.time() - pub_start) * 1000
                    mask_found = True
                    
                    # Sample query points
                    sample_start = time.time()
                    query_points = self.sample_query_points(mask, self.num_query_points)
                    sample_time += (time.time() - sample_start) * 1000
                    
                    if len(query_points) > 0:
                        # Store for logging
                        self._last_query_points = query_points
                        self._last_obj_id = out_obj_id
                        # Publish query points for predictor
                        self.publish_query_points(query_points)
            
            if not mask_found:
                self.get_logger().warn(f'No mask found for frame {self.current_frame_idx}')
            
            total_time = (time.time() - total_start) * 1000
            
            # Store timing for FPS display (no per-frame logging spam)
            self._last_track_time = track_time
            self._last_total_time = total_time
                    
        except Exception as e:
            self.get_logger().error(f'Error propagating mask: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def sample_query_points(self, mask, num_points):
        """
        Sample query points from eroded mask to avoid edge instability.
        
        Strategies:
        - 'random': Random uniform sampling from mask pixels
        - 'fps': Farthest Point Sampling (more spatially diverse)
        
        Args:
            mask: Binary mask (H, W)
            num_points: Number of points to sample
            
        Returns:
            List of (x, y) tuples normalized to [0, 1]
        """
        # Erode mask to avoid sampling near edges (which can be unstable for tracking)
        if self.mask_erosion_pixels > 0:
            kernel = np.ones((self.mask_erosion_pixels * 2 + 1, self.mask_erosion_pixels * 2 + 1), np.uint8)
            mask_uint8 = (mask.astype(np.uint8) * 255)
            eroded_mask_uint8 = cv2.erode(mask_uint8, kernel, iterations=1)
            eroded_mask = eroded_mask_uint8 > 0
        else:
            eroded_mask = mask
        
        # Get all True pixels in eroded mask
        y_coords, x_coords = np.where(eroded_mask)
        
        if len(y_coords) == 0:
            # If erosion removed all points, fall back to original mask
            self.get_logger().warn(f'Erosion removed all points, falling back to original mask')
            y_coords, x_coords = np.where(mask)
            
            if len(y_coords) == 0:
                return []
        
        if len(y_coords) < num_points:
            # Not enough points, return all
            points = [(x, y) for x, y in zip(x_coords, y_coords)]
        else:
            if self.sampling_strategy == 'fps':
                # Farthest Point Sampling using scipy for efficiency
                points = self._fps_sampling(x_coords, y_coords, num_points)
            else:
                # Random sampling (default)
                indices = np.random.choice(len(y_coords), num_points, replace=False)
                points = [(x_coords[i], y_coords[i]) for i in indices]
        
        # Normalize to [0, 1]
        h, w = mask.shape
        normalized = [(float(x) / w, float(y) / h) for x, y in points]
        
        return normalized
    
    def _fps_sampling(self, x_coords, y_coords, num_points):
        """
        Farthest Point Sampling (FPS) to get spatially diverse query points.
        
        Uses scipy's optimized cdist for fast distance computation.
        
        Args:
            x_coords: X coordinates of all candidate pixels (1D array)
            y_coords: Y coordinates of all candidate pixels (1D array)
            num_points: Number of points to sample
            
        Returns:
            List of (x, y) tuples
        """
        # Convert to Nx2 array
        points = np.column_stack([x_coords, y_coords]).astype(np.float32)
        
        # Start with random first point
        sampled_indices = [np.random.randint(0, len(points))]
        
        # Iteratively select farthest points
        for _ in range(num_points - 1):
            # Get already sampled points
            sampled_pts = points[sampled_indices]
            
            # Compute distances from all points to sampled points
            # Using scipy.spatial.distance.cdist is ~10x faster than naive loop
            distances = cdist(points, sampled_pts, metric='euclidean')
            
            # For each point, get min distance to any sampled point
            min_distances = distances.min(axis=1)
            
            # Select point with maximum min distance (farthest from all sampled)
            farthest_idx = min_distances.argmax()
            sampled_indices.append(farthest_idx)
        
        # Extract selected points
        selected_points = [(int(points[i, 0]), int(points[i, 1])) for i in sampled_indices]
        return selected_points
    
    def publish_contours(self, mask):
        """
        Publish mask contours as Polygon message for visualization
        Much more efficient than publishing full mask image (100s of points vs 50K pixels)
        
        Args:
            mask: Binary mask (H, W) as boolean numpy array
        """
        # Convert mask to uint8 for OpenCV
        mask_uint8 = (mask.astype(np.uint8) * 255)
        
        # Find contours
        contours, _ = cv2.findContours(
            mask_uint8, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) == 0:
            return
        
        # Use the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create Polygon message with contour points
        polygon_msg = Polygon()
        for point in largest_contour.squeeze():
            # Point32 uses x, y, z (we set z=0)
            p = Point32()
            p.x = float(point[0])
            p.y = float(point[1])
            p.z = 0.0
            polygon_msg.points.append(p)
        
        self.contour_publisher.publish(polygon_msg)
    
    def publish_query_points(self, query_points):
        """
        Publish query points for predictor node.
        
        Args:
            query_points: List of (x, y) tuples normalized to [0, 1]
        """
        msg = QueryPoints()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        
        for x, y in query_points:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.0
            msg.points.append(p)
        
        self.publisher.publish(msg)
    
    def reset_tracking_callback(self, request, response):
        """Reset tracking state (service callback)"""
        self.get_logger().info('Resetting tracking...')
        
        try:
            # Clear tracking state
            self.tracking_initialized = False
            self.current_frame_idx = 0
            self.detected_objects = {}
            self.initial_boxes = {}
            
            response.success = True
            response.message = 'Tracking reset successfully'
            self.get_logger().info('✓ Tracking reset complete')
            
        except Exception as e:
            self.get_logger().error(f'Error resetting tracking: {e}')
            response.success = False
            response.message = f'Error: {str(e)}'
        
        return response
    
    def cleanup(self):
        """Cleanup resources"""
        # No temp directory to clean up with real-time version
        pass


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
