#!/usr/bin/env python3
"""
Preprocessing Node - Handles camera image preprocessing.

Subscribes to raw camera feed, crops to square (max retained resolution),
resizes to 224x224, and publishes preprocessed image.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
import cv2
import time


class PreprocessorNode(Node):
    def __init__(self):
        super().__init__('preprocessor_node')
        
        # Declare parameters
        self.declare_parameter('crop_size', 224)
        self.declare_parameter('target_fps', 0.0)  # 0.0 = full throughput
        
        self.crop_size = self.get_parameter('crop_size').value
        target_fps_param = self.get_parameter('target_fps').value
        # Handle None/null/0 as full throughput
        self.target_fps = target_fps_param if target_fps_param and target_fps_param > 0 else None
        
        # FPS tracking for logging
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.actual_fps = 0.0
        
        # Subscribe to raw camera
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            1  # Small queue - always process latest frame
        )
        
        # Publish preprocessed image
        self.publisher = self.create_publisher(
            Image,
            '/preprocessed_image',
            1  # Small queue - drop old frames
        )
        
        self.get_logger().info(f'Preprocessor Node started!')
        self.get_logger().info(f'  Crop size: {self.crop_size}x{self.crop_size}')
        if self.target_fps is not None:
            self.get_logger().info(f'  Target FPS: {self.target_fps:.1f} (throttled)')
            # Timer to process frames at target FPS
            timer_period = 1.0 / self.target_fps
            self.timer = self.create_timer(timer_period, self.process_frame)
            # Store latest frame (updated by callback, processed by timer)
            self.latest_frame = None
            self.latest_frame_lock = False
        else:
            self.get_logger().info(f'  Target FPS: unlimited (full throughput)')
            # Process directly in callback (no throttling)
            self.timer = None
    
    def crop_to_square_max_res(self, image):
        """
        Crop to square while maximally retaining resolution.
        If width > height, crop width. If height > width, crop height.
        """
        h, w = image.shape[:2]
        min_dim = min(h, w)
        
        # Calculate crop coordinates to center
        start_y = (h - min_dim) // 2
        start_x = (w - min_dim) // 2
        
        # Crop to square
        cropped = image[start_y:start_y+min_dim, start_x:start_x+min_dim]
        
        return cropped
    
    def image_callback(self, msg):
        """
        Callback for raw camera images.
        If throttling: store latest frame for timer processing.
        If full throughput: process immediately.
        """
        if self.target_fps is None:
            # Full throughput mode - process immediately
            self.process_image(msg)
        else:
            # Throttled mode - store for timer processing
            if not self.latest_frame_lock:
                self.latest_frame = msg
    
    def process_frame(self):
        """Process the latest frame (called by timer at target FPS when throttling)."""
        if self.latest_frame is None:
            return  # No frame received yet
        
        try:
            # Lock to prevent callback from updating while we're reading
            self.latest_frame_lock = True
            msg = self.latest_frame
            self.latest_frame_lock = False
            
            self.process_image(msg)
            
        except Exception as e:
            self.get_logger().error(f'Error in timer processing: {e}')
            self.latest_frame_lock = False
    
    def process_image(self, msg):
        """Core image processing logic."""
        try:
            height, width = msg.height, msg.width
            img_array = np.frombuffer(msg.data, dtype=np.uint8)
            img_array = img_array.reshape((height, width, 3))
            
            # Crop to square (max resolution retained)
            img_square = self.crop_to_square_max_res(img_array)
            
            # Resize to target size (224x224)
            img_resized = cv2.resize(img_square, (self.crop_size, self.crop_size))
            
            # Create output ROS message
            out_msg = Image()
            out_msg.header = msg.header
            out_msg.height = self.crop_size
            out_msg.width = self.crop_size
            out_msg.encoding = msg.encoding  # Keep same encoding (RGB8)
            out_msg.is_bigendian = msg.is_bigendian
            out_msg.step = self.crop_size * 3  # 3 bytes per pixel (RGB)
            out_msg.data = img_resized.tobytes()
            
            # Publish
            self.publisher.publish(out_msg)
            
            # Track actual FPS
            self.frame_count += 1
            fps_elapsed = time.time() - self.fps_start_time
            if fps_elapsed > 2.0:  # Log every 2 seconds
                self.actual_fps = self.frame_count / fps_elapsed
                self.get_logger().info(f'FPS: {self.actual_fps:.1f}')
                self.frame_count = 0
                self.fps_start_time = time.time()
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = PreprocessorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

