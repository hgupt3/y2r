#!/usr/bin/env python3
"""
Simple ROS2 node to test RealSense camera with 224x224 center crop.
Displays the cropped frame and measures FPS.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
import cv2
import time


class CameraViewerNode(Node):
    def __init__(self):
        super().__init__('camera_viewer_node')
        
        # FPS tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0
        
        # Subscribe to camera feed
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10  # QoS queue size
        )
        
        self.get_logger().info('Camera Viewer Node started!')
        self.get_logger().info('Press "q" in the window to quit')
    
    def center_crop_224(self, image):
        """Center crop image to 224x224."""
        h, w = image.shape[:2]
        
        # Calculate crop coordinates
        center_x, center_y = w // 2, h // 2
        crop_size = 224
        half_crop = crop_size // 2
        
        x1 = center_x - half_crop
        y1 = center_y - half_crop
        x2 = x1 + crop_size
        y2 = y1 + crop_size
        
        # Crop
        cropped = image[y1:y2, x1:x2]
        
        return cropped
    
    def image_callback(self, msg):
        """Process incoming camera frame."""
        try:
            # Manually convert ROS Image to numpy array (bypass cv_bridge to avoid library conflicts)
            # ROS Image format: RGB8 means 8-bit RGB
            height = msg.height
            width = msg.width
            
            # Convert byte data to numpy array
            img_array = np.frombuffer(msg.data, dtype=np.uint8)
            img_array = img_array.reshape((height, width, 3))
            
            # Convert RGB to BGR for OpenCV display
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Center crop to 224x224
            cropped = self.center_crop_224(img_bgr)
            
            # Update FPS
            self.frame_count += 1
            elapsed = time.time() - self.start_time
            if elapsed > 1.0:  # Update FPS every second
                self.fps = self.frame_count / elapsed
                self.frame_count = 0
                self.start_time = time.time()
            
            # Add FPS text to image
            display = cropped.copy()
            cv2.putText(
                display,
                f'FPS: {self.fps:.1f}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Display
            cv2.imshow('RealSense 224x224 Center Crop', display)
            
            # Handle key press
            key = cv2.waitKey(1)
            if key == ord('q'):
                self.get_logger().info('Quit requested')
                rclpy.shutdown()
        
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')


def main(args=None):
    rclpy.init(args=args)
    
    node = CameraViewerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

