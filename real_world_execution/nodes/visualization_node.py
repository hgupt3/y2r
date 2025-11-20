#!/usr/bin/env python3
"""
Visualization Node - Displays preprocessed images with FPS counter.

Subscribes to preprocessed images and displays them in real-time.
"""

import os
import sys
import time
import signal
from collections import deque
from pathlib import Path

# Ensure custom tracker_interfaces messages are importable
REAL_WORLD_ROOT = Path(__file__).resolve().parents[1]
CUSTOM_MSG_PATH = REAL_WORLD_ROOT / 'install' / 'tracker_interfaces' / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages'
if CUSTOM_MSG_PATH.exists():
    sys.path.insert(0, str(CUSTOM_MSG_PATH))

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Polygon
from std_srvs.srv import Trigger
from tracker_interfaces.msg import PredictedTracks
import numpy as np
import cv2


class VisualizationNode(Node):
    def __init__(self):
        super().__init__('visualization_node')
        
        # Declare parameters
        self.declare_parameter('window_name', 'Real-Time Tracker')
        self.declare_parameter('display_scale', 2)  # Upscale for visibility
        self.declare_parameter('trail_history_sec', 0.25)
        self.declare_parameter('trail_alpha_floor', 0.2)
        self.declare_parameter('trajectory_near_color', [255, 255, 255])
        self.declare_parameter('trajectory_far_color', [255, 100, 100])
        self.declare_parameter('trajectory_line_thickness', 2)
        
        self.window_name = self.get_parameter('window_name').value
        self.display_scale = self.get_parameter('display_scale').value
        self.trail_history_sec = float(max(0.0, self.get_parameter('trail_history_sec').value))
        self.trail_alpha_floor = float(min(1.0, max(0.0, self.get_parameter('trail_alpha_floor').value)))
        
        # Trajectory visualization parameters
        near_color = self.get_parameter('trajectory_near_color').value
        far_color = self.get_parameter('trajectory_far_color').value
        self.traj_line_thickness = self.get_parameter('trajectory_line_thickness').value
        
        # Convert from list to tuple (BGR for OpenCV)
        self.traj_near_color = tuple(near_color) if isinstance(near_color, list) else near_color
        self.traj_far_color = tuple(far_color) if isinstance(far_color, list) else far_color
        
        # FPS tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0
        
        # Latest contours from perception node
        self.latest_contours = None
        
        # Latest predicted trajectories from predictor node
        self.latest_trajectories = None  # Dict entry containing latest data
        self.trail_history = deque()
        
        # Subscribe to preprocessed image
        self.subscription = self.create_subscription(
            Image,
            '/preprocessed_image',
            self.image_callback,
            1  # Small queue - always show latest frame
        )
        
        # Subscribe to tracked contours from perception
        self.contour_subscription = self.create_subscription(
            Polygon,
            '/tracked_contours',
            self.contour_callback,
            1  # Small queue - always show latest contours
        )
        
        # Subscribe to predicted tracks from predictor
        self.trajectory_subscription = self.create_subscription(
            PredictedTracks,
            '/predicted_tracks',
            self.trajectory_callback,
            1
        )
        
        # Service client for reset tracking
        self.reset_client = self.create_client(Trigger, '/reset_tracking')
        
        self.get_logger().info(f'Visualization Node started! Window: {self.window_name}')
        self.get_logger().info("Press 'q' to quit, 'r' to reset tracking")
    
    def contour_callback(self, msg):
        """Store latest contours from perception node"""
        try:
            # Convert Polygon to numpy array for OpenCV
            if len(msg.points) > 0:
                contours = np.array([[int(p.x), int(p.y)] for p in msg.points], dtype=np.int32)
                self.latest_contours = [contours]  # OpenCV expects list of contours
        except Exception as e:
            self.get_logger().error(f'Error processing contours: {e}')
    
    def trajectory_callback(self, msg):
        """Store latest predicted trajectories from PredictedTracks message"""
        try:
            # Parse PredictedTracks message
            N = msg.num_points
            T = msg.num_timesteps
            
            if N == 0 or T == 0:
                return
            
            # Extract query points (N, 2)
            query_points = np.array([[p.x, p.y] for p in msg.query_points], dtype=np.float32)
            
            # Extract trajectories (N, T, 2)
            traj_x = np.array(msg.trajectory_x, dtype=np.float32).reshape(N, T)
            traj_y = np.array(msg.trajectory_y, dtype=np.float32).reshape(N, T)
            trajectories = np.stack([traj_x, traj_y], axis=-1)
            
            timestamp = self.get_clock().now().nanoseconds * 1e-9
            entry = {
                'timestamp': timestamp,
                'query_points': query_points,
                'pred_tracks': trajectories,
            }
            
            self.latest_trajectories = entry
            if self.trail_history_sec > 0:
                self.trail_history.append(entry)
                self._prune_trail_history(timestamp)
            else:
                self.trail_history.clear()
        except Exception as e:
            self.get_logger().error(f'Error processing trajectories: {e}')
    
    def _prune_trail_history(self, current_time):
        """Remove trail entries older than configured duration."""
        if self.trail_history_sec <= 0:
            self.trail_history.clear()
            return
        
        while self.trail_history and (current_time - self.trail_history[0]['timestamp']) > self.trail_history_sec:
            self.trail_history.popleft()
    
    def _draw_trajectory_entry(self, img, entry, display_width, display_height, alpha=1.0):
        """Draw all trajectories from a stored entry with optional alpha scaling."""
        if entry is None or 'pred_tracks' not in entry:
            return
        
        pred_tracks = entry['pred_tracks']
        if pred_tracks is None:
            return
        
        num_tracks = pred_tracks.shape[0]
        for i in range(num_tracks):
            traj_x = pred_tracks[i, :, 0] * display_width
            traj_y = pred_tracks[i, :, 1] * display_height
            self.draw_gradient_trajectory(img, traj_x, traj_y, thickness=self.traj_line_thickness, alpha=alpha)
    
    def draw_gradient_trajectory(self, img, traj_x, traj_y, thickness=2, alpha=1.0):
        """
        Draw a single trajectory with gradient color.
        
        Args:
            img: image to draw on
            traj_x: x coordinates in pixel space
            traj_y: y coordinates in pixel space
            thickness: line thickness
            alpha: transparency scale applied to the colors
        """
        T = len(traj_x)
        if T < 2:
            return
        
        alpha_scale = float(max(0.0, min(1.0, alpha)))
        
        # Draw line segments with interpolated colors
        for t in range(T - 1):
            # Interpolate color from near to far
            segment_ratio = t / (T - 1)  # 0 to 1
            base_color = [
                self.traj_near_color[i] * (1 - segment_ratio) + self.traj_far_color[i] * segment_ratio
                for i in range(3)
            ]
            color = tuple(int(c * alpha_scale) for c in base_color)
            
            pt1 = (int(traj_x[t]), int(traj_y[t]))
            pt2 = (int(traj_x[t+1]), int(traj_y[t+1]))
            
            cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)
    
    def image_callback(self, msg):
        """Display preprocessed image with FPS overlay."""
        try:
            # Convert ROS Image to numpy
            height, width = msg.height, msg.width
            img_array = np.frombuffer(msg.data, dtype=np.uint8)
            img_array = img_array.reshape((height, width, 3))
            
            # Convert RGB to BGR for OpenCV display
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Update FPS
            self.frame_count += 1
            elapsed = time.time() - self.start_time
            if elapsed > 1.0:
                self.fps = self.frame_count / elapsed
                self.frame_count = 0
                self.start_time = time.time()
            
            # Upscale for better visibility
            display_width = int(width * self.display_scale)
            display_height = int(height * self.display_scale)
            display_size = (display_width, display_height)
            display = cv2.resize(img_bgr, display_size)
            
            # Overlay contours if available
            if self.latest_contours is not None:
                # Scale contours to match display size
                scaled_contours = [
                    (contour * self.display_scale).astype(np.int32) for contour in self.latest_contours
                ]
                
                # Draw white thin contours
                cv2.drawContours(display, scaled_contours, -1, (255, 255, 255), 2)
            
            # Overlay predicted trajectories with trailing history
            latest_entry = None
            if self.trail_history_sec > 0:
                current_time = self.get_clock().now().nanoseconds * 1e-9
                self._prune_trail_history(current_time)
                if self.trail_history:
                    latest_entry = self.trail_history[-1]
                    for entry in list(self.trail_history)[:-1]:
                        age = current_time - entry['timestamp']
                        if age < 0 or age > self.trail_history_sec:
                            continue
                        alpha = 1.0 - (age / self.trail_history_sec)
                        alpha = max(self.trail_alpha_floor, alpha)
                        self._draw_trajectory_entry(display, entry, display_width, display_height, alpha)
            else:
                # History disabled; draw latest only
                latest_entry = self.latest_trajectories
            
            if latest_entry is None and self.latest_trajectories is not None:
                latest_entry = self.latest_trajectories
            
            if latest_entry is not None:
                self._draw_trajectory_entry(display, latest_entry, display_width, display_height, alpha=1.0)
            
            # Add FPS text
            cv2.putText(
                display,
                f'FPS: {self.fps:.1f}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Add resolution info
            cv2.putText(
                display,
                f'Resolution: {width}x{height}',
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            
            # Display
            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(1)
            
            if key == ord('q'):
                self.get_logger().info('Quit requested - shutting down all nodes')
                # Kill entire process group (launch + all nodes)
                try:
                    pgid = os.getpgid(os.getpid())
                    os.killpg(pgid, signal.SIGINT)
                except Exception as e:
                    self.get_logger().error(f'Failed to kill process group: {e}')
                    # Fallback: just exit this node
                    raise KeyboardInterrupt()
            elif key == ord('r'):
                self.call_reset_service()
                
        except Exception as e:
            self.get_logger().error(f'Error in visualization: {e}')
    
    def call_reset_service(self):
        """Call the reset tracking service"""
        self.get_logger().info('Reset tracking requested...')
        
        if not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Reset service not available!')
            return
        
        request = Trigger.Request()
        future = self.reset_client.call_async(request)
        future.add_done_callback(self.reset_response_callback)
    
    def reset_response_callback(self, future):
        """Handle reset service response"""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f'✓ {response.message}')
            else:
                self.get_logger().error(f'✗ Reset failed: {response.message}')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = VisualizationNode()
    
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

