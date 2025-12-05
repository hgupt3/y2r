#!/usr/bin/env python3
"""
Visualization Node - WebSocket server for real-time trajectory visualization.

Streams camera frames and trajectory data to browser clients via WebSocket.
Supports both 2D (OpenCV overlay) and 3D (Three.js) visualization modes.
"""

import os
import sys
import json
import time
import base64
import asyncio
import threading
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
import torch
import torch.nn.functional as F

try:
    import websockets
    from websockets.server import serve
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    print("Warning: websockets not installed. Run: pip install websockets")


class VisualizationNode(Node):
    def __init__(self):
        super().__init__('visualization_node')
        
        # Declare parameters
        self.declare_parameter('mode', '2d')
        self.declare_parameter('websocket_host', '0.0.0.0')
        self.declare_parameter('websocket_port', 8765)
        self.declare_parameter('window_name', 'Real-Time Tracker')
        self.declare_parameter('display_scale', 1)
        self.declare_parameter('trail_history_sec', 0.25)
        self.declare_parameter('trail_alpha_floor', 0.2)
        from rcl_interfaces.msg import ParameterDescriptor
        self.declare_parameter('trajectory_color_stops', [], ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('trajectory_line_thickness', 2)
        
        # Camera intrinsics (for 3D unprojection)
        self.declare_parameter('fx', 615.0)
        self.declare_parameter('fy', 615.0)
        self.declare_parameter('cx', 320.0)
        self.declare_parameter('cy', 240.0)
        
        # Depth range (for 3D mode)
        self.declare_parameter('depth_min', 0.1)
        self.declare_parameter('depth_max', 2.5)
        
        # Get parameters
        self.mode = self.get_parameter('mode').value
        self.ws_host = self.get_parameter('websocket_host').value
        self.ws_port = self.get_parameter('websocket_port').value
        self.window_name = self.get_parameter('window_name').value
        self.display_scale = self.get_parameter('display_scale').value
        self.trail_history_sec = float(max(0.0, self.get_parameter('trail_history_sec').value))
        self.trail_alpha_floor = float(min(1.0, max(0.0, self.get_parameter('trail_alpha_floor').value)))
        self.traj_line_thickness = self.get_parameter('trajectory_line_thickness').value
        
        # Camera intrinsics for 3D mode
        self.intrinsics = {
            'fx': self.get_parameter('fx').value,
            'fy': self.get_parameter('fy').value,
            'cx': self.get_parameter('cx').value,
            'cy': self.get_parameter('cy').value
        }
        
        color_stops_param = self.get_parameter('trajectory_color_stops').value
        self.traj_color_stops = self._load_color_stops(color_stops_param)
        self._ensure_color_stops()
        
        # State
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0
        self.latest_frame = None
        self.latest_depth = None
        self.latest_contours = None
        self.latest_trajectories = None
        self.trail_history = deque(maxlen=50)
        
        # Camera resolution (set on first frame)
        self.cam_width = None
        self.cam_height = None
        
        # Depth range for encoding (meters) - read from parameters
        self.depth_min = self.get_parameter('depth_min').value
        self.depth_max = self.get_parameter('depth_max').value
        
        # GPU blur kernel for depth fill
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.blur_kernel_size = 51
        self.blur_kernel = self._create_gaussian_kernel(self.blur_kernel_size).to(self.device)
        
        # WebSocket clients
        self.ws_clients = set()
        self.ws_loop = None
        self.ws_thread = None
        
        # Subscribe to raw camera image
        self.image_sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.image_callback, 1
        )
        
        # Subscribe to depth for 3D mode
        if self.mode == '3d':
            self.depth_sub = self.create_subscription(
                Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 1
            )
        
        # Subscribe to tracked contours from perception
        self.contour_sub = self.create_subscription(
            Polygon, '/tracked_contours', self.contour_callback, 1
        )
        
        # Subscribe to predicted tracks from predictor
        self.trajectory_sub = self.create_subscription(
            PredictedTracks, '/predicted_tracks', self.trajectory_callback, 1
        )
        
        # Service client for reset tracking
        self.reset_client = self.create_client(Trigger, '/reset_tracking')
        
        # Start WebSocket server
        if WEBSOCKETS_AVAILABLE:
            self._start_websocket_server()
        else:
            self.get_logger().error('WebSocket not available - install websockets package')
        
        self.get_logger().info(f'Visualization Node started!')
        self.get_logger().info(f'  Mode: {self.mode}')
        self.get_logger().info(f'  WebSocket: ws://{self.ws_host}:{self.ws_port}')
    
    def _start_websocket_server(self):
        """Start WebSocket server in a separate thread."""
        self.ws_server = None
        self.ws_shutdown_event = threading.Event()
        
        def run_ws_server():
            self.ws_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.ws_loop)
            
            async def handler(websocket):
                self.ws_clients.add(websocket)
                self.get_logger().info(f'Client connected: {websocket.remote_address}')
                try:
                    async for message in websocket:
                        # Handle incoming messages (e.g., reset command)
                        try:
                            data = json.loads(message)
                            if data.get('command') == 'reset':
                                self._call_reset_service()
                        except json.JSONDecodeError:
                            pass
                except websockets.exceptions.ConnectionClosed:
                    pass
                finally:
                    self.ws_clients.discard(websocket)
                    self.get_logger().info(f'Client disconnected')
            
            async def main():
                self.ws_server = await serve(handler, self.ws_host, self.ws_port)
                # Wait until shutdown is requested
                while not self.ws_shutdown_event.is_set():
                    await asyncio.sleep(0.1)
                self.ws_server.close()
                await self.ws_server.wait_closed()
            
            self.ws_loop.run_until_complete(main())
        
        self.ws_thread = threading.Thread(target=run_ws_server, daemon=True)
        self.ws_thread.start()
    
    def _create_gaussian_kernel(self, kernel_size, sigma=None):
        """Create a Gaussian kernel for GPU convolution."""
        if sigma is None:
            sigma = kernel_size / 6.0
        
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        
        gauss_2d = gauss_1d.unsqueeze(1) @ gauss_1d.unsqueeze(0)
        return gauss_2d.unsqueeze(0).unsqueeze(0)
    
    def _gpu_blur_fill(self, depth_np):
        """Fill invalid depth using GPU-based Gaussian blur."""
        depth_t = torch.from_numpy(depth_np).float().to(self.device)
        
        invalid = (depth_t <= 0) | (depth_t < self.depth_min)
        if not invalid.any():
            return depth_np
        
        depth_for_blur = depth_t.clone()
        depth_for_blur[invalid] = 0
        valid_mask = (~invalid).float()
        
        # Add batch/channel dims: (H, W) -> (1, 1, H, W)
        depth_4d = depth_for_blur.unsqueeze(0).unsqueeze(0)
        mask_4d = valid_mask.unsqueeze(0).unsqueeze(0)
        
        pad = self.blur_kernel_size // 2
        blurred = F.conv2d(depth_4d, self.blur_kernel, padding=pad)
        blurred_mask = F.conv2d(mask_4d, self.blur_kernel, padding=pad)
        
        blurred_mask = blurred_mask.clamp(min=1e-6)
        filled = (blurred / blurred_mask).squeeze()
        
        result = depth_t.clone()
        result[invalid] = filled[invalid]
        
        return result.cpu().numpy()
    
    def shutdown_websocket(self):
        """Shutdown WebSocket server cleanly."""
        if hasattr(self, 'ws_shutdown_event'):
            self.ws_shutdown_event.set()
        if hasattr(self, 'ws_thread') and self.ws_thread.is_alive():
            self.ws_thread.join(timeout=2.0)
    
    def _call_reset_service(self):
        """Call reset tracking service."""
        if not self.reset_client.wait_for_service(timeout_sec=0.5):
            return
        request = Trigger.Request()
        self.reset_client.call_async(request)
    
    def contour_callback(self, msg):
        """Store latest contours. Multiple contours are separated by NaN points."""
        try:
            if len(msg.points) == 0:
                return
            
            # Split points by NaN separators into multiple contours
            contours = []
            current_contour = []
            
            for p in msg.points:
                if np.isnan(p.x) or np.isnan(p.y):
                    # NaN = separator between contours
                    if current_contour:
                        contours.append(current_contour)
                        current_contour = []
                else:
                    current_contour.append([p.x, p.y])
            
            # Don't forget the last contour
            if current_contour:
                contours.append(current_contour)
            
            self.latest_contours = contours if contours else None
        except Exception as e:
            self.get_logger().error(f'Error processing contours: {e}')
    
    def depth_callback(self, msg):
        """Store latest depth frame for 3D mode."""
        try:
            height, width = msg.height, msg.width
            # RealSense depth is 16-bit unsigned (mm)
            depth_array = np.frombuffer(msg.data, dtype=np.uint16).reshape((height, width))
            self.latest_depth = depth_array
        except Exception as e:
            self.get_logger().error(f'Error processing depth: {e}')
    
    def trajectory_callback(self, msg):
        """Store latest predicted trajectories."""
        try:
            N = msg.num_points
            T = msg.num_timesteps
            if N == 0 or T == 0:
                return
            
            # Extract trajectories
            traj_x = np.array(msg.trajectory_x, dtype=np.float32).reshape(N, T)
            traj_y = np.array(msg.trajectory_y, dtype=np.float32).reshape(N, T)
            
            # Check for z coordinates
            has_z = len(msg.trajectory_z) == N * T
            if has_z:
                traj_z = np.array(msg.trajectory_z, dtype=np.float32).reshape(N, T)
            else:
                traj_z = None
            
            # Derive start points from first timestep of trajectories
            start_x = traj_x[:, 0].tolist()
            start_y = traj_y[:, 0].tolist()
            start_z = traj_z[:, 0].tolist() if traj_z is not None else [0.0] * N
            
            timestamp = time.time()
            entry = {
                'timestamp': timestamp,
                'start_points': [[start_x[i], start_y[i], start_z[i]] for i in range(N)],
                'traj_x': traj_x.tolist(),
                'traj_y': traj_y.tolist(),
                'traj_z': traj_z.tolist() if traj_z is not None else None,
            }
            
            self.latest_trajectories = entry
            if self.trail_history_sec > 0:
                self.trail_history.append(entry)
                self._prune_trail_history(timestamp)
        except Exception as e:
            self.get_logger().error(f'Error processing trajectories: {e}')
    
    def _prune_trail_history(self, current_time):
        """Remove old trail entries."""
        while self.trail_history and (current_time - self.trail_history[0]['timestamp']) > self.trail_history_sec:
            self.trail_history.popleft()
    
    def image_callback(self, msg):
        """Process incoming camera frame and broadcast to clients."""
        try:
            height, width = msg.height, msg.width
            if self.cam_width is None:
                self.cam_width = width
                self.cam_height = height
                self.get_logger().info(f'  Camera resolution: {width}x{height}')
            
            # Convert ROS Image to numpy
            img_array = np.frombuffer(msg.data, dtype=np.uint8).reshape((height, width, 3))
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Update FPS
            self.frame_count += 1
            elapsed = time.time() - self.start_time
            if elapsed > 1.0:
                self.fps = self.frame_count / elapsed
                self.frame_count = 0
                self.start_time = time.time()
            
            # Draw overlays for 2D mode
            if self.mode == '2d':
                self._draw_2d_overlays(img_bgr)
            
            # Encode frame as JPEG (80% quality for good balance of size/quality)
            _, jpeg_data = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_b64 = base64.b64encode(jpeg_data.tobytes()).decode('utf-8')
            
            # Build message for clients
            message = {
                'type': 'frame',
                'mode': self.mode,
                'fps': round(self.fps, 1),
                'width': width,
                'height': height,
                'frame': frame_b64,
            }
            
            # For 3D mode, include trajectory data, intrinsics, and point cloud
            if self.mode == '3d':
                message['intrinsics'] = self.intrinsics
                if self.latest_trajectories:
                    message['trajectories'] = self.latest_trajectories
                if self.latest_contours:
                    message['contours'] = self.latest_contours
                
                # Encode depth as PNG (full resolution, compressed)
                if self.latest_depth is not None:
                    # Convert depth from mm to meters
                    depth_m = self.latest_depth.astype(np.float32) / 1000.0
                    
                    # Fill invalid depth using GPU-based blur
                    depth_m = self._gpu_blur_fill(depth_m)
                    
                    # Clip to valid range
                    depth_m = np.clip(depth_m, self.depth_min, self.depth_max)
                    
                    # Scale to 16-bit range for PNG encoding (no more 0 = invalid)
                    depth_scaled = ((depth_m - self.depth_min) / (self.depth_max - self.depth_min) * 65535).astype(np.uint16)
                    
                    # Encode as 16-bit PNG (lossless compression)
                    _, png_data = cv2.imencode('.png', depth_scaled)
                    depth_b64 = base64.b64encode(png_data.tobytes()).decode('utf-8')
                    
                    message['depth_image'] = depth_b64
                    message['depth_min'] = self.depth_min
                    message['depth_max'] = self.depth_max
            
            # Broadcast to all connected clients (fire and forget)
            if self.ws_clients and self.ws_loop:
                msg_json = json.dumps(message)
                asyncio.run_coroutine_threadsafe(self._broadcast(msg_json), self.ws_loop)
            
        except Exception as e:
            self.get_logger().error(f'Error in image callback: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    async def _broadcast(self, message):
        """Send message to all connected clients."""
        if not self.ws_clients:
            return
        
        # Send to all clients concurrently, ignore failures
        await asyncio.gather(
            *[client.send(message) for client in list(self.ws_clients)],
            return_exceptions=True
        )
    
    def _draw_2d_overlays(self, img):
        """Draw contours and trajectories on image for 2D mode."""
        # Draw contours (multiple contours supported)
        if self.latest_contours:
            for contour_points in self.latest_contours:
                if len(contour_points) >= 3:
                    contour = np.array(contour_points, dtype=np.int32)
                    cv2.polylines(img, [contour], True, (255, 255, 255), 2)
        
        # Draw trajectory history with fading alpha
        if self.trail_history and self.trail_history_sec > 0:
            current_time = time.time()
            
            # Sort by age (oldest first, so newest draws on top)
            trails_to_draw = []
            for entry in self.trail_history:
                age = current_time - entry['timestamp']
                if age > self.trail_history_sec:
                    continue
                # Calculate alpha: 1.0 for newest, trail_alpha_floor for oldest
                alpha = 1.0 - (age / self.trail_history_sec) * (1.0 - self.trail_alpha_floor)
                alpha = max(self.trail_alpha_floor, min(1.0, alpha))
                trails_to_draw.append((age, entry, alpha))
            
            # Sort by age descending (oldest first)
            trails_to_draw.sort(key=lambda x: -x[0])
            
            # Draw each trail with proper alpha blending
            for age, entry, alpha in trails_to_draw:
                traj_x = np.array(entry['traj_x'])
                traj_y = np.array(entry['traj_y'])
                N, T = traj_x.shape
                
                # Create overlay for this trail
                overlay = img.copy()
                for i in range(N):
                    self._draw_gradient_trajectory(overlay, traj_x[i], traj_y[i])
                
                # Blend with alpha
                cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)
        
        # Fallback: draw current trajectories at full opacity if no trail history
        elif self.latest_trajectories:
            traj_x = np.array(self.latest_trajectories['traj_x'])
            traj_y = np.array(self.latest_trajectories['traj_y'])
            N, T = traj_x.shape
            
            for i in range(N):
                self._draw_gradient_trajectory(img, traj_x[i], traj_y[i])
    
    def _draw_gradient_trajectory(self, img, traj_x, traj_y):
        """Draw a trajectory with gradient colors."""
        T = len(traj_x)
        if T < 2:
            return
        
        for i in range(T - 1):
            ratio = i / (T - 1)
            color = self._color_from_stops(ratio)
            pt1 = (int(traj_x[i]), int(traj_y[i]))
            pt2 = (int(traj_x[i + 1]), int(traj_y[i + 1]))
            cv2.line(img, pt1, pt2, color, self.traj_line_thickness, cv2.LINE_AA)
    
    def _load_color_stops(self, stops_param):
        """Parse color stops from parameter."""
        stops = []
        if not isinstance(stops_param, (list, tuple)) or len(stops_param) == 0:
            return stops
        
        if isinstance(stops_param[0], (list, tuple)):
            for color in stops_param:
                if len(color) >= 3:
                    stops.append(tuple(int(c) for c in color[:3]))
        else:
            if len(stops_param) % 3 == 0:
                for i in range(0, len(stops_param), 3):
                    stops.append((int(stops_param[i]), int(stops_param[i+1]), int(stops_param[i+2])))
        return stops
    
    def _color_from_stops(self, ratio):
        """Interpolate color from stops."""
        if not self.traj_color_stops:
            return (255, 255, 255)
        if ratio <= 0:
            return self.traj_color_stops[0]
        if ratio >= 1:
            return self.traj_color_stops[-1]
        
        n = len(self.traj_color_stops) - 1
        scaled = ratio * n
        idx = int(scaled)
        frac = scaled - idx
        idx = min(idx, n - 1)
        
        c0, c1 = self.traj_color_stops[idx], self.traj_color_stops[idx + 1]
        return tuple(int(c0[i] * (1 - frac) + c1[i] * frac) for i in range(3))
    
    def _ensure_color_stops(self):
        """Ensure valid color stops."""
        if len(self.traj_color_stops) >= 2:
            return
        if len(self.traj_color_stops) == 1:
            self.traj_color_stops.append(self.traj_color_stops[0])
        else:
            self.traj_color_stops = [(255, 255, 255), (0, 0, 0)]


def main(args=None):
    rclpy.init(args=args)
    node = VisualizationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown_websocket()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
