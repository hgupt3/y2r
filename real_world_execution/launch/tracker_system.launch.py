#!/usr/bin/env python3
"""
Launch file for Real-Time Tracker System.

Starts all nodes: Camera → Preprocessor → Perception → Predictor → Visualization

Usage:
    ros2 launch real_world_execution tracker_system.launch.py
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get config file path
    config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')
    config_file = os.path.join(config_dir, 'tracker_params.yaml')
    
    # Load config first
    import yaml
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup environment for custom messages
    install_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'install'))
    
    # Build camera profile strings from config
    camera_profile = f"{config['camera']['width']}x{config['camera']['height']}x{config['camera']['fps']}"
    depth_profile = f"{config['camera']['depth_width']}x{config['camera']['depth_height']}x{config['camera']['depth_fps']}"
    
    # Launch RealSense camera
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('realsense2_camera'),
                'launch',
                'rs_launch.py'
            ])
        ]),
        launch_arguments={
            'rgb_camera.color_profile': camera_profile,
            'depth_module.depth_profile': depth_profile,
        }.items()
    )
    
    # Get absolute paths to node scripts and config
    nodes_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'nodes'))
    preprocessor_script = os.path.join(nodes_dir, 'preprocessor_node.py')
    visualization_script = os.path.join(nodes_dir, 'visualization_node.py')
    perception_script = os.path.join(nodes_dir, 'perception_node.py')
    predictor_script = os.path.join(nodes_dir, 'predictor_node.py')
    
    # Preprocessor Node
    from launch.actions import ExecuteProcess
    preprocessor_params = f'--ros-args -p crop_size:={config["preprocessor"]["crop_size"]}'
    # Only add target_fps if not null
    if config["preprocessor"]["target_fps"] is not None:
        preprocessor_params += f' -p target_fps:={config["preprocessor"]["target_fps"]}'
    
    preprocessor_node = ExecuteProcess(
        cmd=['bash', '-c', f'. {install_dir}/setup.bash && python3 {preprocessor_script} {preprocessor_params}'],
        name='preprocessor_node',
        output='screen',
        shell=False
    )
    
    # Visualization Node  
    visualization_node = ExecuteProcess(
        cmd=['bash', '-c', f'. {install_dir}/setup.bash && python3 {visualization_script} --ros-args -p "window_name:={config["visualization"]["window_name"]}" -p display_scale:={config["visualization"]["display_scale"]} -p trail_history_sec:={config["visualization"]["trail_history_sec"]} -p trail_alpha_floor:={config["visualization"]["trail_alpha_floor"]}'],
        name='visualization_node',
        output='screen',
        shell=False
    )
    
    # Perception Node
    perception_node = ExecuteProcess(
        cmd=['bash', '-c', f'. {install_dir}/setup.bash && python3 {perception_script} --ros-args -p device:={config["perception"]["device"]} -p detection_model:={config["perception"]["detection_model"]} -p "text_prompt:={config["perception"]["text_prompt"]}" -p num_query_points:={config["perception"]["num_query_points"]} -p mask_erosion_pixels:={config["perception"]["mask_erosion_pixels"]} -p sampling_strategy:={config["perception"]["sampling_strategy"]} -p target_fps:={config["perception"]["target_fps"]}'],
        name='perception_node',
        output='screen',
        shell=False
    )
    
    # Predictor Node (conditionally launched based on config)
    predictor_node = None
    if config.get("predictor", {}).get("enabled", False):
        predictor_node = ExecuteProcess(
            cmd=['bash', '-c', f'. {install_dir}/setup.bash && python3 {predictor_script} --ros-args -p enabled:={config["predictor"]["enabled"]} -p train_config_path:={config["predictor"]["train_config_path"]} -p checkpoint_path:={config["predictor"]["checkpoint_path"]} -p device:={config["predictor"]["device"]}'],
            name='predictor_node',
            output='screen',
            shell=False
        )
    
    # Build launch description
    launch_list = [
        realsense_launch,
        preprocessor_node,
        perception_node,
        visualization_node,
    ]
    
    # Add predictor node if enabled
    if predictor_node is not None:
        launch_list.append(predictor_node)
    
    return LaunchDescription(launch_list)

