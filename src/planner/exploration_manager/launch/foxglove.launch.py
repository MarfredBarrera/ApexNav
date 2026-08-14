"""
ROS2 Launch file for Foxglove visualization.

Drop-in replacement for rviz.launch.py on headless machines: instead of starting
RViz2 against a local display, it starts foxglove_bridge, which serves the whole
ROS graph over a websocket. Connect from Foxglove (desktop or web) with

    ssh -L 8765:localhost:8765 <host>

then open ws://localhost:8765 and import
config/apexnav_foxglove_layout.json for the equivalent of ApexNav.rviz.

Custom messages (plan_env/MultipleMasksWithConfidence, trajectory_manager/PolyTraj)
decode without extra setup: the bridge sends each topic's schema from the ROS 2
typesupport, which it can see because this launch runs from the sourced workspace.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    port_arg = DeclareLaunchArgument(
        'port',
        default_value='8765',
        description='Websocket port for foxglove_bridge'
    )

    address_arg = DeclareLaunchArgument(
        'address',
        default_value='0.0.0.0',
        description='Bind address; 0.0.0.0 so it is reachable from outside the container'
    )

    foxglove_node = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        output='screen',
        parameters=[{
            'port': LaunchConfiguration('port'),
            'address': LaunchConfiguration('address'),
            # The grid_map point clouds and 640x480 depth frames overrun the default
            # 10 MB buffer as soon as the map grows.
            'send_buffer_limit': 100000000,
            'use_compression': True,
            'max_qos_depth': 10,
            'num_threads': 0,          # 0 = one per core
            'capabilities': ['clientPublish', 'connectionGraph', 'assets'],
        }]
    )

    # Same static transform rviz.launch.py publishes. Keep it: the Foxglove 3D panel
    # needs a real TF frame to render the world-framed clouds against.
    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_world_navigation',
        arguments=['0', '0', '0', '0', '0', '0', 'world', 'navigation']
    )

    return LaunchDescription([
        port_arg,
        address_arg,
        foxglove_node,
        static_tf_node,
    ])
