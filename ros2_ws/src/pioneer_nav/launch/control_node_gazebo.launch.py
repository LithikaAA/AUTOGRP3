#!/usr/bin/env python3
# Launch file for testing control_node in Gazebo with Pioneer robot

import os
import tempfile
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    pkg_pioneer_nav = get_package_share_directory('pioneer_nav')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    # Path to existing Pioneer URDF
    robot_file = '/mnt/c/Users/rahma/Downloads/AUTOGRP3-main/AUTOGRP3/robots/pioneer.urdf'
    mesh_dir = '/mnt/c/Users/rahma/Downloads/AUTOGRP3-main/AUTOGRP3/robots/meshes'
    source_world_file = '/mnt/c/Users/rahma/Downloads/AUTOGRP3-main/worlds/basic_urdf.sdf'

    with open(robot_file, 'r') as infp:
        robot_desc = infp.read()

    # Fix mesh paths to be absolute
    robot_desc = robot_desc.replace('filename="meshes/', f'filename="{mesh_dir}/')
    robot_desc = robot_desc.replace('/home/rahma/AUTOGRP3/robots/meshes', mesh_dir)
    robot_desc = robot_desc.replace('<topic>cmd_vel</topic>', '<topic>/cmd_vel</topic>')
    robot_desc = robot_desc.replace('<odom_topic>odom</odom_topic>', '<odom_topic>/odom</odom_topic>')
    robot_desc = robot_desc.replace('<topic>scan</topic>', '<topic>/scan</topic>')
    processed_robot_file = os.path.join(tempfile.gettempdir(), 'pioneer_gazebo.urdf')
    with open(processed_robot_file, 'w') as outfp:
        outfp.write(robot_desc)

    with open(source_world_file, 'r') as infp:
        world_desc = infp.read()
    world_desc = world_desc.replace(
        '''            <include>
        <uri>file:///mnt/c/Users/rahma/Downloads/Resources/robots/pioneer.urdf</uri>
        <name>pioneer</name>
        <pose>0 0 0.2 0 0 0</pose>
        </include>''',
        ''
    )
    world_file = os.path.join(tempfile.gettempdir(), 'pioneer_world_no_robot.sdf')
    with open(world_file, 'w') as outfp:
        outfp.write(world_desc)

    # ==================== Launch Arguments ====================
    rviz_launch_arg = DeclareLaunchArgument(
        'rviz', default_value='false',
        description='Open RViz.'
    )
    
    control_mode_arg = DeclareLaunchArgument(
        'control_mode', default_value='auto',
        description='Control mode: auto, manual, or joy'
    )

    # ==================== Gazebo ====================
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py'),
        ),
        launch_arguments={'gz_args': world_file}.items(),
    )

    # ==================== Robot State Publisher ====================
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='both',
        parameters=[
            {'use_sim_time': True},
            {'robot_description': robot_desc},
        ]
    )

    # ==================== RViz ====================
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        condition=IfCondition(LaunchConfiguration('rviz')),
        parameters=[
            {'use_sim_time': True},
        ]
    )

    # ==================== Spawn Robot ====================
    robot = ExecuteProcess(
        cmd=[
            "ros2", "run", "ros_gz_sim", "create",
            "-file", processed_robot_file,
            "-name", "pioneer",
            "-z", "0.2",
        ],
        name="spawn_robot",
        output="both"
    )

    # ==================== Control Node ====================
    control_node = Node(
        package='pioneer_nav',
        executable='control_node',
        name='pioneer_control_node',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            {'joy_topic': '/joy'},
            {'scan_topic': '/scan'},
            {'odom_topic': '/odom'},
            {'cmd_vel_topic': '/cmd_vel'},
            {'forward_speed': 0.3},
            {'reverse_speed': -0.25},
            {'turn_speed_deg': 35.0},
            {'return_to_center_speed': 0.25},
            {'return_to_center_turn_speed_deg': 30.0},
        ]
    )

    # ==================== Gazebo/ROS Topic Bridge ====================
    gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gazebo_ros_bridge',
        output='screen',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
            '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
        ],
        parameters=[
            {'use_sim_time': True},
        ],
    )

    # ==================== Joy Node (for joystick input, optional) ====================
    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        output='screen',
        parameters=[
            {'dev': '/dev/input/js0'},  # Change if using different joystick device
            {'deadzone': 0.1},
        ]
    )

    return LaunchDescription([
        rviz_launch_arg,
        control_mode_arg,
        gazebo,
        robot,
        robot_state_publisher,
        rviz,
        gz_bridge,
        control_node,
        # Uncomment below if you have a joystick connected
        # joy_node,
    ])
