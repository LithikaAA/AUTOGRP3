#!/usr/bin/env python3
# Launch file for testing control_node in Gazebo with Pioneer robot

import os
import tempfile
import xml.etree.ElementTree as ET
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_pioneer_nav = get_package_share_directory('pioneer_nav')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    # Packaged Pioneer robot/world assets.
    robot_file = os.path.join(pkg_pioneer_nav, 'robots', 'pioneer.urdf')
    mesh_dir = os.path.join(pkg_pioneer_nav, 'robots', 'meshes')
    source_world_file = os.path.join(pkg_pioneer_nav, 'worlds', 'basic_urdf.sdf')

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

    world_file = os.path.join(tempfile.gettempdir(), 'pioneer_world_no_robot.sdf')
    world_tree = ET.parse(source_world_file)
    world = world_tree.getroot().find('world')
    for include in list(world.findall('include')):
        name = include.findtext('name', default='')
        uri = include.findtext('uri', default='')
        if name == 'pioneer' or 'pioneer.urdf' in uri:
            world.remove(include)
    ET.indent(world_tree, space='    ')
    world_tree.write(world_file, encoding='unicode', xml_declaration=True)

    # ==================== Launch Arguments ====================
    rviz_launch_arg = DeclareLaunchArgument(
        'rviz', default_value='false',
        description='Open RViz.'
    )
    
    control_mode_arg = DeclareLaunchArgument(
        'control_mode', default_value='auto',
        description='Control mode: auto, manual, or joy'
    )

    arena_size_arg = DeclareLaunchArgument(
        'arena_size_m', default_value='15.0',
        description='Square arena side length in metres.'
    )
    coverage_boundary_margin_arg = DeclareLaunchArgument(
        'coverage_boundary_margin_m', default_value='0.5',
        description='Lawnmower margin inside the square arena.'
    )
    coverage_sweep_spacing_arg = DeclareLaunchArgument(
        'coverage_sweep_spacing_m', default_value='1.4',
        description='Spacing between lawnmower rows.'
    )
    coverage_scan_spin_arg = DeclareLaunchArgument(
        'coverage_scan_spin_s', default_value='6.0',
        description='Seconds to rotate in place at each coverage waypoint.'
    )
    coverage_scan_turn_speed_arg = DeclareLaunchArgument(
        'coverage_scan_turn_speed', default_value='0.65',
        description='Angular speed in rad/s for coverage scan rotations.'
    )
    coverage_row_midpoint_scans_arg = DeclareLaunchArgument(
        'coverage_row_midpoint_scans', default_value='true',
        description='Add a scan waypoint at the middle of every lawnmower row.'
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
            {'arena_size_m': ParameterValue(LaunchConfiguration('arena_size_m'), value_type=float)},
            {'coverage_boundary_margin_m': ParameterValue(LaunchConfiguration('coverage_boundary_margin_m'), value_type=float)},
            {'coverage_sweep_spacing_m': ParameterValue(LaunchConfiguration('coverage_sweep_spacing_m'), value_type=float)},
            {'coverage_scan_spin_s': ParameterValue(LaunchConfiguration('coverage_scan_spin_s'), value_type=float)},
            {'coverage_scan_turn_speed': ParameterValue(LaunchConfiguration('coverage_scan_turn_speed'), value_type=float)},
            {'coverage_row_midpoint_scans': ParameterValue(LaunchConfiguration('coverage_row_midpoint_scans'), value_type=bool)},
            {'center_arena_on_start': False},
            {'arena_origin_x': 0.0},
            {'arena_origin_y': 0.0},
            {'use_gazebo_tf_pose': True},
            {'gazebo_tf_topic': '/world/pioneer_world/dynamic_pose/info'},
            {'gazebo_tf_frame_match': 'pioneer'},
            {'gazebo_tf_allow_unmatched': True},
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
            '/model/pioneer/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
            '/world/pioneer_world/dynamic_pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
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
        arena_size_arg,
        coverage_boundary_margin_arg,
        coverage_sweep_spacing_arg,
        coverage_scan_spin_arg,
        coverage_scan_turn_speed_arg,
        coverage_row_midpoint_scans_arg,
        gazebo,
        robot,
        robot_state_publisher,
        rviz,
        gz_bridge,
        control_node,
        # Uncomment below if you have a joystick connected
        # joy_node,
    ])
