#!/usr/bin/env python3

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_pioneer_nav = get_package_share_directory('pioneer_nav')

    use_sim_time = LaunchConfiguration('use_sim_time')
    scan_frame = LaunchConfiguration('scan_frame')
    aria = LaunchConfiguration('aria')
    aria_port = LaunchConfiguration('aria_port')
    control = LaunchConfiguration('control')
    gui = LaunchConfiguration('gui')
    estop = LaunchConfiguration('estop')
    oak_camera = LaunchConfiguration('oak_camera')
    detector = LaunchConfiguration('detector')
    camera_topic = LaunchConfiguration('camera_topic')
    depth_topic = LaunchConfiguration('depth_topic')
    waypoint_driver = LaunchConfiguration('waypoint_driver')
    obstacle_waypoint_file = LaunchConfiguration('obstacle_waypoint_file')
    waypoint_goal_tolerance = LaunchConfiguration('waypoint_goal_tolerance')
    waypoint_linear_speed = LaunchConfiguration('waypoint_linear_speed')
    waypoint_slow_linear_speed = LaunchConfiguration('waypoint_slow_linear_speed')
    coverage_area_size_m = LaunchConfiguration('coverage_area_size_m')
    coverage_boundary_margin_m = LaunchConfiguration('coverage_boundary_margin_m')
    coverage_sweep_spacing_m = LaunchConfiguration('coverage_sweep_spacing_m')
    coverage_pattern = LaunchConfiguration('coverage_pattern')
    coverage_scan_spin_s = LaunchConfiguration('coverage_scan_spin_s')
    coverage_scan_turn_speed = LaunchConfiguration('coverage_scan_turn_speed')
    coverage_adaptive_viewpoints = LaunchConfiguration('coverage_adaptive_viewpoints')
    coverage_occlusion_probe_offset_m = LaunchConfiguration('coverage_occlusion_probe_offset_m')
    coverage_max_adaptive_viewpoints = LaunchConfiguration('coverage_max_adaptive_viewpoints')
    coverage_initial_arc_scan_s = LaunchConfiguration('coverage_initial_arc_scan_s')
    coverage_initial_arc_linear_speed = LaunchConfiguration('coverage_initial_arc_linear_speed')
    coverage_initial_arc_turn_speed = LaunchConfiguration('coverage_initial_arc_turn_speed')
    enable_waypoint_obstacle_avoidance = LaunchConfiguration('enable_waypoint_obstacle_avoidance')
    waypoint_obstacle_linear_speed = LaunchConfiguration('waypoint_obstacle_linear_speed')
    waypoint_obstacle_turn_speed = LaunchConfiguration('waypoint_obstacle_turn_speed')
    front_obstacle_dist_m = LaunchConfiguration('front_obstacle_dist_m')
    critical_obstacle_dist_m = LaunchConfiguration('critical_obstacle_dist_m')
    front_obstacle_fov_deg = LaunchConfiguration('front_obstacle_fov_deg')
    waypoint_lidar_self_filter_min_range = LaunchConfiguration('waypoint_lidar_self_filter_min_range')
    control_obstacle_buffer = LaunchConfiguration('control_obstacle_buffer')
    control_emergency_stop_distance = LaunchConfiguration('control_emergency_stop_distance')
    control_lidar_front_angle_deg = LaunchConfiguration('control_lidar_front_angle_deg')
    control_lidar_self_filter_min_range = LaunchConfiguration('control_lidar_self_filter_min_range')
    slam_start_delay = LaunchConfiguration('slam_start_delay')
    odom_tf_stamp_with_current_time = LaunchConfiguration('odom_tf_stamp_with_current_time')

    # ── Detector low-light tuning ──────────────────────────────────────────
    brightness_threshold  = LaunchConfiguration('brightness_threshold')
    min_paper_brightness  = LaunchConfiguration('min_paper_brightness')
    min_dark_ratio        = LaunchConfiguration('min_dark_ratio')
    confident_duration_s  = LaunchConfiguration('confident_duration_s')

    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                pkg_pioneer_nav,
                'launch',
                'slam_mapping.launch.py',
            ])
        ),
        launch_arguments={
            'use_sim_time':                    use_sim_time,
            'rviz':                            'false',
            'scan_frame':                      scan_frame,
            'odom_tf_stamp_with_current_time': odom_tf_stamp_with_current_time,
            'slam_start_delay':                slam_start_delay,
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('scan_frame', default_value='sick_laser'),
        DeclareLaunchArgument('aria', default_value='false'),
        DeclareLaunchArgument('aria_port', default_value='/dev/ttyUSB0'),
        DeclareLaunchArgument('control', default_value='true'),
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('estop', default_value='true'),
        DeclareLaunchArgument('oak_camera', default_value='true'),
        DeclareLaunchArgument('detector', default_value='true'),
        DeclareLaunchArgument('camera_topic', default_value='/oak/rgb/image_raw'),
        DeclareLaunchArgument('depth_topic', default_value='/oak/stereo/image_raw'),
        DeclareLaunchArgument('waypoint_driver', default_value='true'),
        DeclareLaunchArgument('obstacle_waypoint_file', default_value=''),
        DeclareLaunchArgument('waypoint_goal_tolerance', default_value='0.8'),
        DeclareLaunchArgument('waypoint_linear_speed', default_value='0.18'),
        DeclareLaunchArgument('waypoint_slow_linear_speed', default_value='0.04'),
        DeclareLaunchArgument('coverage_area_size_m', default_value='15.0'),
        DeclareLaunchArgument('coverage_boundary_margin_m', default_value='0.5'),
        DeclareLaunchArgument('coverage_sweep_spacing_m', default_value='1.4'),
        DeclareLaunchArgument('coverage_pattern', default_value='serpentine'),
        DeclareLaunchArgument('coverage_scan_spin_s', default_value='8.0'),
        DeclareLaunchArgument('coverage_scan_turn_speed', default_value='0.8'),
        DeclareLaunchArgument('coverage_adaptive_viewpoints', default_value='true'),
        DeclareLaunchArgument('coverage_occlusion_probe_offset_m', default_value='2.0'),
        DeclareLaunchArgument('coverage_max_adaptive_viewpoints', default_value='4'),
        DeclareLaunchArgument('coverage_initial_arc_scan_s', default_value='8.0'),
        DeclareLaunchArgument('coverage_initial_arc_linear_speed', default_value='0.08'),
        DeclareLaunchArgument('coverage_initial_arc_turn_speed', default_value='0.22'),
        DeclareLaunchArgument('enable_waypoint_obstacle_avoidance', default_value='true'),
        DeclareLaunchArgument('waypoint_obstacle_linear_speed', default_value='0.04'),
        DeclareLaunchArgument('waypoint_obstacle_turn_speed', default_value='0.45'),
        DeclareLaunchArgument('front_obstacle_dist_m', default_value='0.45'),
        DeclareLaunchArgument('critical_obstacle_dist_m', default_value='0.22'),
        DeclareLaunchArgument('front_obstacle_fov_deg', default_value='60.0'),
        DeclareLaunchArgument('waypoint_lidar_self_filter_min_range', default_value='0.18'),
        DeclareLaunchArgument('control_obstacle_buffer', default_value='0.25'),
        DeclareLaunchArgument('control_emergency_stop_distance', default_value='0.18'),
        DeclareLaunchArgument('control_lidar_front_angle_deg', default_value='35.0'),
        DeclareLaunchArgument('control_lidar_self_filter_min_range', default_value='0.25'),
        DeclareLaunchArgument('slam_start_delay', default_value='8.0'),
        DeclareLaunchArgument('odom_tf_stamp_with_current_time', default_value='true'),

        # ── Detector tuning ───────────────────────────────────────────────────
        # Normal daylight:  brightness_threshold=170  min_paper_brightness=150  min_dark_ratio=0.02
        # Low light / dusk: brightness_threshold=100  min_paper_brightness=100  min_dark_ratio=0.01
        DeclareLaunchArgument('brightness_threshold',
            default_value='170',
            description='Brightness threshold for white paper detection (lower for darker conditions)'),
        DeclareLaunchArgument('min_paper_brightness',
            default_value='150',
            description='Min mean brightness of candidate region (lower for darker conditions)'),
        DeclareLaunchArgument('min_dark_ratio',
            default_value='0.02',
            description='Min dark pixel fraction to accept a region as paper (lower for faint letters)'),
        DeclareLaunchArgument('confident_duration_s',
            default_value='2.0',
            description='Seconds a detection must be held before logging coordinates'),

        # ── Nodes ─────────────────────────────────────────────────────────────
        Node(
            package='ariaNode',
            executable='ariaNode',
            name='aria_node',
            output='screen',
            condition=IfCondition(aria),
            arguments=['-rp', aria_port],
        ),
        slam_launch,
        Node(
            package='pioneer_nav',
            executable='control_node',
            name='pioneer_control_node',
            output='screen',
            condition=IfCondition(control),
            parameters=[
                {'joy_topic':                  '/joy'},
                {'scan_topic':                 '/scan'},
                {'odom_topic':                 '/odom'},
                {'cmd_vel_topic':              '/cmd_vel'},
                {'estop_status_topic':         '/estop_status'},
                {'robot_pose_topic':           '/robot/pose'},
                {'arena_status_topic':         '/arena_status'},
                {'robot_state_topic':          '/robot_state'},
                {'mission_command_topic':      '/mission_command'},
                {'publish_gui_topics':         True},
                {'external_manual_control':    True},
                {'forward_speed':              0.50},
                {'reverse_speed':              -0.12},
                {'turn_speed_deg':             20.0},
                {'return_to_center_speed':     0.12},
                {'return_to_center_turn_speed_deg': 20.0},
                {'arena_size_m':                 ParameterValue(coverage_area_size_m, value_type=float)},
                {'coverage_boundary_margin_m': ParameterValue(coverage_boundary_margin_m, value_type=float)},
                {'coverage_sweep_spacing_m':   ParameterValue(coverage_sweep_spacing_m, value_type=float)},
                {'coverage_linear_speed':      0.18},
                {'coverage_turn_speed_deg':    20.0},
                {'coverage_scan_spin_s':       ParameterValue(coverage_scan_spin_s, value_type=float)},
                {'coverage_scan_turn_speed':   ParameterValue(coverage_scan_turn_speed, value_type=float)},
                {'coverage_row_midpoint_scans': True},
                {'obstacle_buffer':            ParameterValue(control_obstacle_buffer, value_type=float)},
                {'emergency_stop_distance':    ParameterValue(control_emergency_stop_distance, value_type=float)},
                {'lidar_front_angle_deg':      ParameterValue(control_lidar_front_angle_deg, value_type=float)},
                {'lidar_self_filter_min_range': ParameterValue(control_lidar_self_filter_min_range, value_type=float)},
                {'center_arena_on_start':      True},
                {'use_gazebo_tf_pose':         False},
            ],
        ),
        Node(
            package='pioneer_nav',
            executable='estoplidar',
            name='estoplidar',
            output='screen',
            condition=IfCondition(estop),
            parameters=[
                {'stop_distance_m': 0.18},
                {'warning_distance_m': 0.35},
            ],
        ),
        Node(
            package='pioneer_nav',
            executable='oak_camera',
            name='oak_camera',
            output='screen',
            condition=IfCondition(oak_camera),
            parameters=[
                {'rgb_topic':       camera_topic},
                {'depth_topic':     depth_topic},
                {'frame_id':        'oak_camera'},
                {'publish_rate_hz': 15.0},
                {'webcam_fallback': False},
            ],
        ),
        Node(
            package='pioneer_nav',
            executable='unified_detector',
            name='unified_detector',
            output='screen',
            condition=IfCondition(detector),
            parameters=[
                {'topic':                 camera_topic},
                {'depth_topic':           depth_topic},
                {'require_mapping_state': False},
                {'brightness_threshold':  ParameterValue(brightness_threshold,  value_type=int)},
                {'min_paper_brightness':  ParameterValue(min_paper_brightness,  value_type=int)},
                {'min_dark_ratio':        ParameterValue(min_dark_ratio,        value_type=float)},
                {'confident_duration_s':  ParameterValue(confident_duration_s,  value_type=float)},
            ],
        ),
        TimerAction(
            period=2.0,
            actions=[
                Node(
                    package='pioneer_nav',
                    executable='waypoint_controller',
                    name='waypoint_controller',
                    output='screen',
                    condition=IfCondition(waypoint_driver),
                    parameters=[
                        {'scan_topic': '/scan'},
                        {'odom_topic': '/odom'},
                        {'cmd_vel_topic': '/cmd_vel'},
                        {'mission_command_topic': '/mission_command'},
                        {'robot_state_topic': '/robot_state'},
                        {'ignore_coverage_commands': True},
                        {'obstacle_waypoint_file': obstacle_waypoint_file},
                        {'relative_to_start': True},
                        {'use_test_waypoint': False},
                        {'coverage_area_size_m': ParameterValue(coverage_area_size_m, value_type=float)},
                        {'coverage_boundary_margin_m': ParameterValue(coverage_boundary_margin_m, value_type=float)},
                        {'coverage_sweep_spacing_m': ParameterValue(coverage_sweep_spacing_m, value_type=float)},
                        {'coverage_pattern': coverage_pattern},
                        {'coverage_scan_spin_s': ParameterValue(coverage_scan_spin_s, value_type=float)},
                        {'coverage_scan_turn_speed': ParameterValue(coverage_scan_turn_speed, value_type=float)},
                        {'coverage_adaptive_viewpoints': ParameterValue(coverage_adaptive_viewpoints, value_type=bool)},
                        {'coverage_occlusion_probe_offset_m': ParameterValue(coverage_occlusion_probe_offset_m, value_type=float)},
                        {'coverage_max_adaptive_viewpoints': ParameterValue(coverage_max_adaptive_viewpoints, value_type=int)},
                        {'coverage_initial_arc_scan_s': ParameterValue(coverage_initial_arc_scan_s, value_type=float)},
                        {'coverage_initial_arc_linear_speed': ParameterValue(coverage_initial_arc_linear_speed, value_type=float)},
                        {'coverage_initial_arc_turn_speed': ParameterValue(coverage_initial_arc_turn_speed, value_type=float)},
                        {'waypoint_goal_tolerance': ParameterValue(waypoint_goal_tolerance, value_type=float)},
                        {'waypoint_linear_speed': ParameterValue(waypoint_linear_speed, value_type=float)},
                        {'waypoint_slow_linear_speed': ParameterValue(waypoint_slow_linear_speed, value_type=float)},
                        {'enable_waypoint_obstacle_avoidance': ParameterValue(enable_waypoint_obstacle_avoidance, value_type=bool)},
                        {'waypoint_obstacle_linear_speed': ParameterValue(waypoint_obstacle_linear_speed, value_type=float)},
                        {'waypoint_obstacle_turn_speed': ParameterValue(waypoint_obstacle_turn_speed, value_type=float)},
                        {'front_obstacle_dist_m': ParameterValue(front_obstacle_dist_m, value_type=float)},
                        {'critical_obstacle_dist_m': ParameterValue(critical_obstacle_dist_m, value_type=float)},
                        {'front_obstacle_fov_deg': ParameterValue(front_obstacle_fov_deg, value_type=float)},
                        {'lidar_self_filter_min_range': ParameterValue(waypoint_lidar_self_filter_min_range, value_type=float)},
                    ],
                ),
                Node(
                    package='pioneer_nav',
                    executable='robot_gui',
                    name='robot_gui',
                    output='screen',
                    condition=IfCondition(gui),
                    additional_env={'PIONEER_GUI_ARENA_SIZE_M': coverage_area_size_m},
                ),
            ],
        ),
    ])
