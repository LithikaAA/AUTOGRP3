#!/usr/bin/env python3

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_pioneer_nav = get_package_share_directory('pioneer_nav')

    gui = LaunchConfiguration('gui')
    estop = LaunchConfiguration('estop')
    detector = LaunchConfiguration('detector')
    camera_topic = LaunchConfiguration('camera_topic')
    depth_topic = LaunchConfiguration('depth_topic')
    waypoint_driver = LaunchConfiguration('waypoint_driver')
    obstacle_waypoint_file = LaunchConfiguration('obstacle_waypoint_file')
    waypoint_source = LaunchConfiguration('waypoint_source')
    detection_log_file = LaunchConfiguration('detection_log_file')
    detection_goal_standoff_m = LaunchConfiguration('detection_goal_standoff_m')
    use_astar = LaunchConfiguration('use_astar')
    map_yaml = LaunchConfiguration('map_yaml')
    binary_map_csv = LaunchConfiguration('binary_map_csv')
    astar_obstacle_inflation_m = LaunchConfiguration('astar_obstacle_inflation_m')
    astar_waypoint_spacing_m = LaunchConfiguration('astar_waypoint_spacing_m')
    coverage_area_size_m = LaunchConfiguration('coverage_area_size_m')
    coverage_boundary_margin_m = LaunchConfiguration('coverage_boundary_margin_m')
    coverage_sweep_spacing_m = LaunchConfiguration('coverage_sweep_spacing_m')
    coverage_pattern = LaunchConfiguration('coverage_pattern')
    coverage_probe_radius_ratio = LaunchConfiguration('coverage_probe_radius_ratio')
    coverage_scan_spin_s = LaunchConfiguration('coverage_scan_spin_s')
    coverage_scan_turn_speed = LaunchConfiguration('coverage_scan_turn_speed')
    coverage_adaptive_viewpoints = LaunchConfiguration('coverage_adaptive_viewpoints')
    coverage_occlusion_probe_offset_m = LaunchConfiguration('coverage_occlusion_probe_offset_m')
    coverage_max_adaptive_viewpoints = LaunchConfiguration('coverage_max_adaptive_viewpoints')
    coverage_initial_arc_scan_s = LaunchConfiguration('coverage_initial_arc_scan_s')
    coverage_initial_arc_linear_speed = LaunchConfiguration('coverage_initial_arc_linear_speed')
    coverage_initial_arc_turn_speed = LaunchConfiguration('coverage_initial_arc_turn_speed')
    waypoint_goal_tolerance = LaunchConfiguration('waypoint_goal_tolerance')
    waypoint_linear_speed = LaunchConfiguration('waypoint_linear_speed')
    waypoint_slow_linear_speed = LaunchConfiguration('waypoint_slow_linear_speed')
    enable_waypoint_obstacle_avoidance = LaunchConfiguration('enable_waypoint_obstacle_avoidance')
    waypoint_obstacle_linear_speed = LaunchConfiguration('waypoint_obstacle_linear_speed')
    waypoint_obstacle_turn_speed = LaunchConfiguration('waypoint_obstacle_turn_speed')
    front_obstacle_dist_m = LaunchConfiguration('front_obstacle_dist_m')
    critical_obstacle_dist_m = LaunchConfiguration('critical_obstacle_dist_m')
    waypoint_critical_turn_speed = LaunchConfiguration('waypoint_critical_turn_speed')
    waypoint_critical_reverse_speed = LaunchConfiguration('waypoint_critical_reverse_speed')
    front_obstacle_fov_deg = LaunchConfiguration('front_obstacle_fov_deg')
    side_obstacle_fov_deg = LaunchConfiguration('side_obstacle_fov_deg')
    use_nav2 = LaunchConfiguration('use_nav2')
    slam_start_delay = LaunchConfiguration('slam_start_delay')

    control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                pkg_pioneer_nav,
                'launch',
                'control_node_gazebo.launch.py',
            ])
        ),
        launch_arguments={
            'rviz': 'false',
            'arena_size_m': coverage_area_size_m,
            'coverage_boundary_margin_m': coverage_boundary_margin_m,
            'coverage_sweep_spacing_m': coverage_sweep_spacing_m,
            'coverage_scan_spin_s': coverage_scan_spin_s,
            'coverage_scan_turn_speed': coverage_scan_turn_speed,
            'coverage_row_midpoint_scans': 'true',
            'use_nav2': use_nav2,
        }.items(),
    )

    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                pkg_pioneer_nav,
                'launch',
                'slam_mapping.launch.py',
            ])
        ),
        launch_arguments={
            'use_sim_time': 'true',
            'rviz': 'false',
            'scan_frame': 'pioneer/base_link/laser',
            'odom_tf_stamp_with_current_time': 'true',
            'slam_start_delay': slam_start_delay,
        }.items(),
    )
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                pkg_pioneer_nav,
                'launch',
                'nav2_navigation.launch.py',
            ])
        ),
        condition=IfCondition(use_nav2),
        launch_arguments={
            'use_sim_time': 'true',
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'gui',
            default_value='true',
            description='Start the PyQt robot monitor GUI.',
        ),
        DeclareLaunchArgument(
            'estop',
            default_value='true',
            description='Start LiDAR e-stop status node.',
        ),
        DeclareLaunchArgument('detector', default_value='true'),
        DeclareLaunchArgument('camera_topic', default_value='/camera/image'),
        DeclareLaunchArgument('depth_topic', default_value='/oak/stereo/image_raw'),
        DeclareLaunchArgument(
            'waypoint_driver',
            default_value='true',
            description='Start distbug waypoint command listener.',
        ),
        DeclareLaunchArgument('obstacle_waypoint_file', default_value=''),
        DeclareLaunchArgument(
            'waypoint_source',
            default_value='detections_json',
            description='Use detections_json for unified detector JSON/JSONL object goals, or file for waypoint text file.',
        ),
        DeclareLaunchArgument('detection_log_file', default_value=''),
        DeclareLaunchArgument('detection_goal_standoff_m', default_value='0.6'),
        DeclareLaunchArgument('use_astar', default_value='true'),
        DeclareLaunchArgument('map_yaml', default_value=''),
        DeclareLaunchArgument('binary_map_csv', default_value=''),
        DeclareLaunchArgument('astar_obstacle_inflation_m', default_value='0.25'),
        DeclareLaunchArgument('astar_waypoint_spacing_m', default_value='0.35'),
        DeclareLaunchArgument('coverage_area_size_m', default_value='10.0'),
        DeclareLaunchArgument('coverage_boundary_margin_m', default_value='0.5'),
        DeclareLaunchArgument('coverage_sweep_spacing_m', default_value='1.4'),
        DeclareLaunchArgument('coverage_pattern', default_value='serpentine'),
        DeclareLaunchArgument('coverage_probe_radius_ratio', default_value='0.45'),
        DeclareLaunchArgument('coverage_scan_spin_s', default_value='8.0'),
        DeclareLaunchArgument('coverage_scan_turn_speed', default_value='0.8'),
        DeclareLaunchArgument('coverage_adaptive_viewpoints', default_value='true'),
        DeclareLaunchArgument('coverage_occlusion_probe_offset_m', default_value='2.0'),
        DeclareLaunchArgument('coverage_max_adaptive_viewpoints', default_value='4'),
        DeclareLaunchArgument('coverage_initial_arc_scan_s', default_value='8.0'),
        DeclareLaunchArgument('coverage_initial_arc_linear_speed', default_value='0.08'),
        DeclareLaunchArgument('coverage_initial_arc_turn_speed', default_value='0.22'),
        DeclareLaunchArgument('waypoint_goal_tolerance', default_value='0.8'),
        DeclareLaunchArgument('waypoint_linear_speed', default_value='0.18'),
        DeclareLaunchArgument('waypoint_slow_linear_speed', default_value='0.04'),
        DeclareLaunchArgument('enable_waypoint_obstacle_avoidance', default_value='true'),
        DeclareLaunchArgument('waypoint_obstacle_linear_speed', default_value='0.04'),
        DeclareLaunchArgument('waypoint_obstacle_turn_speed', default_value='0.45'),
        DeclareLaunchArgument('front_obstacle_dist_m', default_value='0.45'),
        DeclareLaunchArgument('critical_obstacle_dist_m', default_value='0.22'),
        DeclareLaunchArgument('waypoint_critical_turn_speed', default_value='0.55'),
        DeclareLaunchArgument('waypoint_critical_reverse_speed', default_value='-0.25'),
        DeclareLaunchArgument('front_obstacle_fov_deg', default_value='130.0'),
        DeclareLaunchArgument('side_obstacle_fov_deg', default_value='90.0'),
        DeclareLaunchArgument('use_nav2', default_value='true'),
        DeclareLaunchArgument(
            'slam_start_delay',
            default_value='8.0',
            description='Seconds to wait before starting slam_toolbox.',
        ),
        control_launch,
        TimerAction(
            period=2.0,
            actions=[
                Node(
                    package='ros_gz_bridge',
                    executable='parameter_bridge',
                    name='camera_bridge',
                    output='screen',
                    arguments=[
                        '/camera/image@sensor_msgs/msg/Image@gz.msgs.Image',
                    ],
                ),
                Node(
                    package='pioneer_nav',
                    executable='estoplidar',
                    name='estoplidar',
                    output='screen',
                    condition=IfCondition(estop),
                    parameters=[
                        {'bag_directory': '/tmp/pioneer_estop/bags'},
                        {'incident_log': '/tmp/pioneer_estop/incidents.txt'},
                        {'stop_distance_m': 0.5},
                        {'warning_distance_m': 1.0},
                    ],
                ),
                Node(
                    package='pioneer_nav',
                    executable='unified_detector',
                    name='unified_detector',
                    output='screen',
                    condition=IfCondition(detector),
                    parameters=[
                        {'topic': camera_topic},
                        {'depth_topic': depth_topic},
                        {'require_mapping_state': False},
                    ],
                ),
            ],
        ),
        TimerAction(
            period=2.0,
            actions=[slam_launch],
        ),
        TimerAction(
            period=PythonExpression([slam_start_delay, ' + 4.0']),
            actions=[nav2_launch],
        ),
        TimerAction(
            period=4.0,
            actions=[
                Node(
                    package='pioneer_nav',
                    executable='waypoint_controller',
                    name='waypoint_controller',
                    output='screen',
                    condition=IfCondition(waypoint_driver),
                    parameters=[
                        {'scan_topic': '/scan'},
                        {'image_topic': '/oak/rgb/image_raw'},
                        {'imu_topic': '/imu'},
                        {'odom_topic': '/odom'},
                        {'cmd_vel_topic': '/cmd_vel'},
                        {'mission_command_topic': '/mission_command'},
                        {'robot_state_topic': '/robot_state'},
                        {'ignore_coverage_commands': True},
                        {'use_gazebo_tf_pose': True},
                        {'gazebo_tf_topic': '/world/pioneer_world/dynamic_pose/info'},
                        {'gazebo_tf_frame_match': 'pioneer'},
                        {'gazebo_tf_allow_unmatched': True},
                        {'obstacle_waypoint_file': obstacle_waypoint_file},
                        {'waypoint_source': waypoint_source},
                        {'detection_log_file': detection_log_file},
                        {'detection_goal_standoff_m': ParameterValue(detection_goal_standoff_m, value_type=float)},
                        {'use_astar': ParameterValue(use_astar, value_type=bool)},
                        {'map_yaml': map_yaml},
                        {'binary_map_csv': binary_map_csv},
                        {'astar_obstacle_inflation_m': ParameterValue(astar_obstacle_inflation_m, value_type=float)},
                        {'astar_waypoint_spacing_m': ParameterValue(astar_waypoint_spacing_m, value_type=float)},
                        {'coverage_area_size_m': ParameterValue(coverage_area_size_m, value_type=float)},
                        {'coverage_boundary_margin_m': ParameterValue(coverage_boundary_margin_m, value_type=float)},
                        {'coverage_sweep_spacing_m': ParameterValue(coverage_sweep_spacing_m, value_type=float)},
                        {'coverage_pattern': coverage_pattern},
                        {'coverage_probe_radius_ratio': ParameterValue(coverage_probe_radius_ratio, value_type=float)},
                        {'coverage_scan_spin_s': ParameterValue(coverage_scan_spin_s, value_type=float)},
                        {'coverage_scan_turn_speed': ParameterValue(coverage_scan_turn_speed, value_type=float)},
                        {'coverage_adaptive_viewpoints': ParameterValue(coverage_adaptive_viewpoints, value_type=bool)},
                        {'coverage_occlusion_probe_offset_m': ParameterValue(coverage_occlusion_probe_offset_m, value_type=float)},
                        {'coverage_max_adaptive_viewpoints': ParameterValue(coverage_max_adaptive_viewpoints, value_type=int)},
                        {'coverage_initial_arc_scan_s': ParameterValue(coverage_initial_arc_scan_s, value_type=float)},
                        {'coverage_initial_arc_linear_speed': ParameterValue(coverage_initial_arc_linear_speed, value_type=float)},
                        {'coverage_initial_arc_turn_speed': ParameterValue(coverage_initial_arc_turn_speed, value_type=float)},
                        {'relative_to_start': True},
                        {'use_test_waypoint': False},
                        {'waypoint_goal_tolerance': ParameterValue(waypoint_goal_tolerance, value_type=float)},
                        {'waypoint_linear_speed': ParameterValue(waypoint_linear_speed, value_type=float)},
                        {'waypoint_slow_linear_speed': ParameterValue(waypoint_slow_linear_speed, value_type=float)},
                        {'enable_waypoint_obstacle_avoidance': ParameterValue(enable_waypoint_obstacle_avoidance, value_type=bool)},
                        {'waypoint_obstacle_linear_speed': ParameterValue(waypoint_obstacle_linear_speed, value_type=float)},
                        {'waypoint_obstacle_turn_speed': ParameterValue(waypoint_obstacle_turn_speed, value_type=float)},
                        {'front_obstacle_dist_m': ParameterValue(front_obstacle_dist_m, value_type=float)},
                        {'critical_obstacle_dist_m': ParameterValue(critical_obstacle_dist_m, value_type=float)},
                        {'waypoint_critical_turn_speed': ParameterValue(waypoint_critical_turn_speed, value_type=float)},
                        {'waypoint_critical_reverse_speed': ParameterValue(waypoint_critical_reverse_speed, value_type=float)},
                        {'front_obstacle_fov_deg': ParameterValue(front_obstacle_fov_deg, value_type=float)},
                        {'side_obstacle_fov_deg': ParameterValue(side_obstacle_fov_deg, value_type=float)},
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
