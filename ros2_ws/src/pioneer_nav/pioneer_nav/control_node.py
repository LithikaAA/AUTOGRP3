#!/usr/bin/env python3
import json
import math
import random
import threading
import time
from enum import Enum

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Joy, LaserScan
from std_msgs.msg import String, Int8
from tf2_msgs.msg import TFMessage
from nav2_msgs.action import NavigateToPose

# Configuration defaults
INITIAL_TURN_ANGLE = 45
INITIAL_DRIVE_DISTANCE = 5.0
BOUNDARY_BUFFER = 1.0
MAP_SIZE = 15.0
MAP_HALF_SIZE = MAP_SIZE / 2
MAP_CENTER = (0.0, 0.0)

OBSTACLE_BUFFER = 0.6
EMERGENCY_STOP_DISTANCE = 0.3
LIDAR_FRONT_ANGLE_FOV = math.radians(35)
LIDAR_SIDE_ANGLE_FOV = math.radians(30)
LIDAR_SELF_FILTER_MIN_RANGE = 0.25
OBSTACLE_REVERSE_DURATION = 0.7
OBSTACLE_TURN_DURATION = 1.5
PREDICTIVE_BUFFER = 4.0

TURN_SPEED_DEG = 35.0
FORWARD_SPEED = 0.5
REVERSE_SPEED = -0.25
ANGLE_TOLERANCE = 7.0
TURN_TIMEOUT_MARGIN = 2.0
ODOM_STALE_TIMEOUT = 1.0
TRANSITION_STOP_DURATION = 0.3
REVERSE_DURATION = 2.0
MIN_TURN_SPEED_FACTOR = 0.2
MAX_TURN_ANGLE_FOR_FULL_SPEED = 45.0
MIN_DRIVE_DISTANCE = 2.0
MAX_DRIVE_DISTANCE = 5.0
MAX_WANDERING_TURN_ANGLE = 120.0
BOUNDARY_ESCAPE_DRIVE_DISTANCE = 3.0
CENTER_BIAS_STRENGTH = 0.7
MIN_CENTER_ANGLE = -45
MAX_CENTER_ANGLE = 45
RETURN_TO_CENTER_SPEED = 0.25
RETURN_TO_CENTER_TURN_SPEED_DEG = 30.0
RETURN_TO_CENTER_MIN_DISTANCE = 0.5

ESTOP_CLEAR = 0
ESTOP_WARNING = 1
ESTOP_ACTIVE = 2


class DRIVE_MODE(Enum):
    MANUAL = 1
    AUTO = 2


class AUTO_STATE(Enum):
    INITIAL_TURN = 0
    INITIAL_DRIVE = 1
    WANDERING_TURN = 2
    WANDERING_DRIVE = 3
    BOUNDARY_REVERSE = 4
    BOUNDARY_ESCAPE_TURN = 5
    BOUNDARY_ESCAPE_DRIVE = 6
    OBSTACLE_REVERSE = 7
    OBSTACLE_TURN = 8
    RETURN_TO_CENTER = 9
    COVERAGE_PLANNING = 10  # NAV2 planning lawnmower
    COVERAGE_EXECUTING = 11  # NAV2 executing lawnmower


def quaternion_to_yaw(orientation):
    x = orientation.x
    y = orientation.y
    z = orientation.z
    w = orientation.w
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp)) % 360


def signed_angle_diff(target_yaw, current_yaw):
    """Return shortest signed yaw error in degrees."""
    angle_diff = (target_yaw - current_yaw + 360) % 360
    if angle_diff > 180:
        angle_diff -= 360
    return angle_diff


class ControlNode(Node):
    def __init__(self):
        super().__init__('pioneer_control_node')

        self.declare_parameter('joy_topic', '/joy')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('joy_deadman_axis', 5)
        self.declare_parameter('joy_manual_button', 0)
        self.declare_parameter('joy_auto_button', 1)
        self.declare_parameter('joy_stop_button', 2)
        self.declare_parameter('external_manual_control', True)
        self.declare_parameter('forward_speed', FORWARD_SPEED)
        self.declare_parameter('reverse_speed', REVERSE_SPEED)
        self.declare_parameter('turn_speed_deg', TURN_SPEED_DEG)
        self.declare_parameter('return_to_center_speed', RETURN_TO_CENTER_SPEED)
        self.declare_parameter('return_to_center_turn_speed_deg', RETURN_TO_CENTER_TURN_SPEED_DEG)
        self.declare_parameter('lidar_front_angle_deg', math.degrees(LIDAR_FRONT_ANGLE_FOV))
        self.declare_parameter('lidar_side_angle_deg', math.degrees(LIDAR_SIDE_ANGLE_FOV))
        self.declare_parameter('lidar_self_filter_min_range', LIDAR_SELF_FILTER_MIN_RANGE)
        self.declare_parameter('center_arena_on_start', True)
        self.declare_parameter('arena_origin_x', 0.0)
        self.declare_parameter('arena_origin_y', 0.0)
        self.declare_parameter('use_gazebo_tf_pose', False)
        self.declare_parameter('gazebo_tf_topic', '/model/pioneer/tf')
        self.declare_parameter('gazebo_tf_frame_match', 'pioneer')
        self.declare_parameter('gazebo_tf_allow_unmatched', False)
        self.declare_parameter('estop_status_topic', '/estop_status')
        self.declare_parameter('robot_pose_topic', '/robot/pose')
        self.declare_parameter('arena_status_topic', '/arena_status')
        self.declare_parameter('robot_state_topic', '/robot_state')
        self.declare_parameter('mission_command_topic', '/mission_command')
        self.declare_parameter('publish_gui_topics', True)
        self.declare_parameter('use_nav2', True)

        joy_topic = self.get_parameter('joy_topic').value
        scan_topic = self.get_parameter('scan_topic').value
        odom_topic = self.get_parameter('odom_topic').value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        estop_status_topic = self.get_parameter('estop_status_topic').value
        robot_pose_topic = self.get_parameter('robot_pose_topic').value
        arena_status_topic = self.get_parameter('arena_status_topic').value
        robot_state_topic = self.get_parameter('robot_state_topic').value
        mission_command_topic = self.get_parameter('mission_command_topic').value

        self.joy_deadman_axis = int(self.get_parameter('joy_deadman_axis').value)
        self.joy_auto_button = int(self.get_parameter('joy_auto_button').value)
        self.joy_manual_button = int(self.get_parameter('joy_manual_button').value)
        self.joy_stop_button = int(self.get_parameter('joy_stop_button').value)
        self.external_manual_control = bool(self.get_parameter('external_manual_control').value)
        self.forward_speed = float(self.get_parameter('forward_speed').value)
        self.reverse_speed = float(self.get_parameter('reverse_speed').value)
        self.turn_speed_deg = float(self.get_parameter('turn_speed_deg').value)
        self.return_to_center_speed = float(self.get_parameter('return_to_center_speed').value)
        self.return_to_center_turn_speed_deg = float(self.get_parameter('return_to_center_turn_speed_deg').value)
        self.lidar_front_angle_fov = math.radians(float(self.get_parameter('lidar_front_angle_deg').value))
        self.lidar_side_angle_fov = math.radians(float(self.get_parameter('lidar_side_angle_deg').value))
        self.lidar_self_filter_min_range = float(self.get_parameter('lidar_self_filter_min_range').value)
        self.center_arena_on_start = bool(self.get_parameter('center_arena_on_start').value)
        self.configured_arena_origin_x = float(self.get_parameter('arena_origin_x').value)
        self.configured_arena_origin_y = float(self.get_parameter('arena_origin_y').value)
        self.use_gazebo_tf_pose = bool(self.get_parameter('use_gazebo_tf_pose').value)
        self.gazebo_tf_topic = self.get_parameter('gazebo_tf_topic').value
        self.gazebo_tf_frame_match = self.get_parameter('gazebo_tf_frame_match').value
        self.gazebo_tf_allow_unmatched = bool(self.get_parameter('gazebo_tf_allow_unmatched').value)
        self.publish_gui_topics = bool(self.get_parameter('publish_gui_topics').value)
        self.use_nav2 = bool(self.get_parameter('use_nav2').value)

        # ---- publishers ----
        self.cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.pose_pub = self.create_publisher(Pose, robot_pose_topic, 10)
        self.arena_status_pub = self.create_publisher(String, arena_status_topic, 10)
        self.robot_state_pub = self.create_publisher(String, robot_state_topic, 10)

        # ---- subscribers ----
        self.create_subscription(Joy, joy_topic, self.joy_cb, 10)
        self.create_subscription(Odometry, odom_topic, self.odom_cb, 10)
        self.create_subscription(LaserScan, scan_topic, self.lidar_cb, 10)
        self.create_subscription(Int8, estop_status_topic, self.estop_status_cb, 10)
        self.create_subscription(String, mission_command_topic, self.mission_command_cb, 10)
        self.create_subscription(Path, '/plan', self.planned_path_cb, 10)  # Monitor NAV2 planning
        if self.use_gazebo_tf_pose:
            self.create_subscription(TFMessage, self.gazebo_tf_topic, self.gazebo_tf_cb, 10)

        # ---- NAV2 Integration ----
        if self.use_nav2:
            self.nav2_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
            self.get_logger().info('NAV2 integration enabled. Waiting for navigate_to_pose action server...')
        else:
            self.nav2_client = None

        self._nav2_goal_handle = None
        self._send_goal_future = None

        self.drive_mode = DRIVE_MODE.AUTO
        self.auto_state = AUTO_STATE.INITIAL_TURN
        self.current_yaw = 0.0
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_orientation = None
        self.map_origin_x = None if self.center_arena_on_start else self.configured_arena_origin_x
        self.map_origin_y = None if self.center_arena_on_start else self.configured_arena_origin_y
        self.have_odom = False
        self.last_odom_time = None
        self.have_gazebo_tf_pose = False
        self.last_gazebo_tf_time = None
        self._reported_gazebo_tf_pose = False
        self.pose_source = 'odom'
        self.target_yaw = None
        self.start_position = (0.0, 0.0)
        self.drive_distance = 0.0
        self.wandering_target_yaw = None

        self.turn_start_time = None
        self.turn_start_yaw = None
        self.reverse_start_time = 0.0
        self.obstacle_maneuver_start_time = 0.0
        self.transition_stop_end_time = 0.0
        self.last_linear = 0.0
        self.last_angular = 0.0
        self.trigger = False
        self.emergency_stop = False
        self.external_estop_status = ESTOP_CLEAR
        self.external_waypoint_active = False
        self.reached_home = False
        self.coverage_active = False  # NAV2 coverage in progress
        self.nav2_path = None  # Planned path from NAV2
        self._last_auto_button = False
        self._last_manual_button = False
        self._last_stop_button = False
        self._last_published_state = None

        self.front_min_distance = float('inf')
        self.left_min_distance = float('inf')
        self.right_min_distance = float('inf')

        self.mutex = threading.Lock()
        self.create_timer(0.1, self.control_loop)

        self.get_logger().info('Pioneer control node started')
        self.get_logger().info(f'Joy: {joy_topic}, scan: {scan_topic}, odom: {odom_topic}, cmd_vel: {cmd_vel_topic}')
        self.get_logger().info(
            f'Speeds: forward={self.forward_speed:.2f} m/s, reverse={self.reverse_speed:.2f} m/s, '
            f'turn={self.turn_speed_deg:.1f} deg/s'
        )
        if self.use_nav2:
            self.get_logger().info('NAV2 coverage enabled for lawnmower mapping.')

    # ------------------------------------------------------------------ #
    #  MISSION COMMAND & ESTOP CALLBACKS                                  #
    # ------------------------------------------------------------------ #

    def mission_command_cb(self, msg: String):
        """Handle GUI mission commands: start_wandering, go_home, drive_waypoints, start_coverage"""
        command = msg.data.strip().lower()
        with self.mutex:
            if command == 'start_wandering':
                self.external_waypoint_active = False
                self.coverage_active = False
                self.reached_home = False
                if self.external_estop_status == ESTOP_ACTIVE:
                    self.get_logger().warn('Ignoring start_wandering command while external e-stop is active.')
                    return
                self.emergency_stop = False
                self.drive_mode = DRIVE_MODE.AUTO
                self.auto_state = AUTO_STATE.WANDERING_TURN
                self.target_yaw = None
                self.start_position = (self.current_x, self.current_y)
                self.transition_stop_end_time = time.time() + TRANSITION_STOP_DURATION
                self.publish_twist(0.0, 0.0)
                self.publish_robot_state()
                self.get_logger().info('GUI command: Start Wandering accepted. Resuming autonomous mapping.')

            elif command == 'go_home':
                self.external_waypoint_active = False
                self.coverage_active = False
                self.reached_home = False
                if self.external_estop_status == ESTOP_ACTIVE:
                    self.get_logger().warn('Ignoring go_home command while external e-stop is active.')
                    return
                self.emergency_stop = False
                self.drive_mode = DRIVE_MODE.AUTO
                self.auto_state = AUTO_STATE.RETURN_TO_CENTER
                self.target_yaw = None
                self.transition_stop_end_time = time.time() + TRANSITION_STOP_DURATION
                self.publish_twist(0.0, 0.0)
                self.publish_robot_state()
                self.get_logger().info('GUI command: Go Home accepted. Returning to map center with obstacle avoidance.')

            elif command == 'drive_waypoints':
                self.external_waypoint_active = True
                self.coverage_active = False
                self.reached_home = False
                self.drive_mode = DRIVE_MODE.MANUAL
                self.publish_twist(0.0, 0.0)
                self.publish_robot_state()
                self.get_logger().info('GUI command: Paused for waypoint driving.')

            elif command == 'stop_coverage':
                if self.coverage_active:
                    self.cancel_coverage_goal()
                    self.coverage_active = False
                    self.reached_home = False
                    self.drive_mode = DRIVE_MODE.AUTO
                    self.auto_state = AUTO_STATE.WANDERING_TURN
                    self.publish_twist(0.0, 0.0)
                    self.publish_robot_state()
                    self.get_logger().info('GUI command: Coverage planning cancelled. Returning to wandering mode.')
                else:
                    self.get_logger().info('GUI command: No active coverage to cancel.')

            elif command == 'start_lawnmower' or command == 'start_coverage':
                if not self.use_nav2:
                    self.get_logger().warn('NAV2 not enabled. Cannot start coverage planning.')
                    return
                if self.external_estop_status == ESTOP_ACTIVE:
                    self.get_logger().warn('Ignoring coverage command while external e-stop is active.')
                    return
                self.external_waypoint_active = True
                self.coverage_active = True
                self.reached_home = False
                self.emergency_stop = False
                self.drive_mode = DRIVE_MODE.MANUAL  # Yield control to NAV2
                self.auto_state = AUTO_STATE.COVERAGE_PLANNING
                self.publish_twist(0.0, 0.0)
                self.publish_robot_state()
                self.get_logger().info('GUI command: Starting NAV2 coverage planning for lawnmower pattern.')
                # Send goal to Nav2 to trigger coverage planning
                self.send_coverage_goal()

    def estop_status_cb(self, msg: Int8):
        """Handle external LiDAR e-stop status"""
        with self.mutex:
            previous = self.external_estop_status
            self.external_estop_status = int(msg.data)
            if self.external_estop_status == ESTOP_ACTIVE:
                self.emergency_stop = True
                self.drive_mode = DRIVE_MODE.MANUAL
                if self.coverage_active:
                    self.cancel_coverage_goal()
                self.coverage_active = False
                self.publish_twist(0.0, 0.0)
                if previous != ESTOP_ACTIVE:
                    self.get_logger().error('External LiDAR E-STOP active. Motion halted. Cancelling coverage.')
            elif self.external_estop_status == ESTOP_WARNING:
                self.publish_twist(0.0, 0.0)
                if previous != ESTOP_WARNING:
                    self.get_logger().warn('External LiDAR warning active. Pausing motion.')
            elif previous != ESTOP_CLEAR:
                self.get_logger().info('External LiDAR e-stop clear.')

    def planned_path_cb(self, msg: Path):
        """Monitor NAV2 planned path"""
        with self.mutex:
            self.nav2_path = msg
            if self.coverage_active and self.auto_state == AUTO_STATE.COVERAGE_PLANNING:
                self.get_logger().info(f'NAV2 coverage path received with {len(msg.poses)} waypoints.')
                self.auto_state = AUTO_STATE.COVERAGE_EXECUTING
                self.get_logger().info('Coverage execution started. Control yielded to NAV2.')

    def send_coverage_goal(self):
        """Send a coverage goal to Nav2's navigate_to_pose action server"""
        if self.nav2_client is None:
            self.get_logger().error('NAV2 client not initialized!')
            return

        # Wait for action server (with timeout)
        if not self.nav2_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error('Nav2 navigate_to_pose action server not available! Is Nav2 running?')
            self.coverage_active = False
            return

        # Create goal: send robot to arena center for coverage
        from geometry_msgs.msg import PoseStamped, Quaternion

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()

        # Goal is to arrive at map center (arena_origin) with default orientation
        if self.map_origin_x is not None and self.map_origin_y is not None:
            goal.pose.pose.position.x = self.map_origin_x
            goal.pose.pose.position.y = self.map_origin_y
        else:
            # Fallback to 0,0 if not centered
            goal.pose.pose.position.x = 0.0
            goal.pose.pose.position.y = 0.0

        goal.pose.pose.position.z = 0.0
        goal.pose.pose.orientation.x = 0.0
        goal.pose.pose.orientation.y = 0.0
        goal.pose.pose.orientation.z = 0.0
        goal.pose.pose.orientation.w = 1.0

        # Send goal asynchronously
        self._send_goal_future = self.nav2_client.send_goal_async(
            goal,
            feedback_callback=self.coverage_feedback_cb
        )
        # When goal response received, call goal_response_cb
        self._send_goal_future.add_done_callback(self.goal_response_cb)

        self.get_logger().info(
            f'Coverage goal sent to Nav2 (target: {goal.pose.pose.position.x:.2f}, '
            f'{goal.pose.pose.position.y:.2f}). Waiting for execution...'
        )

    def goal_response_cb(self, future):
        """Callback when Nav2 responds to coverage goal request"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Nav2 rejected coverage goal! Check Nav2 parameters and map.')
            with self.mutex:
                self.coverage_active = False
                self.auto_state = AUTO_STATE.WANDERING_TURN
                self.publish_twist(0.0, 0.0)
            return

        self.get_logger().info('✓ Nav2 accepted coverage goal. Planner now computing lawnmower path...')
        self._nav2_goal_handle = goal_handle

    def coverage_feedback_cb(self, feedback_msg):
        """Monitor Nav2 coverage execution feedback"""
        # Nav2 provides feedback on path following progress
        # This is called repeatedly during execution
        pass

    def cancel_coverage_goal(self):
        """Cancel the active Nav2 coverage goal"""
        if self._nav2_goal_handle is not None:
            self.get_logger().info('Cancelling Nav2 coverage goal...')
            cancel_future = self._nav2_goal_handle.cancel_goal_async()
            cancel_future.add_done_callback(self._cancel_done_cb)
            self._nav2_goal_handle = None
        else:
            self.get_logger().warn('No active Nav2 goal to cancel.')

    def _cancel_done_cb(self, future):
        """Callback when goal cancellation completes"""
        cancel_response = future.result()
        if cancel_response.return_code == cancel_response.ERROR_NONE:
            self.get_logger().info('✓ Nav2 coverage goal cancelled successfully.')
        else:
            self.get_logger().warn(f'Failed to cancel Nav2 goal. Error code: {cancel_response.return_code}')

    # ------------------------------------------------------------------ #
    #  GUI HELPERS                                                         #
    # ------------------------------------------------------------------ #

    def publish_robot_state(self):
        """Publish robot state to /robot_state — feeds GUI state badge."""
        if not self.publish_gui_topics:
            return

        if self.emergency_stop or self.external_estop_status == ESTOP_ACTIVE:
            state = 'ESTOP'
        elif self.external_estop_status == ESTOP_WARNING:
            state = 'STOPPED'
        elif self.coverage_active:
            if self.auto_state == AUTO_STATE.COVERAGE_PLANNING:
                state = 'COVERAGE_PLANNING'
            else:
                state = 'COVERAGE'
        elif self.external_waypoint_active:
            state = 'WAYPOINT'
        elif self.reached_home:
            state = 'REACHED_HOME'
        elif self.drive_mode == DRIVE_MODE.AUTO:
            state = 'MAPPING'
        else:
            state = 'IDLE'

        if state == self._last_published_state:
            return

        self._last_published_state = state
        msg = String()
        msg.data = state
        self.robot_state_pub.publish(msg)
        self.get_logger().info(f'State → {state}')

    def publish_robot_pose(self):
        """Publish current pose to /robot/pose — feeds GUI map arrow."""
        if not self.publish_gui_topics or self.current_orientation is None:
            return
        msg = Pose()
        msg.position.x = self.current_x
        msg.position.y = self.current_y
        msg.position.z = 0.0
        msg.orientation = self.current_orientation
        self.pose_pub.publish(msg)

    def publish_arena_status(self, rel_x, rel_y):
        """Publish arena debug data to /arena_status — feeds GUI arena panel."""
        distance_from_center = math.hypot(rel_x, rel_y)
        outside_x = max(0.0, abs(rel_x) - MAP_HALF_SIZE)
        outside_y = max(0.0, abs(rel_y) - MAP_HALF_SIZE)
        outside_distance = math.hypot(outside_x, outside_y)
        clearance_to_edge = min(MAP_HALF_SIZE - abs(rel_x), MAP_HALF_SIZE - abs(rel_y))

        self.get_logger().info(
            f'Arena status: state={self.auto_state.name} '
            f'rel=({rel_x:.2f}, {rel_y:.2f})m '
            f'center_dist={distance_from_center:.2f}m '
            f'edge_clearance={clearance_to_edge:.2f}m '
            f'outside={outside_distance:.2f}m '
            f'yaw={self.current_yaw:.1f}deg '
            f'pose_source={self.pose_source}',
            throttle_duration_sec=1.0
        )

        if self.publish_gui_topics:
            msg = String()
            msg.data = json.dumps({
                'state': self.auto_state.name,
                'rel_x': round(rel_x, 2),
                'rel_y': round(rel_y, 2),
                'center_dist': round(distance_from_center, 2),
                'edge_clearance': round(clearance_to_edge, 2),
                'outside': round(outside_distance, 2),
                'yaw': round(self.current_yaw, 1),
                'pose_source': self.pose_source,
            })
            self.arena_status_pub.publish(msg)

    # ------------------------------------------------------------------ #
    #  CALLBACKS                                                           #
    # ------------------------------------------------------------------ #

    def joy_cb(self, msg: Joy):
        with self.mutex:
            max_button = max(self.joy_auto_button, self.joy_manual_button, self.joy_stop_button)
            if max_button >= len(msg.buttons):
                self.get_logger().warn(
                    f'Button index {max_button} out of range (controller has {len(msg.buttons)} buttons).',
                    throttle_duration_sec=5.0
                )
                return

            manual_pressed = bool(msg.buttons[self.joy_manual_button])
            auto_pressed = bool(msg.buttons[self.joy_auto_button])
            stop_pressed = bool(msg.buttons[self.joy_stop_button])

            if stop_pressed and not self._last_stop_button:
                self.emergency_stop = True
                self.drive_mode = DRIVE_MODE.MANUAL
                if self.coverage_active:
                    self.cancel_coverage_goal()
                self.coverage_active = False
                self.publish_twist(0.0, 0.0)
                self.publish_robot_state()
                self.get_logger().warn('PS4 Square pressed: emergency stop latched')

            if manual_pressed and not self._last_manual_button:
                self.emergency_stop = False
                self.drive_mode = DRIVE_MODE.MANUAL
                if self.coverage_active:
                    self.cancel_coverage_goal()
                self.coverage_active = False
                self.publish_twist(0.0, 0.0)
                self.publish_robot_state()
                self.get_logger().info('PS4 X pressed: control node paused for joystick/manual control')

            if auto_pressed and not self._last_auto_button:
                self.emergency_stop = False
                self.coverage_active = False
                self.drive_mode = DRIVE_MODE.AUTO
                self.auto_state = AUTO_STATE.INITIAL_TURN
                if self.have_odom and self.center_arena_on_start:
                    self.map_origin_x = self.current_x
                    self.map_origin_y = self.current_y
                self.target_yaw = None
                self.turn_start_time = None
                self.turn_start_yaw = None
                self.publish_twist(0.0, 0.0)
                self.publish_robot_state()
                self.get_logger().info('PS4 O pressed: autonomous mode activated')

            self._last_manual_button = manual_pressed
            self._last_auto_button = auto_pressed
            self._last_stop_button = stop_pressed

            if self.emergency_stop:
                self.publish_twist(0.0, 0.0)
                return

            if self.drive_mode == DRIVE_MODE.MANUAL and self.external_manual_control:
                return

            deadman_pressed = False
            if self.joy_deadman_axis < len(msg.axes):
                deadman_pressed = msg.axes[self.joy_deadman_axis] < -0.9

            if self.drive_mode == DRIVE_MODE.MANUAL:
                if not deadman_pressed:
                    if self.trigger:
                        self.publish_twist(0.0, 0.0)
                        self.get_logger().warn('Deadman released! Stopping.')
                    self.trigger = False
                    return
                self.trigger = True

                lin_input = msg.axes[1] if len(msg.axes) > 1 else 0.0
                ang_input = msg.axes[3] if len(msg.axes) > 3 else 0.0
                linear = -(lin_input ** 3) * 0.8
                angular = -(ang_input ** 3) * 1.5
                if abs(linear - self.last_linear) > 0.1:
                    linear = self.last_linear + math.copysign(0.1, linear - self.last_linear)
                if abs(angular - self.last_angular) > 0.15:
                    angular = self.last_angular + math.copysign(0.15, angular - self.last_angular)
                self.last_linear = linear
                self.last_angular = angular
                self.publish_twist(-linear, -angular)

    def lidar_cb(self, msg: LaserScan):
        with self.mutex:
            ranges = msg.ranges
            if not ranges:
                self.front_min_distance = float('inf')
                self.left_min_distance = float('inf')
                self.right_min_distance = float('inf')
                return

            front = []
            left = []
            right = []
            front_half = self.lidar_front_angle_fov / 2.0
            side_width = self.lidar_side_angle_fov

            for i, reading in enumerate(ranges):
                if not math.isfinite(reading) or reading <= max(msg.range_min, self.lidar_self_filter_min_range):
                    continue
                if msg.range_max > 0.0 and reading > msg.range_max:
                    continue

                angle = msg.angle_min + i * msg.angle_increment
                angle = math.atan2(math.sin(angle), math.cos(angle))

                if abs(angle) <= front_half:
                    front.append(reading)
                elif front_half < angle <= front_half + side_width:
                    left.append(reading)
                elif -front_half - side_width <= angle < -front_half:
                    right.append(reading)

            self.front_min_distance = min(front) if front else float('inf')
            self.left_min_distance = min(left) if left else float('inf')
            self.right_min_distance = min(right) if right else float('inf')

    def odom_cb(self, msg: Odometry):
        with self.mutex:
            if self.use_gazebo_tf_pose and self.have_gazebo_tf_pose:
                self.last_odom_time = time.time()
                return
            self.current_x = msg.pose.pose.position.x
            self.current_y = msg.pose.pose.position.y
            self.current_yaw = quaternion_to_yaw(msg.pose.pose.orientation)
            self.current_orientation = msg.pose.pose.orientation
            self.pose_source = 'odom'
            if self.center_arena_on_start and (self.map_origin_x is None or self.map_origin_y is None):
                self.map_origin_x = self.current_x
                self.map_origin_y = self.current_y
                self.get_logger().info(
                    f'15x15 arena centered at odom ({self.map_origin_x:.2f}, {self.map_origin_y:.2f}).'
                )
            self.have_odom = True
            self.last_odom_time = time.time()
            self.publish_robot_pose()

    def gazebo_tf_cb(self, msg: TFMessage):
        with self.mutex:
            if not msg.transforms:
                return

            selected = None
            match = self.gazebo_tf_frame_match
            for transform in msg.transforms:
                child = transform.child_frame_id or ''
                parent = transform.header.frame_id or ''
                if match in child or match in parent:
                    selected = transform
                    break

            if selected is None and (len(msg.transforms) == 1 or self.gazebo_tf_allow_unmatched):
                selected = msg.transforms[0]
                if len(msg.transforms) > 1:
                    self.get_logger().warn(
                        f'Gazebo TF frames are unnamed; using first transform from {self.gazebo_tf_topic}.',
                        throttle_duration_sec=5.0
                    )

            if selected is None:
                frames = ', '.join(
                    (t.child_frame_id or t.header.frame_id or '<blank>') for t in msg.transforms[:8]
                )
                self.get_logger().warn(
                    f'Gazebo TF topic active, but no frame matched "{match}". Frames seen: {frames}',
                    throttle_duration_sec=5.0
                )
                return

            self.current_x = selected.transform.translation.x
            self.current_y = selected.transform.translation.y
            self.current_yaw = quaternion_to_yaw(selected.transform.rotation)
            self.current_orientation = selected.transform.rotation
            self.pose_source = 'gazebo_tf'
            if self.center_arena_on_start and (self.map_origin_x is None or self.map_origin_y is None):
                self.map_origin_x = self.current_x
                self.map_origin_y = self.current_y
                self.get_logger().info(
                    f'15x15 arena centered at Gazebo world pose ({self.map_origin_x:.2f}, {self.map_origin_y:.2f}).'
                )

            self.have_gazebo_tf_pose = True
            self.have_odom = True
            self.last_gazebo_tf_time = time.time()
            self.last_odom_time = self.last_gazebo_tf_time
            self.publish_robot_pose()

            if not self._reported_gazebo_tf_pose:
                self._reported_gazebo_tf_pose = True
                frame = selected.child_frame_id or selected.header.frame_id or '<blank>'
                self.get_logger().info(f'Using Gazebo world pose from {self.gazebo_tf_topic}, frame "{frame}".')

    # ------------------------------------------------------------------ #
    #  CONTROL LOOP                                                        #
    # ------------------------------------------------------------------ #

    def relative_position(self):
        if self.map_origin_x is None or self.map_origin_y is None:
            return 0.0, 0.0
        return self.current_x - self.map_origin_x, self.current_y - self.map_origin_y

    def control_loop(self):
        with self.mutex:
            self.publish_robot_state()

            if self.emergency_stop:
                self.publish_twist(0.0, 0.0)
                return

            if self.external_estop_status == ESTOP_ACTIVE:
                self.emergency_stop = True
                self.drive_mode = DRIVE_MODE.MANUAL
                self.coverage_active = False
                self.publish_twist(0.0, 0.0)
                return

            if self.external_estop_status == ESTOP_WARNING:
                self.publish_twist(0.0, 0.0)
                return

            # ---- NAV2 Coverage in progress: yield control only after safety gates ----
            if self.coverage_active and self.auto_state == AUTO_STATE.COVERAGE_EXECUTING:
                if self.front_min_distance < EMERGENCY_STOP_DISTANCE:
                    self.get_logger().error('EMERGENCY STOP during coverage! Obstacle detected.')
                    self.emergency_stop = True
                    self.coverage_active = False
                    self.publish_twist(0.0, 0.0)
                return

            if self.drive_mode != DRIVE_MODE.AUTO:
                return

            if not self.have_odom:
                self.publish_twist(0.0, 0.0)
                self.get_logger().warn('Waiting for odometry before autonomous movement.', throttle_duration_sec=2.0)
                return

            if self.last_odom_time is None or time.time() - self.last_odom_time > ODOM_STALE_TIMEOUT:
                self.publish_twist(0.0, 0.0)
                self.get_logger().warn('Odometry is stale; stopping autonomous movement.', throttle_duration_sec=2.0)
                return

            if self.front_min_distance < EMERGENCY_STOP_DISTANCE:
                self.get_logger().warn(f'EMERGENCY STOP! Obstacle at {self.front_min_distance:.2f}m.')
                self.publish_twist(0.0, 0.0)
                if self.auto_state == AUTO_STATE.RETURN_TO_CENTER:
                    self.handle_return_obstacle_avoidance(emergency=True)
                    return
                if self.auto_state not in [AUTO_STATE.OBSTACLE_REVERSE, AUTO_STATE.OBSTACLE_TURN,
                                           AUTO_STATE.BOUNDARY_REVERSE, AUTO_STATE.BOUNDARY_ESCAPE_TURN,
                                           AUTO_STATE.BOUNDARY_ESCAPE_DRIVE, AUTO_STATE.RETURN_TO_CENTER]:
                    self.auto_state = AUTO_STATE.OBSTACLE_REVERSE
                    self.obstacle_maneuver_start_time = time.time()
                return

            if self.auto_state == AUTO_STATE.OBSTACLE_REVERSE:
                self.handle_obstacle_reverse()
                return
            if self.auto_state == AUTO_STATE.OBSTACLE_TURN:
                self.handle_obstacle_turn()
                return

            if time.time() < self.transition_stop_end_time:
                self.publish_twist(0.0, 0.0)
                return

            rel_x, rel_y = self.relative_position()
            self.publish_arena_status(rel_x, rel_y)

            if not (-MAP_HALF_SIZE <= rel_x <= MAP_HALF_SIZE and -MAP_HALF_SIZE <= rel_y <= MAP_HALF_SIZE) and self.auto_state != AUTO_STATE.RETURN_TO_CENTER:
                self.get_logger().warn(f'Robot outside 15x15 arena at relative ({rel_x:.2f}, {rel_y:.2f}). Returning to start center.')
                self.auto_state = AUTO_STATE.RETURN_TO_CENTER
                self.target_yaw = None
                self.publish_twist(0.0, 0.0)
                self.transition_stop_end_time = time.time() + TRANSITION_STOP_DURATION
                return

            if self.auto_state not in [AUTO_STATE.INITIAL_TURN, AUTO_STATE.RETURN_TO_CENTER] and self.near_boundary(BOUNDARY_BUFFER) and self.auto_state not in [AUTO_STATE.BOUNDARY_REVERSE, AUTO_STATE.BOUNDARY_ESCAPE_TURN, AUTO_STATE.BOUNDARY_ESCAPE_DRIVE]:
                self.get_logger().warn(f'HARD BOUNDARY hit at relative ({rel_x:.2f}, {rel_y:.2f}) with yaw {self.current_yaw:.1f} deg. Initiating boundary REVERSE.')
                self.auto_state = AUTO_STATE.BOUNDARY_REVERSE
                self.transition_stop_end_time = time.time() + TRANSITION_STOP_DURATION
                self.reverse_start_time = time.time()
                return

            if self.front_min_distance < OBSTACLE_BUFFER and self.auto_state not in [AUTO_STATE.BOUNDARY_REVERSE, AUTO_STATE.BOUNDARY_ESCAPE_TURN, AUTO_STATE.BOUNDARY_ESCAPE_DRIVE, AUTO_STATE.RETURN_TO_CENTER]:
                self.get_logger().info(f'Obstacle detected in front (LIDAR) at {self.front_min_distance:.2f}m. Initiating avoidance.')
                self.auto_state = AUTO_STATE.OBSTACLE_REVERSE
                self.obstacle_maneuver_start_time = time.time()
                self.publish_twist(0.0, 0.0)
                self.transition_stop_end_time = time.time() + TRANSITION_STOP_DURATION
                return

            if self.auto_state == AUTO_STATE.INITIAL_TURN:
                self.handle_initial_turn()
            elif self.auto_state == AUTO_STATE.INITIAL_DRIVE:
                self.handle_initial_drive()
            elif self.auto_state == AUTO_STATE.WANDERING_TURN:
                self.handle_wandering_turn()
            elif self.auto_state == AUTO_STATE.WANDERING_DRIVE:
                self.handle_wandering_drive()
            elif self.auto_state == AUTO_STATE.BOUNDARY_REVERSE:
                self.handle_boundary_reverse()
            elif self.auto_state == AUTO_STATE.BOUNDARY_ESCAPE_TURN:
                self.handle_boundary_escape_turn()
            elif self.auto_state == AUTO_STATE.BOUNDARY_ESCAPE_DRIVE:
                self.handle_boundary_escape_drive()
            elif self.auto_state == AUTO_STATE.RETURN_TO_CENTER:
                self.handle_return_to_center()
            elif self.auto_state == AUTO_STATE.COVERAGE_PLANNING:
                # Waiting for NAV2 to compute and execute coverage path
                # Transition to COVERAGE_EXECUTING happens when /plan topic is published
                self.get_logger().info(
                    'Awaiting NAV2 coverage plan... (watch /plan topic for path visualization)',
                    throttle_duration_sec=5.0
                )

    # ------------------------------------------------------------------ #
    #  STATE HANDLERS (existing code - no changes needed)                  #
    # ------------------------------------------------------------------ #

    def handle_obstacle_reverse(self):
        if time.time() - self.obstacle_maneuver_start_time < OBSTACLE_REVERSE_DURATION:
            self.publish_twist(self.reverse_speed, 0.0)
        else:
            self.publish_twist(0.0, 0.0)
            self.transition_stop_end_time = time.time() + TRANSITION_STOP_DURATION
            self.auto_state = AUTO_STATE.OBSTACLE_TURN
            self.obstacle_maneuver_start_time = time.time()
            turn_angle = 90 if self.left_min_distance > self.right_min_distance else -90
            self.target_yaw = (self.current_yaw + turn_angle) % 360
            self._start_turn_timer()
            self.get_logger().info(f'Finished obstacle reverse. Initiating obstacle turn to {self.target_yaw:.1f}°.')

    def handle_obstacle_turn(self):
        angle_diff = signed_angle_diff(self.target_yaw, self.current_yaw)
        abs_angle_diff = abs(angle_diff)
        if abs_angle_diff < ANGLE_TOLERANCE or self._turn_timed_out():
            self.publish_twist(0.0, 0.0)
            self.transition_stop_end_time = time.time() + TRANSITION_STOP_DURATION
            self.auto_state = AUTO_STATE.WANDERING_DRIVE
            self.target_yaw = None
            self.turn_start_time = None
            self.turn_start_yaw = None
            self.get_logger().info('Finished obstacle turn. Resuming wandering drive.')
        else:
            angular_speed = math.copysign(math.radians(self.turn_speed_deg), angle_diff)
            self.publish_twist(0.0, angular_speed)

    def handle_initial_turn(self):
        if self.target_yaw is None:
            self.target_yaw = (self.current_yaw + INITIAL_TURN_ANGLE) % 360
            self._start_turn_timer()
            self.get_logger().info(f'Initial turn to {self.target_yaw:.1f}° from {self.current_yaw:.1f}°')
            self.publish_twist(0.0, 0.0)
            self.transition_stop_end_time = time.time() + TRANSITION_STOP_DURATION
            return
        self.execute_smooth_turn(AUTO_STATE.INITIAL_DRIVE)

    def handle_initial_drive(self):
        distance = math.hypot(self.current_x - self.start_position[0], self.current_y - self.start_position[1])
        if distance >= INITIAL_DRIVE_DISTANCE:
            self.auto_state = AUTO_STATE.WANDERING_TURN
            self.transition_stop_end_time = time.time() + TRANSITION_STOP_DURATION
            self.target_yaw = None
            self.get_logger().info('Initial drive complete. Starting wandering turn.')
        else:
            self.publish_twist(self.forward_speed, 0.0)

    def handle_wandering_turn(self):
        if self.target_yaw is None:
            if self.wandering_target_yaw is not None:
                self.target_yaw = self.wandering_target_yaw
                self.wandering_target_yaw = None
                self.get_logger().info(f'Wandering turn (predictive) to {self.target_yaw:.1f}° from {self.current_yaw:.1f}°.')
            else:
                random_angle = random.uniform(-MAX_WANDERING_TURN_ANGLE, MAX_WANDERING_TURN_ANGLE)
                self.target_yaw = (self.current_yaw + random_angle) % 360
                self.get_logger().info(f'Wandering turn (random) to {self.target_yaw:.1f}° from {self.current_yaw:.1f}°.')
            self._start_turn_timer()
            self.drive_distance = random.uniform(MIN_DRIVE_DISTANCE, MAX_DRIVE_DISTANCE)
            self.publish_twist(0.0, 0.0)
            self.transition_stop_end_time = time.time() + TRANSITION_STOP_DURATION
            return
        self.execute_smooth_turn(AUTO_STATE.WANDERING_DRIVE)

    def handle_wandering_drive(self):
        if self.predictive_boundary_check():
            self.get_logger().info('Predictive avoidance: Adjusting course')
            self.wandering_target_yaw = self._calculate_predictive_avoidance_yaw()
            self.auto_state = AUTO_STATE.WANDERING_TURN
            self.transition_stop_end_time = time.time() + TRANSITION_STOP_DURATION
            self.publish_twist(0.0, 0.0)
            return
        distance = math.hypot(self.current_x - self.start_position[0], self.current_y - self.start_position[1])
        if distance >= self.drive_distance:
            self.auto_state = AUTO_STATE.WANDERING_TURN
            self.target_yaw = None
            self.transition_stop_end_time = time.time() + TRANSITION_STOP_DURATION
            self.get_logger().info('Wandering drive distance met. Starting next wandering turn.')
        else:
            self.publish_twist(self.forward_speed, 0.0)

    def handle_boundary_reverse(self):
        if time.time() - self.reverse_start_time < REVERSE_DURATION:
            self.publish_twist(self.reverse_speed, 0.0)
        else:
            self.publish_twist(0.0, 0.0)
            self.auto_state = AUTO_STATE.BOUNDARY_ESCAPE_TURN
            self.transition_stop_end_time = time.time() + TRANSITION_STOP_DURATION
            self.target_yaw = None
            self.get_logger().info('Reverse complete. Initiating boundary escape turn.')

    def _start_turn_timer(self):
        self.turn_start_time = time.time()
        self.turn_start_yaw = self.current_yaw

    def _turn_timed_out(self):
        if self.turn_start_time is None or self.turn_start_yaw is None or self.target_yaw is None:
            return False
        requested_angle = abs(signed_angle_diff(self.target_yaw, self.turn_start_yaw))
        expected_duration = requested_angle / max(self.turn_speed_deg, 1.0)
        timeout = max(OBSTACLE_TURN_DURATION, expected_duration + TURN_TIMEOUT_MARGIN)
        if time.time() - self.turn_start_time <= timeout:
            return False
        remaining = abs(signed_angle_diff(self.target_yaw, self.current_yaw))
        self.get_logger().warn(
            f'Turn timed out with {remaining:.1f} deg remaining. Continuing to avoid spinning forever.',
            throttle_duration_sec=2.0
        )
        return True

    def execute_aggressive_turn(self, next_state):
        angle_diff = signed_angle_diff(self.target_yaw, self.current_yaw)
        abs_angle_diff = abs(angle_diff)
        if abs_angle_diff < ANGLE_TOLERANCE or self._turn_timed_out():
            self.publish_twist(0.0, 0.0)
            if self.auto_state != AUTO_STATE.OBSTACLE_TURN:
                self.auto_state = next_state
                self.start_position = (self.current_x, self.current_y)
                self.transition_stop_end_time = time.time() + TRANSITION_STOP_DURATION
                self.target_yaw = None
                self.turn_start_time = None
                self.turn_start_yaw = None
                self.get_logger().info(f'Aggressive turn complete. Transitioning to {next_state.name}.')
            return
        angular_speed = math.copysign(math.radians(self.turn_speed_deg), angle_diff)
        self.publish_twist(0.0, angular_speed)

    def handle_boundary_escape_turn(self):
        if self.target_yaw is None:
            rel_x, rel_y = self.relative_position()
            boundaries = {
                'east': MAP_HALF_SIZE - rel_x,
                'west': rel_x + MAP_HALF_SIZE,
                'north': MAP_HALF_SIZE - rel_y,
                'south': rel_y + MAP_HALF_SIZE,
            }
            closest_boundary = min(boundaries, key=boundaries.get)
            base_angle = {'east': 180, 'west': 0, 'north': 270, 'south': 90}[closest_boundary]
            center_yaw = self._calculate_general_inward_direction()
            angle_offset = random.uniform(MIN_CENTER_ANGLE, MAX_CENTER_ANGLE)
            self.target_yaw = (base_angle * (1 - CENTER_BIAS_STRENGTH) + center_yaw * CENTER_BIAS_STRENGTH + angle_offset) % 360
            self._start_turn_timer()
            self.get_logger().info(f'Boundary escape: Facing {closest_boundary} boundary, turning to {self.target_yaw:.1f}°')
            self.publish_twist(0.0, 0.0)
            self.transition_stop_end_time = time.time() + TRANSITION_STOP_DURATION
            return
        self.execute_aggressive_turn(AUTO_STATE.BOUNDARY_ESCAPE_DRIVE)

    def handle_boundary_escape_drive(self):
        distance = math.hypot(self.current_x - self.start_position[0], self.current_y - self.start_position[1])
        if distance >= BOUNDARY_ESCAPE_DRIVE_DISTANCE:
            self.auto_state = AUTO_STATE.WANDERING_TURN
            self.target_yaw = None
            self.transition_stop_end_time = time.time() + TRANSITION_STOP_DURATION
            self.get_logger().info('Boundary escape drive complete. Now returning to wandering.')
        else:
            self.publish_twist(self.forward_speed, 0.0)

    def handle_return_to_center(self):
        """Return to map center with obstacle avoidance. Called by 'go_home' command."""
        rel_x, rel_y = self.relative_position()
        distance_to_center = math.hypot(rel_x - MAP_CENTER[0], rel_y - MAP_CENTER[1])

        # Check if we've reached home
        if distance_to_center < RETURN_TO_CENTER_MIN_DISTANCE:
            self.get_logger().info(
                f'REACHED_HOME - returned to map center. '
                f'pose=({self.current_x:.2f}, {self.current_y:.2f}, yaw={self.current_yaw:.1f} deg), '
                f'rel=({rel_x:.2f}, {rel_y:.2f}), center_error={distance_to_center:.2f}m. '
                f'Waiting for Start Wandering or Drive Waypoints.'
            )
            self.reached_home = True
            self.drive_mode = DRIVE_MODE.MANUAL
            self.target_yaw = None
            self.transition_stop_end_time = time.time() + TRANSITION_STOP_DURATION
            self.publish_twist(0.0, 0.0)
            self.publish_robot_state()
            return

        # Check for obstacles while returning home
        if self.front_min_distance < OBSTACLE_BUFFER:
            self.handle_return_obstacle_avoidance()
            return

        # Calculate target heading to center
        vec_x = MAP_CENTER[0] - rel_x
        vec_y = MAP_CENTER[1] - rel_y
        target_yaw_to_center_rad = math.atan2(vec_y, vec_x)
        target_yaw_to_center_deg = math.degrees(target_yaw_to_center_rad)
        target_yaw_to_center_norm = (target_yaw_to_center_deg + 360) % 360

        # Update target heading if needed
        if self.target_yaw is None or abs((self.target_yaw - target_yaw_to_center_norm + 360) % 360) > ANGLE_TOLERANCE / 2:
            self.target_yaw = target_yaw_to_center_norm
            self.get_logger().info(f'Returning to center. Recalculating target yaw to {self.target_yaw:.1f}°.')
            if self.transition_stop_end_time <= time.time():
                self.publish_twist(0.0, 0.0)
                self.transition_stop_end_time = time.time() + 0.1

        # Execute turn + drive toward center
        angle_diff = signed_angle_diff(self.target_yaw, self.current_yaw)
        abs_angle_diff = abs(angle_diff)

        if abs_angle_diff > ANGLE_TOLERANCE:
            angular_speed = math.copysign(math.radians(self.return_to_center_turn_speed_deg), angle_diff)
            linear_speed = self.return_to_center_speed * 0.1
        else:
            angular_speed = 0.0
            linear_speed = self.return_to_center_speed

        self.publish_twist(linear_speed, angular_speed)

    def handle_return_obstacle_avoidance(self, emergency=False):
        """Obstacle avoidance while returning to center."""
        if self.left_min_distance > self.right_min_distance:
            turn_direction = 1.0
            side_name = 'left'
        else:
            turn_direction = -1.0
            side_name = 'right'

        angular_speed = turn_direction * math.radians(self.return_to_center_turn_speed_deg)

        if emergency:
            linear_speed = self.reverse_speed * 0.6
            self.get_logger().warn(
                f'Go Home obstacle avoidance: obstacle {self.front_min_distance:.2f}m ahead; '
                f'reversing and turning {side_name}.',
                throttle_duration_sec=1.0
            )
        else:
            linear_speed = 0.0
            self.get_logger().info(
                f'Go Home obstacle avoidance: obstacle {self.front_min_distance:.2f}m ahead; '
                f'turning {side_name} toward clearer side.',
                throttle_duration_sec=1.0
            )

        self.target_yaw = None
        self.publish_twist(linear_speed, angular_speed)

    def near_boundary(self, buffer_distance):
        rel_x, rel_y = self.relative_position()
        return (rel_x < -MAP_HALF_SIZE + buffer_distance or
                rel_x > MAP_HALF_SIZE - buffer_distance or
                rel_y < -MAP_HALF_SIZE + buffer_distance or
                rel_y > MAP_HALF_SIZE - buffer_distance)

    def predictive_boundary_check(self):
        if self.near_boundary(PREDICTIVE_BUFFER) and not self.near_boundary(BOUNDARY_BUFFER):
            return self.is_heading_towards_boundary(PREDICTIVE_BUFFER)
        return False

    def _calculate_predictive_avoidance_yaw(self):
        angle_to_center = self._calculate_general_inward_direction()
        rel_x, rel_y = self.relative_position()
        yaw_rad = math.radians(self.current_yaw)
        dir_x = math.cos(yaw_rad)
        dir_y = math.sin(yaw_rad)
        escape_angle = None
        if rel_x < -MAP_HALF_SIZE + PREDICTIVE_BUFFER and dir_x < 0:
            escape_angle = (self.current_yaw + random.uniform(90, 180)) % 360
        elif rel_x > MAP_HALF_SIZE - PREDICTIVE_BUFFER and dir_x > 0:
            escape_angle = (self.current_yaw - random.uniform(90, 180)) % 360
        elif rel_y < -MAP_HALF_SIZE + PREDICTIVE_BUFFER and dir_y < 0:
            escape_angle = (self.current_yaw - random.uniform(90, 180)) % 360
        elif rel_y > MAP_HALF_SIZE - PREDICTIVE_BUFFER and dir_y > 0:
            escape_angle = (self.current_yaw + random.uniform(90, 180)) % 360
        if escape_angle is not None:
            angle_diff_to_center = (angle_to_center - escape_angle + 360) % 360
            if angle_diff_to_center > 180:
                angle_diff_to_center -= 360
            return (escape_angle + angle_diff_to_center * 0.2) % 360
        return (self.current_yaw + random.uniform(-MAX_WANDERING_TURN_ANGLE, MAX_WANDERING_TURN_ANGLE)) % 360

    def _calculate_general_inward_direction(self):
        rel_x, rel_y = self.relative_position()
        vec_x = MAP_CENTER[0] - rel_x
        vec_y = MAP_CENTER[1] - rel_y
        angle_rad = math.atan2(vec_y, vec_x)
        return (math.degrees(angle_rad) + 360) % 360

    def is_heading_towards_boundary(self, check_buffer):
        rel_x, rel_y = self.relative_position()
        yaw_rad = math.radians(self.current_yaw)
        dir_x = math.cos(yaw_rad)
        dir_y = math.sin(yaw_rad)
        if rel_y < -MAP_HALF_SIZE + check_buffer and dir_y < 0:
            return True
        if rel_y > MAP_HALF_SIZE - check_buffer and dir_y > 0:
            return True
        if rel_x < -MAP_HALF_SIZE + check_buffer and dir_x < 0:
            return True
        if rel_x > MAP_HALF_SIZE - check_buffer and dir_x > 0:
            return True
        return False

    def execute_smooth_turn(self, next_state):
        angle_diff = signed_angle_diff(self.target_yaw, self.current_yaw)
        abs_angle_diff = abs(angle_diff)
        if abs_angle_diff < ANGLE_TOLERANCE or self._turn_timed_out():
            self.publish_twist(0.0, 0.0)
            self.auto_state = next_state
            self.start_position = (self.current_x, self.current_y)
            self.transition_stop_end_time = time.time() + TRANSITION_STOP_DURATION
            self.target_yaw = None
            self.turn_start_time = None
            self.turn_start_yaw = None
            self.get_logger().info(f'Turn complete. Transitioning to {next_state.name}.')
            return
        if abs_angle_diff > MAX_TURN_ANGLE_FOR_FULL_SPEED:
            turn_factor = 1.0
        else:
            normalized_diff = abs_angle_diff / MAX_TURN_ANGLE_FOR_FULL_SPEED
            turn_factor = max(MIN_TURN_SPEED_FACTOR, normalized_diff ** 1.5)
        angular_speed = math.copysign(self.turn_speed_deg * turn_factor, angle_diff)
        self.publish_twist(0.0, math.radians(angular_speed))

    def publish_twist(self, linear, angular):
        cmd = Twist()
        MIN_LINEAR_CMD = 0.05
        MIN_ANGULAR_CMD = math.radians(5.0)
        cmd.linear.x = float(math.copysign(MIN_LINEAR_CMD, linear) if 0 < abs(linear) < MIN_LINEAR_CMD else linear)
        cmd.angular.z = float(math.copysign(MIN_ANGULAR_CMD, angular) if 0 < abs(angular) < MIN_ANGULAR_CMD else angular)
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
