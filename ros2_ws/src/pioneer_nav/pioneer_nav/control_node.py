#!/usr/bin/env python3
import math
import random
import threading
import time
from enum import Enum

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Joy, LaserScan

# Configuration defaults
INITIAL_TURN_ANGLE = 45
INITIAL_DRIVE_DISTANCE = 5.0
BOUNDARY_BUFFER = 1.0
MAP_SIZE = 15.0
MAP_CENTER = (MAP_SIZE / 2, MAP_SIZE / 2)

OBSTACLE_BUFFER = 0.6
EMERGENCY_STOP_DISTANCE = 0.3
LIDAR_FRONT_ANGLE_FOV = math.radians(60)
LIDAR_SIDE_ANGLE_FOV = math.radians(30)
OBSTACLE_REVERSE_DURATION = 0.7
OBSTACLE_TURN_DURATION = 1.5
PREDICTIVE_BUFFER = 4.0

TURN_SPEED_DEG = 70
FORWARD_SPEED = 0.8
REVERSE_SPEED = -0.7
ANGLE_TOLERANCE = 7.0
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
RETURN_TO_CENTER_SPEED = 0.5
RETURN_TO_CENTER_TURN_SPEED_DEG = 45
RETURN_TO_CENTER_MIN_DISTANCE = 0.5


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


def quaternion_to_yaw(orientation):
    x = orientation.x
    y = orientation.y
    z = orientation.z
    w = orientation.w
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp)) % 360


class ControlNode(Node):
    def __init__(self):
        super().__init__('pioneer_control_node')

        self.declare_parameter('joy_topic', '/joy')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('joy_deadman_axis', 5)
        self.declare_parameter('joy_auto_button', 0)
        self.declare_parameter('joy_manual_button', 1)

        joy_topic = self.get_parameter('joy_topic').value
        scan_topic = self.get_parameter('scan_topic').value
        odom_topic = self.get_parameter('odom_topic').value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').value

        self.joy_deadman_axis = int(self.get_parameter('joy_deadman_axis').value)
        self.joy_auto_button = int(self.get_parameter('joy_auto_button').value)
        self.joy_manual_button = int(self.get_parameter('joy_manual_button').value)

        self.cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)

        self.create_subscription(Joy, joy_topic, self.joy_cb, 10)
        self.create_subscription(Odometry, odom_topic, self.odom_cb, 10)
        self.create_subscription(LaserScan, scan_topic, self.lidar_cb, 10)

        self.drive_mode = DRIVE_MODE.AUTO
        self.auto_state = AUTO_STATE.INITIAL_TURN
        self.current_yaw = 0.0
        self.current_x = 0.0
        self.current_y = 0.0
        self.target_yaw = None
        self.start_position = (0.0, 0.0)
        self.drive_distance = 0.0
        self.wandering_target_yaw = None

        self.turn_start_time = None
        self.reverse_start_time = 0.0
        self.obstacle_maneuver_start_time = 0.0
        self.transition_stop_end_time = 0.0
        self.last_linear = 0.0
        self.last_angular = 0.0
        self.trigger = False

        self.front_min_distance = float('inf')
        self.left_min_distance = float('inf')
        self.right_min_distance = float('inf')

        self.mutex = threading.Lock()
        self.create_timer(0.1, self.control_loop)

        self.get_logger().info('Pioneer control node started')
        self.get_logger().info(f'Joy: {joy_topic}, scan: {scan_topic}, odom: {odom_topic}, cmd_vel: {cmd_vel_topic}')

    def joy_cb(self, msg: Joy):
        with self.mutex:
            if self.joy_auto_button < len(msg.buttons) and msg.buttons[self.joy_auto_button]:
                if self.drive_mode != DRIVE_MODE.AUTO:
                    self.drive_mode = DRIVE_MODE.AUTO
                    self.auto_state = AUTO_STATE.INITIAL_TURN
                    self.target_yaw = None
                    self.get_logger().info('Auto mode activated')

            elif self.joy_manual_button < len(msg.buttons) and msg.buttons[self.joy_manual_button]:
                if self.drive_mode != DRIVE_MODE.MANUAL:
                    self.drive_mode = DRIVE_MODE.MANUAL
                    self.publish_twist(0.0, 0.0)
                    self.get_logger().info('Manual mode activated')

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

            n = len(ranges)
            angle_min = msg.angle_min
            angle_inc = msg.angle_increment
            center_idx = n // 2
            half_front_idx = int((LIDAR_FRONT_ANGLE_FOV / 2) / angle_inc)

            front_indices = [i % n for i in range(center_idx - half_front_idx, center_idx + half_front_idx + 1)]
            self.front_min_distance = min([ranges[i] for i in front_indices if not math.isinf(ranges[i]) and ranges[i] > 0.01] or [float('inf')])

            left_start_idx = int(((LIDAR_FRONT_ANGLE_FOV / 2) - angle_min) / angle_inc)
            left_end_idx = int(((LIDAR_FRONT_ANGLE_FOV / 2 + LIDAR_SIDE_ANGLE_FOV) - angle_min) / angle_inc)
            left_indices = [i % n for i in range(left_start_idx, left_end_idx + 1)]
            self.left_min_distance = min([ranges[i] for i in left_indices if not math.isinf(ranges[i]) and ranges[i] > 0.01] or [float('inf')])

            right_start_idx = int(((-LIDAR_FRONT_ANGLE_FOV / 2) - angle_min) / angle_inc)
            right_end_idx = int(((-LIDAR_FRONT_ANGLE_FOV / 2 - LIDAR_SIDE_ANGLE_FOV) - angle_min) / angle_inc)
            right_indices = [i % n for i in range(right_start_idx, right_end_idx + 1)]
            self.right_min_distance = min([ranges[i] for i in right_indices if not math.isinf(ranges[i]) and ranges[i] > 0.01] or [float('inf')])

    def odom_cb(self, msg: Odometry):
        with self.mutex:
            self.current_x = msg.pose.pose.position.x
            self.current_y = msg.pose.pose.position.y
            self.current_yaw = quaternion_to_yaw(msg.pose.pose.orientation)

    def control_loop(self):
        with self.mutex:
            if self.drive_mode != DRIVE_MODE.AUTO:
                return

            if self.front_min_distance < EMERGENCY_STOP_DISTANCE:
                self.get_logger().warn(f'EMERGENCY STOP! Obstacle at {self.front_min_distance:.2f}m.')
                self.publish_twist(0.0, 0.0)
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

            if not (0 <= self.current_x <= MAP_SIZE and 0 <= self.current_y <= MAP_SIZE) and self.auto_state != AUTO_STATE.RETURN_TO_CENTER:
                self.get_logger().warn(f'Robot outside map bounds at ({self.current_x:.2f}, {self.current_y:.2f}). Returning to center.')
                self.auto_state = AUTO_STATE.RETURN_TO_CENTER
                self.target_yaw = None
                self.publish_twist(0.0, 0.0)
                self.transition_stop_end_time = time.time() + TRANSITION_STOP_DURATION
                return

            if self.auto_state not in [AUTO_STATE.INITIAL_TURN, AUTO_STATE.INITIAL_DRIVE, AUTO_STATE.RETURN_TO_CENTER] and self.near_boundary(BOUNDARY_BUFFER) and self.auto_state not in [AUTO_STATE.BOUNDARY_REVERSE, AUTO_STATE.BOUNDARY_ESCAPE_TURN, AUTO_STATE.BOUNDARY_ESCAPE_DRIVE]:
                self.get_logger().warn(f'HARD BOUNDARY hit at ({self.current_x:.2f}, {self.current_y:.2f}) with yaw {self.current_yaw:.1f}°. Initiating boundary REVERSE.')
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

    def handle_obstacle_reverse(self):
        if time.time() - self.obstacle_maneuver_start_time < OBSTACLE_REVERSE_DURATION:
            self.publish_twist(REVERSE_SPEED, 0.0)
        else:
            self.publish_twist(0.0, 0.0)
            self.transition_stop_end_time = time.time() + TRANSITION_STOP_DURATION
            self.auto_state = AUTO_STATE.OBSTACLE_TURN
            self.obstacle_maneuver_start_time = time.time()
            if self.left_min_distance > self.right_min_distance:
                turn_angle = 90
            else:
                turn_angle = -90
            self.target_yaw = (self.current_yaw + turn_angle) % 360
            self.get_logger().info(f'Finished obstacle reverse. Initiating obstacle turn to {self.target_yaw:.1f}°.')

    def handle_obstacle_turn(self):
        angle_diff = (self.target_yaw - self.current_yaw + 360) % 360
        if angle_diff > 180:
            angle_diff -= 360
        abs_angle_diff = abs(angle_diff)
        if abs_angle_diff < ANGLE_TOLERANCE or (time.time() - self.obstacle_maneuver_start_time > OBSTACLE_TURN_DURATION):
            self.publish_twist(0.0, 0.0)
            self.transition_stop_end_time = time.time() + TRANSITION_STOP_DURATION
            self.auto_state = AUTO_STATE.WANDERING_DRIVE
            self.target_yaw = None
            self.get_logger().info('Finished obstacle turn. Resuming wandering drive.')
        else:
            angular_speed = math.copysign(math.radians(TURN_SPEED_DEG), angle_diff)
            self.publish_twist(0.0, angular_speed)

    def handle_initial_turn(self):
        if self.target_yaw is None:
            self.target_yaw = (self.current_yaw + INITIAL_TURN_ANGLE) % 360
            self.get_logger().info(f'Initial turn to {self.target_yaw:.1f}° from {self.current_yaw:.1f}°')
            self.publish_twist(0.0, 0.0)
            self.transition_stop_end_time = time.time() + TRANSITION_STOP_DURATION
            return
        self.execute_smooth_turn(AUTO_STATE.INITIAL_DRIVE, FORWARD_SPEED)

    def handle_initial_drive(self):
        distance = math.hypot(self.current_x - self.start_position[0], self.current_y - self.start_position[1])
        if distance >= INITIAL_DRIVE_DISTANCE:
            self.auto_state = AUTO_STATE.WANDERING_TURN
            self.transition_stop_end_time = time.time() + TRANSITION_STOP_DURATION
            self.target_yaw = None
            self.get_logger().info('Initial drive complete. Starting wandering turn.')
        else:
            self.publish_twist(FORWARD_SPEED, 0.0)

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
            self.drive_distance = random.uniform(MIN_DRIVE_DISTANCE, MAX_DRIVE_DISTANCE)
            self.publish_twist(0.0, 0.0)
            self.transition_stop_end_time = time.time() + TRANSITION_STOP_DURATION
            return
        self.execute_smooth_turn(AUTO_STATE.WANDERING_DRIVE, FORWARD_SPEED)

    def handle_wandering_drive(self):
        if self.predictive_boundary_check():
            self.get_logger().info('Predictive avoidance: Adjusting course')
            avoidance_angle = self._calculate_predictive_avoidance_yaw()
            self.wandering_target_yaw = avoidance_angle
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
            self.publish_twist(FORWARD_SPEED, 0.0)

    def handle_boundary_reverse(self):
        if time.time() - self.reverse_start_time < REVERSE_DURATION:
            self.publish_twist(REVERSE_SPEED, 0.0)
        else:
            self.publish_twist(0.0, 0.0)
            self.auto_state = AUTO_STATE.BOUNDARY_ESCAPE_TURN
            self.transition_stop_end_time = time.time() + TRANSITION_STOP_DURATION
            self.target_yaw = None
            self.get_logger().info('Reverse complete. Initiating boundary escape turn.')

    def execute_aggressive_turn(self, next_state):
        angle_diff = (self.target_yaw - self.current_yaw + 360) % 360
        if angle_diff > 180:
            angle_diff -= 360
        abs_angle_diff = abs(angle_diff)
        if abs_angle_diff < ANGLE_TOLERANCE:
            self.publish_twist(0.0, 0.0)
            if self.auto_state != AUTO_STATE.OBSTACLE_TURN:
                self.auto_state = next_state
                self.start_position = (self.current_x, self.current_y)
                self.transition_stop_end_time = time.time() + TRANSITION_STOP_DURATION
                self.target_yaw = None
                self.get_logger().info(f'Aggressive turn complete. Transitioning to {next_state.name}.')
            return
        angular_speed = math.copysign(math.radians(TURN_SPEED_DEG), angle_diff)
        self.publish_twist(0.0, angular_speed)

    def handle_boundary_escape_turn(self):
        if self.target_yaw is None:
            boundaries = {
                'east': MAP_SIZE - self.current_x,
                'west': self.current_x,
                'north': MAP_SIZE - self.current_y,
                'south': self.current_y,
            }
            closest_boundary = min(boundaries, key=boundaries.get)
            if closest_boundary == 'east':
                base_angle = 180
            elif closest_boundary == 'west':
                base_angle = 0
            elif closest_boundary == 'north':
                base_angle = 270
            else:
                base_angle = 90
            center_yaw = self._calculate_general_inward_direction()
            angle_offset = random.uniform(MIN_CENTER_ANGLE, MAX_CENTER_ANGLE)
            self.target_yaw = (base_angle * (1 - CENTER_BIAS_STRENGTH) + center_yaw * CENTER_BIAS_STRENGTH + angle_offset) % 360
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
            self.publish_twist(FORWARD_SPEED, 0.0)

    def handle_return_to_center(self):
        distance_to_center = math.hypot(self.current_x - MAP_CENTER[0], self.current_y - MAP_CENTER[1])
        if distance_to_center < RETURN_TO_CENTER_MIN_DISTANCE:
            self.get_logger().info('Successfully returned to map center. Resuming wandering.')
            self.auto_state = AUTO_STATE.WANDERING_TURN
            self.target_yaw = None
            self.transition_stop_end_time = time.time() + TRANSITION_STOP_DURATION
            self.publish_twist(0.0, 0.0)
            return
        vec_x = MAP_CENTER[0] - self.current_x
        vec_y = MAP_CENTER[1] - self.current_y
        target_yaw_to_center_rad = math.atan2(vec_y, vec_x)
        target_yaw_to_center_deg = math.degrees(target_yaw_to_center_rad)
        target_yaw_to_center_norm = (target_yaw_to_center_deg + 360) % 360
        if self.target_yaw is None or abs((self.target_yaw - target_yaw_to_center_norm + 360) % 360) > ANGLE_TOLERANCE / 2:
            self.target_yaw = target_yaw_to_center_norm
            self.get_logger().info(f'Returning to center. Recalculating target yaw to {self.target_yaw:.1f}°.')
            if self.transition_stop_end_time <= time.time():
                self.publish_twist(0.0, 0.0)
                self.transition_stop_end_time = time.time() + 0.1
        angle_diff = (self.target_yaw - self.current_yaw + 360) % 360
        if angle_diff > 180:
            angle_diff -= 360
        abs_angle_diff = abs(angle_diff)
        if abs_angle_diff > ANGLE_TOLERANCE:
            angular_speed = math.copysign(math.radians(RETURN_TO_CENTER_TURN_SPEED_DEG), angle_diff)
            linear_speed = RETURN_TO_CENTER_SPEED * 0.1
        else:
            angular_speed = 0.0
            linear_speed = RETURN_TO_CENTER_SPEED
        self.publish_twist(linear_speed, angular_speed)

    def near_boundary(self, buffer_distance):
        return (self.current_x < buffer_distance or
                self.current_x > MAP_SIZE - buffer_distance or
                self.current_y < buffer_distance or
                self.current_y > MAP_SIZE - buffer_distance)

    def predictive_boundary_check(self):
        if self.near_boundary(PREDICTIVE_BUFFER) and not self.near_boundary(BOUNDARY_BUFFER):
            return self.is_heading_towards_boundary(PREDICTIVE_BUFFER)
        return False

    def _calculate_predictive_avoidance_yaw(self):
        angle_to_center = self._calculate_general_inward_direction()
        yaw_rad = math.radians(self.current_yaw)
        dir_x = math.cos(yaw_rad)
        dir_y = math.sin(yaw_rad)
        escape_angle = None
        if self.current_x < PREDICTIVE_BUFFER and dir_x < 0:
            escape_angle = (self.current_yaw + random.uniform(90, 180)) % 360
        elif self.current_x > MAP_SIZE - PREDICTIVE_BUFFER and dir_x > 0:
            escape_angle = (self.current_yaw - random.uniform(90, 180)) % 360
        elif self.current_y < PREDICTIVE_BUFFER and dir_y < 0:
            escape_angle = (self.current_yaw - random.uniform(90, 180)) % 360
        elif self.current_y > MAP_SIZE - PREDICTIVE_BUFFER and dir_y > 0:
            escape_angle = (self.current_yaw + random.uniform(90, 180)) % 360
        if escape_angle is not None:
            angle_diff_to_center = (angle_to_center - escape_angle + 360) % 360
            if angle_diff_to_center > 180:
                angle_diff_to_center -= 360
            return (escape_angle + angle_diff_to_center * 0.2) % 360
        return (self.current_yaw + random.uniform(-MAX_WANDERING_TURN_ANGLE, MAX_WANDERING_TURN_ANGLE)) % 360

    def _calculate_general_inward_direction(self):
        vec_x = MAP_CENTER[0] - self.current_x
        vec_y = MAP_CENTER[1] - self.current_y
        angle_rad = math.atan2(vec_y, vec_x)
        return (math.degrees(angle_rad) + 360) % 360

    def is_heading_towards_boundary(self, check_buffer):
        yaw_rad = math.radians(self.current_yaw)
        dir_x = math.cos(yaw_rad)
        dir_y = math.sin(yaw_rad)
        if self.current_y < check_buffer and dir_y < 0:
            return True
        if self.current_y > MAP_SIZE - check_buffer and dir_y > 0:
            return True
        if self.current_x < check_buffer and dir_x < 0:
            return True
        if self.current_x > MAP_SIZE - check_buffer and dir_x > 0:
            return True
        return False

    def execute_smooth_turn(self, next_state, linear_speed):
        angle_diff = (self.target_yaw - self.current_yaw + 360) % 360
        if angle_diff > 180:
            angle_diff -= 360
        abs_angle_diff = abs(angle_diff)
        if abs_angle_diff < ANGLE_TOLERANCE:
            self.publish_twist(0.0, 0.0)
            self.auto_state = next_state
            self.start_position = (self.current_x, self.current_y)
            self.transition_stop_end_time = time.time() + TRANSITION_STOP_DURATION
            self.target_yaw = None
            self.get_logger().info(f'Turn complete. Transitioning to {next_state.name}.')
            return
        if abs_angle_diff > MAX_TURN_ANGLE_FOR_FULL_SPEED:
            turn_factor = 1.0
        else:
            normalized_diff = abs_angle_diff / MAX_TURN_ANGLE_FOR_FULL_SPEED
            turn_factor = max(MIN_TURN_SPEED_FACTOR, normalized_diff ** 1.5)
        angular_speed = math.copysign(TURN_SPEED_DEG * turn_factor, angle_diff)
        self.publish_twist(linear_speed, math.radians(angular_speed))

    def publish_twist(self, linear, angular):
        cmd = Twist()
        MIN_LINEAR_CMD = 0.05
        MIN_ANGULAR_CMD = math.radians(5.0)
        if abs(linear) > 0 and abs(linear) < MIN_LINEAR_CMD:
            cmd.linear.x = math.copysign(MIN_LINEAR_CMD, linear)
        else:
            cmd.linear.x = float(linear)
        if abs(angular) > 0 and abs(angular) < MIN_ANGULAR_CMD:
            cmd.angular.z = math.copysign(MIN_ANGULAR_CMD, angular)
        else:
            cmd.angular.z = float(angular)
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
