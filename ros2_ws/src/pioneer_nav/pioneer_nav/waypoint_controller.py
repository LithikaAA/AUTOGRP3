#!/usr/bin/env python3
"""
Basic waypoint follower for Pioneer 3-AT.

Default behaviour:
- waits for /mission_command == "drive_waypoints"
- loads detections from ~/part3_logs/detections_log.jsonl, written by unified_detector
- converts detected objects into goals relative to the robot's home/start pose
- stops a little short of each object, then continues to the next nearest object
- falls back to ros2_ws/maps/latest_obstacle_waypoints.txt if no detections are usable

Fallback waypoint file format:
    target_x target_y [ignored...]

The latest_obstacle_waypoints.txt file has four columns:
    target_rel_x target_rel_y obstacle_rel_x obstacle_rel_y
Only the first two columns are used for driving.
"""

import json
import math
import os
import csv
import heapq
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path as PathMsg
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int8, String
from tf2_msgs.msg import TFMessage


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wrap_to_pi(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def angle_in_sector(angle: float, centre: float, width: float) -> bool:
    return abs(wrap_to_pi(angle - centre)) <= width / 2.0


def yaw_from_quaternion(q) -> float:
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


ESTOP_CLEAR = 0
ESTOP_WARNING = 1
ESTOP_ACTIVE = 2
OBSTACLE_REVERSE_DURATION = 0.7
OBSTACLE_TURN_TIMEOUT = 4.0
OBSTACLE_TURN_ANGLE = math.radians(90.0)
OBSTACLE_TURN_TOLERANCE = math.radians(7.0)


class WaypointController(Node):
    def __init__(self):
        super().__init__("waypoint_controller")

        self.declare_parameter("waypoint_file", "")
        self.declare_parameter("obstacle_waypoint_file", "")
        self.declare_parameter("waypoint_source", "detections_json")
        self.declare_parameter("detection_log_file", "")
        self.declare_parameter("detection_goal_standoff_m", 0.6)
        self.declare_parameter("detection_min_distance_m", 0.05)
        self.declare_parameter("detection_max_age_s", 0.0)
        self.declare_parameter("use_astar", True)
        self.declare_parameter("map_yaml", "")
        self.declare_parameter("binary_map_csv", "")
        self.declare_parameter("astar_obstacle_inflation_m", 0.25)
        self.declare_parameter("astar_waypoint_spacing_m", 0.35)
        self.declare_parameter("coverage_area_size_m", 15.0)
        self.declare_parameter("coverage_boundary_margin_m", 0.75)
        self.declare_parameter("coverage_sweep_spacing_m", 1.5)
        self.declare_parameter("coverage_pattern", "serpentine")
        self.declare_parameter("coverage_probe_radius_ratio", 0.45)
        self.declare_parameter("coverage_scan_spin_s", 3.0)
        self.declare_parameter("coverage_scan_turn_speed", 0.45)
        self.declare_parameter("coverage_adaptive_viewpoints", True)
        self.declare_parameter("coverage_occlusion_probe_offset_m", 2.0)
        self.declare_parameter("coverage_max_adaptive_viewpoints", 4)
        self.declare_parameter("coverage_initial_arc_scan_s", 8.0)
        self.declare_parameter("coverage_initial_arc_linear_speed", 0.08)
        self.declare_parameter("coverage_initial_arc_turn_speed", 0.22)
        self.declare_parameter("pose_topic", "/odom")
        self.declare_parameter("odom_topic", "")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("planned_path_topic", "/planned_path")
        self.declare_parameter("mission_command_topic", "/mission_command")
        self.declare_parameter("robot_state_topic", "/robot_state")
        self.declare_parameter("estop_status_topic", "/estop_status")
        self.declare_parameter("use_gazebo_tf_pose", False)
        self.declare_parameter("gazebo_tf_topic", "/world/pioneer_world/dynamic_pose/info")
        self.declare_parameter("gazebo_tf_frame_match", "pioneer")
        self.declare_parameter("gazebo_tf_allow_unmatched", False)
        self.declare_parameter("relative_to_start", True)
        self.declare_parameter("auto_start", False)
        self.declare_parameter("use_test_waypoint", False)
        self.declare_parameter("goal_tolerance", 0.35)
        self.declare_parameter("waypoint_goal_tolerance", 0.0)
        self.declare_parameter("heading_tolerance_deg", 10.0)
        self.declare_parameter("linear_speed", 0.16)
        self.declare_parameter("waypoint_linear_speed", 0.0)
        self.declare_parameter("slow_linear_speed", 0.05)
        self.declare_parameter("waypoint_slow_linear_speed", 0.0)
        self.declare_parameter("enable_waypoint_obstacle_avoidance", True)
        self.declare_parameter("waypoint_obstacle_linear_speed", 0.0)
        self.declare_parameter("waypoint_obstacle_turn_speed", 0.0)
        self.declare_parameter("front_obstacle_dist_m", 0.6)
        self.declare_parameter("critical_obstacle_dist_m", 0.3)
        self.declare_parameter("waypoint_critical_turn_speed", 0.55)
        self.declare_parameter("waypoint_critical_reverse_speed", -0.25)
        self.declare_parameter("front_obstacle_fov_deg", 90.0)
        self.declare_parameter("side_obstacle_fov_deg", 90.0)
        self.declare_parameter("angular_gain", 1.4)
        self.declare_parameter("max_angular_speed", 0.55)
        self.declare_parameter("control_rate_hz", 10.0)

        self.waypoint_file = str(self.get_parameter("waypoint_file").value)
        obstacle_waypoint_file = str(self.get_parameter("obstacle_waypoint_file").value)
        if obstacle_waypoint_file and not self.waypoint_file:
            self.waypoint_file = obstacle_waypoint_file
        self.waypoint_source = str(self.get_parameter("waypoint_source").value)
        self.detection_log_file = str(self.get_parameter("detection_log_file").value)
        self.detection_goal_standoff_m = max(
            0.0, float(self.get_parameter("detection_goal_standoff_m").value)
        )
        self.detection_min_distance_m = max(
            0.0, float(self.get_parameter("detection_min_distance_m").value)
        )
        self.detection_max_age_s = max(
            0.0, float(self.get_parameter("detection_max_age_s").value)
        )
        self.use_astar = bool(self.get_parameter("use_astar").value)
        self.map_yaml = str(self.get_parameter("map_yaml").value)
        self.binary_map_csv = str(self.get_parameter("binary_map_csv").value)
        self.astar_obstacle_inflation_m = max(
            0.0, float(self.get_parameter("astar_obstacle_inflation_m").value)
        )
        self.astar_waypoint_spacing_m = max(
            0.05, float(self.get_parameter("astar_waypoint_spacing_m").value)
        )
        self.coverage_area_size_m = max(1.0, float(self.get_parameter("coverage_area_size_m").value))
        self.coverage_boundary_margin_m = max(
            0.0, float(self.get_parameter("coverage_boundary_margin_m").value)
        )
        self.coverage_sweep_spacing_m = max(
            0.25, float(self.get_parameter("coverage_sweep_spacing_m").value)
        )
        self.coverage_pattern = str(self.get_parameter("coverage_pattern").value).strip().lower()
        self.coverage_probe_radius_ratio = clamp(
            float(self.get_parameter("coverage_probe_radius_ratio").value), 0.1, 0.9
        )
        self.coverage_scan_spin_s = max(
            0.0, float(self.get_parameter("coverage_scan_spin_s").value)
        )
        self.coverage_scan_turn_speed = abs(
            float(self.get_parameter("coverage_scan_turn_speed").value)
        )
        self.coverage_adaptive_viewpoints = bool(
            self.get_parameter("coverage_adaptive_viewpoints").value
        )
        self.coverage_occlusion_probe_offset_m = max(
            0.5, float(self.get_parameter("coverage_occlusion_probe_offset_m").value)
        )
        self.coverage_max_adaptive_viewpoints = max(
            0, int(self.get_parameter("coverage_max_adaptive_viewpoints").value)
        )
        self.coverage_initial_arc_scan_s = max(
            0.0, float(self.get_parameter("coverage_initial_arc_scan_s").value)
        )
        self.coverage_initial_arc_linear_speed = max(
            0.0, float(self.get_parameter("coverage_initial_arc_linear_speed").value)
        )
        self.coverage_initial_arc_turn_speed = float(
            self.get_parameter("coverage_initial_arc_turn_speed").value
        )
        self.pose_topic = str(self.get_parameter("pose_topic").value)
        odom_topic = str(self.get_parameter("odom_topic").value)
        if odom_topic:
            self.pose_topic = odom_topic
        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.planned_path_topic = str(self.get_parameter("planned_path_topic").value)
        self.mission_command_topic = str(self.get_parameter("mission_command_topic").value)
        self.robot_state_topic = str(self.get_parameter("robot_state_topic").value)
        self.estop_status_topic = str(self.get_parameter("estop_status_topic").value)
        self.use_gazebo_tf_pose = bool(self.get_parameter("use_gazebo_tf_pose").value)
        self.gazebo_tf_topic = str(self.get_parameter("gazebo_tf_topic").value)
        self.gazebo_tf_frame_match = str(self.get_parameter("gazebo_tf_frame_match").value)
        self.gazebo_tf_allow_unmatched = bool(self.get_parameter("gazebo_tf_allow_unmatched").value)
        self.relative_to_start = bool(self.get_parameter("relative_to_start").value)
        self.auto_start = bool(self.get_parameter("auto_start").value)
        self.goal_tolerance = float(self.get_parameter("goal_tolerance").value)
        waypoint_goal_tolerance = float(self.get_parameter("waypoint_goal_tolerance").value)
        if waypoint_goal_tolerance > 0.0:
            self.goal_tolerance = waypoint_goal_tolerance
        self.heading_tolerance = math.radians(float(self.get_parameter("heading_tolerance_deg").value))
        self.linear_speed = float(self.get_parameter("linear_speed").value)
        waypoint_linear_speed = float(self.get_parameter("waypoint_linear_speed").value)
        if waypoint_linear_speed > 0.0:
            self.linear_speed = waypoint_linear_speed
        self.slow_linear_speed = float(self.get_parameter("slow_linear_speed").value)
        waypoint_slow_linear_speed = float(self.get_parameter("waypoint_slow_linear_speed").value)
        if waypoint_slow_linear_speed > 0.0:
            self.slow_linear_speed = waypoint_slow_linear_speed
        self.enable_waypoint_obstacle_avoidance = bool(
            self.get_parameter("enable_waypoint_obstacle_avoidance").value
        )
        self.waypoint_obstacle_linear_speed = float(
            self.get_parameter("waypoint_obstacle_linear_speed").value
        )
        if self.waypoint_obstacle_linear_speed <= 0.0:
            self.waypoint_obstacle_linear_speed = min(self.slow_linear_speed, 0.04)
        self.waypoint_obstacle_turn_speed = float(
            self.get_parameter("waypoint_obstacle_turn_speed").value
        )
        if self.waypoint_obstacle_turn_speed <= 0.0:
            self.waypoint_obstacle_turn_speed = 0.45
        self.front_obstacle_dist_m = float(self.get_parameter("front_obstacle_dist_m").value)
        self.critical_obstacle_dist_m = float(self.get_parameter("critical_obstacle_dist_m").value)
        self.waypoint_critical_turn_speed = float(
            self.get_parameter("waypoint_critical_turn_speed").value
        )
        self.waypoint_critical_reverse_speed = float(
            self.get_parameter("waypoint_critical_reverse_speed").value
        )
        self.front_obstacle_fov = math.radians(float(self.get_parameter("front_obstacle_fov_deg").value))
        self.side_obstacle_fov = math.radians(float(self.get_parameter("side_obstacle_fov_deg").value))
        self.angular_gain = float(self.get_parameter("angular_gain").value)
        self.max_angular_speed = float(self.get_parameter("max_angular_speed").value)
        control_rate_hz = max(1.0, float(self.get_parameter("control_rate_hz").value))

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.robot_state_pub = self.create_publisher(String, self.robot_state_topic, 10)
        self.path_pub = self.create_publisher(PathMsg, self.planned_path_topic, 10)
        self.create_subscription(Odometry, self.pose_topic, self.odom_callback, 10)
        self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, 10)
        self.create_subscription(String, self.mission_command_topic, self.command_callback, 10)
        self.create_subscription(Int8, self.estop_status_topic, self.estop_status_callback, 10)
        if self.use_gazebo_tf_pose:
            self.create_subscription(TFMessage, self.gazebo_tf_topic, self.gazebo_tf_callback, 10)
        self.create_timer(1.0 / control_rate_hz, self.control_loop)

        self.have_pose = False
        self.have_gazebo_tf_pose = False
        self.have_scan = False
        self.pose_source = "odom"
        self._reported_gazebo_tf_pose = False
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.front_min = float("inf")
        self.left_min = float("inf")
        self.right_min = float("inf")
        self.obstacle_avoidance_active = False
        self.obstacle_avoidance_phase = "clear"
        self.obstacle_avoidance_started = 0.0
        self.obstacle_turn_target_yaw = None
        self.external_estop_status = ESTOP_CLEAR

        self.active = False
        self.goal_achieved = False
        self.relative_waypoints: List[Tuple[float, float]] = []
        self.world_waypoints: List[Tuple[float, float]] = []
        self.current_idx = 0
        self.start_x = 0.0
        self.start_y = 0.0
        self.home_start_idx = 0
        self.returning_home_reported = False
        self.coverage_mode = False
        self.coverage_scan_until = 0.0
        self.coverage_scan_active_idx = -1
        self.coverage_arc_scan_until = 0.0
        self.coverage_arc_scan_active = False
        self.coverage_pending_candidates: Dict[Tuple[int, int], Tuple[float, float, float]] = {}
        self.coverage_added_adaptive = 0
        self._last_status_log = 0.0

        self.get_logger().info(
            f"Waypoint controller ready. pose={self.pose_topic}, scan={self.scan_topic}, cmd_vel={self.cmd_vel_topic}, "
            f"source={self.waypoint_source}, detections={self.detection_log_file or self.default_detection_log_file()}, "
            f"fallback_file={self.waypoint_file or self.default_waypoint_file()}, astar={self.use_astar}, "
            f"estop={self.estop_status_topic}, obstacle_avoidance={self.enable_waypoint_obstacle_avoidance}"
        )
        if self.use_gazebo_tf_pose:
            self.get_logger().info(
                f"Gazebo pose enabled: using {self.gazebo_tf_topic} "
                f'(match="{self.gazebo_tf_frame_match}") when available.'
            )

        if self.auto_start:
            self.start_waypoints()

    def default_waypoint_file(self) -> str:
        candidates = [
            Path("/ros2_ws/maps/latest_obstacle_waypoints.txt"),
            Path.cwd() / "maps" / "latest_obstacle_waypoints.txt",
            Path.cwd() / "ros2_ws" / "maps" / "latest_obstacle_waypoints.txt",
        ]
        here = Path(__file__).resolve()
        for parent in [here] + list(here.parents):
            if parent.name == "ros2_ws":
                candidates.insert(0, parent / "maps" / "latest_obstacle_waypoints.txt")
                break
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return str(candidates[0])

    def default_detection_log_file(self) -> str:
        return str(Path.home() / "part3_logs" / "detections_log.jsonl")

    def default_map_path(self, filename: str) -> str:
        candidates = [
            Path("/ros2_ws") / "maps" / filename,
            Path.cwd() / "maps" / filename,
            Path.cwd() / "ros2_ws" / "maps" / filename,
        ]
        here = Path(__file__).resolve()
        for parent in [here] + list(here.parents):
            if parent.name == "ros2_ws":
                candidates.insert(0, parent / "maps" / filename)
                break
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return str(candidates[0])

    def odom_callback(self, msg: Odometry):
        if self.use_gazebo_tf_pose and self.have_gazebo_tf_pose:
            return
        self.have_pose = True
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.current_yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        self.pose_source = "odom"

    def gazebo_tf_callback(self, msg: TFMessage):
        if not msg.transforms:
            return

        selected = None
        match = self.gazebo_tf_frame_match
        for transform in msg.transforms:
            child = transform.child_frame_id or ""
            parent = transform.header.frame_id or ""
            if match in child or match in parent:
                selected = transform
                break

        if selected is None and (len(msg.transforms) == 1 or self.gazebo_tf_allow_unmatched):
            selected = msg.transforms[0]

        if selected is None:
            frames = ", ".join(
                (t.child_frame_id or t.header.frame_id or "<blank>") for t in msg.transforms[:8]
            )
            self.get_logger().warn(
                f'Gazebo TF topic active, but no frame matched "{match}". Frames seen: {frames}',
                throttle_duration_sec=5.0,
            )
            return

        self.have_pose = True
        self.have_gazebo_tf_pose = True
        self.pose_source = "gazebo_tf"
        self.current_x = selected.transform.translation.x
        self.current_y = selected.transform.translation.y
        self.current_yaw = yaw_from_quaternion(selected.transform.rotation)
        if not self._reported_gazebo_tf_pose:
            self._reported_gazebo_tf_pose = True
            frame = selected.child_frame_id or selected.header.frame_id or "<blank>"
            self.get_logger().info(f'Waypoint controller using Gazebo pose frame "{frame}".')

    def scan_callback(self, msg: LaserScan):
        self.have_scan = True
        if not msg.ranges:
            self.front_min = float("inf")
            self.left_min = float("inf")
            self.right_min = float("inf")
            return

        front = []
        left = []
        right = []
        left_centre = math.pi / 2.0
        right_centre = -math.pi / 2.0
        for idx, raw_range in enumerate(msg.ranges):
            if not math.isfinite(raw_range) or raw_range <= 0.01:
                continue
            if msg.range_min > 0.0 and raw_range < msg.range_min:
                continue
            if msg.range_max > 0.0 and raw_range > msg.range_max:
                continue

            angle = wrap_to_pi(msg.angle_min + idx * msg.angle_increment)
            if angle_in_sector(angle, 0.0, self.front_obstacle_fov):
                front.append(raw_range)
            elif angle_in_sector(angle, left_centre, self.side_obstacle_fov):
                left.append(raw_range)
            elif angle_in_sector(angle, right_centre, self.side_obstacle_fov):
                right.append(raw_range)

        self.front_min = min(front) if front else float("inf")
        self.left_min = min(left) if left else float("inf")
        self.right_min = min(right) if right else float("inf")
        self.record_coverage_occlusions(msg)

    def estop_status_callback(self, msg: Int8):
        previous = self.external_estop_status
        self.external_estop_status = int(msg.data)
        if self.external_estop_status == ESTOP_ACTIVE:
            self.stop_robot()
            self.publish_robot_state("ESTOP")
            if previous != ESTOP_ACTIVE:
                self.get_logger().error("Waypoint controller halted by external LiDAR E-STOP.")
        elif self.external_estop_status == ESTOP_WARNING:
            self.stop_robot()
            self.publish_robot_state("STOPPED")
            if previous != ESTOP_WARNING:
                self.get_logger().warn("Waypoint controller paused by external LiDAR warning.")
        elif previous != ESTOP_CLEAR:
            self.get_logger().info("External LiDAR e-stop clear. Waypoint controller may resume.")

    def record_coverage_occlusions(self, msg: LaserScan):
        if (
            not self.coverage_mode
            or not self.coverage_adaptive_viewpoints
            or (self.coverage_scan_active_idx != self.current_idx and not self.coverage_arc_scan_active)
            or self.coverage_added_adaptive >= self.coverage_max_adaptive_viewpoints
            or not self.have_pose
        ):
            return

        half = self.coverage_area_size_m / 2.0
        margin = min(self.coverage_boundary_margin_m, max(0.0, half - 0.25))
        min_coord = -half + margin
        max_coord = half - margin
        if min_coord >= max_coord:
            return

        range_max = msg.range_max if msg.range_max > 0.0 else 25.0
        for idx, raw_range in enumerate(msg.ranges):
            if not math.isfinite(raw_range) or raw_range <= max(msg.range_min, 0.2):
                continue
            if raw_range >= range_max * 0.96:
                continue

            angle = msg.angle_min + idx * msg.angle_increment
            world_angle = self.current_yaw + angle
            hit_x = self.current_x + raw_range * math.cos(world_angle)
            hit_y = self.current_y + raw_range * math.sin(world_angle)
            rel_hit_x = hit_x - self.start_x
            rel_hit_y = hit_y - self.start_y
            if not (min_coord <= rel_hit_x <= max_coord and min_coord <= rel_hit_y <= max_coord):
                continue

            from_view_x = hit_x - self.current_x
            from_view_y = hit_y - self.current_y
            norm = math.hypot(from_view_x, from_view_y)
            if norm < 0.2:
                continue
            dir_x = from_view_x / norm
            dir_y = from_view_y / norm
            candidate_rel_x = clamp(
                rel_hit_x + dir_x * self.coverage_occlusion_probe_offset_m,
                min_coord,
                max_coord,
            )
            candidate_rel_y = clamp(
                rel_hit_y + dir_y * self.coverage_occlusion_probe_offset_m,
                min_coord,
                max_coord,
            )

            if math.hypot(candidate_rel_x - rel_hit_x, candidate_rel_y - rel_hit_y) < 0.6:
                continue
            if self.coverage_viewpoint_exists(candidate_rel_x, candidate_rel_y):
                continue

            key = (round(candidate_rel_x / 0.5), round(candidate_rel_y / 0.5))
            score = min(
                max_coord - rel_hit_x,
                rel_hit_x - min_coord,
                max_coord - rel_hit_y,
                rel_hit_y - min_coord,
            )
            previous = self.coverage_pending_candidates.get(key)
            if previous is None or score > previous[2]:
                self.coverage_pending_candidates[key] = (candidate_rel_x, candidate_rel_y, score)

    def coverage_viewpoint_exists(self, rel_x: float, rel_y: float) -> bool:
        threshold = max(self.goal_tolerance * 1.5, 0.75)
        world_x = self.start_x + rel_x
        world_y = self.start_y + rel_y
        for existing_x, existing_y in self.world_waypoints:
            if math.hypot(existing_x - world_x, existing_y - world_y) < threshold:
                return True
        return False

    def append_adaptive_coverage_viewpoint(self) -> bool:
        if (
            not self.coverage_mode
            or not self.coverage_adaptive_viewpoints
            or self.coverage_added_adaptive >= self.coverage_max_adaptive_viewpoints
            or not self.coverage_pending_candidates
            or self.home_start_idx >= len(self.world_waypoints)
        ):
            return False

        candidates = sorted(
            self.coverage_pending_candidates.values(),
            key=lambda item: (-item[2], math.hypot(item[0], item[1])),
        )
        self.coverage_pending_candidates.clear()
        for rel_x, rel_y, _score in candidates:
            if self.coverage_viewpoint_exists(rel_x, rel_y):
                continue

            world_goal = (self.start_x + rel_x, self.start_y + rel_y)
            insert_idx = self.home_start_idx
            self.world_waypoints.insert(insert_idx, world_goal)
            self.home_start_idx += 1
            self.coverage_added_adaptive += 1
            self.publish_planned_path()
            self.get_logger().info(
                f"Added adaptive mapping viewpoint {self.coverage_added_adaptive}/"
                f"{self.coverage_max_adaptive_viewpoints} at ({world_goal[0]:.2f}, {world_goal[1]:.2f}) "
                "to inspect space occluded by a LiDAR obstacle return."
            )
            return True

        return False

    def command_callback(self, msg: String):
        command = msg.data.strip().lower()
        if command == "drive_waypoints":
            self.start_waypoints(coverage_mode=False)
        elif command in {"drive_coverage", "start_exploration", "explore_world"}:
            self.start_waypoints(coverage_mode=True)
        elif command in {"go_home", "start_wandering", "stop_waypoints"}:
            if self.active:
                self.get_logger().info(f"Stopping waypoint controller because command '{command}' was received.")
            self.active = False
            self.obstacle_avoidance_active = False
            self.coverage_mode = False
            self.stop_robot()

    def load_waypoints(self) -> List[Tuple[float, float]]:
        path = self.waypoint_file or self.default_waypoint_file()
        waypoints = []
        try:
            with open(path, "r", encoding="utf-8") as waypoint_file:
                for line in waypoint_file:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.replace(",", " ").split()
                    if len(parts) < 2:
                        continue
                    waypoints.append((float(parts[0]), float(parts[1])))
        except (OSError, ValueError) as exc:
            self.get_logger().error(f"Could not load waypoint file '{path}': {exc}")
            return []

        self.get_logger().info(f"Loaded {len(waypoints)} waypoint(s) from {path}")
        for idx, (x, y) in enumerate(waypoints, start=1):
            self.get_logger().info(f"Waypoint {idx}/{len(waypoints)} relative/file target: ({x:.2f}, {y:.2f})")
        return waypoints

    def load_detection_records(self) -> List[Dict[str, Any]]:
        path = self.detection_log_file or self.default_detection_log_file()
        records: List[Dict[str, Any]] = []
        try:
            with open(path, "r", encoding="utf-8") as detection_file:
                text = detection_file.read().strip()
        except OSError as exc:
            self.get_logger().warn(f"Could not read detection log '{path}': {exc}")
            return []

        if not text:
            self.get_logger().warn(f"Detection log '{path}' is empty.")
            return []

        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                records = [item for item in parsed if isinstance(item, dict)]
            elif isinstance(parsed, dict):
                for key in ("detections", "objects", "records"):
                    if isinstance(parsed.get(key), list):
                        records = [item for item in parsed[key] if isinstance(item, dict)]
                        break
                if not records:
                    records = [parsed]
        except json.JSONDecodeError:
            for line_number, line in enumerate(text.splitlines(), start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    self.get_logger().warn(
                        f"Skipping invalid JSON in detection log line {line_number}: {exc}"
                    )
                    continue
                if isinstance(record, dict):
                    records.append(record)

        self.get_logger().info(f"Loaded {len(records)} detection record(s) from {path}")
        return records

    def record_timestamp_seconds(self, record: Dict[str, Any]) -> Optional[float]:
        value = record.get("timestamp")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                import datetime

                return datetime.datetime.fromisoformat(value).timestamp()
            except ValueError:
                return None
        return None

    def detection_world_position(self, record: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        obj_x = record.get("object_x", record.get("x"))
        obj_y = record.get("object_y", record.get("y"))
        if obj_x is not None and obj_y is not None:
            try:
                return float(obj_x), float(obj_y)
            except (TypeError, ValueError):
                return None

        distance = record.get("distance_m", record.get("distance"))
        bearing = record.get("bearing_deg", record.get("bearing"))
        robot_x = record.get("robot_x")
        robot_y = record.get("robot_y")
        robot_yaw_deg = record.get("robot_yaw_deg", 0.0)
        if distance is None or bearing is None or robot_x is None or robot_y is None:
            return None

        try:
            distance_m = float(distance)
            bearing_rad = math.radians(float(bearing))
            robot_yaw = math.radians(float(robot_yaw_deg))
            base_x = float(robot_x)
            base_y = float(robot_y)
        except (TypeError, ValueError):
            return None

        angle = robot_yaw + bearing_rad
        return base_x + distance_m * math.cos(angle), base_y + distance_m * math.sin(angle)

    def load_detection_waypoints(self) -> List[Tuple[float, float]]:
        now_wall = None
        records = self.load_detection_records()
        objects: Dict[str, Tuple[float, float, float, Dict[str, Any]]] = {}

        if self.detection_max_age_s > 0.0:
            import time

            now_wall = time.time()

        for idx, record in enumerate(records):
            if self.detection_max_age_s > 0.0 and now_wall is not None:
                timestamp = self.record_timestamp_seconds(record)
                if timestamp is not None and now_wall - timestamp > self.detection_max_age_s:
                    continue

            world_pos = self.detection_world_position(record)
            if world_pos is None:
                continue

            obj_x, obj_y = world_pos
            rel_x = obj_x - self.start_x
            rel_y = obj_y - self.start_y
            home_distance = math.hypot(rel_x, rel_y)
            if home_distance < self.detection_min_distance_m:
                continue

            standoff = min(self.detection_goal_standoff_m, max(0.0, home_distance - self.goal_tolerance))
            if standoff > 0.0:
                scale = (home_distance - standoff) / home_distance
                rel_goal_x = rel_x * scale
                rel_goal_y = rel_y * scale
            else:
                rel_goal_x = rel_x
                rel_goal_y = rel_y

            label = str(record.get("name") or record.get("label") or record.get("type") or f"object_{idx}")
            rounded_key = f"{label}:{round(obj_x, 1)}:{round(obj_y, 1)}"
            previous = objects.get(rounded_key)
            if previous is None or home_distance < previous[2]:
                objects[rounded_key] = (rel_goal_x, rel_goal_y, home_distance, record)

        ordered = sorted(objects.values(), key=lambda item: item[2])
        waypoints = [(rel_x, rel_y) for rel_x, rel_y, _distance, _record in ordered]

        if waypoints:
            self.get_logger().info(
                f"Using {len(waypoints)} object waypoint(s) from unified detector, "
                f"home=({self.start_x:.2f}, {self.start_y:.2f}), standoff={self.detection_goal_standoff_m:.2f}m"
            )
            for idx, (rel_x, rel_y, distance, record) in enumerate(ordered, start=1):
                label = record.get("name") or record.get("label") or record.get("type") or "object"
                self.get_logger().info(
                    f"Object waypoint {idx}/{len(ordered)} {label}: "
                    f"home_relative_goal=({rel_x:.2f}, {rel_y:.2f}), object_distance_from_home={distance:.2f}m"
                )
        else:
            self.get_logger().warn("No usable detector object waypoints found.")

        return waypoints

    def generate_coverage_waypoints(self) -> List[Tuple[float, float]]:
        half = self.coverage_area_size_m / 2.0
        margin = min(self.coverage_boundary_margin_m, max(0.0, half - 0.25))
        min_coord = -half + margin
        max_coord = half - margin
        if min_coord >= max_coord:
            return []

        if self.coverage_pattern in {"adaptive_arc", "arc", "blank_spot"}:
            self.get_logger().info(
                f"Generated adaptive-arc mapping seed for "
                f"{self.coverage_area_size_m:.1f}x{self.coverage_area_size_m:.1f}m arena. "
                "Extra viewpoints will be added only when LiDAR returns indicate occluded space."
            )
            return [(0.0, 0.0)]

        if self.coverage_pattern in {"visibility_probe", "probe", "sparse"}:
            probe_radius = min(max_coord, half * self.coverage_probe_radius_ratio)
            candidates = [
                (0.0, 0.0),
                (probe_radius, probe_radius),
                (-probe_radius, probe_radius),
                (-probe_radius, -probe_radius),
                (probe_radius, -probe_radius),
            ]
            route: List[Tuple[float, float]] = []
            for x, y in candidates:
                x = clamp(x, min_coord, max_coord)
                y = clamp(y, min_coord, max_coord)
                if not route or math.hypot(x - route[-1][0], y - route[-1][1]) > self.goal_tolerance:
                    route.append((x, y))

            self.get_logger().info(
                f"Generated {len(route)} visibility-probe waypoint(s) for "
                f"{self.coverage_area_size_m:.1f}x{self.coverage_area_size_m:.1f}m arena, "
                f"probe_radius={probe_radius:.2f}m, scan_spin={self.coverage_scan_spin_s:.1f}s."
            )
            return route

        rows = []
        y = min_coord
        while y <= max_coord + 1e-6:
            rows.append(y)
            y += self.coverage_sweep_spacing_m
        if not rows or abs(rows[-1] - max_coord) > self.coverage_sweep_spacing_m * 0.35:
            rows.append(max_coord)

        route: List[Tuple[float, float]] = []
        left_to_right = True
        for y in rows:
            endpoints = [(min_coord, y), (max_coord, y)]
            if not left_to_right:
                endpoints.reverse()
            route.extend(endpoints)
            left_to_right = not left_to_right

        if route and math.hypot(route[-1][0], route[-1][1]) < math.hypot(route[0][0], route[0][1]):
            route.reverse()

        self.get_logger().info(
            f"Generated {len(route)} coverage waypoint(s) for "
            f"{self.coverage_area_size_m:.1f}x{self.coverage_area_size_m:.1f}m arena, "
            f"spacing={self.coverage_sweep_spacing_m:.2f}m, margin={margin:.2f}m."
        )
        return route

    def load_map_metadata(self) -> Tuple[float, float, float]:
        yaml_path = self.map_yaml or self.default_map_path("my_map.yaml")
        resolution = 0.05
        origin_x = 0.0
        origin_y = 0.0
        try:
            with open(yaml_path, "r", encoding="utf-8") as yaml_file:
                for line in yaml_file:
                    stripped = line.strip()
                    if stripped.startswith("resolution:"):
                        resolution = float(stripped.split(":", 1)[1].strip())
                    elif stripped.startswith("origin:"):
                        raw = stripped.split(":", 1)[1].strip().strip("[]")
                        vals = [float(v.strip()) for v in raw.split(",")]
                        if len(vals) >= 2:
                            origin_x, origin_y = vals[0], vals[1]
        except (OSError, ValueError) as exc:
            self.get_logger().warn(f"Could not read map YAML '{yaml_path}': {exc}")
        return resolution, origin_x, origin_y

    def load_binary_map(self) -> Optional[List[List[int]]]:
        csv_path = self.binary_map_csv or self.default_map_path("my_map_binary.csv")
        grid: List[List[int]] = []
        try:
            with open(csv_path, "r", encoding="utf-8", newline="") as csv_file:
                for row in csv.reader(csv_file):
                    if row:
                        grid.append([1 if int(value) else 0 for value in row])
        except (OSError, ValueError) as exc:
            self.get_logger().warn(f"Could not load binary map CSV '{csv_path}': {exc}")
            return None

        if not grid or not grid[0]:
            self.get_logger().warn(f"Binary map CSV '{csv_path}' is empty.")
            return None

        width = len(grid[0])
        if any(len(row) != width for row in grid):
            self.get_logger().warn(f"Binary map CSV '{csv_path}' has uneven row lengths.")
            return None

        self.get_logger().info(f"Loaded A* map {csv_path}: {width}x{len(grid)} cells")
        return grid

    def inflate_grid(self, grid: List[List[int]], resolution: float) -> List[List[int]]:
        radius_cells = int(math.ceil(self.astar_obstacle_inflation_m / resolution))
        if radius_cells <= 0:
            return [row[:] for row in grid]

        height = len(grid)
        width = len(grid[0])
        inflated = [row[:] for row in grid]
        radius_sq = radius_cells * radius_cells
        occupied = [
            (row, col)
            for row in range(height)
            for col in range(width)
            if grid[row][col] != 0
        ]
        for row, col in occupied:
            for dr in range(-radius_cells, radius_cells + 1):
                for dc in range(-radius_cells, radius_cells + 1):
                    if dr * dr + dc * dc > radius_sq:
                        continue
                    rr = row + dr
                    cc = col + dc
                    if 0 <= rr < height and 0 <= cc < width:
                        inflated[rr][cc] = 1
        return inflated

    def world_to_grid(
        self, x: float, y: float, resolution: float, origin_x: float, origin_y: float, height: int
    ) -> Tuple[int, int]:
        col = int(math.floor((x - origin_x) / resolution))
        row_from_bottom = int(math.floor((y - origin_y) / resolution))
        row = height - 1 - row_from_bottom
        return row, col

    def grid_to_world(
        self, row: int, col: int, resolution: float, origin_x: float, origin_y: float, height: int
    ) -> Tuple[float, float]:
        x = origin_x + (col + 0.5) * resolution
        y = origin_y + (height - row - 0.5) * resolution
        return x, y

    def nearest_free_cell(
        self, grid: List[List[int]], cell: Tuple[int, int], max_radius: int = 40
    ) -> Optional[Tuple[int, int]]:
        height = len(grid)
        width = len(grid[0])
        row, col = cell
        if 0 <= row < height and 0 <= col < width and grid[row][col] == 0:
            return cell

        best = None
        best_dist = float("inf")
        for radius in range(1, max_radius + 1):
            for rr in range(row - radius, row + radius + 1):
                for cc in range(col - radius, col + radius + 1):
                    if not (0 <= rr < height and 0 <= cc < width):
                        continue
                    if abs(rr - row) != radius and abs(cc - col) != radius:
                        continue
                    if grid[rr][cc] != 0:
                        continue
                    dist = math.hypot(rr - row, cc - col)
                    if dist < best_dist:
                        best = (rr, cc)
                        best_dist = dist
            if best is not None:
                return best
        return None

    def astar_cells(
        self, grid: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        height = len(grid)
        width = len(grid[0])
        moves = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (1, 1, math.sqrt(2.0)),
        ]
        open_heap = [(0.0, start)]
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score = {start: 0.0}

        while open_heap:
            _priority, current = heapq.heappop(open_heap)
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            current_g = g_score[current]
            for dr, dc, step_cost in moves:
                nr = current[0] + dr
                nc = current[1] + dc
                if not (0 <= nr < height and 0 <= nc < width):
                    continue
                if grid[nr][nc] != 0:
                    continue
                if dr != 0 and dc != 0:
                    if grid[current[0]][nc] != 0 or grid[nr][current[1]] != 0:
                        continue

                neighbor = (nr, nc)
                tentative_g = current_g + step_cost
                if tentative_g >= g_score.get(neighbor, float("inf")):
                    continue

                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                heuristic = math.hypot(goal[0] - nr, goal[1] - nc)
                heapq.heappush(open_heap, (tentative_g + heuristic, neighbor))

        return None

    def sparsify_waypoints(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if len(points) <= 2:
            return points

        sparse = [points[0]]
        last_x, last_y = points[0]
        for x, y in points[1:-1]:
            if math.hypot(x - last_x, y - last_y) >= self.astar_waypoint_spacing_m:
                sparse.append((x, y))
                last_x, last_y = x, y
        sparse.append(points[-1])
        return sparse

    def astar_route_to_goals(
        self, relative_goals: List[Tuple[float, float]]
    ) -> Optional[Tuple[List[Tuple[float, float]], int]]:
        if not relative_goals:
            return []

        base_grid = self.load_binary_map()
        if base_grid is None:
            return None

        resolution, origin_x, origin_y = self.load_map_metadata()
        grid = self.inflate_grid(base_grid, resolution)
        height = len(grid)

        remaining = [
            (self.start_x + rel_x, self.start_y + rel_y)
            for rel_x, rel_y in relative_goals
        ]
        current_world = (self.current_x, self.current_y)
        route_world: List[Tuple[float, float]] = []
        planned_order: List[Tuple[float, float]] = []

        while remaining:
            start_cell_raw = self.world_to_grid(
                current_world[0], current_world[1], resolution, origin_x, origin_y, height
            )
            start_cell = self.nearest_free_cell(grid, start_cell_raw)
            if start_cell is None:
                self.get_logger().warn(f"A* start pose is outside free map: {current_world}")
                return None

            best_idx = -1
            best_path = None
            best_length = float("inf")
            best_goal_cell = None

            for idx, goal_world in enumerate(remaining):
                goal_cell_raw = self.world_to_grid(
                    goal_world[0], goal_world[1], resolution, origin_x, origin_y, height
                )
                goal_cell = self.nearest_free_cell(grid, goal_cell_raw)
                if goal_cell is None:
                    self.get_logger().warn(f"Skipping A* goal outside free map: {goal_world}")
                    continue

                path = self.astar_cells(grid, start_cell, goal_cell)
                if path is None:
                    self.get_logger().warn(f"No A* path to goal {goal_world}")
                    continue

                path_length = max(0, len(path) - 1)
                if path_length < best_length:
                    best_idx = idx
                    best_path = path
                    best_length = path_length
                    best_goal_cell = goal_cell

            if best_idx < 0 or best_path is None or best_goal_cell is None:
                return None

            segment = [
                self.grid_to_world(row, col, resolution, origin_x, origin_y, height)
                for row, col in best_path
            ]
            if route_world:
                segment = segment[1:]
            route_world.extend(segment)
            current_world = self.grid_to_world(
                best_goal_cell[0], best_goal_cell[1], resolution, origin_x, origin_y, height
            )
            planned_order.append(remaining[best_idx])
            remaining.pop(best_idx)

        home_start_idx = len(route_world)
        home_cell = self.nearest_free_cell(
            grid,
            self.world_to_grid(self.start_x, self.start_y, resolution, origin_x, origin_y, height),
        )
        current_cell = self.nearest_free_cell(
            grid,
            self.world_to_grid(current_world[0], current_world[1], resolution, origin_x, origin_y, height),
        )
        if home_cell is None or current_cell is None:
            self.get_logger().warn("Could not locate a free A* cell for the return-home leg.")
            return None

        home_path = self.astar_cells(grid, current_cell, home_cell)
        if home_path is None:
            self.get_logger().warn("No A* path found for return-home leg.")
            return None

        home_segment = [
            self.grid_to_world(row, col, resolution, origin_x, origin_y, height)
            for row, col in home_path
        ]
        if route_world:
            home_segment = home_segment[1:]
        route_world.extend(home_segment)

        route = self.sparsify_waypoints(route_world)
        home_start_world = route_world[home_start_idx] if home_start_idx < len(route_world) else (self.start_x, self.start_y)
        home_start_route_idx = 0
        best_home_start_dist = float("inf")
        for idx, (x, y) in enumerate(route):
            dist = math.hypot(x - home_start_world[0], y - home_start_world[1])
            if dist < best_home_start_dist:
                best_home_start_dist = dist
                home_start_route_idx = idx

        self.get_logger().info(
            f"A* planned {len(route)} follow waypoint(s) through {len(planned_order)} object goal(s), then home."
        )
        for idx, (goal_x, goal_y) in enumerate(planned_order, start=1):
            self.get_logger().info(f"A* object visit order {idx}/{len(planned_order)}: ({goal_x:.2f}, {goal_y:.2f})")
        return route, home_start_route_idx

    def start_waypoints(self, coverage_mode: bool = False):
        if not self.have_pose:
            self.get_logger().warn("Drive Waypoints requested, but no pose has arrived yet.")
            self.stop_robot()
            return

        self.start_x = self.current_x
        self.start_y = self.current_y
        self.coverage_mode = coverage_mode

        if coverage_mode:
            self.relative_waypoints = self.generate_coverage_waypoints()
        elif self.waypoint_source in {"detections_json", "detector_json", "unified_detector"}:
            self.relative_waypoints = self.load_detection_waypoints()
            if not self.relative_waypoints:
                self.get_logger().warn("Falling back to waypoint text file.")
                self.relative_waypoints = self.load_waypoints()
        else:
            self.relative_waypoints = self.load_waypoints()

        if not self.relative_waypoints:
            self.active = False
            self.stop_robot()
            return

        astar_route = None
        if self.use_astar and not self.coverage_mode:
            astar_result = self.astar_route_to_goals(self.relative_waypoints)
            if astar_result:
                astar_route, raw_home_start_idx = astar_result
                self.world_waypoints = [
                    (x, y)
                    for x, y in astar_route
                    if math.hypot(x - self.current_x, y - self.current_y) > self.goal_tolerance * 0.5
                ]
                skipped = len(astar_route) - len(self.world_waypoints)
                self.home_start_idx = max(0, raw_home_start_idx - skipped)
                if not self.world_waypoints:
                    self.world_waypoints = astar_route[-1:]
                    self.home_start_idx = 0
            else:
                self.get_logger().warn("A* route unavailable; falling back to direct waypoint driving.")

        if self.use_astar and astar_route:
            pass
        elif self.relative_to_start:
            self.world_waypoints = [
                (self.start_x + rel_x, self.start_y + rel_y)
                for rel_x, rel_y in self.relative_waypoints
            ]
            self.home_start_idx = len(self.world_waypoints)
            self.world_waypoints.append((self.start_x, self.start_y))
        else:
            self.world_waypoints = list(self.relative_waypoints)
            self.home_start_idx = len(self.world_waypoints)
            self.world_waypoints.append((self.start_x, self.start_y))

        self.current_idx = 0
        self.goal_achieved = False
        self.returning_home_reported = False
        self.obstacle_avoidance_active = False
        self.obstacle_avoidance_phase = "clear"
        self.obstacle_turn_target_yaw = None
        self.coverage_scan_until = 0.0
        self.coverage_scan_active_idx = -1
        self.coverage_arc_scan_until = 0.0
        self.coverage_arc_scan_active = False
        self.coverage_pending_candidates.clear()
        self.coverage_added_adaptive = 0
        if self.coverage_mode and self.coverage_initial_arc_scan_s > 0.0:
            now = self.get_clock().now().nanoseconds / 1e9
            self.coverage_arc_scan_until = now + self.coverage_initial_arc_scan_s
            self.coverage_arc_scan_active = True
        self.active = True
        self._last_status_log = 0.0
        self.publish_robot_state("MAPPING" if self.coverage_mode else "WAYPOINT")
        self.stop_robot()

        self.get_logger().info(
            f"{'Coverage exploration' if self.coverage_mode else 'Waypoint drive'} started "
            f"from pose=({self.start_x:.2f}, {self.start_y:.2f}, "
            f"yaw={math.degrees(self.current_yaw):.1f} deg), relative_to_start={self.relative_to_start}"
        )
        if self.coverage_arc_scan_active:
            self.get_logger().info(
                f"Starting smooth initial mapping arc for {self.coverage_initial_arc_scan_s:.1f}s "
                f"(linear={self.coverage_initial_arc_linear_speed:.2f}m/s, "
                f"angular={self.coverage_initial_arc_turn_speed:.2f}rad/s)."
            )
        for idx, (x, y) in enumerate(self.world_waypoints, start=1):
            label = "return home" if idx - 1 >= self.home_start_idx else "object/path"
            self.get_logger().info(
                f"Waypoint {idx}/{len(self.world_waypoints)} {label} world goal: ({x:.2f}, {y:.2f})"
            )
        self.publish_planned_path()

    def control_loop(self):
        if not self.active:
            return
        if self.external_estop_status == ESTOP_ACTIVE:
            self.stop_robot()
            self.publish_robot_state("ESTOP")
            return
        if self.external_estop_status == ESTOP_WARNING:
            self.stop_robot()
            self.publish_robot_state("STOPPED")
            return
        if not self.have_pose:
            self.stop_robot()
            self.get_logger().warn("Waypoint controller waiting for pose.", throttle_duration_sec=2.0)
            return

        now = self.get_clock().now().nanoseconds / 1e9
        if self.enable_waypoint_obstacle_avoidance:
            obstacle_cmd = self.obstacle_avoidance_cmd(now)
            if obstacle_cmd is not None:
                self.cmd_pub.publish(obstacle_cmd)
                return

        if self.coverage_arc_scan_active:
            if now < self.coverage_arc_scan_until:
                cmd = Twist()
                cmd.linear.x = self.coverage_initial_arc_linear_speed
                cmd.angular.z = self.coverage_initial_arc_turn_speed
                self.cmd_pub.publish(cmd)
                if now - self._last_status_log >= 1.0:
                    self._last_status_log = now
                    self.get_logger().info(
                        "Initial mapping arc active: gathering visible free space and occlusion candidates."
                    )
                return

            self.coverage_arc_scan_active = False
            self.coverage_arc_scan_until = 0.0
            self.stop_robot()
            added = self.append_adaptive_coverage_viewpoint()
            if (
                self.world_waypoints
                and self.current_idx == 0
                and math.hypot(
                    self.world_waypoints[0][0] - self.start_x,
                    self.world_waypoints[0][1] - self.start_y,
                ) <= self.goal_tolerance
            ):
                self.current_idx = 1
            self.get_logger().info(
                "Initial mapping arc complete. "
                + ("Added an occlusion viewpoint from LiDAR gaps." if added else "No extra occlusion viewpoint needed yet.")
            )
            return

        if self.current_idx >= len(self.world_waypoints):
            self.finish_waypoints()
            return

        if self.current_idx >= self.home_start_idx and not self.returning_home_reported:
            self.returning_home_reported = True
            self.publish_robot_state("RETURNING_HOME")
            mission_label = "coverage viewpoints" if self.coverage_mode else "object waypoints"
            self.get_logger().info(f"All {mission_label} reached. Returning home.")

        goal_x, goal_y = self.world_waypoints[self.current_idx]
        dx = goal_x - self.current_x
        dy = goal_y - self.current_y
        distance = math.hypot(dx, dy)
        target_yaw = math.atan2(dy, dx)
        heading_error = wrap_to_pi(target_yaw - self.current_yaw)
        waypoint_number = self.current_idx + 1
        waypoint_count = len(self.world_waypoints)

        if self.coverage_scan_active_idx == self.current_idx:
            if now < self.coverage_scan_until:
                cmd = Twist()
                cmd.angular.z = self.coverage_scan_turn_speed
                self.cmd_pub.publish(cmd)
                return

            self.get_logger().info(
                f"Completed scan turn at waypoint {waypoint_number}/{waypoint_count}."
            )
            self.coverage_scan_active_idx = -1
            self.coverage_scan_until = 0.0
            self.append_adaptive_coverage_viewpoint()
            self.current_idx += 1
            self.stop_robot()
            if self.current_idx >= len(self.world_waypoints):
                self.finish_waypoints()
            return

        if now - self._last_status_log >= 1.0:
            self._last_status_log = now
            self.get_logger().info(
                f"Waypoint {waypoint_number}/{waypoint_count}: "
                f"pose=({self.current_x:.2f}, {self.current_y:.2f}, yaw={math.degrees(self.current_yaw):.1f} deg), "
                f"goal=({goal_x:.2f}, {goal_y:.2f}), "
                f"distance={distance:.2f}m, heading_error={math.degrees(heading_error):.1f} deg, "
                f"pose_source={self.pose_source}"
            )

        if distance <= self.goal_tolerance:
            self.get_logger().info(
                f"Reached waypoint {waypoint_number}/{waypoint_count}: "
                f"pose=({self.current_x:.2f}, {self.current_y:.2f}), "
                f"goal=({goal_x:.2f}, {goal_y:.2f}), final_error={distance:.2f}m"
            )
            self.current_idx += 1
            self.stop_robot()
            if (
                self.coverage_mode
                and waypoint_number <= self.home_start_idx
                and self.coverage_scan_spin_s > 0.0
                and self.coverage_scan_turn_speed > 0.0
            ):
                self.current_idx -= 1
                self.coverage_scan_active_idx = self.current_idx
                self.coverage_scan_until = now + self.coverage_scan_spin_s
                self.publish_robot_state("MAPPING")
                self.get_logger().info(
                    f"Scanning from coverage viewpoint {waypoint_number}/{waypoint_count} "
                    f"for {self.coverage_scan_spin_s:.1f}s."
                )
                return
            if self.current_idx >= len(self.world_waypoints):
                self.finish_waypoints()
            return

        angular = clamp(
            self.angular_gain * heading_error,
            -self.max_angular_speed,
            self.max_angular_speed,
        )
        if abs(heading_error) > self.heading_tolerance:
            linear = 0.0 if abs(heading_error) > math.radians(30.0) else self.slow_linear_speed
        else:
            linear = min(self.linear_speed, distance * 0.5)

        if self.enable_waypoint_obstacle_avoidance and self.obstacle_avoidance_active:
            self.obstacle_avoidance_active = False
            self.obstacle_avoidance_phase = "clear"
            self.obstacle_turn_target_yaw = None
            if self.current_idx >= self.home_start_idx:
                state = "RETURNING_HOME"
            else:
                state = "MAPPING" if self.coverage_mode else "WAYPOINT"
            self.publish_robot_state(state)
            self.get_logger().info("Waypoint obstacle avoidance clear. Resuming planned route.")

        cmd = Twist()
        cmd.linear.x = linear
        cmd.angular.z = angular
        self.cmd_pub.publish(cmd)

    def obstacle_avoidance_cmd(self, now: float) -> Optional[Twist]:
        if not self.have_scan:
            return None

        if self.obstacle_avoidance_phase == "clear" and self.front_min >= self.front_obstacle_dist_m:
            return None

        turn_dir = -1.0 if self.left_min < self.right_min else 1.0
        clearer_side = "left" if turn_dir > 0 else "right"

        if self.obstacle_avoidance_phase == "clear":
            self.obstacle_avoidance_phase = "reverse"
            self.obstacle_avoidance_started = now
            self.obstacle_turn_target_yaw = None
            self.obstacle_avoidance_active = True
            self.publish_robot_state("OBSTACLE_AVOIDANCE")
            self.get_logger().warn(
                f"Waypoint obstacle avoidance: obstacle {self.front_min:.2f}m ahead; "
                f"reversing before turning {clearer_side}.",
                throttle_duration_sec=1.0,
            )

        cmd = Twist()
        if self.obstacle_avoidance_phase == "reverse":
            if now - self.obstacle_avoidance_started < OBSTACLE_REVERSE_DURATION:
                cmd.linear.x = min(self.waypoint_critical_reverse_speed, -0.05)
                cmd.angular.z = 0.0
                return cmd

            self.obstacle_avoidance_phase = "turn"
            self.obstacle_avoidance_started = now
            self.obstacle_turn_target_yaw = wrap_to_pi(self.current_yaw + OBSTACLE_TURN_ANGLE * turn_dir)
            self.stop_robot()
            self.get_logger().info(
                f"Waypoint obstacle avoidance: reverse complete; turning {clearer_side} before resuming route."
            )
            return Twist()

        if self.obstacle_avoidance_phase == "turn":
            if self.obstacle_turn_target_yaw is None:
                self.obstacle_turn_target_yaw = wrap_to_pi(self.current_yaw + OBSTACLE_TURN_ANGLE * turn_dir)
            angle_error = wrap_to_pi(self.obstacle_turn_target_yaw - self.current_yaw)
            if abs(angle_error) <= OBSTACLE_TURN_TOLERANCE or now - self.obstacle_avoidance_started >= OBSTACLE_TURN_TIMEOUT:
                self.obstacle_avoidance_phase = "clear"
                self.obstacle_turn_target_yaw = None
                self.stop_robot()
                return Twist()

            cmd.linear.x = 0.0
            cmd.angular.z = math.copysign(max(abs(self.waypoint_obstacle_turn_speed), 0.35), angle_error)
            return cmd

        return None

    def finish_waypoints(self):
        if self.goal_achieved:
            self.stop_robot()
            return
        self.active = False
        self.goal_achieved = True
        self.stop_robot()
        self.publish_robot_state("REACHED_HOME")
        self.get_logger().info("GOAL ACHIEVED - all waypoints visited and robot returned home.")

    def stop_robot(self):
        self.cmd_pub.publish(Twist())

    def publish_robot_state(self, state: str):
        self.robot_state_pub.publish(String(data=state))

    def publish_planned_path(self):
        path = PathMsg()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = "map"
        for x, y in self.world_waypoints:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self.path_pub.publish(path)


def main(args=None):
    rclpy.init(args=args)
    node = WaypointController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
