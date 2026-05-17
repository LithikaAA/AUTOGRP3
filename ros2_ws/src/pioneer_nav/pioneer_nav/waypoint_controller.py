#!/usr/bin/env python3
"""
Basic waypoint follower for Pioneer 3-AT.

Default behaviour:
- waits for /mission_command == "drive_waypoints"
- loads ros2_ws/maps/latest_obstacle_waypoints.txt
- treats waypoint x/y as centre-relative metres
- anchors those relative waypoints at the robot's current pose when started
- turns to face each waypoint, drives with heading correction, stops at the end

Waypoint file format:
    target_x target_y [ignored...]

The latest_obstacle_waypoints.txt file has four columns:
    target_rel_x target_rel_y obstacle_rel_x obstacle_rel_y
Only the first two columns are used for driving.
"""

import math
import os
from pathlib import Path
from typing import List, Tuple

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wrap_to_pi(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def yaw_from_quaternion(q) -> float:
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class WaypointController(Node):
    def __init__(self):
        super().__init__("waypoint_controller")

        self.declare_parameter("waypoint_file", "")
        self.declare_parameter("obstacle_waypoint_file", "")
        self.declare_parameter("pose_topic", "/odom")
        self.declare_parameter("odom_topic", "")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("mission_command_topic", "/mission_command")
        self.declare_parameter("robot_state_topic", "/robot_state")
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
        self.declare_parameter("waypoint_obstacle_linear_speed", 0.0)
        self.declare_parameter("waypoint_obstacle_turn_speed", 0.0)
        self.declare_parameter("angular_gain", 1.4)
        self.declare_parameter("max_angular_speed", 0.55)
        self.declare_parameter("control_rate_hz", 10.0)

        self.waypoint_file = str(self.get_parameter("waypoint_file").value)
        obstacle_waypoint_file = str(self.get_parameter("obstacle_waypoint_file").value)
        if obstacle_waypoint_file and not self.waypoint_file:
            self.waypoint_file = obstacle_waypoint_file
        self.pose_topic = str(self.get_parameter("pose_topic").value)
        odom_topic = str(self.get_parameter("odom_topic").value)
        if odom_topic:
            self.pose_topic = odom_topic
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.mission_command_topic = str(self.get_parameter("mission_command_topic").value)
        self.robot_state_topic = str(self.get_parameter("robot_state_topic").value)
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
        self.angular_gain = float(self.get_parameter("angular_gain").value)
        self.max_angular_speed = float(self.get_parameter("max_angular_speed").value)
        control_rate_hz = max(1.0, float(self.get_parameter("control_rate_hz").value))

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.robot_state_pub = self.create_publisher(String, self.robot_state_topic, 10)
        self.create_subscription(Odometry, self.pose_topic, self.odom_callback, 10)
        self.create_subscription(String, self.mission_command_topic, self.command_callback, 10)
        if self.use_gazebo_tf_pose:
            self.create_subscription(TFMessage, self.gazebo_tf_topic, self.gazebo_tf_callback, 10)
        self.create_timer(1.0 / control_rate_hz, self.control_loop)

        self.have_pose = False
        self.have_gazebo_tf_pose = False
        self.pose_source = "odom"
        self._reported_gazebo_tf_pose = False
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0

        self.active = False
        self.goal_achieved = False
        self.relative_waypoints: List[Tuple[float, float]] = []
        self.world_waypoints: List[Tuple[float, float]] = []
        self.current_idx = 0
        self.start_x = 0.0
        self.start_y = 0.0
        self._last_status_log = 0.0

        self.get_logger().info(
            f"Waypoint controller ready. pose={self.pose_topic}, cmd_vel={self.cmd_vel_topic}, "
            f"file={self.waypoint_file or self.default_waypoint_file()}"
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

    def command_callback(self, msg: String):
        command = msg.data.strip().lower()
        if command == "drive_waypoints":
            self.start_waypoints()
        elif command in {"go_home", "start_wandering", "stop_waypoints"}:
            if self.active:
                self.get_logger().info(f"Stopping waypoint controller because command '{command}' was received.")
            self.active = False
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

    def start_waypoints(self):
        if not self.have_pose:
            self.get_logger().warn("Drive Waypoints requested, but no pose has arrived yet.")
            self.stop_robot()
            return

        self.relative_waypoints = self.load_waypoints()
        if not self.relative_waypoints:
            self.active = False
            self.stop_robot()
            return

        self.start_x = self.current_x
        self.start_y = self.current_y
        if self.relative_to_start:
            self.world_waypoints = [
                (self.start_x + rel_x, self.start_y + rel_y)
                for rel_x, rel_y in self.relative_waypoints
            ]
        else:
            self.world_waypoints = list(self.relative_waypoints)

        self.current_idx = 0
        self.goal_achieved = False
        self.active = True
        self._last_status_log = 0.0
        self.publish_robot_state("WAYPOINT")
        self.stop_robot()

        self.get_logger().info(
            f"Waypoint drive started from pose=({self.start_x:.2f}, {self.start_y:.2f}, "
            f"yaw={math.degrees(self.current_yaw):.1f} deg), relative_to_start={self.relative_to_start}"
        )
        for idx, (x, y) in enumerate(self.world_waypoints, start=1):
            self.get_logger().info(f"Waypoint {idx}/{len(self.world_waypoints)} world goal: ({x:.2f}, {y:.2f})")

    def control_loop(self):
        if not self.active:
            return
        if not self.have_pose:
            self.stop_robot()
            self.get_logger().warn("Waypoint controller waiting for pose.", throttle_duration_sec=2.0)
            return

        if self.current_idx >= len(self.world_waypoints):
            self.finish_waypoints()
            return

        goal_x, goal_y = self.world_waypoints[self.current_idx]
        dx = goal_x - self.current_x
        dy = goal_y - self.current_y
        distance = math.hypot(dx, dy)
        target_yaw = math.atan2(dy, dx)
        heading_error = wrap_to_pi(target_yaw - self.current_yaw)
        waypoint_number = self.current_idx + 1
        waypoint_count = len(self.world_waypoints)

        now = self.get_clock().now().nanoseconds / 1e9
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

        cmd = Twist()
        cmd.linear.x = linear
        cmd.angular.z = angular
        self.cmd_pub.publish(cmd)

    def finish_waypoints(self):
        if self.goal_achieved:
            self.stop_robot()
            return
        self.active = False
        self.goal_achieved = True
        self.stop_robot()
        self.publish_robot_state("GOAL_ACHIEVED")
        self.get_logger().info("GOAL ACHIEVED - final waypoint reached. Press Go Home to return to centre.")

    def stop_robot(self):
        self.cmd_pub.publish(Twist())

    def publish_robot_state(self, state: str):
        self.robot_state_pub.publish(String(data=state))


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
