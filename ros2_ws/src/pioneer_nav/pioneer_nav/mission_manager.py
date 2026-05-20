#!/usr/bin/env python3
"""
AUTO4508 Part 3 - Mission Manager
Switches between mapping phase and waypoint phase.
Does NOT depend on GPS. Teleop/deadman handled separately.
"""

import json
import math
import os
import subprocess
import time
from collections import deque

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from std_srvs.srv import Trigger
from geometry_msgs.msg import Pose
from sensor_msgs.msg import LaserScan


class MissionState:
    IDLE    = "IDLE"
    MAPPING = "MAPPING"
    WAYPOINT = "WAYPOINT"
    STOPPED = "STOPPED"


class MissionManager(Node):
    def __init__(self):
        super().__init__("mission_manager")

        self.state = MissionState.IDLE
        self.get_logger().info("Mission Manager started. State: IDLE")

        # ── Publishers ────────────────────────────────────────────────────────
        # Broadcasts current robot state to all other nodes (UI, safety, etc.)
        self.state_pub = self.create_publisher(String, "/robot_state", 10)

        # Tells exploration_node to start/stop
        self.mapping_enable_pub = self.create_publisher(Bool, "/mapping/enable", 10)

        # Tells waypoint_node to start/stop
        self.waypoint_enable_pub = self.create_publisher(Bool, "/waypoint/enable", 10)

        # ── Subscribers ───────────────────────────────────────────────────────
        # Safety node publishes True when e-stop triggered
        self.create_subscription(Bool, "/estop/triggered", self.estop_callback, 10)

        # Mapping node publishes True when exploration is complete
        self.create_subscription(Bool, "/mapping/complete", self.mapping_complete_callback, 10)

        # Waypoint node publishes True when all waypoints visited and home reached
        self.create_subscription(Bool, "/waypoint/complete", self.waypoint_complete_callback, 10)

        # Pose and scan — buffered so we have the last 5 s ready for e-stop saves
        self.create_subscription(Pose,      "/robot/pose", self._buffer_pose, 10)
        self.create_subscription(LaserScan, "/scan",       self._buffer_scan, 10)

        # ── E-stop rolling buffer ─────────────────────────────────────────────
        # Stores the last 5 seconds of pose + scan snapshots as dicts.
        # Flushed to disk the moment an e-stop fires.
        self._estop_buffer = deque()
        self._estop_log_dir = os.path.expanduser("~/part3_logs/estop_events")
        os.makedirs(self._estop_log_dir, exist_ok=True)

        # Drop entries older than 5 s every second
        self.create_timer(1.0, self._prune_buffer)

        # ── Services ──────────────────────────────────────────────────────────
        # Call these to switch phases (e.g. from a button press or launch file)
        self.create_service(Trigger, "/mission/start_mapping",  self.start_mapping_cb)
        self.create_service(Trigger, "/mission/start_waypoint", self.start_waypoint_cb)
        self.create_service(Trigger, "/mission/stop",           self.stop_cb)

        # Publish state at 2 Hz so UI always has latest
        self.create_timer(0.5, self.publish_state)

        # ── Journey recorder ──────────────────────────────────────────────────
        # Start a rosbag recording immediately so the whole run is captured.
        # The bag can be replayed later with: ros2 bag play <bag_dir>
        self._bag_proc = None
        self._start_bag_recording()

    # ── Service callbacks ─────────────────────────────────────────────────────

    def start_mapping_cb(self, request, response):
        if self.state == MissionState.STOPPED:
            response.success = False
            response.message = "E-stop is active. Reset before starting."
            return response

        self.get_logger().info("Starting MAPPING phase")
        self._set_state(MissionState.MAPPING)
        self.mapping_enable_pub.publish(Bool(data=True))
        self.waypoint_enable_pub.publish(Bool(data=False))

        response.success = True
        response.message = "Mapping phase started"
        return response

    def start_waypoint_cb(self, request, response):
        if self.state == MissionState.STOPPED:
            response.success = False
            response.message = "E-stop is active. Reset before starting."
            return response

        self.get_logger().info("Starting WAYPOINT phase")
        self._set_state(MissionState.WAYPOINT)
        self.mapping_enable_pub.publish(Bool(data=False))
        self.waypoint_enable_pub.publish(Bool(data=True))

        response.success = True
        response.message = "Waypoint phase started"
        return response

    def stop_cb(self, request, response):
        self.get_logger().info("Stopping all phases")
        self._set_state(MissionState.IDLE)
        self.mapping_enable_pub.publish(Bool(data=False))
        self.waypoint_enable_pub.publish(Bool(data=False))

        response.success = True
        response.message = "Stopped"
        return response

    # ── Subscriber callbacks ──────────────────────────────────────────────────

    def estop_callback(self, msg: Bool):
        if msg.data and self.state != MissionState.STOPPED:
            self.get_logger().warn("E-STOP triggered! Halting all phases.")
            self._set_state(MissionState.STOPPED)
            self.mapping_enable_pub.publish(Bool(data=False))
            self.waypoint_enable_pub.publish(Bool(data=False))
            # Save the last 5 seconds of telemetry immediately
            self._flush_estop_buffer()

    def mapping_complete_callback(self, msg: Bool):
        if msg.data and self.state == MissionState.MAPPING:
            self.get_logger().info("Mapping complete. Returning to IDLE.")
            self._set_state(MissionState.IDLE)
            self.mapping_enable_pub.publish(Bool(data=False))

    def waypoint_complete_callback(self, msg: Bool):
        if msg.data and self.state == MissionState.WAYPOINT:
            self.get_logger().info("Waypoint run complete. Mission finished.")
            self._set_state(MissionState.IDLE)
            self.waypoint_enable_pub.publish(Bool(data=False))

    # ── E-stop buffer methods ─────────────────────────────────────────────────

    def _buffer_pose(self, msg: Pose):
        """Store every incoming pose in the rolling buffer."""
        q    = msg.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw  = math.degrees(math.atan2(siny, cosy))
        self._estop_buffer.append({
            "t":    time.time(),
            "type": "pose",
            "x":    round(msg.position.x, 4),
            "y":    round(msg.position.y, 4),
            "yaw":  round(yaw, 2),
        })

    def _buffer_scan(self, msg: LaserScan):
        """Store a downsampled scan (every 10th ray) in the rolling buffer."""
        self._estop_buffer.append({
            "t":      time.time(),
            "type":   "scan",
            "state":  self.state,
            "ranges": [round(r, 3) for r in msg.ranges[::10]],
        })

    def _prune_buffer(self):
        """Drop anything older than 5 seconds. Called by a 1 Hz timer."""
        cutoff = time.time() - 5.0
        while self._estop_buffer and self._estop_buffer[0]["t"] < cutoff:
            self._estop_buffer.popleft()

    def _flush_estop_buffer(self):
        """
        Write every buffered entry to a timestamped JSONL file.
        Each line is one pose or scan snapshot from the last 5 seconds.
        Replay by reading the file line-by-line and parsing JSON.
        """
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path  = os.path.join(self._estop_log_dir, f"estop_{stamp}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for entry in self._estop_buffer:
                f.write(json.dumps(entry) + "\n")
        self.get_logger().info(
            f"E-stop: saved {len(self._estop_buffer)} records → {path}")

    # ── Journey recorder methods ──────────────────────────────────────────────

    def _start_bag_recording(self):
        """
        Launch ros2 bag record as a background subprocess.
        Records all the important topics for post-run playback.
        Bag files land in ~/part3_logs/bags/run_<timestamp>/
        """
        stamp   = time.strftime("%Y%m%d_%H%M%S")
        bag_dir = os.path.expanduser(f"~/part3_logs/bags/run_{stamp}")
        topics  = (
            "/robot_state "
            "/robot/pose "
            "/scan "
            "/map "
            "/detected_letter "
            "/detections/colour "
            "/detections/image "
            "/oak/rgb/image_raw "
            "/planned_path "
            "/arena_status"
        )
        cmd = f"ros2 bag record -o {bag_dir} {topics}"
        self._bag_proc = subprocess.Popen(cmd, shell=True)
        self.get_logger().info(f"Bag recording started → {bag_dir}")
        self.get_logger().info("To replay: ros2 bag play <bag_dir>")

    def _stop_bag_recording(self):
        """Stop the rosbag subprocess cleanly on shutdown."""
        if self._bag_proc and self._bag_proc.poll() is None:
            self._bag_proc.terminate()
            self._bag_proc = None
            self.get_logger().info("Bag recording stopped.")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _set_state(self, new_state: str):
        self.state = new_state
        self.get_logger().info(f"State -> {self.state}")

    def publish_state(self):
        msg      = String()
        msg.data = self.state
        self.state_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MissionManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop_bag_recording()   # cleanly close the bag before exit
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()