#!/usr/bin/env python3
"""
AUTO4508 Part 3 - Mission Manager
Switches between mapping phase and waypoint phase.
Does NOT depend on GPS. Teleop/deadman handled separately.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from std_srvs.srv import Trigger


class MissionState:
    IDLE = "IDLE"
    MAPPING = "MAPPING"
    WAYPOINT = "WAYPOINT"
    STOPPED = "STOPPED"


class MissionManager(Node):
    def __init__(self):
        super().__init__("mission_manager")

        self.state = MissionState.IDLE
        self.get_logger().info("Mission Manager started. State: IDLE")

        # --- Publishers ---
        # Broadcasts current robot state to all other nodes (UI, safety, etc.)
        self.state_pub = self.create_publisher(String, "/robot_state", 10)

        # Tells exploration_node to start/stop
        self.mapping_enable_pub = self.create_publisher(Bool, "/mapping/enable", 10)

        # Tells waypoint_node to start/stop
        self.waypoint_enable_pub = self.create_publisher(Bool, "/waypoint/enable", 10)

        # --- Subscribers ---
        # Safety node publishes True when e-stop triggered
        self.create_subscription(Bool, "/estop/triggered", self.estop_callback, 10)

        # Mapping node publishes True when exploration is complete
        self.create_subscription(Bool, "/mapping/complete", self.mapping_complete_callback, 10)

        # Waypoint node publishes True when all waypoints visited and home reached
        self.create_subscription(Bool, "/waypoint/complete", self.waypoint_complete_callback, 10)

        # --- Services ---
        # Call these to switch phases (e.g. from a button press or launch file)
        self.create_service(Trigger, "/mission/start_mapping", self.start_mapping_cb)
        self.create_service(Trigger, "/mission/start_waypoint", self.start_waypoint_cb)
        self.create_service(Trigger, "/mission/stop", self.stop_cb)

        # Publish state at 2 Hz so UI always has latest
        self.create_timer(0.5, self.publish_state)

    # ------------------------------------------------------------------
    # Service callbacks — triggered by button press or launch script
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Subscriber callbacks
    # ------------------------------------------------------------------

    def estop_callback(self, msg: Bool):
        if msg.data and self.state != MissionState.STOPPED:
            self.get_logger().warn("E-STOP triggered! Halting all phases.")
            self._set_state(MissionState.STOPPED)
            self.mapping_enable_pub.publish(Bool(data=False))
            self.waypoint_enable_pub.publish(Bool(data=False))

    def mapping_complete_callback(self, msg: Bool):
        if msg.data and self.state == MissionState.MAPPING:
            self.get_logger().info("Mapping complete. Returning to IDLE. Ready for waypoint phase.")
            self._set_state(MissionState.IDLE)
            self.mapping_enable_pub.publish(Bool(data=False))

    def waypoint_complete_callback(self, msg: Bool):
        if msg.data and self.state == MissionState.WAYPOINT:
            self.get_logger().info("Waypoint run complete. Mission finished.")
            self._set_state(MissionState.IDLE)
            self.waypoint_enable_pub.publish(Bool(data=False))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_state(self, new_state: str):
        self.state = new_state
        self.get_logger().info(f"State -> {self.state}")

    def publish_state(self):
        msg = String()
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
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()