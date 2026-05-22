#!/usr/bin/env python3

"""
LiDAR E-STOP -> moving obstacle detection

- Robot drives forward continuously (will be replaced with usual drive)
- Checks ALL directions for moving obstacles
- Two zones:
      -> within 5m: warning, stop and wait until clear
      -> within 1m: EMERGENCY STOP, immediate halt, log incident + save rosbag

How moving detection works (THIS WILL NEED TO BE ALTERED because when robot is moving things move)
ok ok ok new tech
we do similar, except not ego motion where we 
but we usee a threshold and ignore things that are moving at a constant rate
not specifically at the like wall
also only FRONT CONE cuz side kinda useless bruh

- We save the previous lidar scan
- We compare it to the current scan
- If multiple readings changed significantly, something moved
- If that moving thing is also within range, we stop

How rosbag rolling buffer works:
- A rosbag is ALWAYS recording in the background
- Every 10 seconds it restarts (so we never have a huge file)
- When estop triggers, we stop the current bag and save it
- That bag contains up to the last 10 seconds before the estop
- Then we start a fresh bag again for next time

Topics used:
- Subscribes to: /scan
- Publishes to: /cmd_vel
"""

import math
import subprocess
import os
import time
from datetime import datetime

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from std_msgs.msg import Int8
from std_msgs.msg import String

clearstate = 0
warnstate = 1
estopstate = 2

# settings
stopdist = 0.5          # metres -> motion estop zone (moving/suddenly-appearing obstacles)
critical_dist = 0.22    # metres -> proximity estop (static, last-resort; below avoidance threshold)
warndist = 1.0          # metres -> warning zone (slow down / warn)
fwdspeed = 0.2          # m/s
front_half_angle_deg = 45.0   # ±45° = 90° total front cone
min_hits = 3

# motion detection threshold
movethres = 0.18 # metres

# rosbag settings
default_bagdirect = os.path.expanduser("~/pioneer_estop/bags") # where to save bags
bagtime = 10 # seconds before restarting the rolling bag
bagtops = ["/scan", "/cmd_vel"] # what topics to record

# where to save incident logs
default_incidentlog = os.path.expanduser("~/pioneer_estop/incidents.txt")


class LidarEstop(Node):

    def __init__(self):
        # create ROS2 node
        super().__init__("estoplidar")
        self.declare_parameter("publish_forward_when_clear", False)
        self.declare_parameter("bag_directory", default_bagdirect)
        self.declare_parameter("incident_log", default_incidentlog)
        self.declare_parameter("stop_distance_m", stopdist)
        self.declare_parameter("critical_distance_m", critical_dist)
        self.declare_parameter("warning_distance_m", warndist)
        self.declare_parameter("front_half_angle_deg", front_half_angle_deg)
        self.declare_parameter("min_hits", min_hits)
        self.declare_parameter("latch_estop", True)
        self.publish_forward_when_clear = bool(
            self.get_parameter("publish_forward_when_clear").value
        )
        self.bagdirect = self.get_parameter("bag_directory").value
        self.incidentlog = self.get_parameter("incident_log").value
        self.stopdist = max(0.01, float(self.get_parameter("stop_distance_m").value))
        self.criticaldist = max(0.01, float(self.get_parameter("critical_distance_m").value))
        self.warndist = max(self.stopdist, float(self.get_parameter("warning_distance_m").value))
        self.front_half_angle = math.radians(
            max(1.0, float(self.get_parameter("front_half_angle_deg").value))
        )
        self.min_hits = max(1, int(self.get_parameter("min_hits").value))
        self.latch_estop = bool(self.get_parameter("latch_estop").value)

        # proxEstop: auto-clearing — something is in the front cone right now
        # estopON:   latching — something *moved* toward the robot (requires reset_estop)
        # warnON:    auto-clearing warning zone
        self.proxEstop = False
        self.estopON = False
        self.warnON  = False

        # stores the previous scan so we can compare
        self.prevranges = None

        # rosbag subprocess handle
        self.bagproc = None

        # make sure bag directory exists
        os.makedirs(self.bagdirect, exist_ok=True)
        os.makedirs(os.path.dirname(self.incidentlog), exist_ok=True)

        # create publisher for robot velocity
        self.cmdpub = self.create_publisher(
            Twist,
            "/cmd_vel",
            10
        )
        self.statuspub = self.create_publisher(Int8, "/estop_status", 10)
        self.triggeredpub = self.create_publisher(Bool, "/estop/triggered", 10)

        # subscribe to lidar scan topic
        self.create_subscription(
            LaserScan,
            "/scan",
            self.lidarcb,
            10
        )
        self.create_subscription(String, "/mission_command", self.commandcb, 10)

        # timer: runs every 0.1 seconds (10 Hz)
        # continuously sends velocity commands
        self.create_timer(0.1, self.controloop)

        # timer: restarts the rolling bag every bagtime seconds
        self.create_timer(float(bagtime), self.restartbag)

        # start the rolling bag straight away
        self.startbag()

        self.get_logger().info("LiDAR e-stop node started. Watching for moving obstacles...")

    def commandcb(self, msg: String):
        if msg.data.strip() != "reset_estop":
            return
        self.estopON = False
        self.proxEstop = False
        self.warnON = False
        self.sendvelo(0.0)
        self.statuspub.publish(Int8(data=clearstate))
        self.triggeredpub.publish(Bool(data=False))
        self.get_logger().info("LiDAR e-stop reset by /mission_command.")

    def front_sector_stats(self, msg: LaserScan):
        """Count rays in front cone within critical (last-resort) and warning distances."""
        closest = float("inf")
        stop_hits = 0
        warn_hits = 0

        angle = msg.angle_min
        for rangenow in msg.ranges:
            if -self.front_half_angle <= angle <= self.front_half_angle:
                if not (math.isnan(rangenow) or math.isinf(rangenow) or rangenow <= 0.1):
                    closest = min(closest, rangenow)
                    if rangenow <= self.criticaldist:
                        stop_hits += 1
                    elif rangenow <= self.warndist:
                        warn_hits += 1
            angle += msg.angle_increment

        return closest, stop_hits, warn_hits

    # rosbag helpers

    def startbag(self):
        # name each bag by timestamp (with microseconds so rapid restarts don't clash)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        bagpath = os.path.join(self.bagdirect, f"rolling_{timestamp}")

        try:
            self.bagproc = subprocess.Popen(
                ["ros2", "bag", "record", "-o", bagpath, "--topics"] + bagtops
            )
            self.get_logger().info(f"Rosbag started: {bagpath}")
        except Exception as e:
            self.get_logger().warn(f"Could not start rosbag: {e}")

    def stopbag(self):
        # stop the current bag subprocess
        if self.bagproc is not None:
            self.bagproc.terminate()
            self.bagproc.wait()   # wait for it to fully close and flush to disk
            self.bagproc = None

    def restartbag(self):
        # rolling restart: stop current bag, start a fresh one
        self.stopbag()
        self.startbag()

    def savethebag(self):
        # on estop: stop current bag (saves whatever it recorded up to now)
        # then immediately start a new rolling bag for next time
        self.get_logger().info("Saving incident rosbag...")
        self.stopbag()
        self.startbag()

    # lidar callback, runs every time a new scan arrives
    def lidarcb(self, msg: LaserScan):

        curr = msg.ranges
        static_closest, static_stop_hits, static_warn_hits = self.front_sector_stats(msg)

        # ── Proximity estop (AUTO-CLEARING) ──────────────────────────────────
        # Anything in the front cone within stopdist → stop immediately.
        # Clears on its own when the obstacle moves away — no manual reset needed.
        if static_stop_hits >= self.min_hits and not self.estopON:
            if not self.proxEstop:
                self.proxEstop = True
                self.sendvelo(0.0)
                self.get_logger().warn(
                    f"PROXIMITY ESTOP: obstacle at {static_closest:.2f}m "
                    f"({static_stop_hits} front rays) — will auto-clear when path is free."
                )
                self.logincident(static_closest, static_stop_hits)
                self.savethebag()
        elif self.proxEstop and static_stop_hits < self.min_hits:
            self.proxEstop = False
            self.get_logger().info("Proximity estop cleared — path is free, resuming.")

        # Warning zone (auto-clearing)
        if static_warn_hits >= self.min_hits and not self.estopON and not self.proxEstop:
            if not self.warnON:
                self.warnON = True
                self.get_logger().warn(
                    f"Warning: obstacle at {static_closest:.2f}m "
                    f"({static_warn_hits} front rays)"
                )
        elif self.warnON and static_warn_hits < self.min_hits and not self.proxEstop:
            self.warnON = False
            self.get_logger().info("Warning zone clear — resuming.")

        # ── Motion estop (LATCHING) ───────────────────────────────────────────
        # Something suddenly moved close — latches until reset_estop command.
        if self.prevranges is None:
            self.prevranges = curr
            return

        if len(curr) != len(self.prevranges):
            self.prevranges = curr
            return

        motion_stop_hits = 0
        motion_warn_hits = 0
        motion_closest = float("inf")
        angle = msg.angle_min

        for i, rangenow in enumerate(curr):
            rangeprev = self.prevranges[i]
            angle_i = msg.angle_min + i * msg.angle_increment

            if (
                math.isnan(rangenow) or math.isinf(rangenow) or rangenow <= 0.1 or
                math.isnan(rangeprev) or math.isinf(rangeprev) or rangeprev <= 0.1
            ):
                continue

            # only check the front cone for motion too
            if abs(angle_i) > self.front_half_angle:
                continue

            change = abs(rangenow - rangeprev)
            if change < 0.05:
                continue

            if change >= movethres:
                if rangenow <= self.stopdist:
                    motion_stop_hits += 1
                    motion_closest = min(motion_closest, rangenow)
                elif rangenow <= self.warndist:
                    motion_warn_hits += 1
                    motion_closest = min(motion_closest, rangenow)

        if motion_stop_hits >= self.min_hits and not self.estopON:
            self.estopON = True
            self.proxEstop = False
            self.warnON = False
            self.sendvelo(0.0)
            self.get_logger().warn(
                f"MOTION ESTOP: moving obstacle at {motion_closest:.2f}m "
                f"({motion_stop_hits} rays) — send reset_estop to clear."
            )
            self.logincident(motion_closest, motion_stop_hits)
            self.savethebag()

        self.prevranges = curr

    # write incident to a log file
    def logincident(self, closest: float, hits: int):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] ESTOP triggered! - Moving obstacle at {closest:.2f}m ({hits} ray hits)\n"

        try:
            with open(self.incidentlog, "a") as f:
                f.write(line)
            self.get_logger().info(f"Incident logged to {self.incidentlog}")
        except Exception as e:
            self.get_logger().warn(f"Could not write incident log: {e}")

    # control loop, runs on the timer
    # handles steady state, the immediate stops happen in lidarcb
    def controloop(self):

        # estop overrides everything (latching motion estop OR proximity estop)
        if self.estopON or self.proxEstop:
            self.sendvelo(0.0)
            self.statuspub.publish(Int8(data=estopstate))
            self.triggeredpub.publish(Bool(data=True))

        # warning zone — stop and wait
        elif self.warnON:
            self.sendvelo(0.0)
            self.statuspub.publish(Int8(data=warnstate))
            self.triggeredpub.publish(Bool(data=False))

        # all clear, go forward
        else:
            self.statuspub.publish(Int8(data=clearstate))
            self.triggeredpub.publish(Bool(data=False))
            if self.publish_forward_when_clear:
                self.sendvelo(fwdspeed)

    # sends a velocity command
    def sendvelo(self, speed: float):
        twist = Twist()
        twist.linear.x = speed
        self.cmdpub.publish(twist)


# main function
def main():
    rclpy.init()
    node = LidarEstop()

    try:
        # keep node running
        rclpy.spin(node)

    except KeyboardInterrupt:
        # ctrl+c
        pass

    finally:
        # safety stop before shutdown
        # wrapped in try/except so a timing issue on ctrl+c doesn't crash it
        try:
            node.sendvelo(0.0)
        except Exception:
            pass
        # stop the rosbag cleanly
        node.stopbag()
        # clean up
        node.destroy_node()
        rclpy.shutdown()


# run program
if __name__ == "__main__":
    main()
