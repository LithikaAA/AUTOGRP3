#!/usr/bin/env python3

"""
LiDAR E-STOP -> moving obstacle detection

- Robot drives forward continuously (will be replaced with usual drive)
- Checks ALL directions for moving obstacles
- Two zones:
      -> within 4m: warning, stop and wait until clear
      -> within 1m: EMERGENCY STOP, immediate halt, log incident + save rosbag

How moving detection works (THIS WILL NEED TO BE ALTERED because when robot is moving things move)
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

# settings
stopdist = 1.0 # metres -> emergency stop zone
warndist = 4.0 # metres -> warning zone (stop and wait)
fwdspeed = 0.2 # m/s

# motion detection threshold
movethres = 0.18 # metres

# how long to hold estop before allowing resume (seconds)
estop_hold = 4.0

# rosbag settings
bagdirect = "/ros2_ws/bags" # where to save bags
bagtime = 10 # seconds before restarting the rolling bag
bagtops = ["/scan", "/cmd_vel"] # what topics to record

# where to save incident logs
incidentlog = "/ros2_ws/incidents.txt"


class LidarEstop(Node):

    def __init__(self):
        # create ROS2 node
        super().__init__("estoplidar")

        # two separate states now:
        # estop_active -> full emergency stop (1m), latches for estop_hold seconds
        # warn_active  -> warning zone stop (4m), clears as soon as path is clear
        self.estop_active = False
        self.warn_active  = False

        # timestamp of when estop last triggered
        self.estop_time = 0.0

        # stores the previous scan so we can compare
        self.prev_ranges = None

        # rosbag subprocess handle
        self.bag_proc = None

        # make sure bag directory exists
        os.makedirs(bagdirect, exist_ok=True)

        # create publisher for robot velocity
        self.cmd_pub = self.create_publisher(
            Twist,
            "/cmd_vel",
            10
        )

        # subscribe to lidar scan topic
        self.create_subscription(
            LaserScan,
            "/scan",
            self.lidarcb,
            10
        )

        # timer: runs every 0.1 seconds (10 Hz)
        # continuously sends velocity commands
        self.create_timer(0.1, self.controloop)

        # timer: restarts the rolling bag every bagtime seconds
        self.create_timer(float(bagtime), self.restartbag)

        # start the rolling bag straight away
        self.startbag()

        self.get_logger().info("LiDAR e-stop node started. Watching for moving obstacles...")

    # rosbag helpers

    def startbag(self):
        # name each bag by timestamp (with microseconds so rapid restarts don't clash)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        bagpath = os.path.join(bagdirect, f"rolling_{timestamp}")

        try:
            self.bag_proc = subprocess.Popen(
                ["ros2", "bag", "record", "-o", bagpath] + bagtops
            )
            self.get_logger().info(f"Rosbag started: {bagpath}")
        except Exception as e:
            self.get_logger().warn(f"Could not start rosbag: {e}")

    def stopbag(self):
        # stop the current bag subprocess
        if self.bag_proc is not None:
            self.bag_proc.terminate()
            self.bag_proc.wait()   # wait for it to fully close and flush to disk
            self.bag_proc = None

    def restartbag(self):
        # rolling restart: stop current bag, start a fresh one
        self.stopbag()
        self.startbag()

    def save_incident_bag(self):
        # on estop: stop current bag (saves whatever it recorded up to now)
        # then immediately start a new rolling bag for next time
        self.get_logger().info("Saving incident rosbag...")
        self.stopbag()
        self.startbag()

    # lidar callback, runs every time a new scan arrives
    def lidarcb(self, msg: LaserScan):

        curr = msg.ranges

        # first scan ever — nothing to compare against yet, just save and wait
        if self.prev_ranges is None:
            self.prev_ranges = curr
            return

        # make sure scan lengths match (they always should, but just in case)
        if len(curr) != len(self.prev_ranges):
            self.prev_ranges = curr
            return

        # track closest moving hit and how many rays agree per zone
        movehits_warn  = 0 # rays with motion within warn zone (4m)
        movehits_estop = 0 # rays with motion within estop zone (1m)
        hitclosest = float("inf")

        for i in range(len(curr)):

            rangenow  = curr[i]
            rangeprev = self.prev_ranges[i]

            # skip invalid readings (esp 0 readings)
            if (
                math.isnan(rangenow)  or math.isinf(rangenow)  or rangenow  <= 0.1 or
                math.isnan(rangeprev) or math.isinf(rangeprev) or rangeprev <= 0.1
            ):
                continue

            # how much did this ray change since last scan?
            change = abs(rangenow - rangeprev)

            # ignore tiny jitter completely
            if change < 0.05:
                continue

            # count hits per zone
            if change >= movethres:

                if rangenow <= stopdist:
                    # emergency zone
                    movehits_estop += 1
                    hitclosest = min(hitclosest, rangenow)

                elif rangenow <= warndist:
                    # warning zone
                    movehits_warn += 1
                    hitclosest = min(hitclosest, rangenow)

        # needs multiple rays to agree
        estop_trig = movehits_estop >= 5
        warn_trig  = movehits_warn  >= 5

        # --- EMERGENCY STOP (within 1m) ---
        if estop_trig:
            print(f"[ESTOP] moving object detected (hits={movehits_estop}, closest={hitclosest:.2f}m)")

            # send stop immediately right here, dont wait for controloop
            self.sendvelo(0.0)

            if not self.estop_active:
                self.get_logger().info(f"EMERGENCY STOP - moving obstacle at {hitclosest:.2f}m")
                self.estop_active = True
                self.warn_active  = False  # estop overrides warn
                self.estop_time   = time.time()
                # log incident to file
                self.log_incident(hitclosest, movehits_estop)
                # save the rosbag
                self.save_incident_bag()

        # --- WARNING ZONE (within 4m) ---
        # only applies if not already in estop
        elif warn_trig and not self.estop_active:
            print(f"[WARN] moving object in warning zone (hits={movehits_warn}, closest={hitclosest:.2f}m)")

            # stop immediately too
            self.sendvelo(0.0)

            if not self.warn_active:
                self.get_logger().info(f"Moving obstacle in warning zone at {hitclosest:.2f}m — stopping")
                self.warn_active = True

        # --- ALL CLEAR ---
        else:
            # clear warn zone immediately
            if self.warn_active:
                self.get_logger().info("Warning zone clear - resuming")
                self.warn_active = False

            # only clear estop after hold time has passed
            if self.estop_active:
                held_for = time.time() - self.estop_time
                if held_for >= estop_hold:
                    self.get_logger().info("Path clear - resuming")
                    self.estop_active = False

        # save current scan for next comparison
        self.prev_ranges = curr

    # write incident to a log file
    def log_incident(self, closest: float, hits: int):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] ESTOP triggered - moving obstacle at {closest:.2f}m ({hits} ray hits)\n"

        try:
            with open(incidentlog, "a") as f:
                f.write(line)
            self.get_logger().info(f"Incident logged to {incidentlog}")
        except Exception as e:
            self.get_logger().warn(f"Could not write incident log: {e}")

    # control loop, runs on the timer
    # handles steady state — the immediate stops happen in lidarcb
    def controloop(self):

        # estop overrides everything
        if self.estop_active:
            self.sendvelo(0.0)

        # warning zone — stop and wait
        elif self.warn_active:
            self.sendvelo(0.0)

        # all clear, go forward
        else:
            self.sendvelo(fwdspeed)

    # sends a velocity command
    def sendvelo(self, speed: float):
        twist = Twist()
        twist.linear.x = speed
        self.cmd_pub.publish(twist)


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
