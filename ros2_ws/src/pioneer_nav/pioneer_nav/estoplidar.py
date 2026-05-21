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

clearstate = 0
warnstate = 1
estopstate = 2

# settings
stopdist = 0.18 # metres -> emergency stop zone
warndist = 0.35 # metres -> warning zone (stop and wait)
fwdspeed = 0.2 # m/s

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
        self.declare_parameter("warning_distance_m", warndist)
        self.publish_forward_when_clear = bool(
            self.get_parameter("publish_forward_when_clear").value
        )
        self.bagdirect = self.get_parameter("bag_directory").value
        self.incidentlog = self.get_parameter("incident_log").value
        self.stopdist = max(0.01, float(self.get_parameter("stop_distance_m").value))
        self.warndist = max(self.stopdist, float(self.get_parameter("warning_distance_m").value))

        # two separate states now:
        # estopON -> full emergency stop (1m), latches for estophold seconds
        # warnON  -> warning zone stop (5m), clears as soon as path is clear
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

        # first scan ever, nothing to compare against yet, just save and wait
        if self.prevranges is None:
            self.prevranges = curr
            return

        # make sure scan lengths match (they always should, but just in case)
        if len(curr) != len(self.prevranges):
            self.prevranges = curr
            return

        # track closest moving hit and how many rays agree per zone
        warnMOOVEhits  = 0 # rays with motion within warn zone (5m)
        estopMOOVEhits = 0 # rays with motion within estop zone (1m)
        hitclosest = float("inf")

        for i in range(len(curr)):

            rangenow  = curr[i]
            rangeprev = self.prevranges[i]

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

                if rangenow <= self.stopdist:
                    # emergency zone
                    estopMOOVEhits += 1
                    hitclosest = min(hitclosest, rangenow)

                elif rangenow <= self.warndist:
                    # warning zone
                    warnMOOVEhits += 1
                    hitclosest = min(hitclosest, rangenow)

        # needs multiple rays to agree
        estoptrigger = estopMOOVEhits >= 5
        warntrigger = warnMOOVEhits >= 5

        # EMERGENCY STOP
        if estoptrigger and not self.estopON:

            print(f"Moving object detected (hits={estopMOOVEhits}, closest={hitclosest:.2f}m)")

            # stop immediately
            self.sendvelo(0.0)

            self.get_logger().info(
                f"EMERGENCY STOP - moving obstacle at {hitclosest:.2f}m"
            )

            self.estopON = True
            self.warnON  = False

            # log + save evidence
            self.logincident(hitclosest, estopMOOVEhits)
            self.savethebag()

        # WARNING ZONE (5m)
        # only applies if not already in estop
        elif warntrigger and not self.estopON:
            print(f"[WARN] moving object in warning zone (hits={warnMOOVEhits}, closest={hitclosest:.2f}m)")

            # stop immediately too
            self.sendvelo(0.0)

            if not self.warnON:
                self.get_logger().info(f"Moving obstacle in warning zone at {hitclosest:.2f}m — stopping")
                self.warnON = True

        # ALL CLEAR
        else:
            # clear warn zone immediately
            if self.warnON:
                self.get_logger().info("Warning zone clear - resuming")
                self.warnON = False

        # save current scan for next comparison
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

        # estop overrides everything
        if self.estopON:
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
