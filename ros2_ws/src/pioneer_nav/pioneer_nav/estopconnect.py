#!/usr/bin/env python3

"""
LiDAR E-STOP -> moving obstacle detection

- checks the front cone only (30 deg either way) for moving obstacles
- two zones:
    -> within 5m: warning, stop and wait until clear
    -> within 1m: EMERGENCY STOP, immediate halt, log incident + save rosbag
        > this is PERMANENT, robot does not resume, human must reset

- publishes /estop_status so other nodes know what state we're in:
    0 = all clear
    1 = warning (moving obstacle 1-5m), will resume when clear
    2 = emergency stop (moving obstacle within 1m), PERMANENT

How moving detection works:
- we compare each lidar scan to the previous one
- stationary objects while the robot moves usually change at a consistent rate
  (example: walls slowly getting closer as the robot drives forward)
- sudden larger changes are treated as moving obstacles
- needs 5 rays to agree before triggering (reduces false positives)
- only checks the front 30 deg cone since side objects are less important

How rosbag rolling buffer works (lowkey might replace this):
- a rosbag is ALWAYS recording in the background
- restarts every 5 seconds so when estop triggers, bag has last ~5 seconds
- when estop triggers, we stop (save) the current bag then start a fresh one
- bag records /scan and /cmd_vel so you can see exactly what lidar saw
  and what commands were being sent right before the estop

Topics:
- subscribes to: /scan
- publishes to:  /cmd_vel, /estop_status
"""

import math
import subprocess
import os
import collections
from datetime import datetime

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Int8

# /estop_status values
clearstate = 0
warnstate  = 1
estopstate = 2

# settings
stopdist = 1.0  # metres -> emergency stop zone
warndist = 5.0  # metres -> warning zone (stop and wait)

# front cone
conehalfdeg = 30.0  # check 30 deg either side of dead ahead

# consistency based motion detection
historylen = 5  # how many past deltas per ray to average for drift baseline
drifttol = 0.15 # metres -> how much a ray can spike above its drift before flagged
                # TUNE: lower = more sensitive, higher = fewer false positives
minhits = 5 # how many flagged rays needed to actually trigger

# rosbag settings
bagdirect = "/ros2_ws/bags"
bagsecs = 5 # rolling window, so saved bag = last ~5 seconds
bagtops = ["/scan", "/cmd_vel"] # what to record

# where to save incident logs
incidentlog = "/ros2_ws/incidents.txt"


class LidarEstop(Node):

    def __init__(self):
        super().__init__("estoplidar")

        # two states:
        # estopON -> full emergency stop (1m), PERMANENT, never clears automatically
        # warnON  -> warning zone (5m), clears once path is clear
        self.estopON = False
        self.warnON = False

        # last scan - used to compute per-ray deltas
        self.prevranges = None

        # rolling delta history per ray, built on first scan
        # each ray gets a deque of its last historylen deltas
        # used to work out the "normal" drift rate so we can spot sudden spikes
        self.deltahistory = None

        # rosbag subprocess handle
        self.bagproc = None

        os.makedirs(bagdirect, exist_ok=True)

        # pubs
        self.cmdpub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.statuspub = self.create_publisher(Int8, "/estop_status", 10)

        self.create_subscription(LaserScan, "/scan", self.lidarcb, 10)

        # control loop at 10hz
        self.create_timer(0.1, self.controloop)

        # rolling bag restart every bagsecs seconds
        self.create_timer(float(bagsecs), self.restartbag)

        self.startbag()

        self.get_logger().info("LiDAR E-STOP node started. Watching for moving obstacles...")


    # rosbag helpers

    def startbag(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        bagpath = os.path.join(bagdirect, f"rolling_{timestamp}")
        try:
            self.bagproc = subprocess.Popen(
                ["ros2", "bag", "record", "-o", bagpath, "--topics"] + bagtops
            )
            self.get_logger().info(f"Rosbag started: {bagpath}")
        except Exception as e:
            self.get_logger().warn(f"Could not start rosbag: {e}")

    def stopbag(self):
        if self.bagproc is not None:
            self.bagproc.terminate()
            self.bagproc.wait()  # wait for it to flush to disk
            self.bagproc = None

    def restartbag(self):
        self.stopbag()
        self.startbag()

    def savethebag(self):
        # stops current bag (saves last ~5 seconds of /scan + /cmd_vel)
        # then kicks off a fresh one so recording continues
        self.get_logger().info("Saving incident rosbag...")
        self.stopbag()
        self.startbag()


    # lidar callback, runs every time a new scan arrives
    def lidarcb(self, msg: LaserScan):
        curr  = msg.ranges
        nrays = len(curr)

        # first scan, nothing to compare against yet, just set up history
        if self.prevranges is None:
            self.prevranges = curr
            # one deque per ray, pre-filled with 0 (no drift assumed yet)
            self.deltahistory = [
                collections.deque([0.0] * historylen, maxlen=historylen)
                for _ in range(nrays)
            ]
            return

        # scan size changed? shouldn't happen but handle it
        if len(curr) != nrays:
            self.prevranges = curr
            return

        # work out which ray indices fall in the front cone
        conerad = math.radians(conehalfdeg)
        idxahead = int(round(-msg.angle_min / msg.angle_increment))
        idxahead = max(0, min(nrays - 1, idxahead))
        idxhalf = int(math.ceil(conerad / msg.angle_increment))
        idxstart = max(0,     idxahead - idxhalf)
        idxend = min(nrays, idxahead + idxhalf + 1)  # exclusive

        warnmoovehits = 0
        estopmoovehits = 0
        hitclosest = float("inf")

        for i in range(idxstart, idxend):

            rangenow = curr[i]
            rangeprev = self.prevranges[i]

            # skip invalid readings
            if (
                math.isnan(rangenow) or math.isinf(rangenow) or rangenow  <= 0.1 or math.isnan(rangeprev) or math.isinf(rangeprev) or rangeprev <= 0.1
            ):
                self.deltahistory[i].append(0.0)
                continue

            instantchange = abs(rangenow - rangeprev)

            # average drift for this ray over the last historylen scans
            # this is the "expected" change from ego motion
            avgdrift = sum(self.deltahistory[i]) / len(self.deltahistory[i])

            # update history before using avgdrift (don't pollute your own baseline)
            self.deltahistory[i].append(instantchange)

            # how much did this ray spike above its normal drift?
            deviation = instantchange - avgdrift

            # only flag if the spike is significant (sudden change, not steady drift)
            if deviation >= drifttol:
                if rangenow <= stopdist:
                    estopmoovehits += 1
                    hitclosest = min(hitclosest, rangenow)
                elif rangenow <= warndist:
                    warnmoovehits += 1
                    hitclosest = min(hitclosest, rangenow)

        estoptrigger = estopmoovehits >= minhits
        warntrigger = warnmoovehits  >= minhits

        # emergency stop
        if estoptrigger and not self.estopON:
            self.sendvelo(0.0)
            self.get_logger().info(
                f"EMERGENCY STOP - moving obstacle at {hitclosest:.2f}m ({estopmoovehits} ray hits)"
            )
            self.estopON = True
            self.warnON = False
            self.logincident(hitclosest, estopmoovehits)
            self.savethebag()

        # warning zone (5m), only if not already estopped
        elif warntrigger and not self.estopON:
            self.sendvelo(0.0)
            if not self.warnON:
                self.get_logger().info(
                    f"Moving obstacle in warning zone at {hitclosest:.2f}m - stopping"
                )
                self.warnON = True

        # all clear
        else:
            # estopON is NEVER cleared here, robot stays stopped until human resets
            if self.warnON:
                self.get_logger().info("Warning zone clear - resuming")
                self.warnON = False

        self.prevranges = curr


    # write incident to log file

    def logincident(self, closest: float, hits: int):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] ESTOP triggered! - Moving obstacle at {closest:.2f}m ({hits} ray hits)\n"
        try:
            with open(incidentlog, "a") as f:
                f.write(line)
            self.get_logger().info(f"Incident logged to {incidentlog}")
        except Exception as e:
            self.get_logger().warn(f"Could not write incident log: {e}")


    # control loop, runs on the timer
    # steady state only, immediate stops happen in lidarcb

    def controloop(self):
        if self.estopON:
            self.sendvelo(0.0)
            self.statuspub.publish(Int8(data=estopstate))
        elif self.warnON:
            self.sendvelo(0.0)
            self.statuspub.publish(Int8(data=warnstate))
        else:
            self.statuspub.publish(Int8(data=clearstate))

    def sendvelo(self, speed: float):
        twist = Twist()
        twist.linear.x = speed
        self.cmdpub.publish(twist)


# main
def main():
    rclpy.init()
    node = LidarEstop()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.sendvelo(0.0)
        except Exception:
            pass
        node.stopbag()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
