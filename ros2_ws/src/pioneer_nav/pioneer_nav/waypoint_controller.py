#!/usr/bin/env python3
"""
waypoint_controller.py
Stripped-down waypoint-following controller for Pioneer 3-AT — Python/ROS2.

Pose sources (set via ROS2 parameters):
  pose_topic  : odometry topic  (default /odom)
  pose_frame  : if set, reads TF map→base_link  ← set to "map" with slam_toolbox
  base_frame  : robot base frame                 (default base_link)
  waypoint_file: path to waypoints text file

Waypoint file format (one per line):
  x_m  y_m  [yaw_rad]
  # lines starting with # are ignored
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from tf2_ros import Buffer, TransformListener, TransformException

# ── Tuning constants ──────────────────────────────────────────────────────────
GOAL_TOLERANCE          = 150    # mm — distance that counts as "arrived"

DRIVE_SPEED             = 300    # mm/s  normal cruise
FINAL_SPEED             =  95    # mm/s  inside final-approach zone

MAX_GOAL_STEER          =  30    # deg   cap on raw goal angle during normal driving
MAX_TOTAL_STEER         =  55    # deg   absolute heading cap
STEER_DEADBAND          =   2    # deg   below this, don't bother rotating
HEADING_LIMIT           =  20    # deg   max change per control cycle (smoothing)
TURN_SPEED              =  80    # deg/s cap on angular velocity published

FINAL_APPROACH_DISTANCE = 1000   # mm    switch to slow/align inside this radius
FINAL_ALIGN_STOP_ANGLE  =    8   # deg   stop linear motion and rotate in place
FINAL_ALIGN_SLOW_ANGLE  =    3   # deg   slow to 30 % speed while aligning

WAYPOINT_PROGRESS_EPS   =   60   # mm    improvement smaller than this doesn't count
WAYPOINT_STUCK_TIMEOUT  =    6.0 # s     no progress for this long → skip waypoint
BLOCKED_WAYPOINT_DIST   = 1200   # mm    only check stuck when closer than this

CONTROL_PERIOD_MS       =   50   # ms    control loop rate
# ─────────────────────────────────────────────────────────────────────────────


def normalise_angle(angle_deg: float) -> float:
    """Wrap angle to (-180, 180]."""
    while angle_deg <= -180:
        angle_deg += 360
    while angle_deg > 180:
        angle_deg -= 360
    return angle_deg


def clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def pythag_mm(dx: int, dy: int) -> int:
    return int(round(math.sqrt(dx * dx + dy * dy)))


def quat_to_yaw_deg(x, y, z, w) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return normalise_angle(math.degrees(math.atan2(siny_cosp, cosy_cosp)))


def smooth_heading(desired: float, previous: float) -> float:
    """Limit per-cycle heading change then clamp to MAX_TOTAL_STEER."""
    delta = desired - previous
    if delta > HEADING_LIMIT:
        desired = previous + HEADING_LIMIT
    elif delta < -HEADING_LIMIT:
        desired = previous - HEADING_LIMIT
    return clamp(desired, MAX_TOTAL_STEER)


def load_waypoints(path: str) -> list:
    """
    Load waypoints from a text file.
    Returns list of dicts with keys x, y, yaw (all floats, metres / radians).
    """
    waypoints = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                wp = {
                    'x':   float(parts[0]),
                    'y':   float(parts[1]),
                    'yaw': float(parts[2]) if len(parts) > 2 else 0.0,
                }
                waypoints.append(wp)
    except (OSError, ValueError):
        pass
    return waypoints


class WaypointController(Node):

    def __init__(self):
        super().__init__('waypoint_controller')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('waypoint_file', '')
        self.declare_parameter('pose_topic',    '/odom')
        self.declare_parameter('pose_frame',    '')        # set to "map" for SLAM
        self.declare_parameter('base_frame',    'base_link')
        self.get_logger().info(f"WAYPOINT FILE USED: {wp_file}")
        wp_file         = self.get_parameter('waypoint_file').value
        self.pose_topic = self.get_parameter('pose_topic').value
        self.pose_frame = self.get_parameter('pose_frame').value
        self.base_frame = self.get_parameter('base_frame').value

        # ── Waypoints ─────────────────────────────────────────────────────────
        self.waypoints = load_waypoints(wp_file)
        if not self.waypoints:
            self.get_logger().warn(
                f"No waypoints loaded from '{wp_file}'; using fallback (5 m ahead).")
            self.waypoints = [{'x': 5.0, 'y': 0.0, 'yaw': 0.0}]

        # ── State ─────────────────────────────────────────────────────────────
        self.have_odom          = False
        self.have_imu           = False
        self.origin             = None          # (x_mm, y_mm) anchored on first pose
        self.current_wp_idx     = 0
        self.current_goal       = None          # (x_mm, y_mm) world-frame
        self.mission_complete   = False
        self.prev_heading       = 0.0
        self.best_dist_mm       = 1_000_000_000
        self.last_progress_time = self.get_clock().now()
        self.imu_yaw_deg        = 0.0
        self.using_tf_last      = None          # for one-shot log on source switch

        # Latest sensor messages
        self.odom_msg  = None

        # ── TF (only when pose_frame is set) ──────────────────────────────────
        if self.pose_frame:
            self.tf_buffer   = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
        else:
            self.tf_buffer   = None
            self.tf_listener = None

        # ── ROS interfaces ────────────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.create_subscription(
            Odometry, self.pose_topic, self._odom_cb, 10)

        self.create_subscription(
            Imu, '/imu', self._imu_cb,
            rclpy.qos.qos_profile_sensor_data)

        self.create_timer(
            CONTROL_PERIOD_MS / 1000.0, self._control_loop)

        self.get_logger().info(
            f"Loaded {len(self.waypoints)} waypoint(s). "
            f"Pose source: {self.pose_frame or self.pose_topic}. "
            f"First goal: ({self.waypoints[0]['x']:.2f}, {self.waypoints[0]['y']:.2f})")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        self.odom_msg  = msg
        self.have_odom = True

    def _imu_cb(self, msg: Imu):
        o = msg.orientation
        # Skip zero-initialised messages (sensor not ready yet)
        if o.w == 0.0 and o.z == 0.0:
            return
        self.imu_yaw_deg = quat_to_yaw_deg(o.x, o.y, o.z, o.w)
        self.have_imu    = True

    # ── Pose retrieval ────────────────────────────────────────────────────────

    def _pose_from_odom(self):
        """Return (x_mm, y_mm, yaw_deg) from latest odometry + optional IMU yaw."""
        p   = self.odom_msg.pose.pose
        x   = int(round(p.position.x * 1000))
        y   = int(round(p.position.y * 1000))
        o   = p.orientation
        yaw = quat_to_yaw_deg(o.x, o.y, o.z, o.w)
        if self.have_imu:
            yaw = self.imu_yaw_deg     # IMU yaw is more accurate on real hardware
        return x, y, yaw

    def _pose_from_tf(self):
        """
        Return (x_mm, y_mm, yaw_deg) from TF map→base_link, or None if unavailable.
        This is the pose to use when slam_toolbox is running — it gives a
        globally-consistent position even as the map is being built.
        """
        if not self.tf_buffer:
            return None
        try:
            tf = self.tf_buffer.lookup_transform(
                self.pose_frame, self.base_frame, rclpy.time.Time())
            t  = tf.transform.translation
            r  = tf.transform.rotation
            x  = int(round(t.x * 1000))
            y  = int(round(t.y * 1000))
            yaw = quat_to_yaw_deg(r.x, r.y, r.z, r.w)
            return x, y, yaw
        except TransformException:
            return None

    def _get_pose(self):
        """
        Get the best available pose. TF (map frame) takes priority when available;
        falls back to odom + IMU.
        Returns (x_mm, y_mm, yaw_deg, using_tf: bool).
        """
        if self.pose_frame:
            result = self._pose_from_tf()
            if result:
                return (*result, True)

        return (*self._pose_from_odom(), False)

    # ── Navigation helpers ────────────────────────────────────────────────────

    def _world_goal(self, idx: int):
        """Convert waypoint (relative to origin) to world-frame mm coordinates."""
        wp = self.waypoints[idx]
        return (
            self.origin[0] + int(round(wp['x'] * 1000)),
            self.origin[1] + int(round(wp['y'] * 1000)),
        )

    def _goal_relative(self, curr_x, curr_y, phi_deg):
        """
        Compute distance (mm) and relative angle (deg) from current pose to goal.
        Relative angle is positive = goal is to the left, negative = right.
        """
        dx    = self.current_goal[0] - curr_x
        dy    = self.current_goal[1] - curr_y
        theta = normalise_angle(math.degrees(math.atan2(dy, dx)))
        dist  = pythag_mm(int(dx), int(dy))
        angle = normalise_angle(theta - phi_deg)
        return dist, angle

    def _at_goal(self, curr_x, curr_y) -> bool:
        dx = self.current_goal[0] - curr_x
        dy = self.current_goal[1] - curr_y
        return pythag_mm(int(dx), int(dy)) <= GOAL_TOLERANCE

    def _advance_waypoint(self, reason: str):
        """Move to the next waypoint, or end the mission."""
        self.best_dist_mm       = 1_000_000_000
        self.last_progress_time = self.get_clock().now()
        self.prev_heading       = 0.0

        if self.current_wp_idx + 1 >= len(self.waypoints):
            self.mission_complete = True
            self._publish_stop()
            self.get_logger().info(f'Mission complete: {reason}')
            return

        self.current_wp_idx += 1
        self.current_goal    = self._world_goal(self.current_wp_idx)
        self._publish_stop()
        wp = self.waypoints[self.current_wp_idx]
        self.get_logger().info(
            f'{reason} → next waypoint ({wp["x"]:.2f}, {wp["y"]:.2f})')

    def _publish_stop(self):
        self.cmd_pub.publish(Twist())

    # ── Control loop ──────────────────────────────────────────────────────────

    def _control_loop(self):
        if self.mission_complete:
            self._publish_stop()
            return

        if not self.have_odom:
            self.get_logger().info(
                f'Waiting for odometry on {self.pose_topic}…',
                throttle_duration_sec=2.0)
            self._publish_stop()
            return

        # ── Pose ──────────────────────────────────────────────────────────────
        curr_x, curr_y, phi, using_tf = self._get_pose()

        if using_tf != self.using_tf_last:
            if using_tf:
                self.get_logger().info(
                    f'Using TF pose  ({self.pose_frame} → {self.base_frame})')
            else:
                self.get_logger().info(
                    f'Using odom/IMU pose  (TF not yet available)')
            self.using_tf_last = using_tf

        # ── Anchor origin on first valid pose ─────────────────────────────────
        if self.origin is None:
            self.origin       = (curr_x, curr_y)
            self.current_goal = self._world_goal(0)
            self.get_logger().info(
                f'Origin anchored at ({curr_x/1000:.2f}, {curr_y/1000:.2f}). '
                f'First world goal ({self.current_goal[0]/1000:.2f}, '
                f'{self.current_goal[1]/1000:.2f}).')

        # ── Goal reached? ─────────────────────────────────────────────────────
        if self._at_goal(curr_x, curr_y):
            is_last = self.current_wp_idx + 1 >= len(self.waypoints)
            self._advance_waypoint('Goal reached' if is_last else 'Waypoint reached')
            return

        # ── Distance / angle to goal ──────────────────────────────────────────
        dist_mm, goal_angle = self._goal_relative(curr_x, curr_y, phi)

        # Progress tracking for stuck detection
        if dist_mm + WAYPOINT_PROGRESS_EPS < self.best_dist_mm:
            self.best_dist_mm       = dist_mm
            self.last_progress_time = self.get_clock().now()

        elapsed = (self.get_clock().now() - self.last_progress_time).nanoseconds / 1e9
        stuck = (dist_mm < BLOCKED_WAYPOINT_DIST and elapsed > WAYPOINT_STUCK_TIMEOUT)
        if stuck:
            wp = self.waypoints[self.current_wp_idx]
            self.get_logger().warn(
                f'Stuck {self.best_dist_mm/1000:.2f} m from waypoint '
                f'({wp["x"]:.2f}, {wp["y"]:.2f}) — skipping.')
            self._advance_waypoint('Stuck — advancing')
            return

        # ── Heading ───────────────────────────────────────────────────────────
        final_approach = dist_mm < FINAL_APPROACH_DISTANCE

        if final_approach:
            raw_heading = clamp(goal_angle, MAX_TOTAL_STEER)
        else:
            raw_heading = clamp(goal_angle, MAX_GOAL_STEER)

        heading = smooth_heading(raw_heading, self.prev_heading)
        self.prev_heading = heading

        # ── Speed ─────────────────────────────────────────────────────────────
        base_speed = FINAL_SPEED if final_approach else DRIVE_SPEED
        linear_mps = base_speed / 1000.0
        angular_rps = math.radians(clamp(heading * 2, TURN_SPEED))

        # Alignment-based speed shaping
        if final_approach and abs(goal_angle) >= FINAL_ALIGN_STOP_ANGLE:
            linear_mps = 0.0                   # rotate in place to align
        elif final_approach and abs(goal_angle) >= FINAL_ALIGN_SLOW_ANGLE:
            linear_mps *= 0.3
        elif abs(heading) >= 18:
            linear_mps = 0.0                   # large heading error → rotate in place
        elif abs(heading) >= 10:
            linear_mps *= 0.35
        elif abs(heading) >= 5:
            linear_mps *= 0.65

        # Tiny heading error? Don't bother rotating at all
        if abs(heading) < STEER_DEADBAND:
            angular_rps = 0.0

        # ── Publish ───────────────────────────────────────────────────────────
        cmd           = Twist()
        cmd.linear.x  = linear_mps
        cmd.angular.z = angular_rps
        self.cmd_pub.publish(cmd)

        self.get_logger().info(
            f'pos=({curr_x/1000:.2f}, {curr_y/1000:.2f})  '
            f'goal=({self.current_goal[0]/1000:.2f}, {self.current_goal[1]/1000:.2f})  '
            f'dist={dist_mm}mm  goal_ang={goal_angle:.1f}°  heading={heading:.1f}°',
            throttle_duration_sec=1.0)


def main(args=None):
    rclpy.init(args=args)
    node = WaypointController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()