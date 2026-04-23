#!/usr/bin/env python3

# AUTO4508 Part 2 - Mission Controller
# Self-contained - no launch file needed.
# Run with:
#   ros2 run pioneer_nav part2_mission_controller
#
# PS4 Controller:
#   X (buttons[0])  = AUTO mode
#   O (buttons[1])  = MANUAL mode
#   L1 (buttons[4]) = deadman (hold in AUTO)
#   D-pad up/down   = forward / back
#   D-pad left/right = turn left / right
#
# Keyboard fallback (if no controller):
#   a = AUTO    m = MANUAL    d = deadman toggle
#   w/s = fwd/back    q/e = turn    x = stop
#
# Waypoints — edit WAYPOINTS list below to change destinations

import math
import os
import time
import sys
import termios
import tty
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist
from sensor_msgs.msg import NavSatFix, LaserScan, Image, Joy, Imu
from cv_bridge import CvBridge


# =============================================================================
# WAYPOINTS — edit these to change where the robot goes
# Format: (latitude, longitude)
# Last waypoint should match first to return to start
# =============================================================================
WAYPOINTS = [
    (-31.9804902, 115.8174017),   # WP1
    (-31.9805027, 115.8174483),   # WP2
    (-31.9803921, 115.8174755),   # WP3
    (-31.9804214, 115.8173444),   # WP4
    (-31.9804902, 115.8174017),   # return to start
]

# =============================================================================
# TOPICS — change these if your robot uses different topic names
# =============================================================================
GPS_TOPIC      = "/fix"
SCAN_TOPIC     = "/scan"
IMAGE_TOPIC    = "/camera/image"
IMU_TOPIC      = "/imu/data"
JOY_TOPIC      = "/joy"
CMD_VEL_TOPIC  = "/cmd_vel"
ANNOTATED_TOPIC = "/camera/annotated"   # published back so you can view on laptop

# =============================================================================
# PS4 BUTTON MAPPING
# Run: ros2 topic echo /joy   then press each button to verify indices
# =============================================================================
BTN_AUTO     = 0    # X     -> AUTO mode
BTN_MANUAL   = 1    # O     -> MANUAL mode
BTN_DEADMAN  = 4    # L1    -> hold to move in AUTO
BTN_FORWARD  = 11   # D-pad up
BTN_BACK     = 12   # D-pad down
BTN_LEFT     = 13   # D-pad left
BTN_RIGHT    = 14   # D-pad right

# =============================================================================
# TUNING — adjust these if the robot behaves unexpectedly
# =============================================================================
MAX_LINEAR_SPEED         = 0.5    # m/s top speed in auto
MAX_ANGULAR_SPEED        = 1.0    # rad/s top turn speed
MANUAL_LINEAR_SPEED      = 0.3    # m/s when holding D-pad
MANUAL_ANGULAR_SPEED     = 0.6    # rad/s when holding D-pad
GOAL_RADIUS_M            = 1.5    # metres - how close = waypoint reached
CONE_STOP_DISTANCE_M     = 1.4    # metres - stop this close to cone
OBJECT_SEARCH_RADIUS_M   = 4.0    # metres - max range to look for object
FRONT_OBSTACLE_DIST_M    = 0.9    # metres - soft obstacle avoidance
CRITICAL_OBSTACLE_DIST_M = 0.5    # metres - hard stop and turn

# =============================================================================
# CAMERA / VISION
# =============================================================================
CAMERA_HFOV_RAD  = 1.089    # horizontal field of view (matches OAK-D default)
CONE_MIN_AREA    = 800.0    # min pixel area to count as a cone
OBJECT_MIN_AREA  = 500.0    # min pixel area to count as an object
PHOTOS_DIR       = "/ros2_ws/photos"   # where to save photos on the robot

# HSV colour ranges — tune these if detection is wrong outdoors
# Use the keyboard shortcut 'v' to toggle debug view
ORANGE_LOWER = np.array([5,  120,  80], dtype=np.uint8)
ORANGE_UPPER = np.array([22, 255, 255], dtype=np.uint8)
BLUE_LOWER   = np.array([95,  80,  60], dtype=np.uint8)
BLUE_UPPER   = np.array([130, 255, 255], dtype=np.uint8)
OTHER_RANGES = [
    (np.array([0,   120, 80]), np.array([10,  255, 255])),   # red low
    (np.array([165, 120, 80]), np.array([179, 255, 255])),   # red high
    (np.array([18,  120, 80]), np.array([35,  255, 255])),   # yellow
    (np.array([38,   80, 50]), np.array([88,  255, 255])),   # green
]


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class Detection:
    label: str
    center_x: float
    center_y: float
    area: float
    bearing_rad: float
    bbox: Tuple[int, int, int, int]
    confidence: float = 1.0
    distance_hint_m: float = 0.0


@dataclass
class WaypointResult:
    waypoint_index: int
    cone_photo: Optional[str] = None
    object_label: Optional[str] = None
    object_photo: Optional[str] = None
    object_distance_m: Optional[float] = None
    object_bearing_rad: Optional[float] = None


# =============================================================================
# Helper functions
# =============================================================================

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def wrap_to_pi(a):
    while a >  math.pi: a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a

def is_finite(x):
    return math.isfinite(x) and not math.isnan(x)

def bearing_from_x(x_px, width, hfov):
    return ((x_px - width / 2.0) / (width / 2.0)) * (hfov / 2.0)

def get_btn(msg, idx):
    return 0 <= idx < len(msg.buttons) and msg.buttons[idx] == 1

def apply_morph(mask, k=7):
    kernel = np.ones((k, k), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def biggest_contour(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(cnts, key=cv2.contourArea) if cnts else None

def all_contours(mask, min_area):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in cnts if cv2.contourArea(c) >= min_area]

def shape_name(cnt):
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    v = len(approx)
    if v == 3: return "triangle"
    if v == 4:
        x, y, w, h = cv2.boundingRect(approx)
        return "square" if 0.85 <= w/max(h,1) <= 1.15 else "rectangle"
    if v > 6: return "circle"
    return "polygon"

def cone_confidence(cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    if w == 0 or h == 0: return 0.0
    aspect = h / float(w)
    fill   = cv2.contourArea(cnt) / max(w * h, 1)
    return (1 - min(abs(aspect - 1.7) / 1.5, 1.0)) * 0.6 + \
           (1 - min(abs(fill   - 0.5) / 0.4, 1.0)) * 0.4

def area_to_dist(area, real_w=0.3, focal=600.0):
    if area <= 0: return 99.0
    return (real_w * focal) / max(math.sqrt(area), 1)

def latlon_to_xy(lat, lon, lat0, lon0):
    R = 6378137.0
    x = R * math.radians(lon - lon0) * math.cos(math.radians((lat + lat0) / 2))
    y = R * math.radians(lat - lat0)
    return x, y


# =============================================================================
# Vision — detect objects
# =============================================================================

def detect_cone(hsv, width, hfov, min_area=CONE_MIN_AREA):
    mask = apply_morph(cv2.inRange(hsv, ORANGE_LOWER, ORANGE_UPPER))
    cnt  = biggest_contour(mask)
    if cnt is None or cv2.contourArea(cnt) < min_area:
        return None
    x, y, w, h = cv2.boundingRect(cnt)
    cx = x + w / 2.0
    return Detection(
        label="orange_cone", center_x=cx, center_y=y + h / 2.0,
        area=cv2.contourArea(cnt),
        bearing_rad=bearing_from_x(cx, width, hfov),
        bbox=(x, y, w, h),
        confidence=cone_confidence(cnt),
        distance_hint_m=area_to_dist(cv2.contourArea(cnt)),
    )


def detect_bucket(hsv, width, hfov, min_area=OBJECT_MIN_AREA):
    mask = apply_morph(cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER))
    cnt  = biggest_contour(mask)
    if cnt is None or cv2.contourArea(cnt) < min_area:
        return None
    x, y, w, h = cv2.boundingRect(cnt)
    cx = x + w / 2.0
    return Detection(
        label=f"blue_bucket_{shape_name(cnt)}",
        center_x=cx, center_y=y + h / 2.0,
        area=cv2.contourArea(cnt),
        bearing_rad=bearing_from_x(cx, width, hfov),
        bbox=(x, y, w, h),
        confidence=0.8,
        distance_hint_m=area_to_dist(cv2.contourArea(cnt), real_w=0.35),
    )


def detect_other(hsv, width, hfov, min_area=OBJECT_MIN_AREA):
    combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in OTHER_RANGES:
        combined |= cv2.inRange(hsv, lo, hi)
    # remove orange and blue
    combined = cv2.bitwise_and(
        combined,
        cv2.bitwise_not(
            cv2.inRange(hsv, ORANGE_LOWER, ORANGE_UPPER) |
            cv2.inRange(hsv, BLUE_LOWER,   BLUE_UPPER)
        )
    )
    combined = apply_morph(combined)
    cnt = biggest_contour(combined)
    if cnt is None or cv2.contourArea(cnt) < min_area:
        return None
    x, y, w, h = cv2.boundingRect(cnt)
    cx = x + w / 2.0
    return Detection(
        label=shape_name(cnt),
        center_x=cx, center_y=y + h / 2.0,
        area=cv2.contourArea(cnt),
        bearing_rad=bearing_from_x(cx, width, hfov),
        bbox=(x, y, w, h),
        confidence=0.6,
        distance_hint_m=area_to_dist(cv2.contourArea(cnt)),
    )


def detect_all_cones(hsv, width, hfov, min_area=400.0):
    mask = apply_morph(cv2.inRange(hsv, ORANGE_LOWER, ORANGE_UPPER), k=5)
    cnts = all_contours(mask, min_area)
    dets = []
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w / 2.0
        dets.append(Detection(
            label="orange_cone", center_x=cx, center_y=y + h / 2.0,
            area=cv2.contourArea(cnt),
            bearing_rad=bearing_from_x(cx, width, hfov),
            bbox=(x, y, w, h),
            confidence=cone_confidence(cnt),
            distance_hint_m=area_to_dist(cv2.contourArea(cnt)),
        ))
    return sorted(dets, key=lambda d: d.bearing_rad)


def annotate_frame(bgr, cone, obj, mode, state, wp_idx, total_wps):
    """Draw detection boxes and status on frame for remote viewing."""
    out = bgr.copy()

    if cone:
        x, y, w, h = cone.bbox
        cv2.rectangle(out, (x, y), (x+w, y+h), (0, 165, 255), 2)
        cv2.putText(out, f"CONE {cone.confidence:.2f} ~{cone.distance_hint_m:.1f}m",
                    (x, max(y-8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,165,255), 2)

    if obj:
        x, y, w, h = obj.bbox
        col = (255, 80, 0) if "bucket" in obj.label else (0, 220, 0)
        cv2.rectangle(out, (x, y), (x+w, y+h), col, 2)
        cv2.putText(out, f"{obj.label} ~{obj.distance_hint_m:.1f}m",
                    (x, max(y-8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

    # status bar at top
    status = f"MODE:{mode}  STATE:{state}  WP:{wp_idx+1}/{total_wps}"
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (30, 30, 30), -1)
    cv2.putText(out, status, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

    return out


def save_photo(bgr, detection, photos_dir, prefix):
    """Save annotated photo to disk. Returns file path."""
    os.makedirs(photos_dir, exist_ok=True)
    ts   = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(photos_dir, f"{prefix}_{ts}.png")
    img  = bgr.copy()
    if detection:
        x, y, w, h = detection.bbox
        col = (0, 165, 255) if "cone" in detection.label else \
              (255, 80, 0)  if "bucket" in detection.label else (0, 220, 0)
        cv2.rectangle(img, (x, y), (x+w, y+h), col, 3)
        cv2.putText(img, f"{detection.label} conf:{detection.confidence:.2f}",
                    (x, max(y-8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    cv2.imwrite(path, img)
    return path


# =============================================================================
# Main controller node
# =============================================================================

class Part2MissionController(Node):

    def __init__(self):
        super().__init__("part2_mission_controller")

        os.makedirs(PHOTOS_DIR, exist_ok=True)

        # ---- state ----
        self.mode       = "MANUAL"
        self.auto_state = "NAVIGATE"
        self.wp_idx     = 0

        self._last_auto_btn   = False
        self._last_manual_btn = False

        # sensor data
        self.have_gps   = False
        self.have_scan  = False
        self.have_image = False
        self.have_imu   = False

        self.lat = 0.0;  self.lon = 0.0
        self.origin_lat = None;  self.origin_lon = None
        self.heading    = 0.0
        self.last_lat   = None;  self.last_lon  = None
        self.last_gps_t = None

        self.front_min = float("inf")
        self.left_min  = float("inf")
        self.right_min = float("inf")
        self.last_scan = None

        self.latest_bgr  = None
        self.latest_hsv  = None
        self.img_w       = None
        self.img_h       = None
        self.bridge      = CvBridge()

        self.cone_det:   Optional[Detection] = None
        self.obj_det:    Optional[Detection] = None
        self.all_cones:  List[Detection]     = []

        self.manual_lin  = 0.0
        self.manual_ang  = 0.0
        self.deadman     = False

        self.cone_photo_done = False
        self.obj_photo_done  = False
        self._wp_result      = None

        self.journey: List[WaypointResult] = []
        self.start_time = time.time()

        # ---- pubs / subs ----
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.cmd_pub  = self.create_publisher(Twist, CMD_VEL_TOPIC, 10)
        self.ann_pub  = self.create_publisher(Image, ANNOTATED_TOPIC, 10)

        self.create_subscription(NavSatFix, GPS_TOPIC,   self._gps_cb,   qos)
        self.create_subscription(LaserScan, SCAN_TOPIC,  self._scan_cb,  qos)
        self.create_subscription(Image,     IMAGE_TOPIC, self._image_cb, qos)
        self.create_subscription(Joy,       JOY_TOPIC,   self._joy_cb,   10)
        self.create_subscription(Imu,       IMU_TOPIC,   self._imu_cb,   qos)

        self.timer = self.create_timer(0.1, self._loop)

        self.get_logger().info("=" * 50)
        self.get_logger().info("Part 2 Mission Controller started")
        self.get_logger().info(f"Waypoints: {len(WAYPOINTS)}")
        self.get_logger().info(f"Photos dir: {PHOTOS_DIR}")
        self.get_logger().info("PS4: X=AUTO  O=MANUAL  L1=deadman")
        self.get_logger().info("     D-pad = drive in MANUAL")
        self.get_logger().info("Keyboard:  a=AUTO  m=MANUAL  d=deadman  w/s/q/e/x")
        self.get_logger().info("=" * 50)

        threading.Thread(target=self._kbd_thread, daemon=True).start()

    # =========================================================================
    # Sensor callbacks
    # =========================================================================

    def _gps_cb(self, msg: NavSatFix):
        if not is_finite(msg.latitude) or not is_finite(msg.longitude):
            return
        self.have_gps = True
        now = self.get_clock().now().nanoseconds * 1e-9

        if self.origin_lat is None:
            self.origin_lat = msg.latitude
            self.origin_lon = msg.longitude
            self.get_logger().info(
                f"GPS origin set: {self.origin_lat:.7f}, {self.origin_lon:.7f}"
            )

        # heading from GPS motion if no IMU
        if not self.have_imu and self.last_lat is not None and self.last_gps_t:
            if now - self.last_gps_t > 0.1:
                dx, dy = latlon_to_xy(msg.latitude, msg.longitude,
                                      self.last_lat, self.last_lon)
                if math.hypot(dx, dy) > 0.05:
                    self.heading = math.atan2(dy, dx)

        self.lat       = msg.latitude
        self.lon       = msg.longitude
        self.last_lat  = msg.latitude
        self.last_lon  = msg.longitude
        self.last_gps_t = now

    def _imu_cb(self, msg: Imu):
        q = msg.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.heading  = math.atan2(siny, cosy)
        self.have_imu = True

    def _scan_cb(self, msg: LaserScan):
        self.have_scan = True
        self.last_scan = msg
        ranges = [r if is_finite(r) else float("inf") for r in msg.ranges]
        n = len(ranges)
        if n == 0: return
        mid = n // 2
        fw  = max(10, n // 16)
        self.front_min = min(ranges[max(0,mid-fw):min(n,mid+fw)] or [float("inf")])
        self.left_min  = min(ranges[min(n-1,mid+fw):]            or [float("inf")])
        self.right_min = min(ranges[:max(1,mid-fw)]              or [float("inf")])

    def _image_cb(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(str(e), throttle_duration_sec=2.0)
            return

        self.have_image = True
        self.latest_bgr = bgr
        self.latest_hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        self.img_h, self.img_w = bgr.shape[:2]

        hsv = self.latest_hsv

        self.cone_det  = detect_cone(hsv, self.img_w, CAMERA_HFOV_RAD, CONE_MIN_AREA)
        bucket         = detect_bucket(hsv, self.img_w, CAMERA_HFOV_RAD, OBJECT_MIN_AREA)
        self.obj_det   = bucket if bucket else detect_other(hsv, self.img_w, CAMERA_HFOV_RAD, OBJECT_MIN_AREA)
        self.all_cones = detect_all_cones(hsv, self.img_w, CAMERA_HFOV_RAD, CONE_MIN_AREA * 0.5)

        # publish annotated frame for remote viewing
        try:
            ann = annotate_frame(bgr, self.cone_det, self.obj_det,
                                 self.mode, self.auto_state,
                                 self.wp_idx, len(WAYPOINTS))
            ann_msg = self.bridge.cv2_to_imgmsg(ann, encoding="bgr8")
            ann_msg.header = msg.header
            self.ann_pub.publish(ann_msg)
        except Exception:
            pass

    def _joy_cb(self, msg: Joy):
        # mode switching — rising edge only
        auto_btn   = get_btn(msg, BTN_AUTO)
        manual_btn = get_btn(msg, BTN_MANUAL)

        if auto_btn and not self._last_auto_btn:
            self.mode       = "AUTO"
            self.auto_state = "NAVIGATE"
            self.get_logger().info("PS4 X: AUTO mode")

        if manual_btn and not self._last_manual_btn:
            self.mode = "MANUAL"
            self.get_logger().info("PS4 O: MANUAL mode")

        self._last_auto_btn   = auto_btn
        self._last_manual_btn = manual_btn

        # deadman — held = active
        self.deadman = get_btn(msg, BTN_DEADMAN)

        # D-pad driving
        fwd  = get_btn(msg, BTN_FORWARD)
        back = get_btn(msg, BTN_BACK)
        left = get_btn(msg, BTN_LEFT)
        rght = get_btn(msg, BTN_RIGHT)

        self.manual_lin = MANUAL_LINEAR_SPEED  if fwd  else \
                         -MANUAL_LINEAR_SPEED  if back else 0.0
        self.manual_ang = MANUAL_ANGULAR_SPEED if left else \
                         -MANUAL_ANGULAR_SPEED if rght else 0.0

    def _kbd_thread(self):
        settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while rclpy.ok():
                k = sys.stdin.read(1)
                if   k == 'a': self.mode = "AUTO";   self.auto_state = "NAVIGATE"; self.get_logger().info("KB: AUTO")
                elif k == 'm': self.mode = "MANUAL";                                self.get_logger().info("KB: MANUAL")
                elif k == 'd': self.deadman = not self.deadman;                     self.get_logger().info(f"KB: deadman={'ON' if self.deadman else 'OFF'}")
                elif k == 'w': self.manual_lin =  MANUAL_LINEAR_SPEED;  self.manual_ang = 0.0
                elif k == 's': self.manual_lin = -MANUAL_LINEAR_SPEED;  self.manual_ang = 0.0
                elif k == 'q': self.manual_lin = 0.0; self.manual_ang =  MANUAL_ANGULAR_SPEED
                elif k == 'e': self.manual_lin = 0.0; self.manual_ang = -MANUAL_ANGULAR_SPEED
                elif k == 'x': self.manual_lin = 0.0; self.manual_ang = 0.0
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

    # =========================================================================
    # Geometry helpers
    # =========================================================================

    def _current_xy(self):
        if self.origin_lat is None: return 0.0, 0.0
        return latlon_to_xy(self.lat, self.lon, self.origin_lat, self.origin_lon)

    def _waypoint_xy(self, idx):
        lat, lon = WAYPOINTS[idx]
        return latlon_to_xy(lat, lon, self.origin_lat, self.origin_lon)

    def _dist_to_wp(self):
        if self.origin_lat is None or self.wp_idx >= len(WAYPOINTS):
            return float("inf")
        x, y   = self._current_xy()
        gx, gy = self._waypoint_xy(self.wp_idx)
        return math.hypot(gx - x, gy - y)

    def _heading_to_wp(self):
        if self.origin_lat is None or self.wp_idx >= len(WAYPOINTS):
            return self.heading
        x, y   = self._current_xy()
        gx, gy = self._waypoint_xy(self.wp_idx)
        return math.atan2(gy - y, gx - x)

    def _lidar_at(self, bearing):
        if self.last_scan is None: return None
        n   = len(self.last_scan.ranges)
        idx = int(round((bearing - self.last_scan.angle_min) /
                         self.last_scan.angle_increment))
        if idx < 0 or idx >= n: return None
        vals = [self.last_scan.ranges[i]
                for i in range(max(0, idx-4), min(n, idx+5))
                if is_finite(self.last_scan.ranges[i])]
        return min(vals) if vals else None

    # =========================================================================
    # Drive helpers
    # =========================================================================

    def _stop(self):   self._cmd(0.0, 0.0)
    def _cmd(self, lin, ang):
        m = Twist()
        m.linear.x  = float(lin)
        m.angular.z = float(ang)
        self.cmd_pub.publish(m)

    def _save_photo(self, prefix, detection=None):
        if self.latest_bgr is None: return None
        path = save_photo(self.latest_bgr, detection, PHOTOS_DIR, prefix)
        self.get_logger().info(f"Photo saved: {path}")
        return path

    # =========================================================================
    # Main control loop
    # =========================================================================

    def _loop(self):
        if self.mode == "MANUAL":
            self._cmd(self.manual_lin, self.manual_ang)
            return

        # AUTO mode
        if not (self.have_gps and self.have_scan and self.have_image):
            self._stop()
            self.get_logger().info("Waiting for sensors...", throttle_duration_sec=3.0)
            return

        if not self.deadman:
            self._stop()
            self.get_logger().warn("Hold L1 (deadman) to move in AUTO",
                                   throttle_duration_sec=2.0)
            return

        if self.wp_idx >= len(WAYPOINTS):
            self.auto_state = "DONE"

        if self.auto_state == "DONE":
            self._stop()
            return

        # hard safety stop
        if self.front_min < CRITICAL_OBSTACLE_DIST_M:
            turn = -1.0 if self.left_min < self.right_min else 1.0
            self._cmd(0.0, 0.8 * turn)
            return

        {
            "NAVIGATE":       self._navigate,
            "WEAVE":          self._weave,
            "APPROACH_CONE":  self._approach_cone,
            "CAPTURE_CONE":   self._capture_cone,
            "FIND_OBJECT":    self._find_object,
            "CAPTURE_OBJECT": self._capture_object,
        }.get(self.auto_state, self._stop)()

    # =========================================================================
    # Behaviour states
    # =========================================================================

    def _navigate(self):
        dist = self._dist_to_wp()
        herr = wrap_to_pi(self._heading_to_wp() - self.heading)

        if dist < GOAL_RADIUS_M:
            self.auto_state      = "APPROACH_CONE"
            self.cone_photo_done = False
            self.obj_photo_done  = False
            self._stop()
            self.get_logger().info(f"Near WP{self.wp_idx+1} -> APPROACH_CONE")
            return

        # weave on first leg (WP0 -> WP1) if cones detected ahead
        if self.wp_idx == 1 and self.cone_det:
            cr = self._lidar_at(self.cone_det.bearing_rad)
            if cr and cr < FRONT_OBSTACLE_DIST_M * 2.0:
                self.auto_state = "WEAVE"
                self.get_logger().info("Entering WEAVE mode")
                return

        if self.front_min < FRONT_OBSTACLE_DIST_M:
            turn = -1.0 if self.left_min < self.right_min else 1.0
            self._cmd(0.08, 0.6 * turn)
            return

        if abs(herr) > 0.35:
            self._cmd(0.0, clamp(1.4 * herr, -0.8, 0.8))
            return

        self._cmd(clamp(0.3 + 0.15 * dist, 0.0, MAX_LINEAR_SPEED),
                  clamp(1.2 * herr, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED))

    def _weave(self):
        cone = self.cone_det
        if cone is None:
            self.auto_state = "NAVIGATE"; return

        cr = self._lidar_at(cone.bearing_rad)
        if cr is None or cr > FRONT_OBSTACLE_DIST_M * 2.5:
            self.auto_state = "NAVIGATE"; return

        if self._dist_to_wp() < GOAL_RADIUS_M:
            self.auto_state      = "APPROACH_CONE"
            self.cone_photo_done = False
            self.obj_photo_done  = False
            self._stop(); return

        desired = 0.3 if self.left_min >= self.right_min else -0.3
        err = wrap_to_pi(cone.bearing_rad - desired)
        self._cmd(0.15 if cr > 1.0 else 0.08,
                  clamp(-1.8 * err, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED))

    def _approach_cone(self):
        cone = self.cone_det
        if cone is None:
            self._cmd(0.0, 0.35); return   # spin to find cone

        berr = wrap_to_pi(cone.bearing_rad - 0.18)  # keep cone slightly right
        cr   = self._lidar_at(cone.bearing_rad)

        if cr is not None and cr < CONE_STOP_DISTANCE_M:
            self.auto_state = "CAPTURE_CONE"
            self._stop()
            self.get_logger().info("Cone reached")
            return

        if self.front_min < FRONT_OBSTACLE_DIST_M * 0.9:
            turn = -1.0 if self.left_min < self.right_min else 1.0
            self._cmd(0.0, 0.5 * turn); return

        self._cmd(0.12 if abs(berr) < 0.35 else 0.04,
                  clamp(1.5 * berr, -0.6, 0.6))

    def _capture_cone(self):
        result = WaypointResult(waypoint_index=self.wp_idx + 1)
        if not self.cone_photo_done:
            path = self._save_photo(f"wp{self.wp_idx+1}_cone", self.cone_det)
            result.cone_photo    = path
            self.cone_photo_done = True
            self._wp_result      = result
            self.get_logger().info(
                f"Cone photo taken. "
                f"conf={self.cone_det.confidence:.2f} "
                f"~{self.cone_det.distance_hint_m:.1f}m"
                if self.cone_det else "Cone photo taken."
            )
        self.auto_state = "FIND_OBJECT"
        self._stop()
        self.get_logger().info("Searching for coloured object...")

    def _find_object(self):
        obj = self.obj_det
        if obj:
            r = self._lidar_at(obj.bearing_rad)
            if r and r <= OBJECT_SEARCH_RADIUS_M:
                self.auto_state = "CAPTURE_OBJECT"
                self._stop(); return
            # also accept area-based distance if lidar angle misses it
            if obj.distance_hint_m <= OBJECT_SEARCH_RADIUS_M:
                self.auto_state = "CAPTURE_OBJECT"
                self._stop(); return
        self._cmd(0.0, 0.3)   # sweep slowly

    def _capture_object(self):
        obj = self.obj_det
        if obj is None:
            self.auto_state = "FIND_OBJECT"; return

        lidar_r = self._lidar_at(obj.bearing_rad)
        dist    = lidar_r if lidar_r else obj.distance_hint_m
        result  = self._wp_result or WaypointResult(waypoint_index=self.wp_idx + 1)

        if not self.obj_photo_done:
            path = self._save_photo(f"wp{self.wp_idx+1}_{obj.label}", obj)
            result.object_label       = obj.label
            result.object_photo       = path
            result.object_distance_m  = dist
            result.object_bearing_rad = obj.bearing_rad
            self.obj_photo_done = True
            self.journey.append(result)

        self.get_logger().info(
            f"WP{self.wp_idx+1}: {obj.label} "
            f"conf={obj.confidence:.2f} "
            f"dist={dist:.2f}m "
            f"bearing={math.degrees(obj.bearing_rad):.1f}deg"
        )

        self.wp_idx += 1
        if self.wp_idx >= len(WAYPOINTS):
            self.auto_state = "DONE"
            self._stop()
            self._summary()
        else:
            self.auto_state = "NAVIGATE"
            self._stop()
            self.get_logger().info(f"Heading to WP{self.wp_idx+1}")

    # =========================================================================
    # Journey summary
    # =========================================================================

    def _summary(self):
        elapsed = time.time() - self.start_time
        m, s = divmod(int(elapsed), 60)
        lines = [
            "", "=" * 55,
            "        MISSION COMPLETE — JOURNEY SUMMARY",
            "=" * 55,
            f"  Waypoints visited : {len(self.journey)} / {len(WAYPOINTS)}",
            f"  Mission time      : {m}m {s}s",
            f"  Photos saved to   : {os.path.abspath(PHOTOS_DIR)}",
            "-" * 55,
        ]
        for r in self.journey:
            lines.append(f"  WP {r.waypoint_index}:")
            lines.append(f"    Cone photo   : {r.cone_photo or 'NOT TAKEN'}")
            if r.object_label:
                lines.append(f"    Object       : {r.object_label}")
                lines.append(f"    Distance     : {r.object_distance_m:.2f}m" if r.object_distance_m else "    Distance     : unknown")
                lines.append(f"    Object photo : {r.object_photo or 'NOT TAKEN'}")
            else:
                lines.append("    Object       : NOT FOUND")
            lines.append("")
        lines.append("=" * 55)
        summary = "\n".join(lines)
        self.get_logger().info(summary)
        print(summary)

    def destroy_node(self):
        self._stop()
        super().destroy_node()


# =============================================================================
# Entry point
# =============================================================================

def main(args=None):
    rclpy.init(args=args)
    node = Part2MissionController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop()
        node._summary()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
