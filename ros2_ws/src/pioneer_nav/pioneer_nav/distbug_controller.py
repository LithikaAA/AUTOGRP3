#!/usr/bin/env python3

# note to self:
# this is a "whole mission" controller for AUTO4508 part 2 style testing.
# it is built to work in simulation first, then be adapted to the real robot.
#
# features included:
# - GPS waypoint driving
# - lidar obstacle avoidance
# - cone detection with camera
# - save photo at each cone
# - keep marker on robot's right while approaching
# - detect nearby coloured object
# - estimate its distance using lidar + camera bearing

import math
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from sensor_msgs.msg import NavSatFix, LaserScan, Image
from cv_bridge import CvBridge


# =========================
# helper functions
# =========================

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wrap_to_pi(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def bearing_from_image_x(x_px: float, width: int, hfov_rad: float) -> float:
    # x at image centre -> 0 rad
    # left negative, right positive
    norm = (x_px - (width / 2.0)) / (width / 2.0)
    return norm * (hfov_rad / 2.0)


def is_finite_number(x: float) -> bool:
    return math.isfinite(x) and not math.isnan(x)


def hsv_mask_orange(hsv_img: np.ndarray) -> np.ndarray:
    # note to self:
    # orange cone range. may need tuning in sim / on real bot.
    lower = np.array([5, 100, 100], dtype=np.uint8)
    upper = np.array([25, 255, 255], dtype=np.uint8)
    return cv2.inRange(hsv_img, lower, upper)


def largest_contour(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


@dataclass
class Detection:
    label: str
    center_x: float
    center_y: float
    area: float
    bearing_rad: float
    bbox: Tuple[int, int, int, int]


# =========================
# main node
# =========================

class Part2MissionController(Node):
    def __init__(self):
        super().__init__("part2_mission_controller")

        # -------------------------
        # params
        # -------------------------
        self.declare_parameter("gps_topic", "/fix")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("image_topic", "/camera/image")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")

        # waypoint list as lat, lon pairs flattened
        self.declare_parameter(
            "gps_waypoints",
            [
                -31.980000, 115.820000,
                -31.980002, 115.820030,
                -31.980010, 115.820030,
                -31.980000, 115.820000
            ]
        )

        # control tuning
        self.declare_parameter("max_linear_speed", 0.5)
        self.declare_parameter("max_angular_speed", 1.0)
        self.declare_parameter("goal_radius_m", 1.2)
        self.declare_parameter("cone_stop_distance_m", 1.4)
        self.declare_parameter("object_search_radius_m", 4.0)

        self.declare_parameter("front_obstacle_dist_m", 0.9)
        self.declare_parameter("critical_obstacle_dist_m", 0.5)

        self.declare_parameter("camera_hfov_rad", 1.089)
        self.declare_parameter("cone_min_area", 500.0)
        self.declare_parameter("object_min_area", 350.0)

        self.declare_parameter("photos_dir", "mission_photos")

        gps_topic = self.get_parameter("gps_topic").value
        scan_topic = self.get_parameter("scan_topic").value
        image_topic = self.get_parameter("image_topic").value
        cmd_vel_topic = self.get_parameter("cmd_vel_topic").value

        flat_wps = list(self.get_parameter("gps_waypoints").value)
        if len(flat_wps) % 2 != 0:
            raise ValueError("gps_waypoints must be [lat1, lon1, lat2, lon2, ...]")

        self.gps_waypoints: List[Tuple[float, float]] = []
        for i in range(0, len(flat_wps), 2):
            self.gps_waypoints.append((float(flat_wps[i]), float(flat_wps[i + 1])))

        self.max_linear_speed = float(self.get_parameter("max_linear_speed").value)
        self.max_angular_speed = float(self.get_parameter("max_angular_speed").value)
        self.goal_radius_m = float(self.get_parameter("goal_radius_m").value)
        self.cone_stop_distance_m = float(self.get_parameter("cone_stop_distance_m").value)
        self.object_search_radius_m = float(self.get_parameter("object_search_radius_m").value)

        self.front_obstacle_dist_m = float(self.get_parameter("front_obstacle_dist_m").value)
        self.critical_obstacle_dist_m = float(self.get_parameter("critical_obstacle_dist_m").value)

        self.camera_hfov_rad = float(self.get_parameter("camera_hfov_rad").value)
        self.cone_min_area = float(self.get_parameter("cone_min_area").value)
        self.object_min_area = float(self.get_parameter("object_min_area").value)

        self.photos_dir = str(self.get_parameter("photos_dir").value)
        os.makedirs(self.photos_dir, exist_ok=True)

        # -------------------------
        # state
        # -------------------------
        self.auto_state = "NAVIGATE"  # NAVIGATE, APPROACH_CONE, CAPTURE_CONE, FIND_OBJECT, CAPTURE_OBJECT, DONE
        self.current_wp_idx = 0

        self.have_gps = False
        self.have_scan = False
        self.have_image = False

        self.current_lat = 0.0
        self.current_lon = 0.0
        self.current_heading = 0.0  # estimated from GPS motion
        self.last_lat = None
        self.last_lon = None
        self.last_gps_time = None

        self.origin_lat = None
        self.origin_lon = None

        self.front_min = float("inf")
        self.left_min = float("inf")
        self.right_min = float("inf")
        self.last_scan = None

        self.latest_bgr = None
        self.latest_hsv = None
        self.image_width = None
        self.image_height = None
        self.bridge = CvBridge()

        self.latest_cone_detection: Optional[Detection] = None
        self.latest_object_detection: Optional[Detection] = None

        self.cone_photo_taken = False
        self.object_photo_taken = False
        self.last_object_result = None

        # -------------------------
        # pubs/subs
        # -------------------------
        self.cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)

        self.create_subscription(NavSatFix, gps_topic, self.gps_callback, 10)
        self.create_subscription(LaserScan, scan_topic, self.scan_callback, 10)
        self.create_subscription(Image, image_topic, self.image_callback, 10)

        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info("Part 2 mission controller started")
        self.get_logger().info(f"Waypoints loaded: {len(self.gps_waypoints)}")
        self.get_logger().info("Start in AUTO mode.")

    # =========================
    # callbacks
    # =========================

    def gps_callback(self, msg: NavSatFix):
        if not is_finite_number(msg.latitude) or not is_finite_number(msg.longitude):
            return

        self.have_gps = True
        now = self.get_clock().now().nanoseconds * 1e-9

        if self.origin_lat is None:
            self.origin_lat = msg.latitude
            self.origin_lon = msg.longitude
            self.get_logger().info(
                f"GPS origin set to lat={self.origin_lat:.8f}, lon={self.origin_lon:.8f}"
            )

        # heading estimate from GPS motion
        if self.last_lat is not None and self.last_lon is not None and self.last_gps_time is not None:
            dt = now - self.last_gps_time
            if dt > 0.1:
                dx, dy = self.latlon_to_local_xy(
                    msg.latitude,
                    msg.longitude,
                    self.last_lat,
                    self.last_lon,
                )
                dist = math.hypot(dx, dy)
                if dist > 0.08:
                    self.current_heading = math.atan2(dy, dx)

        self.current_lat = msg.latitude
        self.current_lon = msg.longitude
        self.last_lat = msg.latitude
        self.last_lon = msg.longitude
        self.last_gps_time = now

    def scan_callback(self, msg: LaserScan):
        self.have_scan = True
        self.last_scan = msg

        ranges = []
        for r in msg.ranges:
            if math.isfinite(r):
                ranges.append(r)
            else:
                ranges.append(float("inf"))

        n = len(ranges)
        if n == 0:
            return

        mid = n // 2
        front_half_width = max(10, n // 16)

        front = ranges[max(0, mid - front_half_width): min(n, mid + front_half_width)]
        left = ranges[min(n - 1, mid + front_half_width):]
        right = ranges[:max(1, mid - front_half_width)]

        self.front_min = min(front) if front else float("inf")
        self.left_min = min(left) if left else float("inf")
        self.right_min = min(right) if right else float("inf")

    def image_callback(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"Image conversion failed: {e}", throttle_duration_sec=2.0)
            return

        self.have_image = True
        self.latest_bgr = bgr
        self.latest_hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        self.image_height, self.image_width = bgr.shape[:2]

        self.latest_cone_detection = self.detect_cone()
        self.latest_object_detection = self.detect_other_object()

    # =========================
    # perception
    # =========================

    def detect_cone(self) -> Optional[Detection]:
        if self.latest_hsv is None or self.image_width is None:
            return None

        mask = hsv_mask_orange(self.latest_hsv)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        cnt = largest_contour(mask)
        if cnt is None:
            return None

        area = cv2.contourArea(cnt)
        if area < self.cone_min_area:
            return None

        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w / 2.0
        cy = y + h / 2.0
        bearing = bearing_from_image_x(cx, self.image_width, self.camera_hfov_rad)

        return Detection(
            label="orange_cone",
            center_x=cx,
            center_y=cy,
            area=area,
            bearing_rad=bearing,
            bbox=(x, y, w, h),
        )

    def detect_other_object(self) -> Optional[Detection]:
        if self.latest_hsv is None or self.latest_bgr is None or self.image_width is None:
            return None

        hsv = self.latest_hsv

        # note to self:
        # simple colored-object detection.
        # we exclude orange so we don't just rediscover the cone.
        # detect red, yellow, blue, green blobs.
        color_ranges = [
            ("red1", np.array([0, 100, 100]), np.array([10, 255, 255])),
            ("red2", np.array([160, 100, 100]), np.array([179, 255, 255])),
            ("yellow", np.array([20, 100, 100]), np.array([35, 255, 255])),
            ("green", np.array([40, 80, 80]), np.array([85, 255, 255])),
            ("blue", np.array([90, 80, 80]), np.array([130, 255, 255])),
        ]

        combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for _, low, high in color_ranges:
            combined |= cv2.inRange(hsv, low, high)

        kernel = np.ones((5, 5), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

        cnt = largest_contour(combined)
        if cnt is None:
            return None

        area = cv2.contourArea(cnt)
        if area < self.object_min_area:
            return None

        shape_label = self.classify_shape(cnt)

        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w / 2.0
        cy = y + h / 2.0
        bearing = bearing_from_image_x(cx, self.image_width, self.camera_hfov_rad)

        return Detection(
            label=shape_label,
            center_x=cx,
            center_y=cy,
            area=area,
            bearing_rad=bearing,
            bbox=(x, y, w, h),
        )

    def classify_shape(self, contour) -> str:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        vertices = len(approx)

        if vertices == 3:
            return "triangle"
        if vertices == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect = w / float(h) if h > 0 else 0.0
            if 0.85 <= aspect <= 1.15:
                return "square"
            return "rectangle"
        if vertices > 6:
            return "circle"
        return "unknown_shape"

    # =========================
    # geometry
    # =========================

    def latlon_to_local_xy(self, lat: float, lon: float, lat0: float, lon0: float) -> Tuple[float, float]:
        # simple local tangent approximation, fine for short campus-scale distances
        r_earth = 6378137.0
        dlat = math.radians(lat - lat0)
        dlon = math.radians(lon - lon0)
        x = r_earth * dlon * math.cos(math.radians((lat + lat0) / 2.0))
        y = r_earth * dlat
        return x, y

    def current_xy(self) -> Tuple[float, float]:
        if self.origin_lat is None:
            return 0.0, 0.0
        return self.latlon_to_local_xy(self.current_lat, self.current_lon, self.origin_lat, self.origin_lon)

    def waypoint_xy(self, idx: int) -> Tuple[float, float]:
        lat, lon = self.gps_waypoints[idx]
        return self.latlon_to_local_xy(lat, lon, self.origin_lat, self.origin_lon)

    def distance_to_current_waypoint(self) -> float:
        if self.origin_lat is None or self.current_wp_idx >= len(self.gps_waypoints):
            return float("inf")
        x, y = self.current_xy()
        gx, gy = self.waypoint_xy(self.current_wp_idx)
        return math.hypot(gx - x, gy - y)

    def heading_to_current_waypoint(self) -> float:
        x, y = self.current_xy()
        gx, gy = self.waypoint_xy(self.current_wp_idx)
        return math.atan2(gy - y, gx - x)

    def lidar_range_at_bearing(self, bearing_rad: float) -> Optional[float]:
        if self.last_scan is None:
            return None

        angle_min = self.last_scan.angle_min
        angle_inc = self.last_scan.angle_increment
        n = len(self.last_scan.ranges)

        idx = int(round((bearing_rad - angle_min) / angle_inc))
        if idx < 0 or idx >= n:
            return None

        window = 4
        vals = []
        for i in range(max(0, idx - window), min(n, idx + window + 1)):
            r = self.last_scan.ranges[i]
            if math.isfinite(r):
                vals.append(r)

        if not vals:
            return None
        return min(vals)

    # =========================
    # actions
    # =========================

    def stop_robot(self):
        self.publish_cmd(0.0, 0.0)

    def publish_cmd(self, linear_x: float, angular_z: float):
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        self.cmd_pub.publish(msg)

    def save_current_image(self, prefix: str) -> Optional[str]:
        if self.latest_bgr is None:
            return None

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.photos_dir, f"{prefix}_{timestamp}.png")
        cv2.imwrite(path, self.latest_bgr)
        self.get_logger().info(f"Saved image: {path}")
        return path

    # =========================
    # behaviour
    # =========================

    def control_loop(self):
        self.run_auto()

    def run_auto(self):
        if not (self.have_gps and self.have_scan and self.have_image):
            self.stop_robot()
            return

        if self.current_wp_idx >= len(self.gps_waypoints):
            self.auto_state = "DONE"

        if self.auto_state == "DONE":
            self.stop_robot()
            return

        if self.front_min < self.critical_obstacle_dist_m:
            # hard safety turn
            turn_dir = -1.0 if self.left_min < self.right_min else 1.0
            self.publish_cmd(0.0, 0.8 * turn_dir)
            return

        if self.auto_state == "NAVIGATE":
            self.do_navigate()
        elif self.auto_state == "APPROACH_CONE":
            self.do_approach_cone()
        elif self.auto_state == "CAPTURE_CONE":
            self.do_capture_cone()
        elif self.auto_state == "FIND_OBJECT":
            self.do_find_object()
        elif self.auto_state == "CAPTURE_OBJECT":
            self.do_capture_object()
        else:
            self.stop_robot()

    def do_navigate(self):
        dist = self.distance_to_current_waypoint()
        goal_heading = self.heading_to_current_waypoint()
        heading_error = wrap_to_pi(goal_heading - self.current_heading)

        # if near waypoint, start looking for the actual cone
        if dist < self.goal_radius_m:
            self.auto_state = "APPROACH_CONE"
            self.cone_photo_taken = False
            self.object_photo_taken = False
            self.last_object_result = None
            self.stop_robot()
            self.get_logger().info(f"Near waypoint {self.current_wp_idx + 1}, switching to cone approach")
            return

        # simple obstacle-aware navigation
        if self.front_min < self.front_obstacle_dist_m:
            turn_dir = -1.0 if self.left_min < self.right_min else 1.0
            self.publish_cmd(0.08, 0.6 * turn_dir)
            return

        if abs(heading_error) > 0.35:
            self.publish_cmd(0.0, clamp(1.4 * heading_error, -0.8, 0.8))
            return

        lin = clamp(0.3 + 0.15 * dist, 0.0, self.max_linear_speed)
        ang = clamp(1.2 * heading_error, -self.max_angular_speed, self.max_angular_speed)
        self.publish_cmd(lin, ang)

    def do_approach_cone(self):
        cone = self.latest_cone_detection

        if cone is None:
            # spin slowly to find cone
            self.publish_cmd(0.0, 0.35)
            return

        # keep marker on robot's right:
        # note to self:
        # instead of centering cone dead ahead, aim so cone sits slightly RIGHT in image.
        desired_bearing = 0.18  # positive -> cone on right side of camera
        bearing_error = wrap_to_pi(cone.bearing_rad - desired_bearing)

        cone_range = self.lidar_range_at_bearing(cone.bearing_rad)
        if cone_range is not None and cone_range < self.cone_stop_distance_m:
            self.auto_state = "CAPTURE_CONE"
            self.stop_robot()
            self.get_logger().info("Cone reached. Capturing marker photo.")
            return

        if self.front_min < self.front_obstacle_dist_m * 0.9:
            turn_dir = -1.0 if self.left_min < self.right_min else 1.0
            self.publish_cmd(0.0, 0.5 * turn_dir)
            return

        lin = 0.12 if abs(bearing_error) < 0.35 else 0.04
        ang = clamp(1.5 * bearing_error, -0.6, 0.6)
        self.publish_cmd(lin, ang)

    def do_capture_cone(self):
        if not self.cone_photo_taken:
            self.save_current_image(f"waypoint_{self.current_wp_idx + 1}_cone")
            self.cone_photo_taken = True

        self.auto_state = "FIND_OBJECT"
        self.stop_robot()
        self.get_logger().info("Now searching for nearby object.")
        return

    def do_find_object(self):
        obj = self.latest_object_detection

        # if we see an object, move to capture state
        if obj is not None:
            obj_range = self.lidar_range_at_bearing(obj.bearing_rad)
            if obj_range is not None and obj_range <= self.object_search_radius_m:
                self.auto_state = "CAPTURE_OBJECT"
                self.stop_robot()
                return

        # otherwise sweep slowly
        self.publish_cmd(0.0, 0.3)

    def do_capture_object(self):
        obj = self.latest_object_detection
        if obj is None:
            self.auto_state = "FIND_OBJECT"
            return

        obj_range = self.lidar_range_at_bearing(obj.bearing_rad)

        if not self.object_photo_taken:
            self.save_current_image(f"waypoint_{self.current_wp_idx + 1}_{obj.label}")
            self.object_photo_taken = True

        self.last_object_result = {
            "waypoint_index": self.current_wp_idx + 1,
            "shape": obj.label,
            "bearing_rad": obj.bearing_rad,
            "distance_m": obj_range,
        }

        self.get_logger().info(
            f"Waypoint {self.current_wp_idx + 1}: object={obj.label}, distance={obj_range}"
        )

        # move to next waypoint
        self.current_wp_idx += 1

        if self.current_wp_idx >= len(self.gps_waypoints):
            self.auto_state = "DONE"
            self.stop_robot()
            self.get_logger().info("Mission complete.")
        else:
            self.auto_state = "NAVIGATE"
            self.stop_robot()
            self.get_logger().info(f"Proceeding to waypoint {self.current_wp_idx + 1}")

    # =========================
    # cleanup
    # =========================

    def destroy_node(self):
        self.stop_robot()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Part2MissionController()

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
