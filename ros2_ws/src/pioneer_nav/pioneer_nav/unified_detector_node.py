#!/usr/bin/env python3
"""
unified_detector_node.py
------------------------
Combines Greek letter detection (ONNX CNN) with red/yellow colour object
detection into a single ROS2 node.

Publishes
---------
/detected_letter        std_msgs/String   — confirmed letter name
/detections/colour      std_msgs/String   — JSON blob for each colour detection
/detections/image       sensor_msgs/Image — annotated BGR frame

Subscribes
----------
/oak/rgb/image_raw      sensor_msgs/Image  — camera feed
/oak/stereo/image_raw   sensor_msgs/Image  — 16UC1 depth map in millimetres
/odom                   nav_msgs/Odometry  — robot pose + heading
/robot_state            std_msgs/String    — optional; "MAPPING" gates detection

Sign region detection — letter-first approach
---------------------------------------------
Rather than finding white paper and hoping the letter is inside it, this
approach finds the letter strokes first (thick dark edges) and works outwards
to locate the paper boundary. This is robust to any background because the
thick black marker strokes are the most consistent feature regardless of
lighting, background colour, or paper placement.

Pipeline:
  1. Canny edge detection on blurred grayscale
  2. Dilate edges to connect nearby strokes into solid blobs
  3. Find contours — look for blobs with the right size and aspect ratio
     to be a hand-drawn Greek letter (not too small, not too large)
  4. Expand the bounding box outward by a margin to estimate the paper area
  5. Verify the expanded region is actually bright (paper is white/light)
  6. Score candidates by size and proximity to image centre

Parameters (all tunable at launch):
  -p min_letter_area:=500        min contour area to be considered a letter stroke
  -p max_letter_area_frac:=0.15  max fraction of image area for a letter stroke
  -p paper_expand:=0.6           how much to expand letter bbox to find paper (fraction)
  -p min_paper_brightness:=80    min mean brightness of expanded paper region
  -p confident_duration_s:=2.0   seconds detection must be held before logging
"""

import json
import math
import os
import time

import cv2
import numpy as np
import onnxruntime as ort
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

# ── HSV colour ranges ────────────────────────────────────────────────────────
RED_LOWER_1  = np.array([  0, 150, 100], dtype=np.uint8)
RED_UPPER_1  = np.array([ 10, 255, 255], dtype=np.uint8)
RED_LOWER_2  = np.array([168, 150, 100], dtype=np.uint8)
RED_UPPER_2  = np.array([179, 255, 255], dtype=np.uint8)
YELLOW_LOWER = np.array([ 22, 150, 150], dtype=np.uint8)
YELLOW_UPPER = np.array([ 32, 255, 255], dtype=np.uint8)

HFOV_RAD = math.radians(71.0)


def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def depth_at_point(depth_image: np.ndarray, cx: int, cy: int,
                   patch: int = 5) -> float | None:
    h, w    = depth_image.shape
    x0, x1  = max(0, cx - patch), min(w, cx + patch)
    y0, y1  = max(0, cy - patch), min(h, cy + patch)
    region  = depth_image[y0:y1, x0:x1].astype(np.float32)
    valid   = region[region > 0]
    if valid.size == 0:
        return None
    return float(np.median(valid)) / 1000.0


def world_position(robot_x, robot_y, robot_yaw, distance_m, bearing_rad):
    angle = robot_yaw + bearing_rad
    return (round(robot_x + distance_m * math.cos(angle), 3),
            round(robot_y + distance_m * math.sin(angle), 3))


class DetectionTracker:
    def __init__(self):
        self.first_seen: float = 0.0
        self.logged:     bool  = False
        self.last_photo: float = 0.0

    def seen(self, now: float):
        if self.first_seen == 0.0:
            self.first_seen = now

    def reset(self):
        self.first_seen = 0.0
        self.logged     = False

    def elapsed(self, now: float) -> float:
        return (now - self.first_seen) if self.first_seen > 0 else 0.0


class UnifiedDetectorNode(Node):

    def __init__(self):
        super().__init__("unified_detector")

        # ── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter("topic",                   "/oak/rgb/image_raw")
        self.declare_parameter("depth_topic",             "/oak/stereo/image_raw")
        self.declare_parameter("min_letter_area",         500)    # min px² for a letter stroke blob
        self.declare_parameter("max_letter_area_frac",    0.15)   # max fraction of image area
        self.declare_parameter("paper_expand",            0.6)    # expand factor to find paper from letter
        self.declare_parameter("min_paper_brightness",    80)     # min mean brightness of paper region
        self.declare_parameter("confidence_threshold",    0.5)
        self.declare_parameter("process_every_n_frames",  3)
        self.declare_parameter("confirmations_required",  3)
        self.declare_parameter("min_colour_area",         3000.0)
        self.declare_parameter("require_mapping_state",   False)
        self.declare_parameter("confident_duration_s",    2.0)
        self.declare_parameter("photo_cooldown_s",        5.0)

        topic                  = self.get_parameter("topic").value
        depth_topic            = self.get_parameter("depth_topic").value
        self.min_letter_area   = int(self.get_parameter("min_letter_area").value)
        self.max_letter_frac   = float(self.get_parameter("max_letter_area_frac").value)
        self.paper_expand      = float(self.get_parameter("paper_expand").value)
        self.min_paper_bright  = int(self.get_parameter("min_paper_brightness").value)
        self.conf_thresh       = self.get_parameter("confidence_threshold").value
        self.process_every     = int(self.get_parameter("process_every_n_frames").value)
        self.confirms_req      = int(self.get_parameter("confirmations_required").value)
        self.min_colour_area   = float(self.get_parameter("min_colour_area").value)
        self.require_mapping   = bool(self.get_parameter("require_mapping_state").value)
        self.confident_dur     = float(self.get_parameter("confident_duration_s").value)
        self.photo_cooldown    = float(self.get_parameter("photo_cooldown_s").value)

        self.get_logger().info(
            f"Letter-first detection: "
            f"min_letter_area={self.min_letter_area}  "
            f"paper_expand={self.paper_expand}  "
            f"min_paper_brightness={self.min_paper_bright}"
        )

        # ── ONNX model ────────────────────────────────────────────────────────
        model_path = os.path.join(os.path.dirname(__file__), "greek_classifier.onnx")
        self.session    = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.get_logger().info("ONNX model loaded")

        self.classes: dict[int, str] = {}
        class_file = os.path.join(os.path.dirname(__file__), "greek_classes.txt")
        with open(class_file) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    self.classes[int(parts[0])] = parts[1]

        # ── State ─────────────────────────────────────────────────────────────
        self.active         = not self.require_mapping
        self.frame_count    = 0
        self.depth_image: np.ndarray | None = None

        self.robot_x   = 0.0
        self.robot_y   = 0.0
        self.robot_yaw = 0.0

        self.letter_last_label  : str | None = None
        self.letter_frame_count : int        = 0
        self.letter_cx          : int        = 0
        self.letter_cy          : int        = 0
        self.letter_tracker = DetectionTracker()

        self.colour_trackers: dict[str, DetectionTracker] = {
            "red_obstacle":    DetectionTracker(),
            "yellow_obstacle": DetectionTracker(),
        }
        self.colour_centres: dict[str, tuple[int, int]] = {
            "red_obstacle":    (0, 0),
            "yellow_obstacle": (0, 0),
        }

        # ── Logging dirs ──────────────────────────────────────────────────────
        self.log_dir   = os.path.expanduser("~/part3_logs")
        self.photo_dir = os.path.join(self.log_dir, "colour_detections")
        os.makedirs(self.photo_dir, exist_ok=True)
        self.log_file  = os.path.join(self.log_dir, "detections_log.jsonl")

        # ── Publishers ────────────────────────────────────────────────────────
        self.letter_pub = self.create_publisher(String, "/detected_letter",   10)
        self.colour_pub = self.create_publisher(String, "/detections/colour", 10)
        self.image_pub  = self.create_publisher(Image,  "/detections/image",  10)

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(Image,    topic,          self.image_callback, 10)
        self.create_subscription(Image,    depth_topic,    self.depth_callback, 10)
        self.create_subscription(Odometry, "/odom",        self.odom_callback,  10)
        self.create_subscription(String,   "/robot_state", self.state_callback, 10)

        self.get_logger().info(f"RGB   topic: {topic}")
        self.get_logger().info(f"Depth topic: {depth_topic}")
        if self.require_mapping:
            self.get_logger().info("Waiting for /robot_state == MAPPING")

    # ── ROS callbacks ────────────────────────────────────────────────────────

    def state_callback(self, msg: String):
        self.active = (msg.data == "MAPPING")

    def odom_callback(self, msg: Odometry):
        self.robot_x   = msg.pose.pose.position.x
        self.robot_y   = msg.pose.pose.position.y
        self.robot_yaw = yaw_from_quaternion(msg.pose.pose.orientation)

    def depth_callback(self, msg: Image):
        raw = np.frombuffer(msg.data, dtype=np.uint16)
        self.depth_image = raw.reshape((msg.height, msg.width))

    def image_callback(self, msg: Image):
        if not self.active:
            return
        self.frame_count += 1
        if self.frame_count % self.process_every != 0:
            return

        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            (msg.height, msg.width, -1))
        bgr  = frame.copy()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        now  = time.monotonic()

        self._process_letter(gray, bgr, msg.width, now)
        self._process_colour(hsv, bgr, msg.width, now)
        self._publish_image(bgr, msg)
        cv2.imwrite("/tmp/unified_detection.png", bgr)

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _get_depth(self, cx: int, cy: int) -> float | None:
        if self.depth_image is None:
            return None
        d = depth_at_point(self.depth_image, cx, cy)
        return d if (d is not None and d > 0.05) else None

    def _log_detection(self, kind: str, name: str, confidence: float,
                       cx: int, cy: int, img_width: int):
        bearing_rad = self._bearing_from_x(cx, img_width)
        bearing_deg = math.degrees(bearing_rad)
        distance_m  = self._get_depth(cx, cy)

        obj_x = obj_y = None
        if distance_m is not None:
            obj_x, obj_y = world_position(
                self.robot_x, self.robot_y, self.robot_yaw,
                distance_m, bearing_rad)

        record = {
            "type":          kind,
            "name":          name,
            "confidence":    round(confidence, 4),
            "distance_m":    round(distance_m, 3) if distance_m else None,
            "bearing_deg":   round(bearing_deg, 2),
            "object_x":      obj_x,
            "object_y":      obj_y,
            "robot_x":       round(self.robot_x, 4),
            "robot_y":       round(self.robot_y, 4),
            "robot_yaw_deg": round(math.degrees(self.robot_yaw), 2),
            "timestamp":     time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        with open(self.log_file, "a") as f:
            f.write(json.dumps(record) + "\n")

        dist_str = f"{distance_m:.2f}m" if distance_m else "depth N/A"
        self.get_logger().info(
            f"LOGGED {kind}/{name}  dist={dist_str}  "
            f"bearing={bearing_deg:.1f}°  "
            f"robot=({self.robot_x:.2f},{self.robot_y:.2f})  "
            f"obj=({obj_x},{obj_y})"
        )

    def _depth_label(self, cx: int, cy: int) -> str:
        d = self._get_depth(cx, cy)
        return f"{d:.2f}m" if d is not None else "?m"

    # ── Letter pipeline ───────────────────────────────────────────────────────

    def _process_letter(self, gray: np.ndarray, bgr: np.ndarray,
                        img_width: int, now: float):
        region_rect = self.find_sign_region(gray)

        if region_rect is None:
            self.letter_last_label  = None
            self.letter_frame_count = 0
            self.letter_tracker.reset()
            return

        x, y, cw, ch = region_rect
        cv2.rectangle(bgr, (x, y), (x + cw, y + ch), (255, 0, 0), 2)

        region = gray[y:y + ch, x:x + cw]
        letter = self.extract_letter(region)
        if letter.size == 0:
            return

        name, confidence = self.classify(letter)

        if confidence <= self.conf_thresh:
            self.letter_last_label  = None
            self.letter_frame_count = 0
            self.letter_tracker.reset()
            return

        if name == self.letter_last_label:
            self.letter_frame_count += 1
        else:
            self.letter_last_label  = name
            self.letter_frame_count = 1
            self.letter_tracker.reset()

        self.letter_cx = x + cw // 2
        self.letter_cy = y + ch // 2
        self.letter_tracker.seen(now)

        if self.letter_frame_count >= self.confirms_req:
            out      = String()
            out.data = name
            self.letter_pub.publish(out)
            cv2.rectangle(bgr, (x, y), (x + cw, y + ch), (0, 255, 0), 2)
            depth_str = self._depth_label(self.letter_cx, self.letter_cy)
            cv2.putText(bgr,
                        f"{name} ({confidence:.2f})  {depth_str}",
                        (x, max(y - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if (self.letter_tracker.elapsed(now) >= self.confident_dur
                and not self.letter_tracker.logged):
            self._log_detection("letter", name, confidence,
                                 self.letter_cx, self.letter_cy, img_width)
            self.letter_tracker.logged = True

    # ── Sign region detection — letter-first ──────────────────────────────────

    def find_sign_region(self, gray: np.ndarray):
        """
        Find the paper sign by locating the letter strokes first.

        Greek letters drawn with thick black marker on white paper produce
        strong, consistent edge blobs regardless of the background or lighting.
        We find these blobs, then expand outward to recover the full paper area.

        Steps:
          1. Blur to reduce noise, then Canny edge detection
          2. Dilate heavily to merge nearby strokes into solid blobs
          3. Find contours — filter for blobs sized like a letter
             (not tiny noise, not huge background objects)
          4. For each letter-sized blob, expand its bounding box by
             paper_expand fraction in each direction to estimate paper area
          5. Check the expanded region is bright (white paper)
          6. Score by size and distance from image centre
        """
        h, w   = gray.shape
        img_cx = w // 2
        img_cy = h // 2
        img_area = h * w

        # Step 1 — blur and edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges   = cv2.Canny(blurred, 30, 100)

        # Step 2 — dilate to connect nearby strokes into solid blobs
        # A larger kernel merges the strokes of a single letter together
        kernel  = np.ones((25, 25), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        cv2.imwrite("/tmp/debug_edges.png", dilated)

        # Step 3 — find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        best       = None
        best_score = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)

            # Filter: blob must be letter-sized
            # Too small = noise, too large = background object
            if not (self.min_letter_area < area < img_area * self.max_letter_frac):
                continue

            lx, ly, lw, lh = cv2.boundingRect(cnt)

            # Filter: aspect ratio of the letter blob itself
            # Greek letters fit roughly in a 0.2–3.0 aspect range
            aspect = lw / lh if lh > 0 else 0
            if not (0.2 < aspect < 3.0):
                continue

            # Step 4 — expand bounding box to estimate paper area
            # The letter is smaller than the paper, so expand outward
            expand_x = int(lw * self.paper_expand)
            expand_y = int(lh * self.paper_expand)

            px = max(0, lx - expand_x)
            py = max(0, ly - expand_y)
            pw = min(w, lx + lw + expand_x) - px
            ph = min(h, ly + lh + expand_y) - py

            if pw <= 0 or ph <= 0:
                continue

            # Filter: expanded region aspect ratio should look like paper
            paper_aspect = pw / ph if ph > 0 else 0
            if not (0.3 < paper_aspect < 2.5):
                continue

            # Step 5 — check the expanded region is actually bright (paper)
            paper_region = gray[py:py + ph, px:px + pw]
            if paper_region.mean() < self.min_paper_bright:
                continue

            # Step 6 — score by size and centre proximity
            cx_r  = px + pw // 2
            cy_r  = py + ph // 2
            dist  = math.hypot(cx_r - img_cx, cy_r - img_cy)
            max_d = math.hypot(img_cx, img_cy)
            score = (area / img_area) * (1 - 0.5 * dist / max_d)

            if score > best_score:
                best_score = score
                best       = (px, py, pw, ph)

        return best

    # ── Letter extraction and classification (unchanged) ──────────────────────

    def extract_letter(self, region: np.ndarray) -> np.ndarray:
        IMG_SIZE = 64
        blur     = cv2.GaussianBlur(region, (5, 5), 0)
        thresh   = cv2.adaptiveThreshold(blur, 255,
                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                       cv2.THRESH_BINARY_INV, 11, 4)
        conts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        lx = region.shape[1]; ly = region.shape[0]; lw = lh = 0
        for cnt in conts:
            if cv2.contourArea(cnt) > 20:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                lx = min(lx, bx); ly = min(ly, by)
                lw = max(lw, bx + bw); lh = max(lh, by + bh)
        lw -= lx; lh -= ly
        letter  = region[ly:ly+lh, lx:lx+lw] if (lw > 0 and lh > 0) else region
        scale   = (IMG_SIZE - 8) / max(letter.shape)
        new_w   = int(letter.shape[1] * scale)
        new_h   = int(letter.shape[0] * scale)
        resized = cv2.resize(letter, (new_w, new_h))
        result  = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8) * 255
        y_off   = (IMG_SIZE - new_h) // 2
        x_off   = (IMG_SIZE - new_w) // 2
        result[y_off:y_off+new_h, x_off:x_off+new_w] = resized
        return result

    def classify(self, letter_region: np.ndarray):
        inp     = letter_region.astype(np.float32) / 255.0
        inp     = inp[np.newaxis, np.newaxis, :, :]
        outputs = self.session.run(None, {self.input_name: inp})
        probs   = outputs[0][0]
        probs   = np.exp(probs - probs.max())
        probs  /= probs.sum()
        idx        = int(np.argmax(probs))
        confidence = float(probs[idx])
        return self.classes[idx], confidence

    # ── Colour pipeline ───────────────────────────────────────────────────────

    def _process_colour(self, hsv: np.ndarray, bgr: np.ndarray,
                        img_width: int, now: float):
        candidates = {
            "red_obstacle":    self._detect_colour(
                hsv, RED_LOWER_1, RED_UPPER_1, RED_LOWER_2, RED_UPPER_2),
            "yellow_obstacle": self._detect_colour(
                hsv, YELLOW_LOWER, YELLOW_UPPER),
        }

        for label, result in candidates.items():
            tracker = self.colour_trackers[label]

            if result is None:
                tracker.reset()
                continue

            contour, (bx, by, bw, bh) = result
            cx_px = int(bx + bw / 2)
            cy_px = int(by + bh / 2)
            self.colour_centres[label] = (cx_px, cy_px)

            tracker.seen(now)
            bearing_rad = self._bearing_from_x(cx_px, img_width)
            depth_str   = self._depth_label(cx_px, cy_px)

            colour = (0, 0, 255) if "red" in label else (0, 255, 255)
            cv2.rectangle(bgr, (bx, by), (bx + bw, by + bh), colour, 3)
            cv2.putText(bgr,
                        f"{label}  {depth_str}",
                        (bx, max(by - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)

            payload      = String()
            payload.data = json.dumps({
                "label":         label,
                "center_x":      cx_px,
                "center_y":      cy_px,
                "bearing_deg":   round(math.degrees(bearing_rad), 2),
                "robot_x":       self.robot_x,
                "robot_y":       self.robot_y,
                "robot_yaw_deg": round(math.degrees(self.robot_yaw), 2),
                "elapsed_s":     round(tracker.elapsed(now), 2),
                "timestamp":     time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
            self.colour_pub.publish(payload)

            if tracker.elapsed(now) >= self.confident_dur and not tracker.logged:
                self._log_detection(label, label, 1.0,
                                    cx_px, cy_px, img_width)
                tracker.logged = True

                if now - tracker.last_photo >= self.photo_cooldown:
                    fname = f"{label}_{time.strftime('%Y%m%d_%H%M%S')}.png"
                    cv2.imwrite(os.path.join(self.photo_dir, fname), bgr)
                    tracker.last_photo = now
                    self.get_logger().info(f"Photo saved: {fname}")

    # ── Image publisher ───────────────────────────────────────────────────────

    def _publish_image(self, bgr: np.ndarray, original_msg: Image):
        out              = Image()
        out.header       = original_msg.header
        out.height       = bgr.shape[0]
        out.width        = bgr.shape[1]
        out.encoding     = "bgr8"
        out.is_bigendian = 0
        out.step         = bgr.shape[1] * 3
        out.data         = bgr.tobytes()
        self.image_pub.publish(out)

    # ── Colour helpers ────────────────────────────────────────────────────────

    def _detect_colour(self, hsv, lower1, upper1, lower2=None, upper2=None):
        mask = cv2.inRange(hsv, lower1, upper1)
        if lower2 is not None:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower2, upper2))
        k    = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        biggest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(biggest) < self.min_colour_area:
            return None
        return biggest, cv2.boundingRect(biggest)

    @staticmethod
    def _bearing_from_x(x_px: float, img_width: int) -> float:
        norm = (x_px - img_width / 2.0) / (img_width / 2.0)
        return norm * (HFOV_RAD / 2.0)


def main(args=None):
    rclpy.init(args=args)
    node = UnifiedDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
