#!/usr/bin/env python3
"""
AUTO4508 Part 3 - Robot Monitor GUI
====================================
Runs on a LAPTOP connected to the same network as the robot.
Subscribes to ROS2 topics published by the robot's nodes and displays
everything in one window. It never publishes anything — it is read-only.

HOW THE NODES CONNECT TO THIS GUI:
-----------------------------------
  mission_manager.py  →  /robot_state        →  state badge + action label
  control_node.py     →  /robot/pose         →  map arrow + position display
  control_node.py     →  /arena_status       →  arena debug panel
  unified_detector    →  /detected_letter    →  detection log + status panel
  unified_detector    →  /detections/colour  →  detection log entry only
  unified_detector    →  /detections/image   →  bottom-right live photo panel
  slam_toolbox        →  /map                →  map panel background
  mission_manager.py  →  /planned_path       →  path overlay on map
  OAK-D driver        →  /oak/rgb/image_raw  →  camera feed panel
  detections_log.jsonl (file, polled every 2s) → map markers (one per obstacle)

Requirements:
    sudo apt install python3-pyqt5 python3-opencv
    Source ROS2 before running:
        source /opt/ros/humble/setup.bash
    Set matching ROS_DOMAIN_ID on robot and laptop:
        export ROS_DOMAIN_ID=42

Run:
    cd ros2_ws/src/pioneer_nav/pioneer_nav
    python3 GUI.py
"""

import sys
import json
import math
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path as FilePath

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from nav_msgs.msg import OccupancyGrid, Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QFrame, QScrollArea, QSizePolicy, QProgressBar, QPushButton
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QPointF, QRect
from PyQt5.QtGui import (
    QImage, QPixmap, QFont, QColor, QPainter,
    QPen, QBrush, QPolygonF
)


# ──────────────────────────────────────────────
#  COLOUR PALETTE
# ──────────────────────────────────────────────
BG        = "#0d1117"
PANEL     = "#161b22"
BORDER    = "#30363d"
ACCENT    = "#58a6ff"
GREEN     = "#3fb950"
YELLOW    = "#d29922"
RED       = "#f85149"
TEXT      = "#e6edf3"
TEXT_DIM  = "#8b949e"

FONT_UI   = "Ubuntu Mono"
FONT_BODY = "DejaVu Sans"

# Path to the detection log written by unified_detector_node
DETECTION_LOG_PATH = os.path.expanduser("~/part3_logs/detections_log.jsonl")
ESTOP_LOG_DIR = os.path.expanduser("~/part3_logs/estop_events")
BAG_DIR = os.path.expanduser("~/part3_logs/bags")
LIDAR_SELF_MASK_MIN_RANGE = float(os.environ.get("PIONEER_GUI_LIDAR_SELF_MASK_MIN_RANGE", "0.18"))
ARENA_SIZE_M = float(os.environ.get("PIONEER_GUI_ARENA_SIZE_M", "8.0"))


# ──────────────────────────────────────────────
#  SIGNALS
# ──────────────────────────────────────────────
class Signals(QObject):
    camera_frame    = pyqtSignal(np.ndarray)
    colour_image    = pyqtSignal(np.ndarray)   # /detections/image → bottom-right panel
    robot_state     = pyqtSignal(str)
    robot_pose      = pyqtSignal(float, float, float)
    letter_detected = pyqtSignal(str)
    colour_detected = pyqtSignal(dict)
    object_detected = pyqtSignal(dict)
    map_updated     = pyqtSignal(object)
    scan_updated    = pyqtSignal(object)
    path_updated    = pyqtSignal(object)
    arena_updated   = pyqtSignal(dict)
    save_status     = pyqtSignal(str, bool)
    mission_command = pyqtSignal(str)


# ──────────────────────────────────────────────
#  ROS2 NODE
# ──────────────────────────────────────────────
class GUINode(Node):
    def __init__(self, signals: Signals):
        super().__init__("robot_gui")
        self.signals = signals
        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        live_map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Camera feed — raw topic for main camera panel
        self.create_subscription(Image,         "/oak/rgb/image_raw", self._cb_camera,       10)
        self.create_subscription(Image,         "/camera/image",      self._cb_camera,       10)

        # Annotated detection image — feeds bottom-right photo panel
        self.create_subscription(Image,         "/detections/image",  self._cb_colour_image, 10)

        self.create_subscription(String,        "/robot_state",       self._cb_state,        10)
        self.create_subscription(Pose,          "/robot/pose",        self._cb_pose,         10)
        self.create_subscription(String,        "/detected_letter",   self._cb_letter,       10)
        self.create_subscription(String,        "/detections/colour", self._cb_colour,       10)
        self.create_subscription(String,        "/detections/object", self._cb_object,       10)
        self.create_subscription(OccupancyGrid, "/map",               self._cb_map,          map_qos)
        self.create_subscription(OccupancyGrid, "/map",               self._cb_map,          live_map_qos)
        self.create_subscription(LaserScan,     "/scan",              self._cb_scan,         10)
        self.create_subscription(Path,          "/planned_path",      self._cb_path,         10)
        self.create_subscription(String,        "/arena_status",      self._cb_arena,        10)
        self.command_pub = self.create_publisher(String, "/mission_command", 10)

    def publish_command(self, command: str):
        self.command_pub.publish(String(data=command))
        self.get_logger().info(f"GUI command published: {command}")

    def _cb_camera(self, msg):
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            (msg.height, msg.width, -1))
        if msg.encoding != "bgr8":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.signals.camera_frame.emit(frame)

    def _cb_colour_image(self, msg):
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            (msg.height, msg.width, -1))
        if msg.encoding != "bgr8":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.signals.colour_image.emit(frame)

    def _cb_state(self, msg):
        self.signals.robot_state.emit(msg.data)

    def _cb_pose(self, msg):
        q    = msg.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw  = math.atan2(siny, cosy)
        self.signals.robot_pose.emit(msg.position.x, msg.position.y, yaw)

    def _cb_letter(self, msg):
        self.signals.letter_detected.emit(msg.data)

    def _cb_colour(self, msg):
        try:
            self.signals.colour_detected.emit(json.loads(msg.data))
        except Exception:
            pass

    def _cb_object(self, msg):
        try:
            self.signals.object_detected.emit(json.loads(msg.data))
        except Exception:
            pass

    def _cb_map(self, msg):
        self.signals.map_updated.emit(msg)

    def _cb_scan(self, msg):
        self.signals.scan_updated.emit(msg)

    def _cb_path(self, msg):
        self.signals.path_updated.emit(msg)

    def _cb_arena(self, msg):
        try:
            self.signals.arena_updated.emit(json.loads(msg.data))
        except Exception:
            pass


# ──────────────────────────────────────────────
#  UI HELPER FUNCTIONS
# ──────────────────────────────────────────────

def make_panel(title: str) -> tuple:
    frame = QFrame()
    frame.setStyleSheet(f"""
        QFrame {{
            background: {PANEL};
            border: 1px solid {BORDER};
            border-radius: 8px;
        }}
    """)
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(10, 8, 10, 10)
    layout.setSpacing(6)

    if title:
        lbl = QLabel(title.upper())
        lbl.setFont(QFont(FONT_UI, 8, QFont.Bold))
        lbl.setStyleSheet(f"color: {TEXT_DIM}; border: none; background: transparent;")
        layout.addWidget(lbl)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"color: {BORDER}; border: none; background: {BORDER}; max-height: 1px;")
        layout.addWidget(line)

    return frame, layout


def status_row(grid: QGridLayout, row: int, key: str, val: str = "—", val_colour: str = TEXT):
    k = QLabel(key)
    k.setFont(QFont(FONT_UI, 9))
    k.setStyleSheet(f"color: {TEXT_DIM}; border: none; background: transparent;")
    v = QLabel(val)
    v.setFont(QFont(FONT_UI, 9, QFont.Bold))
    v.setStyleSheet(f"color: {val_colour}; border: none; background: transparent;")
    grid.addWidget(k, row, 0)
    grid.addWidget(v, row, 1)
    return v


# ──────────────────────────────────────────────
#  MAP WIDGET
# ──────────────────────────────────────────────
class MapWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(f"background: {BG};")

        self._map_img   = None
        self._map_res   = 0.05
        self._map_ox    = 0.0
        self._map_oy    = 0.0
        self._map_w     = 0
        self._map_h     = 0
        self._map_data  = None
        self._last_map_time = None
        self._slam_overlay_img = None
        self._slam_overlay_key = None

        self._robot_x   = 0.0
        self._robot_y   = 0.0
        self._robot_yaw = 0.0
        self._have_pose  = False
        self._arena_origin_x = None
        self._arena_origin_y = None

        self._arena_size = ARENA_SIZE_M
        self._arena_half = self._arena_size / 2.0
        self._coverage_res = 0.25
        self._coverage_n = int(self._arena_size / self._coverage_res)
        self._free_cells = np.zeros((self._coverage_n, self._coverage_n), dtype=bool)
        self._obstacle_cells = np.zeros((self._coverage_n, self._coverage_n), dtype=bool)

        self._path_trace = []
        self._scan_hits = []
        self._max_trace_points = 2000
        self._max_scan_hits = 5000
        self._scan_hit_lifetime = 10.0

        # Each entry: (world_x, world_y, label, colour_str)
        # Rebuilt every 2 seconds from detections_log.jsonl — never appended directly
        self._detections = []

        self._path = []

    def update_map(self, msg):
        self._map_res = msg.info.resolution
        self._map_ox  = msg.info.origin.position.x
        self._map_oy  = msg.info.origin.position.y
        self._map_w   = msg.info.width
        self._map_h   = msg.info.height

        data = np.array(msg.data, dtype=np.int8).reshape((self._map_h, self._map_w))
        self._map_data = data.copy()
        img = np.full((self._map_h, self._map_w, 3), 205, dtype=np.uint8)
        img[data == 0] = [254, 254, 254]
        img[data > 50] = [0, 0, 0]
        img = np.flipud(img)
        h, w, _ = img.shape
        self._map_img = QImage(img.tobytes(), w, h, 3*w, QImage.Format_RGB888).copy()
        self._last_map_time = time.time()
        self._slam_overlay_img = None
        self._slam_overlay_key = None
        self.update()

    def update_pose(self, x, y, yaw):
        self._robot_x   = x
        self._robot_y   = y
        self._robot_yaw = yaw
        self._have_pose = True
        if self._arena_origin_x is None or self._arena_origin_y is None:
            self._arena_origin_x = x
            self._arena_origin_y = y
            self._slam_overlay_img = None
            self._slam_overlay_key = None
        if not self._path_trace or math.hypot(
            x - self._path_trace[-1][0],
            y - self._path_trace[-1][1],
        ) > 0.03:
            self._path_trace.append((x, y))
            if len(self._path_trace) > self._max_trace_points:
                self._path_trace = self._path_trace[-self._max_trace_points:]
        self.update()

    def reset_centre_reference(self):
        if self._have_pose:
            self._arena_origin_x = self._robot_x
            self._arena_origin_y = self._robot_y
        else:
            self._arena_origin_x = None
            self._arena_origin_y = None
        self._slam_overlay_img = None
        self._slam_overlay_key = None
        self._free_cells.fill(False)
        self._obstacle_cells.fill(False)
        self._path_trace.clear()
        self._scan_hits.clear()
        self.update()

    def update_scan(self, msg):
        if not self._have_pose or self._arena_origin_x is None:
            return

        hits = []
        now = time.time()
        min_usable_range = max(msg.range_min, LIDAR_SELF_MASK_MIN_RANGE)
        step = max(1, len(msg.ranges) // 180)
        for i in range(0, len(msg.ranges), step):
            r = msg.ranges[i]
            if math.isnan(r) or r <= min_usable_range:
                continue
            hit_obstacle = math.isfinite(r)
            if not hit_obstacle:
                r = msg.range_max
            elif r > msg.range_max:
                r = msg.range_max
                hit_obstacle = False
            angle = self._robot_yaw + msg.angle_min + i * msg.angle_increment
            wx = self._robot_x + r * math.cos(angle)
            wy = self._robot_y + r * math.sin(angle)
            if hit_obstacle:
                hits.append((wx, wy, now))
            self._paint_lidar_ray(angle, r, msg.range_max, hit_obstacle)

        self._scan_hits.extend(hits)
        cutoff = now - self._scan_hit_lifetime
        self._scan_hits = [hit for hit in self._scan_hits if hit[2] >= cutoff]
        if len(self._scan_hits) > self._max_scan_hits:
            self._scan_hits = self._scan_hits[-self._max_scan_hits:]
        self.update()

    def _world_to_arena(self, wx, wy):
        if self._arena_origin_x is None or self._arena_origin_y is None:
            return 0.0, 0.0
        return wx - self._arena_origin_x, wy - self._arena_origin_y

    def _arena_to_cell(self, ax, ay):
        col = int((ax + self._arena_half) / self._coverage_res)
        row = int((self._arena_half - ay) / self._coverage_res)
        if 0 <= row < self._coverage_n and 0 <= col < self._coverage_n:
            return row, col
        return None

    def _world_to_cell(self, wx, wy):
        ax, ay = self._world_to_arena(wx, wy)
        return self._arena_to_cell(ax, ay)

    def _ray_distance_to_arena_edge(self, angle):
        ax, ay = self._world_to_arena(self._robot_x, self._robot_y)
        dx = math.cos(angle)
        dy = math.sin(angle)
        distances = []
        if abs(dx) > 1e-6:
            distances.append((self._arena_half - ax) / dx)
            distances.append((-self._arena_half - ax) / dx)
        if abs(dy) > 1e-6:
            distances.append((self._arena_half - ay) / dy)
            distances.append((-self._arena_half - ay) / dy)
        positive = [dist for dist in distances if dist > 0.0]
        return min(positive) if positive else self._arena_half * 2.0

    def _paint_lidar_ray(self, angle, distance, sensor_max_range, hit_obstacle=True):
        arena_exit_dist = self._ray_distance_to_arena_edge(angle)
        max_dist = min(distance, sensor_max_range, arena_exit_dist)
        step = max(self._coverage_res * 0.5, 0.05)
        travelled = 0.0
        while travelled < max_dist:
            wx = self._robot_x + travelled * math.cos(angle)
            wy = self._robot_y + travelled * math.sin(angle)
            cell = self._world_to_cell(wx, wy)
            if cell is not None:
                self._free_cells[cell] = True
            travelled += step

        if hit_obstacle and distance < sensor_max_range * 0.97:
            wx = self._robot_x + distance * math.cos(angle)
            wy = self._robot_y + distance * math.sin(angle)
            cell = self._world_to_cell(wx, wy)
            if cell is not None:
                self._obstacle_cells[cell] = True

    def set_detections(self, detections: list):
        """Replace all detection markers. Called by the log poller."""
        self._detections = detections
        self.update()

    def update_path(self, poses):
        self._path = [(p.pose.position.x, p.pose.position.y) for p in poses]
        if (
            self._path
            and (self._arena_origin_x is None or self._arena_origin_y is None)
        ):
            if self._have_pose:
                self._arena_origin_x = self._robot_x
                self._arena_origin_y = self._robot_y
            else:
                # Coverage paths append the start/home pose last; use it to
                # center the GUI before the first /robot/pose callback arrives.
                self._arena_origin_x, self._arena_origin_y = self._path[-1]
            self._slam_overlay_img = None
            self._slam_overlay_key = None
        self.update()

    def _arena_view_rect(self):
        size = max(10, min(self.width(), self.height()) - 20)
        dx = (self.width() - size) // 2
        dy = (self.height() - size) // 2
        return dx, dy, size

    def _world_to_px(self, wx, wy):
        if self._arena_origin_x is not None and self._arena_origin_y is not None:
            dx, dy, size = self._arena_view_rect()
            ax, ay = self._world_to_arena(wx, wy)
            px = int(dx + (ax + self._arena_half) / self._arena_size * size)
            py = int(dy + (self._arena_half - ay) / self._arena_size * size)
            return (px, py)

        if self._map_w > 0 and self._map_h > 0:
            dx, dy, scale = self._map_view_transform()
            gx = (wx - self._map_ox) / self._map_res
            gy = self._map_h - (wy - self._map_oy) / self._map_res
            return (int(dx + gx * scale), int(dy + gy * scale))

        if self._map_w == 0 or self._map_h == 0:
            scan_points = [(x, y) for x, y, _stamp in self._scan_hits]
            points = self._path_trace + scan_points + [(self._robot_x, self._robot_y)]
            if not points:
                return (self.width()//2, self.height()//2)
            min_x = min(p[0] for p in points)
            max_x = max(p[0] for p in points)
            min_y = min(p[1] for p in points)
            max_y = max(p[1] for p in points)
            span = max(max_x - min_x, max_y - min_y, 1.0)
            pad = max(1.0, span * 0.15)
            min_x -= pad; max_x += pad; min_y -= pad; max_y += pad
            sx = self.width() / max(max_x - min_x, 0.1)
            sy = self.height() / max(max_y - min_y, 0.1)
            scale = min(sx, sy)
            used_w = (max_x - min_x) * scale
            used_h = (max_y - min_y) * scale
            px = int((wx - min_x) * scale + (self.width() - used_w) / 2)
            py = int((max_y - wy) * scale + (self.height() - used_h) / 2)
            return (px, py)

        return (self.width() // 2, self.height() // 2)

    def _map_view_transform(self):
        if self._map_w <= 0 or self._map_h <= 0:
            size = max(10, min(self.width(), self.height()) - 20)
            return (self.width() - size) // 2, (self.height() - size) // 2, 1.0
        scale = min(
            max(1, self.width() - 20) / self._map_w,
            max(1, self.height() - 20) / self._map_h,
        )
        used_w = self._map_w * scale
        used_h = self._map_h * scale
        return (self.width() - used_w) / 2, (self.height() - used_h) / 2, scale

    def _draw_coverage_grid(self, painter):
        dx, dy, size = self._arena_view_rect()
        cell = size / self._coverage_n

        painter.fillRect(dx, dy, size, size, QColor(205, 205, 205))

        for row in range(self._coverage_n):
            y = int(dy + row * cell)
            h = max(1, int(math.ceil(cell)))
            for col in range(self._coverage_n):
                if not self._free_cells[row, col] and not self._obstacle_cells[row, col]:
                    continue
                x = int(dx + col * cell)
                w = max(1, int(math.ceil(cell)))
                colour = QColor(20, 20, 20) if self._obstacle_cells[row, col] else QColor(255, 255, 255)
                painter.fillRect(x, y, w, h, colour)

        painter.setPen(QPen(QColor(120, 120, 120, 90), 1))
        for i in range(self._coverage_n + 1):
            pos = int(dx + i * cell)
            painter.drawLine(pos, dy, pos, dy + size)
            painter.drawLine(dx, pos, dx + size, pos)

        metre_step = self._coverage_n / self._arena_size
        painter.setPen(QPen(QColor(88, 166, 255, 100), 1))
        for metre in range(int(self._arena_size) + 1):
            offset = int(metre * metre_step * cell)
            painter.drawLine(dx + offset, dy, dx + offset, dy + size)
            painter.drawLine(dx, dy + offset, dx + size, dy + offset)

        painter.setPen(QPen(QColor(88, 166, 255, 210), 2, Qt.DashLine))
        painter.drawRect(dx, dy, size, size)

        painter.setPen(QPen(QColor(70, 70, 70), 1))
        painter.setFont(QFont(FONT_UI, 9))
        painter.drawText(dx + 8, dy + 18, f"{self._arena_size:g} x {self._arena_size:g} m LiDAR scan")

    def _slam_overlay_image(self):
        if (
            self._map_data is None
            or self._map_w <= 0
            or self._map_h <= 0
            or self._arena_origin_x is None
            or self._arena_origin_y is None
        ):
            return None

        _dx, _dy, size = self._arena_view_rect()
        key = (
            size,
            self._map_w,
            self._map_h,
            self._map_res,
            self._map_ox,
            self._map_oy,
            self._arena_origin_x,
            self._arena_origin_y,
            self._last_map_time,
        )
        if self._slam_overlay_img is not None and self._slam_overlay_key == key:
            return self._slam_overlay_img

        rows, cols = np.indices((size, size), dtype=np.float32)
        arena_x = (cols + 0.5) / size * self._arena_size - self._arena_half
        arena_y = self._arena_half - (rows + 0.5) / size * self._arena_size
        world_x = self._arena_origin_x + arena_x
        world_y = self._arena_origin_y + arena_y

        gx = np.floor((world_x - self._map_ox) / self._map_res).astype(np.int32)
        gy = np.floor((world_y - self._map_oy) / self._map_res).astype(np.int32)
        valid = (gx >= 0) & (gx < self._map_w) & (gy >= 0) & (gy < self._map_h)

        img = np.full((size, size, 3), 205, dtype=np.uint8)
        sampled = np.full((size, size), -1, dtype=np.int16)
        sampled[valid] = self._map_data[gy[valid], gx[valid]]
        img[sampled == 0] = [254, 254, 254]
        img[sampled > 50] = [0, 0, 0]

        self._slam_overlay_img = QImage(
            img.tobytes(),
            size,
            size,
            3 * size,
            QImage.Format_RGB888,
        ).copy()
        self._slam_overlay_key = key
        return self._slam_overlay_img

    def _draw_slam_overlay(self, painter):
        overlay = self._slam_overlay_image()
        if overlay is None:
            return
        dx, dy, size = self._arena_view_rect()
        painter.drawImage(dx, dy, overlay)

    def _draw_empty_arena(self, painter):
        size = min(self.width(), self.height()) - 20
        size = max(10, size)
        dx = (self.width() - size) // 2
        dy = (self.height() - size) // 2
        painter.fillRect(dx, dy, size, size, QColor(205, 205, 205))
        painter.setPen(QPen(QColor(120, 120, 120), 1, Qt.DashLine))
        painter.drawRect(dx, dy, size, size)

        painter.setPen(QPen(QColor(70, 70, 70), 1))
        painter.setFont(QFont(FONT_UI, 9))
        painter.drawText(dx + 8, dy + 18, f"{self._arena_size:g} x {self._arena_size:g} m SLAM map")

    def _draw_slam_map(self, painter):
        if self._map_img is None or self._map_w <= 0 or self._map_h <= 0:
            self._draw_empty_arena(painter)
            return

        if self._arena_origin_x is not None and self._arena_origin_y is not None:
            top_left = self._world_to_px(
                self._map_ox,
                self._map_oy + self._map_h * self._map_res,
            )
            bottom_right = self._world_to_px(
                self._map_ox + self._map_w * self._map_res,
                self._map_oy,
            )
            x = min(top_left[0], bottom_right[0])
            y = min(top_left[1], bottom_right[1])
            w = max(1, abs(bottom_right[0] - top_left[0]))
            h = max(1, abs(bottom_right[1] - top_left[1]))
            painter.drawImage(
                QRect(x, y, w, h),
                self._map_img,
            )
            return

        dx, dy, scale = self._map_view_transform()
        painter.drawImage(
            int(dx), int(dy),
            self._map_img.scaled(
                int(self._map_w * scale),
                int(self._map_h * scale),
                Qt.KeepAspectRatio,
                Qt.FastTransformation,
            )
        )

    def _draw_arena_boundary(self, painter):
        if self._arena_origin_x is None or self._arena_origin_y is None:
            return
        left = self._arena_origin_x - self._arena_half
        right = self._arena_origin_x + self._arena_half
        bottom = self._arena_origin_y - self._arena_half
        top = self._arena_origin_y + self._arena_half
        corners = [
            self._world_to_px(left, bottom),
            self._world_to_px(right, bottom),
            self._world_to_px(right, top),
            self._world_to_px(left, top),
        ]
        painter.setPen(QPen(QColor(88, 166, 255, 180), 2, Qt.DashLine))
        for i in range(len(corners)):
            painter.drawLine(*corners[i], *corners[(i + 1) % len(corners)])

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(205, 205, 205))
        if self._arena_origin_x is not None and self._arena_origin_y is not None:
            self._draw_coverage_grid(painter)
            self._draw_slam_overlay(painter)
            self._draw_arena_boundary(painter)
        else:
            self._draw_slam_map(painter)
            self._draw_arena_boundary(painter)

        if len(self._path) >= 2:
            painter.setPen(QPen(QColor(ACCENT), 2, Qt.DashLine))
            for i in range(len(self._path) - 1):
                p1 = self._world_to_px(*self._path[i])
                p2 = self._world_to_px(*self._path[i+1])
                painter.drawLine(*p1, *p2)

        # Draw detection markers — one per unique obstacle from the log file
        for (wx, wy, label, col) in self._detections:
            px, py = self._world_to_px(wx, wy)
            colour = QColor(RED) if "red" in col else \
                     QColor(YELLOW) if "yellow" in col else \
                     QColor(ACCENT) if col == "letter" else QColor(GREEN)
            painter.setBrush(QBrush(colour))
            painter.setPen(QPen(QColor(TEXT), 1))
            painter.drawEllipse(px - 8, py - 8, 16, 16)
            painter.setFont(QFont(FONT_UI, 7, QFont.Bold))
            painter.setPen(QColor(TEXT))
            display = label.replace("_obstacle", "").replace("_", " ")
            painter.drawText(px + 10, py + 4, display)

        rx, ry = self._world_to_px(self._robot_x, self._robot_y)
        painter.setBrush(QBrush(QColor(ACCENT)))
        painter.setPen(QPen(QColor(TEXT), 2))
        painter.drawEllipse(rx - 5, ry - 5, 10, 10)
        painter.end()


# ──────────────────────────────────────────────
#  DETECTION LOG WIDGET
# ──────────────────────────────────────────────
class DetectionLog(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet(f"""
            QScrollArea {{ border: none; background: transparent; }}
            QScrollBar:vertical {{ background: {PANEL}; width: 6px; }}
            QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 3px; }}
        """)

        self._inner = QWidget()
        self._inner.setStyleSheet("background: transparent;")
        self._inner_layout = QVBoxLayout(self._inner)
        self._inner_layout.setContentsMargins(0, 0, 0, 0)
        self._inner_layout.setSpacing(3)
        self._inner_layout.addStretch()

        self._scroll.setWidget(self._inner)
        layout.addWidget(self._scroll)

    def add_entry(self, text: str, colour: str = TEXT):
        ts  = datetime.now().strftime("%H:%M:%S")
        lbl = QLabel(f"<span style='color:{TEXT_DIM}'>{ts}</span>  "
                     f"<span style='color:{colour}'>{text}</span>")
        lbl.setFont(QFont(FONT_UI, 9))
        lbl.setStyleSheet("background: transparent; border: none;")
        lbl.setWordWrap(True)
        count = self._inner_layout.count()
        self._inner_layout.insertWidget(count - 1, lbl)
        QTimer.singleShot(50, lambda: self._scroll.verticalScrollBar().setValue(
            self._scroll.verticalScrollBar().maximum()))


# ──────────────────────────────────────────────
#  ARENA DEBUG PANEL
# ──────────────────────────────────────────────
class ArenaPanel(QWidget):
    ARENA_HALF = ARENA_SIZE_M / 2.0

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        grid = QGridLayout()
        grid.setSpacing(4)
        grid.setColumnStretch(1, 1)

        self._lbl_state  = status_row(grid, 0, "Drive state",    "—",      ACCENT)
        self._lbl_pos    = status_row(grid, 1, "Position",       "(—, —)", TEXT)
        self._lbl_home   = status_row(grid, 2, "Dist from home", "— m",    TEXT)
        self._lbl_edge   = status_row(grid, 3, "Edge clearance", "— m",    TEXT)
        self._lbl_yaw    = status_row(grid, 4, "Heading",        "—°",     TEXT)
        self._lbl_source = status_row(grid, 5, "Pose source",    "—",      TEXT_DIM)
        layout.addLayout(grid)

        bar_label = QLabel("Edge clearance")
        bar_label.setFont(QFont(FONT_UI, 8))
        bar_label.setStyleSheet(f"color: {TEXT_DIM}; border: none; background: transparent;")
        layout.addWidget(bar_label)

        self._edge_bar = QProgressBar()
        self._edge_bar.setRange(0, 100)
        self._edge_bar.setValue(100)
        self._edge_bar.setTextVisible(False)
        self._edge_bar.setFixedHeight(8)
        self._set_bar_colour(GREEN)
        layout.addWidget(self._edge_bar)

    def _set_bar_colour(self, colour: str):
        self._edge_bar.setStyleSheet(f"""
            QProgressBar {{
                background: {BORDER};
                border: none;
                border-radius: 4px;
            }}
            QProgressBar::chunk {{
                background: {colour};
                border-radius: 4px;
            }}
        """)

    def update(self, data: dict):
        state  = data.get('state', '—')
        rel_x  = data.get('rel_x', 0.0)
        rel_y  = data.get('rel_y', 0.0)
        home   = data.get('center_dist', 0.0)
        edge   = data.get('edge_clearance', 0.0)
        yaw    = data.get('yaw', 0.0)
        source = data.get('pose_source', '—')

        state_colours = {
            'WANDERING_DRIVE':       GREEN,
            'WANDERING_TURN':        ACCENT,
            'MAPPING':               GREEN,
            'WAYPOINT':              ACCENT,
            'COVERAGE_INITIAL_SCAN': ACCENT,
            'COVERAGE_SCAN':         ACCENT,
            'OBSTACLE_REVERSE':      YELLOW,
            'OBSTACLE_TURN':         YELLOW,
            'OBSTACLE_AVOIDANCE':    YELLOW,
            'BOUNDARY_REVERSE':      RED,
            'BOUNDARY_ESCAPE_TURN':  RED,
            'BOUNDARY_ESCAPE_DRIVE': RED,
            'RETURN_TO_CENTER':      YELLOW,
            'RETURNING_HOME':        YELLOW,
            'ESTOP':                 RED,
            'STOPPED':               RED,
            'INITIAL_TURN':          TEXT_DIM,
            'INITIAL_DRIVE':         TEXT_DIM,
        }
        sc = state_colours.get(state, TEXT)
        self._lbl_state.setText(state.replace('_', ' '))
        self._lbl_state.setStyleSheet(
            f"color: {sc}; border: none; background: transparent; font-weight: bold;")
        self._lbl_pos.setText(f"({rel_x:+.2f}, {rel_y:+.2f}) m")
        self._lbl_home.setText(f"{home:.2f} m")
        self._lbl_edge.setText(f"{edge:.2f} m")
        self._lbl_yaw.setText(f"{yaw:.1f}°")
        self._lbl_source.setText(source)

        pct = min(100, int(edge / self.ARENA_HALF * 100))
        self._edge_bar.setValue(pct)
        if pct > 50:
            self._set_bar_colour(GREEN)
        elif pct > 20:
            self._set_bar_colour(YELLOW)
        else:
            self._set_bar_colour(RED)


# ──────────────────────────────────────────────
#  MAIN WINDOW
# ──────────────────────────────────────────────
class RobotGUI(QMainWindow):
    def __init__(self, signals: Signals):
        super().__init__()
        self.signals = signals
        self.setWindowTitle("AUTO4508 — Pioneer 3-AT Monitor")
        self.setMinimumSize(1400, 800)
        self.setStyleSheet(f"background: {BG}; color: {TEXT};")
        self._latest_map_msg = None
        self._last_detection_event_times: dict[str, float] = {}
        self._last_log_mtime: float = 0.0
        self._last_display_state: str = ""

        self._build_ui()
        self._connect_signals()

        # Poll detections_log.jsonl every 2 seconds to update map markers
        self._log_poll_timer = QTimer()
        self._log_poll_timer.timeout.connect(self._poll_detection_log)
        self._log_poll_timer.start(2000)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # ── Header bar ──────────────────────────────
        header = QHBoxLayout()
        title  = QLabel("AUTO4508  ·  Pioneer 3-AT")
        title.setFont(QFont(FONT_BODY, 13, QFont.Bold))
        title.setStyleSheet(f"color: {ACCENT};")
        header.addWidget(title)
        header.addStretch()

        self._state_badge = QLabel("IDLE")
        self._state_badge.setFont(QFont(FONT_UI, 11, QFont.Bold))
        self._state_badge.setAlignment(Qt.AlignCenter)
        self._state_badge.setFixedHeight(32)
        self._state_badge.setStyleSheet(self._badge_style(YELLOW))
        header.addWidget(self._state_badge)

        self._reset_estop_btn = QPushButton("X  RESET E-STOP")
        self._reset_estop_btn.setFont(QFont(FONT_UI, 10, QFont.Bold))
        self._reset_estop_btn.setCursor(Qt.PointingHandCursor)
        self._reset_estop_btn.setAutoDefault(False)
        self._reset_estop_btn.setDefault(False)
        self._reset_estop_btn.setFixedHeight(32)
        self._reset_estop_btn.setStyleSheet(f"""
            QPushButton {{
                background: {RED};
                color: white;
                border: 2px solid {RED};
                border-radius: 6px;
                padding: 2px 16px;
            }}
            QPushButton:hover {{
                background: #c0392b;
                border-color: #c0392b;
            }}
        """)
        header.addWidget(self._reset_estop_btn)
        root.addLayout(header)

        # ── Main content row ────────────────────────
        content = QHBoxLayout()
        content.setSpacing(10)

        # LEFT COLUMN
        left = QVBoxLayout()
        left.setSpacing(10)

        cam_frame, cam_layout = make_panel("Camera Feed")
        self._cam_label = QLabel()
        self._cam_label.setAlignment(Qt.AlignCenter)
        self._cam_label.setMinimumSize(480, 270)
        self._cam_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._cam_label.setStyleSheet("background: #000; border: none;")
        self._cam_label.setText("No camera feed")
        cam_layout.addWidget(self._cam_label)
        left.addWidget(cam_frame, 3)

        status_frame, status_layout = make_panel("Robot Status")
        sg = QGridLayout()
        sg.setSpacing(6)
        self._status_labels = {}
        for i, (key, val) in enumerate([
            ("State",       "IDLE"),
            ("Action",      "—"),
            ("Pos X",       "0.00 m"),
            ("Pos Y",       "0.00 m"),
            ("Last Letter", "—"),
        ]):
            self._status_labels[key] = status_row(sg, i, key, val)
        status_layout.addLayout(sg)
        left.addWidget(status_frame, 1)

        arena_frame, arena_layout = make_panel("Arena Debug")
        self._arena_panel = ArenaPanel()
        arena_layout.addWidget(self._arena_panel)
        left.addWidget(arena_frame, 1)

        content.addLayout(left, 5)

        # MIDDLE COLUMN
        map_frame, map_layout = make_panel("Map")
        self._map_widget = MapWidget()
        map_layout.addWidget(self._map_widget)

        map_controls = QHBoxLayout()
        self._start_wandering_btn = QPushButton("Start Mapping")
        self._start_wandering_btn.setFont(QFont(FONT_UI, 9, QFont.Bold))
        self._start_wandering_btn.setCursor(Qt.PointingHandCursor)
        self._start_wandering_btn.setAutoDefault(False)
        self._start_wandering_btn.setDefault(False)
        self._start_wandering_btn.setStyleSheet(f"""
            QPushButton {{ background: {GREEN}22; color: {GREEN};
                border: 1px solid {GREEN}; border-radius: 6px; padding: 6px 12px; }}
        """)
        self._go_home_btn = QPushButton("Go Home")
        self._go_home_btn.setFont(QFont(FONT_UI, 9, QFont.Bold))
        self._go_home_btn.setCursor(Qt.PointingHandCursor)
        self._go_home_btn.setAutoDefault(False)
        self._go_home_btn.setDefault(False)
        self._go_home_btn.setStyleSheet(f"""
            QPushButton {{ background: {ACCENT}22; color: {ACCENT};
                border: 1px solid {ACCENT}; border-radius: 6px; padding: 6px 12px; }}
        """)
        self._drive_waypoints_btn = QPushButton("Drive Waypoints")
        self._drive_waypoints_btn.setFont(QFont(FONT_UI, 9, QFont.Bold))
        self._drive_waypoints_btn.setCursor(Qt.PointingHandCursor)
        self._drive_waypoints_btn.setAutoDefault(False)
        self._drive_waypoints_btn.setDefault(False)
        self._drive_waypoints_btn.setStyleSheet(f"""
            QPushButton {{ background: {YELLOW}22; color: {YELLOW};
                border: 1px solid {YELLOW}; border-radius: 6px; padding: 6px 12px; }}
        """)
        self._save_map_btn = QPushButton("Save Map")
        self._save_map_btn.setFont(QFont(FONT_UI, 9, QFont.Bold))
        self._save_map_btn.setCursor(Qt.PointingHandCursor)
        self._save_map_btn.setAutoDefault(False)
        self._save_map_btn.setDefault(False)
        self._save_map_btn.setStyleSheet(f"""
            QPushButton {{ background: {GREEN}22; color: {GREEN};
                border: 1px solid {GREEN}; border-radius: 6px; padding: 6px 12px; }}
            QPushButton:disabled {{ background: {BORDER}; color: {TEXT_DIM};
                border: 1px solid {BORDER}; }}
        """)
        self._replay_btn = QPushButton(">  Replay Journey")
        self._replay_btn.setFont(QFont(FONT_UI, 9, QFont.Bold))
        self._replay_btn.setCursor(Qt.PointingHandCursor)
        self._replay_btn.setAutoDefault(False)
        self._replay_btn.setDefault(False)
        self._replay_btn.setStyleSheet(f"""
            QPushButton {{ background: {ACCENT}22; color: {ACCENT};
                border: 1px solid {ACCENT}; border-radius: 6px; padding: 6px 12px; }}
            QPushButton:disabled {{ background: {BORDER}; color: {TEXT_DIM};
                border: 1px solid {BORDER}; }}
        """)
        self._save_map_label = QLabel("Maps save to ros2_ws/maps")
        self._save_map_label.setFont(QFont(FONT_UI, 8))
        self._save_map_label.setStyleSheet(f"color: {TEXT_DIM}; border: none; background: transparent;")
        map_controls.addWidget(self._start_wandering_btn)
        map_controls.addWidget(self._go_home_btn)
        map_controls.addWidget(self._drive_waypoints_btn)
        map_controls.addWidget(self._save_map_btn)
        map_controls.addWidget(self._replay_btn)
        map_controls.addWidget(self._save_map_label, 1)
        map_layout.addLayout(map_controls)
        content.addWidget(map_frame, 4)

        # RIGHT COLUMN
        right = QVBoxLayout()
        right.setSpacing(10)

        log_frame, log_layout = make_panel("Detection Log")
        self._det_log = DetectionLog()
        self._det_log.setMinimumHeight(200)
        log_layout.addWidget(self._det_log)
        right.addWidget(log_frame, 3)

        estop_frame, estop_layout = make_panel("Last E-Stop Event")
        self._estop_log = DetectionLog()
        self._estop_log.setMinimumHeight(80)
        estop_layout.addWidget(self._estop_log)
        right.addWidget(estop_frame, 1)

        photo_frame, photo_layout = make_panel("Last Detection  ( /detections/image )")
        self._photo_label = QLabel()
        self._photo_label.setAlignment(Qt.AlignCenter)
        self._photo_label.setMinimumHeight(160)
        self._photo_label.setStyleSheet("background: #000; border: none;")
        self._photo_label.setText("Waiting for detection...")
        photo_layout.addWidget(self._photo_label)
        right.addWidget(photo_frame, 2)

        content.addLayout(right, 3)
        root.addLayout(content, 1)

        # ── Bottom status bar ───────────────────────
        bottom = QHBoxLayout()
        self._action_label = QLabel("Waiting for robot...")
        self._action_label.setFont(QFont(FONT_UI, 9))
        self._action_label.setStyleSheet(f"color: {TEXT_DIM};")
        bottom.addWidget(self._action_label)
        bottom.addStretch()
        self._clock_label = QLabel()
        self._clock_label.setFont(QFont(FONT_UI, 9))
        self._clock_label.setStyleSheet(f"color: {TEXT_DIM};")
        bottom.addWidget(self._clock_label)
        root.addLayout(bottom)

        self._clock_timer = QTimer()
        self._clock_timer.timeout.connect(self._tick_clock)
        self._clock_timer.start(1000)
        self._tick_clock()

    def _badge_style(self, colour: str) -> str:
        return (f"background: {colour}22; color: {colour}; "
                f"border: 1px solid {colour}; border-radius: 6px; padding: 2px 14px;")

    def _connect_signals(self):
        self.signals.camera_frame.connect(self._on_camera)
        self.signals.colour_image.connect(self._on_colour_image)
        self.signals.robot_state.connect(self._on_state)
        self.signals.robot_pose.connect(self._on_pose)
        self.signals.letter_detected.connect(self._on_letter)
        self.signals.colour_detected.connect(self._on_colour)
        #self.signals.object_detected.connect(self._on_object_detection)
        self.signals.map_updated.connect(self._on_map)
        self.signals.scan_updated.connect(self._on_scan)
        self.signals.path_updated.connect(self._on_path)
        self.signals.arena_updated.connect(self._on_arena)
        self.signals.save_status.connect(self._on_save_status)
        self._start_wandering_btn.clicked.connect(lambda: self._send_mission_command("drive_coverage"))
        self._go_home_btn.clicked.connect(lambda: self._send_mission_command("go_home"))
        self._drive_waypoints_btn.clicked.connect(lambda: self._send_mission_command("drive_waypoints"))
        self._save_map_btn.clicked.connect(self._save_map)
        self._replay_btn.clicked.connect(self._replay_journey)
        self._reset_estop_btn.clicked.connect(self._reset_estop)

    # ── Detection log poller ──────────────────────────────────────────────────

    def _poll_detection_log(self):
        """
        Read detections_log.jsonl every 2 seconds.
        Keeps only the closest reading per label (same logic as the detector).
        Rebuilds map markers from scratch each poll so stale dots never accumulate.
        """
        if not os.path.exists(DETECTION_LOG_PATH):
            return

        # Skip re-reading if the file hasn't changed since last poll
        try:
            mtime = os.path.getmtime(DETECTION_LOG_PATH)
        except OSError:
            return
        if mtime == self._last_log_mtime:
            return
        self._last_log_mtime = mtime

        best: dict[str, dict] = {}
        try:
            with open(DETECTION_LOG_PATH) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        name   = record.get("name", "")
                        dist   = record.get("distance_m") or 999.0
                        if name not in best or dist < (best[name].get("distance_m") or 999.0):
                            best[name] = record
                    except Exception:
                        continue
        except Exception:
            return

        # Build marker list from best records that have valid world coordinates
        detections = []
        for record in best.values():
            obj_x = record.get("object_x")
            obj_y = record.get("object_y")
            name  = record.get("name", "")
            kind  = record.get("type", "")
            shape = record.get("shape", "")
            if obj_x is None or obj_y is None:
                continue
            colour = "red"    if "red"    in name else \
                     "yellow" if "yellow" in name else \
                     "letter" if kind == "letter" else "green"
            display = f"{shape} {name}".strip() if shape else name
            detections.append((obj_x, obj_y, display, colour))

        self._map_widget.set_detections(detections)

        # Update detection count in status bar
        self._save_map_label.setText(
            f"Maps save to ros2_ws/maps  ·  {len(detections)} detection(s) logged")

    # ── Slot handlers ────────────────────────────

    def _on_camera(self, frame: np.ndarray):
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.tobytes(), w, h, ch*w, QImage.Format_RGB888)
        pix  = QPixmap.fromImage(qimg).scaled(
            self._cam_label.width(), self._cam_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._cam_label.setPixmap(pix)

    def _on_colour_image(self, frame: np.ndarray):
        """Bottom-right panel — annotated frame from /detections/image."""
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.tobytes(), w, h, ch*w, QImage.Format_RGB888)
        pix  = QPixmap.fromImage(qimg).scaled(
            self._photo_label.width(), self._photo_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._photo_label.setPixmap(pix)

    def _on_state(self, state: str):
        self._status_labels["State"].setText(state)
        self._state_badge.setText(state)
        colour_map = {
            "MAPPING":          GREEN,
            "WAYPOINT":         ACCENT,
            "RETURNING_HOME":   YELLOW,
            "OBSTACLE_AVOIDANCE": YELLOW,
            "GOAL_ACHIEVED":    GREEN,
            "REACHED_HOME":     GREEN,
            "IDLE":             YELLOW,
            "STOPPED":          RED,
            "ESTOP":            RED,
            "RETURN_TO_CENTER": YELLOW,
        }
        colour = colour_map.get(state, TEXT_DIM)
        self._state_badge.setStyleSheet(self._badge_style(colour))
        action_map = {
            "MAPPING":  "Exploring area and building map...",
            "WAYPOINT": "Driving to waypoints at maximum speed...",
            "GOAL_ACHIEVED": "Goal achieved. Press Go Home to return to centre.",
            "REACHED_HOME": "Reached home. Waiting for next command.",
            "IDLE":     "Standing by.",
            "STOPPED":  "EMERGENCY STOP — all motion halted!",
            "ESTOP":    "EMERGENCY STOP — obstacle detected!",
            "RETURN_TO_CENTER": "Returning to arena centre...",
        }
        action = action_map.get(state, state)
        self._status_labels["Action"].setText(action)
        self._action_label.setText(action)
        if state != self._last_display_state:
            self._last_display_state = state
            self._det_log.add_entry(f"State -> {state}", colour)
            if state in ("STOPPED", "ESTOP"):
                self._load_latest_estop_event()

    def _on_pose(self, x: float, y: float, yaw: float):
        self._status_labels["Pos X"].setText(f"{x:.2f} m")
        self._status_labels["Pos Y"].setText(f"{y:.2f} m")
        self._map_widget.update_pose(x, y, yaw)

    def _should_log_detection_event(self, key: str, cooldown_s: float = 1.5) -> bool:
        now = time.monotonic()
        last = self._last_detection_event_times.get(key, 0.0)
        if now - last < cooldown_s:
            return False
        self._last_detection_event_times[key] = now
        return True

    def _on_letter(self, raw: str):
        try:
            data = json.loads(raw)
            name = data.get("name", raw)
            dist = data.get("distance_m")
            dist_str = f"  dist={dist:.2f}m" if dist else ""
        except (json.JSONDecodeError, TypeError):
            name = raw
            dist_str = ""
        self._status_labels["Last Letter"].setText(name)
        if self._should_log_detection_event(f"letter:{name}", cooldown_s=8.0):
            self._det_log.add_entry(f"Greek letter: {name}{dist_str}", GREEN)

    def _on_colour(self, data: dict):
        """Log entry only — map markers come from the log file poller."""
        label   = data.get("label", "unknown")
        dist    = data.get("distance_m", 0.0)
        bearing = data.get("bearing_deg", 0.0)
        shape   = data.get("shape", "")

        colour = RED if "red" in label else YELLOW
        shape_str = f"  shape={shape}" if shape else ""
        if self._should_log_detection_event(f"colour:{label}", cooldown_s=8.0):
            self._det_log.add_entry(
                f"{label}{shape_str}  dist={dist:.2f}m  bearing={bearing:.1f} deg",
                colour)


    def _on_map(self, msg):
        self._latest_map_msg = msg
        self._map_widget.update_map(msg)

    def _on_scan(self, msg):
        self._map_widget.update_scan(msg)

    def _on_path(self, msg):
        self._map_widget.update_path(msg.poses)

    def _on_arena(self, data: dict):
        self._arena_panel.update(data)

    def _send_mission_command(self, command: str):
        if command in {"start_wandering", "drive_coverage"}:
            self._map_widget.reset_centre_reference()
            self._on_state("MAPPING")
        elif command == "drive_waypoints":
            self._on_state("WAYPOINT")
        elif command == "go_home":
            self._on_state("RETURN_TO_CENTER")
        self.signals.mission_command.emit(command)
        self._det_log.add_entry(f"Command -> {command}", ACCENT)

    def _load_latest_estop_event(self):
        if not os.path.isdir(ESTOP_LOG_DIR):
            self._estop_log.add_entry("No e-stop directory yet.", TEXT_DIM)
            return

        jsonl_files = sorted(
            [f for f in os.listdir(ESTOP_LOG_DIR)
             if f.startswith("estop_") and f.endswith(".jsonl")],
            reverse=True)

        if jsonl_files:
            name = jsonl_files[0]
            path = os.path.join(ESTOP_LOG_DIR, name)
            self._estop_log.add_entry(f"-- {name} --", RED)
            try:
                with open(path, encoding="utf-8") as f:
                    entries = [json.loads(line) for line in f if line.strip()]
                pose_entries = [e for e in entries if e.get("type") == "pose"]
                scan_entries = [e for e in entries if e.get("type") == "scan"]
                self._estop_log.add_entry(
                    f"{len(entries)} records | {len(pose_entries)} pose | {len(scan_entries)} scan",
                    TEXT_DIM)
                if pose_entries:
                    p0, p1 = pose_entries[0], pose_entries[-1]
                    self._estop_log.add_entry(
                        f"({p0['x']:.2f},{p0['y']:.2f}) -> "
                        f"({p1['x']:.2f},{p1['y']:.2f})  yaw={p1['yaw']:.1f} deg",
                        YELLOW)
                if scan_entries:
                    ranges = [r for r in scan_entries[-1].get("ranges", []) if r > 0.05]
                    if ranges:
                        self._estop_log.add_entry(f"Closest obstacle: {min(ranges):.2f} m", RED)
            except Exception as exc:
                self._estop_log.add_entry(f"Error: {exc}", RED)
            return

        incidents_paths = [
            os.path.join(ESTOP_LOG_DIR, "incidents.txt"),
            os.path.expanduser("~/pioneer_estop/incidents.txt"),
        ]
        for path in incidents_paths:
            if not os.path.exists(path):
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
                self._estop_log.add_entry(f"-- {os.path.basename(path)} --", RED)
                for line in lines[-3:]:
                    self._estop_log.add_entry(line, YELLOW)
                return
            except Exception as exc:
                self._estop_log.add_entry(f"Error: {exc}", RED)
                return

        self._estop_log.add_entry("No e-stop files yet.", TEXT_DIM)

    def _replay_journey(self):
        if not os.path.isdir(BAG_DIR):
            self._det_log.add_entry("No bags directory found.", RED)
            return

        bags = sorted(
            [os.path.join(BAG_DIR, d) for d in os.listdir(BAG_DIR)
             if os.path.isdir(os.path.join(BAG_DIR, d))
             and os.path.exists(os.path.join(BAG_DIR, d, "metadata.yaml"))])
        if not bags:
            self._det_log.add_entry("No recorded bags found.", RED)
            return

        self._det_log.add_entry(f"Replaying {len(bags)} bag(s) in order...", ACCENT)
        self._replay_btn.setEnabled(False)
        self._replay_btn.setText(f"Replaying 1/{len(bags)}")

        def _run():
            for i, bag_path in enumerate(bags):
                name = os.path.basename(bag_path)
                QTimer.singleShot(
                    0,
                    lambda n=name, idx=i: self._replay_btn.setText(
                        f"Replaying {idx + 1}/{len(bags)}: {n[-12:]}"))
                subprocess.run(["ros2", "bag", "play", bag_path, "--clock", "--rate", "1.0"],
                               check=False)
            QTimer.singleShot(0, self._on_replay_done)

        threading.Thread(target=_run, daemon=True).start()

    def _on_replay_done(self):
        self._replay_btn.setEnabled(True)
        self._replay_btn.setText(">  Replay Journey")
        self._det_log.add_entry("Replay finished.", TEXT_DIM)

    def _reset_estop(self):
        self._last_display_state = ""
        self.signals.mission_command.emit("reset_estop")
        self._det_log.add_entry("E-stop reset sent.", YELLOW)
        self._on_state("IDLE")

    def _save_map(self):
        if self._latest_map_msg is None:
            self.signals.save_status.emit("No /map received yet", False)
            return
        self._save_map_btn.setEnabled(False)
        self._save_map_label.setText("Saving current /map...")
        self._save_map_label.setStyleSheet(f"color: {TEXT_DIM}; border: none; background: transparent;")

        def worker():
            msg = self._latest_map_msg
            try:
                prefix = self._save_occupancy_grid(msg)
                self.signals.save_status.emit(f"Saved {prefix}.png", True)
            except Exception as exc:
                self.signals.save_status.emit(str(exc), False)

        threading.Thread(target=worker, daemon=True).start()

    def _save_occupancy_grid(self, msg) -> str:
        out_dir = self._map_output_dir()
        os.makedirs(out_dir, exist_ok=True)
        stamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = os.path.join(out_dir, f"pioneer_map_{stamp}")

        width  = msg.info.width
        height = msg.info.height
        data   = np.array(msg.data, dtype=np.int16).reshape((height, width))
        img    = np.full((height, width), 205, dtype=np.uint8)
        img[data == 0]   = 254
        img[data >= 65]  = 0
        img = np.flipud(img)
        self._draw_arena_boundary_on_saved_map(img, msg)

        image_path = f"{prefix}.png"
        yaml_path  = f"{prefix}.yaml"
        if not cv2.imwrite(image_path, img):
            raise RuntimeError(f"Could not write {image_path}")

        yaw = self._yaw_from_quaternion(msg.info.origin.orientation)
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(
                f"image: {os.path.basename(image_path)}\n"
                f"mode: trinary\n"
                f"resolution: {msg.info.resolution:.8f}\n"
                f"origin: [{msg.info.origin.position.x:.8f}, "
                f"{msg.info.origin.position.y:.8f}, {yaw:.8f}]\n"
                f"negate: 0\n"
                f"occupied_thresh: 0.65\n"
                f"free_thresh: 0.25\n"
            )

        return prefix

    def _draw_arena_boundary_on_saved_map(self, img, msg):
        grid = self._map_widget
        if grid._arena_origin_x is None or grid._arena_origin_y is None:
            return

        height, width = img.shape[:2]
        half = grid._arena_half
        left = grid._arena_origin_x - half
        right = grid._arena_origin_x + half
        bottom = grid._arena_origin_y - half
        top = grid._arena_origin_y + half

        def world_to_img(wx, wy):
            gx = int(round((wx - msg.info.origin.position.x) / msg.info.resolution))
            gy = int(round(msg.info.height - (wy - msg.info.origin.position.y) / msg.info.resolution))
            return gx, gy

        corners = [
            world_to_img(left, bottom),
            world_to_img(right, bottom),
            world_to_img(right, top),
            world_to_img(left, top),
        ]
        rect = (0, 0, width - 1, height - 1)
        for i, p1 in enumerate(corners):
            p2 = corners[(i + 1) % len(corners)]
            ok, cp1, cp2 = cv2.clipLine(rect, p1, p2)
            if ok:
                cv2.line(img, cp1, cp2, 0, 2)

    def _write_obstacle_waypoints(self, path, waypoints):
        with open(path, "w", encoding="utf-8") as f:
            f.write("# target_rel_x_m target_rel_y_m obstacle_rel_x_m obstacle_rel_y_m\n")
            f.write("# targets are 1.0 m in front of each obstacle from the centre reference\n")
            for target_x, target_y, obstacle_x, obstacle_y in waypoints:
                f.write(f"{target_x:.3f} {target_y:.3f} {obstacle_x:.3f} {obstacle_y:.3f}\n")

    def _coverage_obstacle_standoff_waypoints(self):
        grid     = self._map_widget
        occupied = grid._obstacle_cells.copy()
        visited  = np.zeros_like(occupied, dtype=bool)
        height, width = occupied.shape
        clusters = []

        for start_row in range(height):
            for start_col in range(width):
                if not occupied[start_row, start_col] or visited[start_row, start_col]:
                    continue
                stack = [(start_row, start_col)]
                visited[start_row, start_col] = True
                cells = []
                while stack:
                    row, col = stack.pop()
                    cells.append((row, col))
                    for drow, dcol in ((1,0),(-1,0),(0,1),(0,-1)):
                        nr, nc = row+drow, col+dcol
                        if (0 <= nr < height and 0 <= nc < width
                                and occupied[nr, nc] and not visited[nr, nc]):
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                if len(cells) >= 3:
                    clusters.append(cells)

        waypoints  = []
        standoff_m = 1.0
        min_spacing = 0.75
        edge_margin = 0.35
        for cells in clusters:
            avg_row    = sum(r for r, _ in cells) / len(cells)
            avg_col    = sum(c for _, c in cells) / len(cells)
            obstacle_x = (avg_col + 0.5) * grid._coverage_res - grid._arena_half
            obstacle_y = grid._arena_half - (avg_row + 0.5) * grid._coverage_res
            dist_from_centre = math.hypot(obstacle_x, obstacle_y)
            if dist_from_centre < standoff_m + 0.2:
                continue
            unit_x   = obstacle_x / dist_from_centre
            unit_y   = obstacle_y / dist_from_centre
            target_x = obstacle_x - unit_x * standoff_m
            target_y = obstacle_y - unit_y * standoff_m
            target_x = max(-grid._arena_half + edge_margin, min(grid._arena_half - edge_margin, target_x))
            target_y = max(-grid._arena_half + edge_margin, min(grid._arena_half - edge_margin, target_y))
            if all(math.hypot(target_x - px, target_y - py) >= min_spacing
                   for px, py, _ox, _oy in waypoints):
                waypoints.append((target_x, target_y, obstacle_x, obstacle_y))

        waypoints.sort(key=lambda item: math.hypot(item[0], item[1]))
        return waypoints[:12]

    def _yaw_from_quaternion(self, q) -> float:
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny, cosy)

    def _map_output_dir(self) -> str:
        env_dir = os.environ.get("PIONEER_MAP_DIR")
        if env_dir:
            return os.path.expanduser(env_dir)
        here = FilePath(__file__).resolve()
        for parent in [here] + list(here.parents):
            if parent.name == "ros2_ws":
                return str(parent / "maps")
        cwd = FilePath.cwd().resolve()
        if cwd.name == "ros2_ws":
            return str(cwd / "maps")
        docker_ws = FilePath("/ros2_ws")
        if docker_ws.exists():
            return str(docker_ws / "maps")
        return str(cwd / "maps")

    def _on_save_status(self, message: str, ok: bool):
        self._save_map_btn.setEnabled(True)
        colour = GREEN if ok else RED
        self._save_map_label.setText(message)
        self._save_map_label.setStyleSheet(f"color: {colour}; border: none; background: transparent;")

    def _tick_clock(self):
        self._clock_label.setText(datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))


# ──────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────
def main():
    rclpy.init()
    signals = Signals()
    node    = GUINode(signals)
    signals.mission_command.connect(node.publish_command)

    ros_thread = threading.Thread(
        target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = RobotGUI(signals)
    win.show()

    exit_code = app.exec_()

    node.destroy_node()
    rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
