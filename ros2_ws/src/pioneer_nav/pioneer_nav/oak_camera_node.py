#!/usr/bin/env python3
"""
Publish OAK-D RGB and stereo depth frames for the GUI and unified detector.

Publishes:
  /oak/rgb/image_raw     sensor_msgs/Image, bgr8
  /oak/stereo/image_raw  sensor_msgs/Image, 16UC1 depth in millimetres
"""

import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

try:
    import depthai as dai
except Exception:
    dai = None


def make_image_msg(frame: np.ndarray, encoding: str, frame_id: str, stamp) -> Image:
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = int(frame.shape[0])
    msg.width = int(frame.shape[1])
    msg.encoding = encoding
    msg.is_bigendian = 0
    channels = 1 if frame.ndim == 2 else frame.shape[2]
    msg.step = int(frame.shape[1] * frame.dtype.itemsize * channels)
    msg.data = frame.tobytes()
    return msg


class OakCameraNode(Node):
    def __init__(self):
        super().__init__("oak_camera")

        self.declare_parameter("rgb_topic", "/oak/rgb/image_raw")
        self.declare_parameter("depth_topic", "/oak/stereo/image_raw")
        self.declare_parameter("frame_id", "oak_camera")
        self.declare_parameter("publish_rate_hz", 15.0)
        self.declare_parameter("webcam_fallback", False)

        self.rgb_topic = str(self.get_parameter("rgb_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        rate_hz = max(1.0, float(self.get_parameter("publish_rate_hz").value))
        self.webcam_fallback = bool(self.get_parameter("webcam_fallback").value)

        self.rgb_pub = self.create_publisher(Image, self.rgb_topic, 10)
        self.depth_pub = self.create_publisher(Image, self.depth_topic, 10)
        self.device = None
        self.q_rgb = None
        self.q_depth = None
        self.cap = None
        self.using_oak = False
        self._last_no_frame_log = 0.0

        if dai is None:
            self.get_logger().error("depthai is not installed; cannot start OAK-D.")
            if self.webcam_fallback:
                self._start_webcam()
        else:
            self._start_oak()

        self.create_timer(1.0 / rate_hz, self.publish_frames)
        self.get_logger().info(
            f"OAK camera publisher ready: rgb={self.rgb_topic}, depth={self.depth_topic}, "
            f"source={'oak-d' if self.using_oak else 'webcam fallback' if self.cap else 'none'}"
        )

    def _start_oak(self):
        try:
            pipeline = dai.Pipeline()

            cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
            mono_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
            mono_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

            stereo = pipeline.create(dai.node.StereoDepth).build(
                left=mono_left.requestOutput((640, 400)),
                right=mono_right.requestOutput((640, 400)),
                presetMode=dai.node.StereoDepth.PresetMode.FAST_ACCURACY,
            )

            self.q_rgb = cam_rgb.requestOutput((640, 480), dai.ImgFrame.Type.BGR888p).createOutputQueue()
            self.q_depth = stereo.depth.createOutputQueue()
            self.device = dai.Device()
            self.device.start(pipeline)
            self.using_oak = True
            self.get_logger().info("OAK-D pipeline started.")
        except Exception as exc:
            self.get_logger().error(f"Could not start OAK-D pipeline: {exc}")
            if self.webcam_fallback:
                self._start_webcam()

    def _start_webcam(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("Webcam fallback requested, but /dev/video0 could not be opened.")
            self.cap = None
            return
        self.get_logger().warn("Using webcam fallback. Depth topic will not publish real depth.")

    def publish_frames(self):
        if self.using_oak:
            rgb_packet = self.q_rgb.tryGet()
            depth_packet = self.q_depth.tryGet()
            if rgb_packet is None:
                self._log_no_frame()
                return

            bgr = rgb_packet.getCvFrame()
            stamp = self.get_clock().now().to_msg()
            self.rgb_pub.publish(make_image_msg(bgr, "bgr8", self.frame_id, stamp))

            if depth_packet is not None:
                depth = depth_packet.getFrame().astype(np.uint16, copy=False)
                self.depth_pub.publish(make_image_msg(depth, "16UC1", self.frame_id, stamp))
            return

        if self.cap is not None:
            ok, frame = self.cap.read()
            if not ok:
                self._log_no_frame()
                return
            self.rgb_pub.publish(make_image_msg(frame, "bgr8", self.frame_id, self.get_clock().now().to_msg()))
            return

        self._log_no_frame()

    def _log_no_frame(self):
        now = time.monotonic()
        if now - self._last_no_frame_log > 2.0:
            self._last_no_frame_log = now
            self.get_logger().warn("No camera frame available.", throttle_duration_sec=2.0)

    def destroy_node(self):
        if self.cap is not None:
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = OakCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
