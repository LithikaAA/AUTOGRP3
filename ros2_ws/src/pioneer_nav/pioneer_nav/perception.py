#!/usr/bin/env python3

# AUTO4508 Part 2 - Vision / Perception Module
# Drop-in replacement for the perception section of part2_mission_controller.py
#
# Detects:
#   - Orange traffic cones  (waypoint markers)
#   - Blue buckets          (coloured objects at each waypoint)
#   - Other coloured shapes (red, yellow, green as fallback objects)
#
# Each detector returns a Detection dataclass with:
#   - label       : what was found ("orange_cone", "blue_bucket", shape name)
#   - center_x/y  : pixel position in image
#   - area        : contour area in pixels (bigger = closer)
#   - bearing_rad : horizontal angle from robot centre (neg=left, pos=right)
#   - bbox         : (x, y, w, h) bounding box
#   - confidence  : 0.0-1.0 score based on shape + colour match
#
# Tuning for outdoors:
#   - HSV ranges are wider than sim to handle sunlight variation
#   - Morphology kernels are larger to clean up noise from grass/shadows
#   - Minimum areas are higher to ignore small false positives at distance
#   - Shape validation checks aspect ratio to confirm cone vs flat blob

import math
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


# =============================================================================
# HSV colour ranges  — tuned for outdoor daylight
# Format: (name, lower_hsv, upper_hsv)
# If detection is wrong outdoors, adjust these first.
# Use the debug_hsv_ranges() function below to visualise live.
# =============================================================================

# Orange cone — wider range to handle sun/shadow
ORANGE_LOWER = np.array([5,  120,  80], dtype=np.uint8)
ORANGE_UPPER = np.array([22, 255, 255], dtype=np.uint8)

# Blue bucket
BLUE_LOWER   = np.array([95,  80,  60], dtype=np.uint8)
BLUE_UPPER   = np.array([130, 255, 255], dtype=np.uint8)

# Other object colours (fallback if no bucket found)
OTHER_COLOUR_RANGES = [
    ("red_lo",  np.array([0,   120, 80]),  np.array([10,  255, 255])),
    ("red_hi",  np.array([165, 120, 80]),  np.array([179, 255, 255])),
    ("yellow",  np.array([18,  120, 80]),  np.array([35,  255, 255])),
    ("green",   np.array([38,   80, 50]),  np.array([88,  255, 255])),
]


# =============================================================================
# Detection result
# =============================================================================

@dataclass
class Detection:
    label: str
    center_x: float
    center_y: float
    area: float
    bearing_rad: float
    bbox: Tuple[int, int, int, int]   # x, y, w, h
    confidence: float = 1.0
    distance_hint_m: Optional[float] = None  # rough estimate from area if lidar unavailable


# =============================================================================
# Helper functions
# =============================================================================

def bearing_from_image_x(x_px: float, width: int, hfov_rad: float) -> float:
    """Convert pixel x position to horizontal bearing angle."""
    norm = (x_px - (width / 2.0)) / (width / 2.0)
    return norm * (hfov_rad / 2.0)


def rough_distance_from_area(area_px: float, known_width_m: float = 0.3,
                              focal_px: float = 600.0) -> float:
    """
    Very rough distance estimate from contour area.
    Works when lidar bearing alignment is uncertain.
    known_width_m: real world width of object in metres (cone ~0.3m)
    focal_px: approximate focal length in pixels (tune per camera)
    """
    if area_px <= 0:
        return float("inf")
    apparent_width_px = math.sqrt(area_px)
    return (known_width_m * focal_px) / apparent_width_px


def apply_morphology(mask: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    """Clean up a binary mask with open then close operations."""
    k = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def largest_contour(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def all_contours_above_area(mask: np.ndarray, min_area: float) -> list:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= min_area]


def classify_shape(contour) -> str:
    """Classify a contour by number of polygon vertices."""
    peri   = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    v = len(approx)
    if v == 3:
        return "triangle"
    if v == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect = w / float(h) if h > 0 else 0.0
        return "square" if 0.85 <= aspect <= 1.15 else "rectangle"
    if v > 6:
        return "circle"
    return "polygon"


def cone_shape_confidence(contour) -> float:
    """
    Score 0.0-1.0 how cone-like a contour is.
    A real cone tapers toward the top — tall aspect ratio, triangular-ish.
    """
    x, y, w, h = cv2.boundingRect(contour)
    if w == 0 or h == 0:
        return 0.0

    aspect = h / float(w)           # cone is taller than wide -> >1.0
    area   = cv2.contourArea(contour)
    rect_area = w * h
    fill   = area / rect_area if rect_area > 0 else 0.0  # cone ~0.5 fill

    # score aspect: ideal ~1.2-2.5
    aspect_score = 1.0 - min(abs(aspect - 1.7) / 1.5, 1.0)
    # score fill: ideal ~0.4-0.6
    fill_score   = 1.0 - min(abs(fill - 0.5) / 0.4, 1.0)

    return (aspect_score * 0.6) + (fill_score * 0.4)


def bucket_shape_confidence(contour) -> float:
    """
    Score 0.0-1.0 how bucket-like a contour is.
    A bucket is roughly cylindrical — roughly square aspect ratio.
    """
    x, y, w, h = cv2.boundingRect(contour)
    if w == 0 or h == 0:
        return 0.0

    aspect = w / float(h)
    area   = cv2.contourArea(contour)
    fill   = area / (w * h) if w * h > 0 else 0.0

    # bucket aspect ~0.7-1.3
    aspect_score = 1.0 - min(abs(aspect - 1.0) / 0.5, 1.0)
    # bucket fill ~0.6-0.85 (mostly filled rectangle)
    fill_score   = 1.0 - min(abs(fill - 0.75) / 0.3, 1.0)

    return (aspect_score * 0.5) + (fill_score * 0.5)


# =============================================================================
# Main detection functions
# =============================================================================

def detect_cone(hsv_img: np.ndarray,
                image_width: int,
                hfov_rad: float,
                min_area: float = 800.0) -> Optional[Detection]:
    """
    Detect the largest orange traffic cone in the image.

    Returns None if no cone found above min_area.
    Confidence score reflects how cone-shaped the blob is.
    """
    # colour mask
    mask = cv2.inRange(hsv_img, ORANGE_LOWER, ORANGE_UPPER)
    mask = apply_morphology(mask, kernel_size=7)

    cnt = largest_contour(mask)
    if cnt is None:
        return None

    area = cv2.contourArea(cnt)
    if area < min_area:
        return None

    x, y, w, h = cv2.boundingRect(cnt)
    cx = x + w / 2.0
    cy = y + h / 2.0
    bearing  = bearing_from_image_x(cx, image_width, hfov_rad)
    conf     = cone_shape_confidence(cnt)
    dist_hint = rough_distance_from_area(area, known_width_m=0.3)

    return Detection(
        label="orange_cone",
        center_x=cx,
        center_y=cy,
        area=area,
        bearing_rad=bearing,
        bbox=(x, y, w, h),
        confidence=conf,
        distance_hint_m=dist_hint,
    )


def detect_bucket(hsv_img: np.ndarray,
                  image_width: int,
                  hfov_rad: float,
                  min_area: float = 600.0) -> Optional[Detection]:
    """
    Detect the blue bucket (the coloured object at each waypoint).

    Returns None if no bucket found above min_area.
    """
    mask = cv2.inRange(hsv_img, BLUE_LOWER, BLUE_UPPER)
    mask = apply_morphology(mask, kernel_size=7)

    cnt = largest_contour(mask)
    if cnt is None:
        return None

    area = cv2.contourArea(cnt)
    if area < min_area:
        return None

    x, y, w, h = cv2.boundingRect(cnt)
    cx = x + w / 2.0
    cy = y + h / 2.0
    bearing   = bearing_from_image_x(cx, image_width, hfov_rad)
    conf      = bucket_shape_confidence(cnt)
    shape     = classify_shape(cnt)
    dist_hint = rough_distance_from_area(area, known_width_m=0.35)

    return Detection(
        label=f"blue_bucket_{shape}",
        center_x=cx,
        center_y=cy,
        area=area,
        bearing_rad=bearing,
        bbox=(x, y, w, h),
        confidence=conf,
        distance_hint_m=dist_hint,
    )


def detect_other_object(hsv_img: np.ndarray,
                        image_width: int,
                        hfov_rad: float,
                        min_area: float = 500.0) -> Optional[Detection]:
    """
    Detect any other coloured object that isn't orange or blue.
    Used as fallback if bucket not found.
    Returns the largest matching blob.
    """
    # combine all non-orange, non-blue colour ranges
    combined = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
    for _, low, high in OTHER_COLOUR_RANGES:
        combined |= cv2.inRange(hsv_img, low, high)

    # subtract orange and blue so we don't double-detect
    orange_mask = cv2.inRange(hsv_img, ORANGE_LOWER, ORANGE_UPPER)
    blue_mask   = cv2.inRange(hsv_img, BLUE_LOWER,   BLUE_UPPER)
    combined    = cv2.bitwise_and(combined, cv2.bitwise_not(orange_mask | blue_mask))

    combined = apply_morphology(combined, kernel_size=7)

    cnt = largest_contour(combined)
    if cnt is None:
        return None

    area = cv2.contourArea(cnt)
    if area < min_area:
        return None

    shape = classify_shape(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    cx = x + w / 2.0
    cy = y + h / 2.0
    bearing   = bearing_from_image_x(cx, image_width, hfov_rad)
    dist_hint = rough_distance_from_area(area, known_width_m=0.3)

    return Detection(
        label=shape,
        center_x=cx,
        center_y=cy,
        area=area,
        bearing_rad=bearing,
        bbox=(x, y, w, h),
        confidence=0.6,
        distance_hint_m=dist_hint,
    )


def detect_all_cones(hsv_img: np.ndarray,
                     image_width: int,
                     hfov_rad: float,
                     min_area: float = 400.0) -> List[Detection]:
    """
    Detect ALL orange cones visible (not just the largest).
    Used for weave mode to track multiple cones at once.
    Returns list sorted left to right by bearing.
    """
    mask = cv2.inRange(hsv_img, ORANGE_LOWER, ORANGE_UPPER)
    mask = apply_morphology(mask, kernel_size=5)

    contours = all_contours_above_area(mask, min_area)
    detections = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w / 2.0
        cy = y + h / 2.0
        bearing = bearing_from_image_x(cx, image_width, hfov_rad)
        conf    = cone_shape_confidence(cnt)

        detections.append(Detection(
            label="orange_cone",
            center_x=cx,
            center_y=cy,
            area=area,
            bearing_rad=bearing,
            bbox=(x, y, w, h),
            confidence=conf,
        ))

    # sort left to right
    detections.sort(key=lambda d: d.bearing_rad)
    return detections


# =============================================================================
# Photo saving
# =============================================================================

def save_detection_photo(bgr_img: np.ndarray,
                         detection: Detection,
                         photos_dir: str,
                         prefix: str) -> Optional[str]:
    """
    Save a photo with the detection bounding box drawn on it.
    Returns the saved file path.
    """
    if bgr_img is None:
        return None

    os.makedirs(photos_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename  = f"{prefix}_{timestamp}.png"
    path      = os.path.join(photos_dir, filename)

    # draw bounding box and label on a copy
    annotated = bgr_img.copy()
    x, y, w, h = detection.bbox

    # colour the box based on what was detected
    if "cone" in detection.label:
        colour = (0, 165, 255)    # orange in BGR
    elif "bucket" in detection.label:
        colour = (255, 100, 0)    # blue in BGR
    else:
        colour = (0, 255, 0)      # green for other

    cv2.rectangle(annotated, (x, y), (x + w, y + h), colour, 3)

    label_text = f"{detection.label} conf:{detection.confidence:.2f}"
    if detection.distance_hint_m and detection.distance_hint_m < 20.0:
        label_text += f" ~{detection.distance_hint_m:.1f}m"

    # draw label background
    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(annotated, (x, y - th - 10), (x + tw + 6, y), colour, -1)
    cv2.putText(annotated, label_text,
                (x + 3, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imwrite(path, annotated)
    return path


# =============================================================================
# Debug visualisation  (run standalone to tune HSV ranges)
# =============================================================================

def debug_hsv_ranges(camera_index: int = 0):
    """
    Live camera feed showing detection masks.
    Run this standalone on the robot to tune HSV values:
        python3 perception.py
    Press q to quit.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Could not open camera")
        return

    print("Debug mode - press Q to quit")
    print("  Green box  = orange cone detection")
    print("  Blue box   = blue bucket detection")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]

        cone   = detect_cone(hsv, w, 1.089)
        bucket = detect_bucket(hsv, w, 1.089)
        other  = detect_other_object(hsv, w, 1.089)

        display = frame.copy()

        if cone:
            x, y, bw, bh = cone.bbox
            cv2.rectangle(display, (x, y), (x+bw, y+bh), (0, 165, 255), 2)
            cv2.putText(display, f"CONE conf:{cone.confidence:.2f} ~{cone.distance_hint_m:.1f}m",
                        (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,165,255), 2)

        if bucket:
            x, y, bw, bh = bucket.bbox
            cv2.rectangle(display, (x, y), (x+bw, y+bh), (255, 100, 0), 2)
            cv2.putText(display, f"BUCKET conf:{bucket.confidence:.2f} ~{bucket.distance_hint_m:.1f}m",
                        (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,100,0), 2)

        if other and not bucket:
            x, y, bw, bh = other.bbox
            cv2.rectangle(display, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
            cv2.putText(display, f"OTHER:{other.label}",
                        (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)

        # show orange mask top-right corner
        orange_mask = cv2.inRange(hsv, ORANGE_LOWER, ORANGE_UPPER)
        orange_small = cv2.resize(orange_mask, (w//4, h//4))
        orange_colour = cv2.cvtColor(orange_small, cv2.COLOR_GRAY2BGR)
        display[0:h//4, 3*w//4:w] = orange_colour

        cv2.putText(display, "orange mask (top right)",
                    (3*w//4, h//4 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

        cv2.imshow("Perception Debug", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    debug_hsv_ranges()
