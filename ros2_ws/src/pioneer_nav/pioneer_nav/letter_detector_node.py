import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import numpy as np
import cv2
import onnxruntime as ort
import os

# Load class map
CLASSES = {}
_class_file = os.path.join(os.path.dirname(__file__), 'greek_classes.txt')
with open(_class_file) as f:
    for line in f:
        parts = line.strip().split(',')
        CLASSES[int(parts[0])] = parts[1]

class LetterDetectorNode(Node):
    def __init__(self):
        super().__init__('letter_detector')

        self.declare_parameter('topic', '/oak/rgb/image_raw')
        self.declare_parameter('brightness_threshold', 170)
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('process_every_n_frames', 3)
        self.declare_parameter('confirmations_required', 3)

        topic               = self.get_parameter('topic').value
        self.bright_thresh  = self.get_parameter('brightness_threshold').value
        self.conf_thresh    = self.get_parameter('confidence_threshold').value
        self.process_every  = self.get_parameter('process_every_n_frames').value
        self.confirms_req   = self.get_parameter('confirmations_required').value

        model_path = os.path.join(os.path.dirname(__file__), 'greek_classifier.onnx')
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.get_logger().info('ONNX model loaded')

        self.result_pub = self.create_publisher(String, '/detected_letter', 10)
        self.frame_count = 0
        self.last_detection = None
        self.detection_count = 0

        self.sub = self.create_subscription(
            Image, topic, self.image_callback, 10)
        self.get_logger().info(f'Listening on: {topic}')

    def find_sign_region(self, gray, frame):
        """
        Find white A4 paper with a letter on it.
        Uses edge detection to find rectangles rather than
        brightness thresholding — much more robust in real environments.
        """
        h, w = gray.shape
        img_cx, img_cy = w // 2, h // 2

        # Step 1 — find bright white regions (paper is much whiter than walls)
        _, bright = cv2.threshold(gray, self.bright_thresh, 255,
                                   cv2.THRESH_BINARY)

        # Step 2 — find edges within bright regions only
        # This avoids detecting windows which are uniformly bright
        edges = cv2.Canny(gray, 50, 150)

        # Step 3 — combine — look for bright regions that also have edges
        # (paper has edges from the letter, windows are uniform)
        kernel = np.ones((20, 20), np.uint8)
        bright_dilated = cv2.dilate(bright, kernel)
        edges_in_bright = cv2.bitwise_and(edges, bright_dilated)

        # Step 4 — find contours of edge-containing bright regions
        kernel2 = np.ones((15, 15), np.uint8)
        filled = cv2.dilate(edges_in_bright, kernel2)
        filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel2)

        contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_score = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if h*w*0.01 < area < h*w*0.6:
                x, y, cw, ch = cv2.boundingRect(cnt)
                aspect = cw / ch if ch > 0 else 0

                # Paper is portrait or landscape A4 — roughly 0.5-2.0 aspect
                if 0.3 < aspect < 2.5:
                    # Check the region is actually bright (white paper)
                    region = gray[y:y+ch, x:x+cw]
                    mean_brightness = region.mean()
                    if mean_brightness < 150:
                        continue  # not white enough to be paper

                    # Check it contains dark pixels (the letter)
                    dark_pixels = np.sum(region < 100)
                    dark_ratio = dark_pixels / region.size
                    if dark_ratio < 0.02 or dark_ratio > 0.5:
                        continue  # too few dark pixels = no letter, too many = not paper

                    # Score: prefer centre, prefer correct size
                    cx = x + cw // 2
                    cy = y + ch // 2
                    dist = np.sqrt((cx-img_cx)**2 + (cy-img_cy)**2)
                    max_dist = np.sqrt(img_cx**2 + img_cy**2)
                    dist_penalty = dist / max_dist

                    # Prefer regions that are roughly A4 paper sized
                    # (between 5% and 40% of image)
                    size_score = min(area/(h*w*0.4), 1.0)
                    score = size_score * (1 - 0.6 * dist_penalty)

                    if score > best_score:
                        best_score = score
                        best = (x, y, cw, ch)

        return best

    def extract_letter(self, region):
      # Use adaptive threshold to handle grey backgrounds
      # better than a fixed value of 100
      blur = cv2.GaussianBlur(region, (5, 5), 0)
      thresh = cv2.adaptiveThreshold(
         blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 4
      )

      conts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
      lx, ly = region.shape[1], region.shape[0]
      lw, lh = 0, 0
      for cnt in conts:
        if cv2.contourArea(cnt) > 50:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            lx = min(lx, bx)
            ly = min(ly, by)
            lw = max(lw, bx+bw)
            lh = max(lh, by+bh)
        lw = lw - lx
        lh = lh - ly
        if lw > 0 and lh > 0:
           return region[ly:ly+lh, lx:lx+lw]
      return region

    def classify(self, letter_region):
        resized = cv2.resize(letter_region, (64, 64))
        inp = resized.astype(np.float32) / 255.0
        inp = inp[np.newaxis, np.newaxis, :, :]
        outputs = self.session.run(None, {self.input_name: inp})
        probs = outputs[0][0]
        probs = np.exp(probs - probs.max())
        probs = probs / probs.sum()
        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])
        return CLASSES[class_id], confidence

    def image_callback(self, msg):
        self.frame_count += 1
        if self.frame_count % self.process_every != 0:
            return

        frame = np.frombuffer(msg.data, dtype=np.uint8)
        frame = frame.reshape((msg.height, msg.width, -1))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        region_rect = self.find_sign_region(gray, frame)

        if region_rect is None:
            self.last_detection = None
            self.detection_count = 0
            return

        x, y, cw, ch = region_rect

        # Draw blue box showing detected sign region
        cv2.rectangle(frame, (x, y), (x+cw, y+ch), (255, 0, 0), 2)

        region = gray[y:y+ch, x:x+cw]
        letter = self.extract_letter(region)
        if letter.size == 0:
            return
        cv2.imwrite(f'/tmp/live_{self.frame_count}.png', letter)   
        cv2.imwrite('/tmp/letter_input.png', letter)
        name, confidence = self.classify(letter)

        if confidence > self.conf_thresh:
            if name == self.last_detection:
                self.detection_count += 1
            else:
                self.last_detection = name
                self.detection_count = 1

            if self.detection_count >= self.confirms_req:
                self.get_logger().info(
                    f'Detected: {name}  confidence: {confidence:.3f}'
                )
                msg_out = String()
                msg_out.data = name
                self.result_pub.publish(msg_out)
                cv2.rectangle(frame, (x, y), (x+cw, y+ch),
                             (0, 255, 0), 2)
                cv2.putText(frame, f'{name} ({confidence:.2f})',
                            (x, max(y-10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2)
        else:
            self.last_detection = None
            self.detection_count = 0

        cv2.imwrite('/tmp/detection_result.png', frame)

def main():
    rclpy.init()
    rclpy.spin(LetterDetectorNode())

if __name__ == '__main__':
    main()
