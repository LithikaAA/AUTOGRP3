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
        idx, name = parts[0], parts[1]
        CLASSES[int(idx)] = name

class LetterDetectorNode(Node):
    def __init__(self):
        super().__init__('letter_detector')

        # Parameters - tune without editing code
        self.declare_parameter('topic', '/oak/rgb/image_raw')
        self.declare_parameter('brightness_threshold', 180)
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('process_every_n_frames', 3)
        self.declare_parameter('confirmations_required', 3)

        topic         = self.get_parameter('topic').value
        self.bright_thresh  = self.get_parameter('brightness_threshold').value
        self.conf_thresh    = self.get_parameter('confidence_threshold').value
        self.process_every  = self.get_parameter('process_every_n_frames').value
        self.confirms_req   = self.get_parameter('confirmations_required').value

        # Load ONNX model
        model_path = os.path.join(
            os.path.dirname(__file__), 'greek_classifier.onnx')
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.get_logger().info(f'ONNX model loaded from {model_path}')

        # Publisher so other nodes can use the detection result
        self.result_pub = self.create_publisher(String, '/detected_letter', 10)

        self.frame_count = 0
        self.last_detection = None
        self.detection_count = 0

        self.sub = self.create_subscription(
            Image, topic, self.image_callback, 10)
        self.get_logger().info(f'Listening on: {topic}')
        self.get_logger().info(
            f'Threshold: {self.bright_thresh} '
            f'Confidence: {self.conf_thresh}')

    def find_sign_region(self, gray):
        _, bright = cv2.threshold(
            gray, self.bright_thresh, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5,5), np.uint8)
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(
            bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = gray.shape
        best = None
        best_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if h*w*0.01 < area < h*w*0.5:
                x, y, cw, ch = cv2.boundingRect(cnt)
                aspect = cw/ch if ch > 0 else 0
                if 0.3 < aspect < 2.5 and area > best_area:
                    best_area = area
                    best = (x, y, cw, ch)
        return best

    def extract_letter(self, region):
        _, dark = cv2.threshold(region, 100, 255, cv2.THRESH_BINARY_INV)
        conts, _ = cv2.findContours(
            dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lx, ly = region.shape[1], region.shape[0]
        lw, lh = 0, 0
        for cnt in conts:
            if cv2.contourArea(cnt) > 100:
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

        region_rect = self.find_sign_region(gray)
        if region_rect is None:
            self.last_detection = None
            self.detection_count = 0
            return

        x, y, cw, ch = region_rect
        region = gray[y:y+ch, x:x+cw]
        letter = self.extract_letter(region)
        if letter.size == 0:
            return

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
                # Publish result for other nodes
                msg_out = String()
                msg_out.data = name
                self.result_pub.publish(msg_out)

                # Save annotated frame
                cv2.rectangle(frame, (x, y), (x+cw, y+ch),
                             (0, 255, 0), 2)
                cv2.putText(frame, f'{name} ({confidence:.2f})',
                            (x, max(y-10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2)
                cv2.imwrite('/tmp/detection_result.png', frame)
        else:
            self.last_detection = None
            self.detection_count = 0

def main():
    rclpy.init()
    rclpy.spin(LetterDetectorNode())

if __name__ == '__main__':
    main()
