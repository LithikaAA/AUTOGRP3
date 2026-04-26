import cv2
import numpy as np
import math

# copy the helper functions here, or just paste this minimal test:

RED_LOWER_1  = np.array([0,   120,  80], dtype=np.uint8)
RED_UPPER_1  = np.array([10,  255, 255], dtype=np.uint8)
RED_LOWER_2  = np.array([165, 120,  80], dtype=np.uint8)
RED_UPPER_2  = np.array([179, 255, 255], dtype=np.uint8)
YELLOW_LOWER = np.array([18,  120,  80], dtype=np.uint8)
YELLOW_UPPER = np.array([35,  255, 255], dtype=np.uint8)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red_mask    = cv2.inRange(hsv, RED_LOWER_1, RED_UPPER_1)
    red_mask   |= cv2.inRange(hsv, RED_LOWER_2, RED_UPPER_2)
    yellow_mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    cv2.imshow("Camera", frame)
    cv2.imshow("Red Mask", red_mask)
    cv2.imshow("Yellow Mask", yellow_mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()