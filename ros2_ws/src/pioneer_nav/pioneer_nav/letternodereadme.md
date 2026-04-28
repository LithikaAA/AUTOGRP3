Running on the Real Pioneer:
Terminal 1 — Start OAK-D camera:

source /opt/ros/jazzy/setup.bash
ros2 launch depthai_ros_driver camera.launch.py


Terminal 2 — Run detector with real robot parameters:

source /opt/ros/jazzy/setup.bash
python3 ~/ros2_ws/src/letter_detector_node.py \
  --ros-args \
  -p topic:=/oak/rgb/image_raw \
  -p brightness_threshold:=180 \
  -p confidence_threshold:=0.5


Terminal 3 — See detection results:

ros2 topic echo /detected_letter



Camera driver through: depthai_ros_driver and topic is: /oak/rgb/image_raw

May need to tune brightness threshold (currently at 180)
