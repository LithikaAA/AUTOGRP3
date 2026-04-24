#!/bin/bash

# chmod +x run_robot.sh

# Source ROS2
source /opt/ros/jazzy/setup.bash
source /ros2_ws/install/setup.bash

echo "Starting RosAria..."
ros2 run ariaNode ariaNode --ros-args -p port:=/dev/ttyUSB0 &
sleep 2

echo "Starting LIDAR..."
ros2 launch sick_scan_xd sick_tim_7xx.launch.py &
sleep 2

echo "Starting DistBug controller..."
ros2 run pioneer_nav distbug_controller &
sleep 2

echo "System running. Press CTRL+C to stop."
wait
