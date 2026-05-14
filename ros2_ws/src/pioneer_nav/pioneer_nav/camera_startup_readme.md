To connect to pioneer:

Terminal 1 (aria node)
1: 
ssh team3@192.168.2.101  

2:
Clone git:
 git clone https://github.com/LithikaAA/AUTOGRP3.git

Then clone branch: 
 cd AUTOGRP3
 cd AUTOGRP3
 git pull
 git checkout camera


3:
 
 docker build -t pioneer_jazzy .


4:

 docker run -it --privileged \
  --device=/dev/ttyUSB0 \
  --volume=/dev/bus/usb:/dev/bus/usb \
  --volume=/run/udev:/run/udev:ro \
  --volume=/home/team3/AUTOGRP3/AUTOGRP3/ros2_ws/src:/ros2_ws/src \
  --network=host pioneer_jazzy



5:

 source /opt/ros/jazzy/setup.bash 
 source /ros2_ws/install/setup.bash 
 ros2 run ariaNode ariaNode -rp /dev/ttyUSB0




Starting New terminal ()
1: 
ssh team3@192.168.2.101  

2:
docker ps

3:
docker exec -it <container_name> bash




2nd terminal: Start camera(wait for "camera is ready")

 docker exec -it $(docker ps -q) bash
 source /opt/ros/jazzy/setup.bash
 ros2 launch depthai_ros_driver camera.launch.py




Terminal 3 (start detector) For unified detector:

 python3 /ros2_ws/src/pioneer_nav/pioneer_nav/unified_detector_node.py \
   --ros-args -p brightness_threshold:=170 -p confident_duration_s:=1.0
 






Terminal 4, To see results of unified:

Results are saved to part3_logs. Detections come in this form in detections_log: 
{"type": "letter", "name": "Alpha", "confidence": 0.9821, "robot_x": 1.23, "robot_y": -0.45, "robot_yaw_deg": 92.3, "timestamp": "2026-05-13T10:34:01"}
{"type": "red_obstacle", "name": "red_obstacle", "confidence": 1.0, "robot_x": 1.67, "robot_y": -0.51, "robot_yaw_deg": 88.1, "timestamp": "2026-05-13T10:34:18", "distance_m": 0.87, "bearing_deg": -5.2}


For live detection results do:
 tail -f ~/part3_logs/detections_log.jsonl, 


If you want to see in terminal detections coming in:
 source /opt/ros/jazzy/setup.bash
 ros2 topic echo /detected_letter        # letters
 ros2 topic echo /detections/colour      # colour JSON


For detection results (being docker exec'd in)
 cat ~/part3_logs/detections_log.jsonl      



If you want to see the result images:
 
 From ssh (not docker) to save images
 docker cp $(docker ps -lq):/root/part3_logs ~/part3_logs

 to get onto Mac:
 scp -r team3@192.168.2.101:~/part3_logs ~/Desktop/

 idk to get onto windows sorry but u can try that above
