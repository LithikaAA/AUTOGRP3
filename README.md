## SETUP AND RUN CONTAINER

###  1. Connect to the Robot

#### **If on Wi‑Fi**
SSH directly into the bot:

```
ssh team3@192.168.2.101   # Bot 1
```

#### **If using hotspot**
First check the bot’s IP (under the docker flags section I believe) - open terminal on bot from bottom left cog button:

```
ifconfig
```

Then SSH in:

```
ssh team3@<ip>
# for example (below is Rosa's phone hotspot)
ssh team3@172.20.10.4
```

---

### 2. Clone the Repository

Note, the repo may already exist;

```
rm -rf AUTOGRP3
# do the above if it alr exists
git clone https://github.com/LithikaAA/AUTOGRP3.git
cd AUTOGRP3
```

To clone a specific branch (e.g., `annacontainer`):

```
git clone -b annacontainer https://github.com/LithikaAA/AUTOGRP3.git
cd AUTOGRP3
```

---

### 3. Build the Docker Image

From inside the repo:

```
docker build -t pioneer_jazzy .
```

---

### 4. Run the Container

```
docker run -it --privileged --device=/dev/ttyUSB0 --network=host pioneer_jazzy

# or sometimes (for controller i think?)
docker run -it --privileged --device=/dev/ttyUSB0 --device=/dev/input/js0 --network=host pioneer_jazzy
```

---

### 5. Opening Additional Terminals

If you need more terminals inside the same container:

1. Check the container name:

```
docker ps
```

2. SSH into the bot again, then exec into the running container:

```
docker exec -it <container_name> bash
```

Example:

```
docker exec -it thirsty_archimedes bash
```

---

### 6. Running the ROS2 Node

Inside the container:

```
ros2 run ariaNode ariaNode --rp /dev/ttyUSB0
```

---
## EXACT STEPS TO RUN THE CONTROLLER ON THE REAL ROBOT

(Assuming you’re SSH’d into the robot and inside the Docker container.)

1. **Start ARIA (robot driver)**
ARIA provides odometry and connects to the Pioneer base
```
ros2 run ariaNode ariaNode --rp /dev/ttyUSB0
```
- Must show **Connected to robot**
- `/odom` should now publish

2. **Start the SICK TiM7xx LIDAR**
Your LIDAR is Ethernet-based at IP 192.168.0.1.
```
ros2 launch sick_scan_xd sick_tim_7xx.launch.py hostname:=192.168.0.1
```
- Requires `ros-jazzy-sick-scan-xd` installed in your container
- `/scan` should begin publishing
- If package missing, add it to Dockerfile and rebuild

3. **Verify all required topics exist**

The controller will not run unless the required data streams are alive.
`ros2 topic list`
You MUST see:
- `/odom`
- `/scan`
- `/camera/image`
- `/cmd_vel` (will appear once controller starts)

4. **Run the DistBug controller (or whatever)**
Once all sensors are publishing, start your controller.
```
ros2 run pioneer_nav distbug_controller
```
- It will explore locally using LiDAR and camera detections
- It only requires local sensors for exploration

5. **Watch the controller output**
Confirm the robot is receiving commands.
`ros2 topic echo /cmd_vel`
- Values should change as the robot moves
- If always zero, check LIDAR, camera, odom, and deadman

6. **Confirm movement**
If `/cmd_vel` is non-zero, ARIA will drive the robot.
- Robot should explore locally
- Will avoid obstacles using LIDAR
- Will stop and log camera detections


FOR WANDERING + MAPPING + SAVING MAP: 
1. Aria Node needs to be activated:
ros2 run ariaNode ariaNode --rp /dev/ttyUSB0

2. Lidar has to be activated:
ros2 launch sick_scan_xd sick_tim_7xx.launch.py hostname:=192.168.0.1 frame_id:=sick_laser

Lidar checks - if there is an issue and map says can't subscribe/subscribing to "cloud": 
Issues that you may enounter is that the lidar link frame_id shows up as cloud - to fix this you need to kill it and check. 
pkill -f sick_scan

ros2 launch sick_scan_xd sick_tim_7xx.launch.py \
  hostname:=192.168.0.1 \
  frame_id:=sick_laser \
  tf_publish_rate:=0.0


3. Start the controller node - for 15x15 wandering
ros2 run pioneer_nav control_node --ros-args \
  -p forward_speed:=0.15 \
  -p reverse_speed:=-0.12 \
  -p turn_speed_deg:=20.0 \
  -p return_to_center_speed:=0.12 \
  -p return_to_center_turn_speed_deg:=20.0


4. Start SLAM
cd /ros2_ws
source install/setup.bash
ros2 launch pioneer_nav slam_mapping.launch.py \
  use_sim_time:=false \
  rviz:=false \
  scan_frame:=sick_laser \
  odom_tf_stamp_with_current_time:=true \
  slam_start_delay:=8.0

CHECKS: if you need to kill existing slam: 
pkill -f slam_toolbox
pkill -f lifecycle_manager
pkill -f odom_tf_broadcaster
pkill -f static_transform_publisher


5. Saving the map (potentially change this later)
save map
mkdir -p maps
ros2 run nav2_map_server map_saver_cli -f maps/pioneer_map --fmt png

6. (This will change later to integrate but for now) opening the map file/saving on your local:
a. checking that it saved/what it saved as:
ls -lh /ros2_ws/maps

(should get something like this:
total 12K
-rw-r--r-- 1 root root 6.4K May 13 10:51 pioneer_map.png
-rw-r--r-- 1 root root  133 May 13 10:51 pioneer_map.yaml
root@pioneer1-NUC11PHi7:/ros2_ws#
)

b. getting it locally
inside docker:
docker cp wizardly_gates:/ros2_ws/maps/pioneer_map.png ~/pioneer_map.png
docker cp wizardly_gates:/ros2_ws/maps/pioneer_map.yaml ~/pioneer_map.yaml
ls -lh ~/pioneer_map.*

then outside docker: 
scp team3@192.168.2.101:~/pioneer_map.png .
scp team3@192.168.2.101:~/pioneer_map.yaml .
will get something like this: 
rahma@MSI:~$ scp team3@192.168.2.101:~/pioneer_map.png .
scp team3@192.168.2.101:~/pioneer_map.yaml .
team3@192.168.2.101's password:
pioneer_map.png                                                  100% 6529   427.0KB/s   00:00
team3@192.168.2.101's password:
pioneer_map.yaml                                                 100%  133    12.3KB/s   00:00

c. then to open it: 
explorer.exe 
this will take you to the file to open 



(need to figure out where this should go but this is the convertor to pbm) 
cd /mnt/c/Users/rahma/Downloads/AUTOGRP3-main/AUTOGRP3
python3 ros2_ws/src/pioneer_nav/scripts/map_to_binary.py \
  ros2_ws/maps/my_map.pgm \
  --pbm ros2_ws/maps/my_map_binary.pbm \
  --csv ros2_ws/maps/my_map_binary.csv




----------------------------------------------------------------------------------------------------------------------------------------------------------------------
  START FROM HERE - CLEAN UP LATER (FOR GUI)
1.  Follow normal steps to SSH in and build and run docker - note potentially when running the docker you may need to do the one compatiable with the camera - play around with this a bit if something isn't working
2.  Start up aria node as normal
3.  Start up LiDAR
4.  To run the GUI (gooooooeeeeeyyy)
```
    ros2 launch pioneer_nav mapping_gui.launch.py \
    use_sim_time:=false \
    scan_frame:=sick_laser \
    gui:=true \
    estop:=true
```
6.  Potential issues:
SLAM: potentially need to kill other sessions and make sure \map topic is up
kill sessions:
```
CHECKS: if you need to kill existing slam: 
pkill -f slam_toolbox
pkill -f lifecycle_manager
pkill -f odom_tf_broadcaster
pkill -f static_transform_publisher
```
check topics: 
```
ros2 topic hz /map
```
7. Accessing the saved map/waypoint txt file: saved here: /ros2_ws/maps
The main files to access:
/ros2_ws/maps/latest_obstacle_waypoints.txt
/ros2_ws/maps/pioneer_map_*.png
/ros2_ws/maps/pioneer_map_*.yaml
/ros2_ws/maps/pioneer_map_*_obstacle_waypoints.txt

```
From pioneer:
cd ~/AUTOGRP3/AUTOGRP3/AUTOGRP3/ros2_ws/maps
ls -lh
Copy onto laptop:
scp team3@192.168.2.101:~/AUTOGRP3/AUTOGRP3/AUTOGRP3/ros2_ws/maps/latest_obstacle_waypoints.txt .
scp team3@192.168.2.101:~/AUTOGRP3/AUTOGRP3/AUTOGRP3/ros2_ws/maps/pioneer_map_*.png .
scp team3@192.168.2.101:~/AUTOGRP3/AUTOGRP3/AUTOGRP3/ros2_ws/maps/pioneer_map_*.yaml .
```
Full process is earlier in this read me as well. 

