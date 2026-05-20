# AUTOGRP3 — Pioneer Robot Setup & Operation Guide

---

## Table of Contents

1. [Connect to the Robot](#1-connect-to-the-robot)
2. [Clone the Repository](#2-clone-the-repository)
3. [Build the Docker Image](#3-build-the-docker-image)
4. [Run the Container](#4-run-the-container)
5. [Open Additional Terminals](#5-open-additional-terminals)
6. [Run the ROS2 Node](#6-run-the-ros2-node)
7. [Run the Controller on the Real Robot](#7-run-the-controller-on-the-real-robot)
8. [Wandering + Mapping + Saving the Map](#8-wandering--mapping--saving-the-map)
9. [GUI Mode](#9-gui-mode)

---

## 1. Connect to the Robot

**On Wi-Fi:**
```bash
ssh team3@192.168.2.101   # Bot 1
```

**On hotspot:**

First, open a terminal on the bot (bottom-left cog button) and check its IP:
```bash
ifconfig
```
Then SSH in:
```bash
ssh team3@<ip>
# Example (Rosa's phone hotspot):
ssh team3@172.20.10.4
```

---

## 2. Clone the Repository

> The repo may already exist — remove it first if so.

```bash
rm -rf AUTOGRP3
git clone https://github.com/LithikaAA/AUTOGRP3.git
cd AUTOGRP3
```

**To clone a specific branch** (e.g. `annacontainer`):
```bash
git clone -b mission-manager-fix https://github.com/LithikaAA/AUTOGRP3.git
cd AUTOGRP3
```

---

## 3. Build the Docker Image

From inside the repo:
```bash
docker build -t pioneer_jazzy .
```

---

## 4. Run the Container

```bash
docker run -it --privileged --device=/dev/ttyUSB0 --network=host pioneer_jazzy
```

If you need controller support:
```bash
docker run -it --privileged --device=/dev/ttyUSB0 --device=/dev/input/js0 --network=host pioneer_jazzy
```

---

## 5. Open Additional Terminals

To open more terminals inside the **same running container**:

1. Check the container name:
```bash
docker ps
```

2. SSH into the bot again, then exec into the container:
```bash
docker exec -it   fervent_hofstadter bash 
# Example:
docker exec -it thirsty_archimedes bash
```

---

## 6. Run the ROS2 Node

Inside the container:
```bash
ros2 run ariaNode ariaNode --rp /dev/ttyUSB0
```

---

## 7. Run the Controller on the Real Robot

> Assumes you are SSH'd into the robot and inside the Docker container.

### Step 1 — Start ARIA (robot driver)

ARIA provides odometry and connects to the Pioneer base.
```bash
ros2 run ariaNode ariaNode --rp /dev/ttyUSB0
```
✅ Must show **Connected to robot**  
✅ `/odom` should now publish

---

### Step 2 — Start the SICK TiM7xx LiDAR

The LiDAR is Ethernet-based at IP `192.168.0.1`.
```bash
ros2 launch sick_scan_xd sick_tim_7xx.launch.py hostname:=192.168.0.1
```
✅ `/scan` should begin publishing  
> Requires `ros-jazzy-sick-scan-xd` installed in your container. If missing, add it to the Dockerfile and rebuild.

---

### Step 3 — Verify all required topics

The controller will not run unless these topics are live:
```bash
ros2 topic list
```
You must see:
- `/odom`
- `/scan`
- `/camera/image`
- `/cmd_vel` *(appears once controller starts)*

---

### Step 4 — Run the DistBug controller

```bash
ros2 run pioneer_nav distbug_controller
```

The controller explores locally using LiDAR and camera detections, and only requires local sensors.

---

### Step 5 — Watch the controller output

```bash
ros2 topic echo /cmd_vel
```
Values should change as the robot moves. If always zero, check LiDAR, camera, odom, and deadman switch.

---

### Step 6 — Confirm movement

If `/cmd_vel` is non-zero, ARIA will drive the robot. The robot will:
- Explore locally
- Avoid obstacles using LiDAR
- Stop and log camera detections

---

## 8. Wandering + Mapping + Saving the Map

### Step 1 — Start ARIA
```bash
ros2 run ariaNode ariaNode --rp /dev/ttyUSB0
```

---

### Step 2 — Start the LiDAR
```bash
ros2 launch sick_scan_xd sick_tim_7xx.launch.py hostname:=192.168.0.1 frame_id:=sick_laser
```

**LiDAR troubleshooting — `frame_id` shows as `cloud`:**
```bash
pkill -f sick_scan

ros2 launch sick_scan_xd sick_tim_7xx.launch.py \
  hostname:=192.168.0.1 \
  frame_id:=sick_laser \
  tf_publish_rate:=0.0
```

---

### Step 3 — Start the controller (15×15 m wandering)
```bash
ros2 run pioneer_nav control_node --ros-args \
  -p forward_speed:=0.15 \
  -p reverse_speed:=-0.12 \
  -p turn_speed_deg:=20.0 \
  -p return_to_center_speed:=0.12 \
  -p return_to_center_turn_speed_deg:=20.0
```

---

### Step 4 — Start SLAM
```bash
cd /ros2_ws
source install/setup.bash

ros2 launch pioneer_nav slam_mapping.launch.py \
  use_sim_time:=false \
  rviz:=false \
  scan_frame:=sick_laser \
  odom_tf_stamp_with_current_time:=true \
  slam_start_delay:=8.0
```

**Kill existing SLAM sessions if needed:**
```bash
pkill -f slam_toolbox
pkill -f lifecycle_manager
pkill -f odom_tf_broadcaster
pkill -f static_transform_publisher
```

---

### Step 5 — Save the map
```bash
mkdir -p maps
ros2 run nav2_map_server map_saver_cli -f maps/pioneer_map --fmt png
```

---

### Step 6 — Retrieve the map

**Check the saved files:**
```bash
ls -lh /ros2_ws/maps
```
Expected output:
```
total 12K
-rw-r--r-- 1 root root 6.4K May 13 10:51 pioneer_map.png
-rw-r--r-- 1 root root  133 May 13 10:51 pioneer_map.yaml
```

**Copy from container to the bot's home directory:**
```bash
docker cp wizardly_gates:/ros2_ws/maps/pioneer_map.png ~/pioneer_map.png
docker cp wizardly_gates:/ros2_ws/maps/pioneer_map.yaml ~/pioneer_map.yaml
ls -lh ~/pioneer_map.*
```

**Copy from bot to your local machine:**
```bash
scp team3@192.168.2.101:~/pioneer_map.png .
scp team3@192.168.2.101:~/pioneer_map.yaml .
```

**Open the file explorer on Windows:**
```bash
explorer.exe .
```

---

### Optional — Convert map to binary `.pbm` / `.csv`
```bash
cd /mnt/c/Users/rahma/Downloads/AUTOGRP3-main/AUTOGRP3

python3 ros2_ws/src/pioneer_nav/scripts/map_to_binary.py \
  ros2_ws/maps/my_map.pgm \
  --pbm ros2_ws/maps/my_map_binary.pbm \
  --csv ros2_ws/maps/my_map_binary.csv
```

---

## 9. GUI Mode

### Steps

1. Follow the normal SSH, build, and Docker run steps above.
2. Start the ARIA node.
3. Start the LiDAR.
4. Launch the GUI:

```bash
ros2 launch pioneer_nav mapping_gui.launch.py \
  use_sim_time:=false \
  scan_frame:=sick_laser \
  gui:=true \
  estop:=true
```
on your laptop 

```bash
cd /mnt/c/Users/Lithi/Desktop/AUTO4408/Project\ 1/AUTOGRP3/ros2_ws
source install/setup.bash
export ROS_DOMAIN_ID=0
ros2 run pioneer_nav robot_gui
```
---

### Troubleshooting SLAM in GUI mode

**Kill existing sessions:**
```bash
pkill -f slam_toolbox
pkill -f lifecycle_manager
pkill -f odom_tf_broadcaster
pkill -f static_transform_publisher
```

**Check the map topic is publishing:**
```bash
ros2 topic hz /map
```

---

### Accessing saved map & waypoint files

All files save to `/ros2_ws/maps`. Key files:

| File | Description |
|------|-------------|
| `latest_obstacle_waypoints.txt` | Most recent obstacle waypoints |
| `pioneer_map_*.png` | Map image |
| `pioneer_map_*.yaml` | Map metadata |
| `pioneer_map_*_obstacle_waypoints.txt` | Timestamped obstacle waypoints |

**On the bot:**
```bash
cd ~/AUTOGRP3/AUTOGRP3/AUTOGRP3/ros2_ws/maps
ls -lh
```

**Copy to your local machine:**
```bash
scp team3@192.168.2.101:~/AUTOGRP3/AUTOGRP3/AUTOGRP3/ros2_ws/maps/latest_obstacle_waypoints.txt .
scp team3@192.168.2.101:~/AUTOGRP3/AUTOGRP3/AUTOGRP3/ros2_ws/maps/pioneer_map_*.png .
scp team3@192.168.2.101:~/AUTOGRP3/AUTOGRP3/AUTOGRP3/ros2_ws/maps/pioneer_map_*.yaml .
```