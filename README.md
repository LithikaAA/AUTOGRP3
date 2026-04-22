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

3. **Start the GPS driver**
GPS provides the `/fix` NavSatFix messages the controller uses for position.
```
ros2 run nmea_navsat_driver nmea_serial_driver --ros-args -p port:=/dev/ttyACM0 -p baud:=9600
```
- `/fix` should publish at 1–5 Hz
- First GPS fix becomes the controller’s origin

4. **Verify all required topics exist**

The controller will not run unless all three data streams are alive.
`ros2 topic list`
You MUST see:
- `/odom`
- `/scan`
- `/fix`
- `/cmd_vel` (will appear once controller starts)

5. **Run the DistBug controller (or whatever)**
Once all sensors are publishing, start your controller.
```
ros2 run pioneer_nav distbug_controller
```
- It will load `waypoint.txt`
- It will print the current waypoint
- It will switch modes: GO_TO_POINT → WALL_FOLLOW → FIX_YAW

6. **Watch the controller output**
Confirm the robot is receiving commands.
`ros2 topic echo /cmd_vel`
- Values should change as the robot moves
- If always zero → missing GPS, LIDAR, or odom

7. **Confirm movement**
If `/cmd_vel` is non-zero, ARIA will drive the robot.
- Robot should move toward GPS waypoint
- Will avoid obstacles using LIDAR
- Will pause at each waypoint
