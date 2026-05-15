# AUTO4508 Robot GUI

A PyQt5-based monitor that runs on your laptop and displays live data from the Pioneer 3-AT robot over ROS2.

---

## What it shows

| Panel | Contents |
|---|---|
| **Camera Feed** | Live stream from the OAK-D camera |
| **Robot Status** | Current state, action, position, heading, last detected letter |
| **Map** | Occupancy grid with robot arrow, detection markers, and planned path |
| **Detection Log** | Timestamped log of every greek letter and colour obstacle detected |
| **Last Detection Photo** | Most recent photo saved by the colour detector |

The state badge at the top changes colour automatically:

- 🟡 **IDLE** — standing by
- 🟢 **MAPPING** — exploring and building map
- 🔵 **WAYPOINT_DRIVING** — driving to waypoints
- 🔴 **ESTOP** — emergency stop, obstacle within 1m

---

## Requirements

Your laptop needs ROS2 Humble installed and sourced, plus:

```bash
sudo apt install python3-pyqt5 python3-opencv
```

The robot must be running and publishing to the topics listed below. Your laptop and the robot must be on the same network with `ROS_DOMAIN_ID` matching.

---

## Testing without the robot

You can check the GUI launches correctly before connecting to the robot:

```bash
source /opt/ros/humble/setup.bash
cd ros2_ws/src/pioneer_nav/pioneer_nav
python3 GUI.py
```

A dark window should open with all panels showing "waiting..." messages. If it opens without crashing, it's working. The panels will populate automatically once the robot nodes start publishing.

---

## Running with the robot

### Step 1 — Network setup
Make sure your laptop and the robot PC are on the same WiFi network, and set a matching `ROS_DOMAIN_ID` on **both machines**:

```bash
export ROS_DOMAIN_ID=42   # use whatever number your team set
```

To make this permanent so you don't have to set it every time:
```bash
echo "export ROS_DOMAIN_ID=42" >> ~/.bashrc
source ~/.bashrc
```

### Step 2 — Start nodes on the robot
SSH into the robot PC and launch your nodes:

```bash
# Terminal 1 — navigation and control
ros2 launch pioneer_nav part2_mission.launch.py

# Terminal 2 — detector (greek letters + colour)
ros2 run pioneer_nav unified_detector_node

# Terminal 3 — SLAM mapping
ros2 run slam_toolbox async_slam_toolbox_node
```

### Step 3 — Check topics are visible from your laptop
On your laptop, verify ROS2 can see the robot's topics:

```bash
ros2 topic list
```

You should see `/oak/rgb/image_raw`, `/robot_state`, `/map` etc. If you don't see them, the network or `ROS_DOMAIN_ID` isn't set up correctly.

### Step 4 — Launch the GUI
```bash
source /opt/ros/humble/setup.bash
cd ros2_ws/src/pioneer_nav/pioneer_nav
python3 GUI.py
```

---

## How the nodes connect to the GUI

The GUI is **passive** — it only subscribes, it never publishes. Your existing nodes don't need any changes. Here's what feeds each panel:

| GUI Panel                         | Fed by node                          | Topic                |
|-----------------------------------|--------------------------------------|----------------------|
| Camera Feed                       | OAK-D camera driver                  | `/oak/rgb/image_raw` |
| State badge                       | `control_node.py`                    | `/robot_state`       |
| Position / heading                | `odom_tf_broadcaster.py`             | `/robot/pose`        |
| Greek letter log                  | `unified_detector_node.py`           | `/detected_letter`   |
| Colour obstacle log + map markers | `unified_detector_node.py`           | `/detections/colour` |
| Map                               | `slam_toolbox`                       | `/map`               |
| Planned path                      | `control_node.py` or mission manager | `/planned_path`      |

The only things you may need to add to your existing nodes:
- Publish `/robot_state` as a `std_msgs/String` from `control_node.py` whenever the robot's state changes (e.g. `"MAPPING"`, `"WAYPOINT_DRIVING"`, `"ESTOP"`)
- Publish `/planned_path` as a `nav_msgs/Path` from your mission manager when in waypoint driving mode
- Make sure `odom_tf_broadcaster.py` is publishing to `/robot/pose` as a `geometry_msgs/Pose`

If any topics are already publishing under different names, just change the topic name at the top of `GUI.py` in the `GUINode.__init__` method.

---

## ROS2 Topics

Full reference of all topics the GUI subscribes to:

| Topic                | Type                     | Used for                                         |
|----------------------|--------------------------|--------------------------------------------------|
| `/oak/rgb/image_raw` | `sensor_msgs/Image`      | Camera feed                                      |
| `/robot_state`       | `std_msgs/String`        | State badge and action label                     |
| `/robot/pose`        | `geometry_msgs/Pose`     | Position and heading display, robot arrow on map |
| `/detected_letter`   | `std_msgs/String`        | Greek letter log entries                         |
| `/detections/colour` | `std_msgs/String` (JSON) | Colour obstacle log, map markers, photos         |
| `/map`               | `nav_msgs/OccupancyGrid` | Map panel                                        |
| `/planned_path`      | `nav_msgs/Path`          | Path overlay on map (waypoint driving phase)     |

---

## Troubleshooting

**No camera feed**
- Check the robot is publishing: `ros2 topic echo /oak/rgb/image_raw`
- Make sure `ROS_DOMAIN_ID` matches on both machines

**Map not showing**
- The map only appears once SLAM starts publishing to `/map`
- Check with: `ros2 topic list | grep map`

**Topics not visible from laptop**
- Run `ros2 topic list` on your laptop — if empty, the network isn't set up correctly
- Make sure both machines are on the same WiFi and `ROS_DOMAIN_ID` matches
- Try pinging the robot PC: `ping <robot-ip>`

**Detection photos not showing**
- Photos are saved by `unified_detector_node.py` to `~/part3_logs/colour_detections/` on the **robot PC**
- The GUI reads them from that path — if running on a separate laptop the path won't exist locally
- Either mount the robot's filesystem over the network, or copy photos across via `scp`

**GUI crashes on startup**
- Make sure ROS2 is sourced before running: `source /opt/ros/humble/setup.bash`
- Check PyQt5 is installed: `python3 -c "import PyQt5"`