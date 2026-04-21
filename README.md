# SETUP AND RUN CONTAINER

##  1. Connect to the Robot

### **If on Wi‑Fi**
SSH directly into the bot:

```
ssh team3@192.168.2.101   # Bot 1
```

### **If using hotspot**
First check the bot’s IP:

```
ifconfig
```

Then SSH in:

```
ssh team3@172.20.10.4
```

---

## 2. Clone the Repository

The repo may already exist, but if not:

```
git clone https://github.com/LithikaAA/AUTOGRP3.git
cd AUTOGRP3
```

To clone a specific branch (e.g., `annacontainer`):

```
git clone -b annacontainer https://github.com/LithikaAA/AUTOGRP3.git
cd AUTOGRP3
```

---

## 3. Build the Docker Image

From inside the repo:

```
docker build -t pioneer_jazzy .
```

---

## 4. Run the Container

```
docker run -it --privileged --device=/dev/ttyUSB0 --network=host pioneer_jazzy
```

---

## 5. Opening Additional Terminals

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

## 6. Running the ROS2 Node

Inside the container:

```
ros2 run ariaNode ariaNode --rp /dev/ttyUSB0
```

---
