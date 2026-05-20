# ROS 2 Jazzy on Ubuntu 24.04
FROM ros:jazzy-ros-core

# Fix apt networking issues (forces IPv4)
RUN echo 'Acquire::ForceIPv4 "true";' > /etc/apt/apt.conf.d/99force-ipv4

# Basic tools + ROS Jazzy packages
RUN apt-get update && apt-get install -y \
    locales \
    git \
    build-essential \
    cmake \
    iputils-ping \
    net-tools \
    python3-pip \
    python3-opencv \
    doxygen \
    python3-colcon-common-extensions \
    ros-jazzy-ros-base \
    ros-jazzy-rviz2 \
    ros-jazzy-robot-state-publisher \
    ros-jazzy-tf2-ros \
    ros-jazzy-tf2-tools \
    ros-jazzy-tf2-msgs \
    ros-jazzy-nav-msgs \
    ros-jazzy-sensor-msgs \
    ros-jazzy-geometry-msgs \
    ros-jazzy-slam-toolbox \
    ros-jazzy-nav2-map-server \
    ros-jazzy-nav2-lifecycle-manager \
    ros-jazzy-joy \
    ros-jazzy-teleop-twist-joy \
    ros-jazzy-nmea-navsat-driver \
    ros-jazzy-teleop-twist-keyboard \
    ros-jazzy-cv-bridge \
    # lidar
    ros-jazzy-sick-scan-xd \
    ros-jazzy-depthai-ros \
    ros-jazzy-diagnostic-updater \
    ros-jazzy-depthai-ros \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install depthai --break-system-packages
# Install Python packages
RUN pip install depthai onnxruntime --break-system-packages
RUN pip install onnxruntime

# Locale
RUN locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8

# Install AriaCoda
RUN git clone https://github.com/reedhedges/AriaCoda.git /opt/AriaCoda && \
    cd /opt/AriaCoda && \
    make -j2 && \
    make install

RUN apt-get update && apt-get install -y \
    # ... existing packages ...
    ros-jazzy-nav2-bringup \
    ros-jazzy-nav2-map-server \
    ros-jazzy-nav2-lifecycle-manager \
    ros-jazzy-nav2-planner \
    ros-jazzy-nav2-controller \
    ros-jazzy-nav2-bt-navigator \
    ros-jazzy-nav2-behaviors \
    ros-jazzy-nav2-costmap-2d \
    ros-jazzy-nav2-util \
    ros-jazzy-nav2-recoveries \
    && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/usr/local/lib

# Copy your project
COPY ros2_ws/src /ros2_ws/src
COPY ros2_ws/basic_urdf.sdf /ros2_ws/basic_urdf.sdf
COPY robots /ros2_ws/robots
COPY ariaNode /ros2_ws/src/ariaNode

# Build workspace
WORKDIR /ros2_ws
RUN . /opt/ros/jazzy/setup.sh && \
    colcon build --symlink-install

# Default location for saved occupancy maps. Bind-mount this path when running
# the container if you want maps to persist on the host.
RUN mkdir -p /ros2_ws/maps

# Auto-source environment
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc && \
    echo "source /ros2_ws/install/setup.bash" >> /root/.bashrc

CMD ["/bin/bash"]
