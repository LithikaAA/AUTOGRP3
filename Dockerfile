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
    python3-pip \
    python3-opencv \
    doxygen \
    python3-colcon-common-extensions \
    ros-jazzy-ros-base \
    ros-jazzy-rviz2 \
    ros-jazzy-tf2-ros \
    ros-jazzy-nav-msgs \
    ros-jazzy-sensor-msgs \
    ros-jazzy-geometry-msgs \
    ros-jazzy-joy \
    ros-jazzy-teleop-twist-joy \
    ros-jazzy-nmea-navsat-driver \
    ros-jazzy-teleop-twist-keyboard \
    python3-opencv \
    ros-jazzy-cv-bridge \
    # lidar
    ros-jazzy-sick-scan-xd \
    ros-jazzy-diagnostic-updater \
    ros-jazzy-sick-scan-xd \
    ros-jazzy-cv-bridge \
    && rm -rf /var/lib/apt/lists/*

# Locale
RUN locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8

# Install AriaCoda
RUN git clone https://github.com/reedhedges/AriaCoda.git /opt/AriaCoda && \
    cd /opt/AriaCoda && \
    make -j2 && \
    make install

ENV LD_LIBRARY_PATH=/usr/local/lib

# Copy your project
COPY ros2_ws/src /ros2_ws/src
COPY ariaNode /ros2_ws/src/ariaNode

# Build workspace
WORKDIR /ros2_ws
RUN . /opt/ros/jazzy/setup.sh && \
    colcon build --symlink-install

# Auto-source environment
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc && \
    echo "source /ros2_ws/install/setup.bash" >> /root/.bashrc

CMD ["/bin/bash"]
