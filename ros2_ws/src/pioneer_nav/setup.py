from setuptools import find_packages, setup

package_name = 'pioneer_nav'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    package_data={
        package_name: [
            'greek_classifier.onnx',
            'greek_classes.txt',
        ],
    },
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/part2_mission.launch.py',
            'launch/sdf.launch.py',
            'launch/control_node_gazebo.launch.py',
            'launch/slam_mapping.launch.py',
            'launch/gazebo_mapping_gui.launch.py',
            'launch/mapping_gui.launch.py',
        ]),
        ('share/' + package_name + '/config', [
            'pioneer_nav/config/test_waypoints.txt',
            'config/slam_toolbox_mapping.yaml',
            'config/slam_mapping.rviz',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lilac',
    maintainer_email='lilac@todo.todo',
    description='Waypoint navigation for Pioneer robot',
    license='TODO: License declaration',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'distbug_controller = pioneer_nav.distbug_controller:main',
            'lidarstop = pioneer_nav.lidarstop:main',
            'waypoint_controller = pioneer_nav.waypoint_controller:main',
            'ps4_joystick = pioneer_nav.ps4_joystick:main',
            'control_node = pioneer_nav.control_node:main',
            'odom_tf_broadcaster = pioneer_nav.odom_tf_broadcaster:main',
            'mission_manager = pioneer_nav.mission_manager:main',
            'unified_detector = pioneer_nav.unified_detector_node:main',
            'colour_detector = pioneer_nav.colour_detector_node:main',
            'estoplidar = pioneer_nav.estoplidar:main',
            'estopconnect = pioneer_nav.estopconnect:main',
            'estoptest = pioneer_nav.estoptest:main',
            'robot_gui = pioneer_nav.GUI:main',
        ],
    },
)
