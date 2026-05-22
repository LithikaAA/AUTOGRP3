from pathlib import Path
import os

from setuptools import find_packages, setup

package_name = 'pioneer_nav'
here = Path(__file__).resolve().parent
workspace_root = here.parents[1]
repo_root = here.parents[2]


def first_existing(*paths):
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def rel(path):
    return os.path.relpath(path, here)


robots_dir = first_existing(workspace_root / 'robots', repo_root / 'robots')
world_file = first_existing(workspace_root / 'basic_urdf.sdf', repo_root / 'ros2_ws' / 'basic_urdf.sdf')

asset_data_files = []
if (robots_dir / 'pioneer.urdf').exists():
    asset_data_files.append(
        ('share/' + package_name + '/robots', [rel(robots_dir / 'pioneer.urdf')])
    )
    meshes_dir = robots_dir / 'meshes'
    if meshes_dir.exists():
        mesh_groups = {}
        for mesh_path in meshes_dir.rglob('*'):
            if mesh_path.is_file():
                rel_dir = mesh_path.parent.relative_to(meshes_dir)
                dest = Path('share') / package_name / 'robots' / 'meshes' / rel_dir
                mesh_groups.setdefault(str(dest), []).append(rel(mesh_path))
        asset_data_files.extend((dest, files) for dest, files in mesh_groups.items())

if world_file.exists():
    asset_data_files.append(
        ('share/' + package_name + '/worlds', [rel(world_file)])
    )

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
            'launch/nav2_navigation.launch.py',
            'launch/gazebo_mapping_gui.launch.py',
            'launch/mapping_gui.launch.py',
            'launch/letter_detector.launch.py',
        ]),
        ('share/' + package_name + '/config', [
            'pioneer_nav/config/test_waypoints.txt',
            'config/slam_toolbox_mapping.yaml',
            'config/nav2_params.yaml',
            'config/slam_mapping.rviz',
        ]),
    ] + asset_data_files,
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
            'oak_camera = pioneer_nav.oak_camera_node:main',
            'unified_detector = pioneer_nav.unified_detector_node:main',
            'colour_detector = pioneer_nav.colour_detector_node:main',
            'estoplidar = pioneer_nav.estoplidar:main',
            'estopconnect = pioneer_nav.estopconnect:main',
            'estoptest = pioneer_nav.estoptest:main',
            'robot_gui = pioneer_nav.GUI:main',
        ],
    },
)
