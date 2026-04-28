from setuptools import find_packages, setup

package_name = 'pioneer_nav'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['waypoint.txt']),
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
            'ps4_basic_drive = pioneer_nav.ps4_basic_drive:main',
            'mission_manager = pioneer_nav.mission_manager:main',
            'colour_detector = pioneer_nav.colour_detector_node:main',
        ],
    },
)
