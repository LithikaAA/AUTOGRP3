EXAMPLE:
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ariaNode',
            executable='ariaNode',
            arguments=['-remoteHost', '192.168.2.213'],
            output='screen'
        ),
        Node(
            package='pioneer_autonomy',
            executable='waypoint_controller',
            output='screen'
        ),
    ])
