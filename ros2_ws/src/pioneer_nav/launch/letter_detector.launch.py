from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('topic', 
            default_value='/oak/rgb/image_raw'),
        DeclareLaunchArgument('brightness_threshold', 
            default_value='180'),
        DeclareLaunchArgument('confidence_threshold', 
            default_value='0.5'),

        Node(
            package='pioneer_nav',
            executable='letter_detector_node',
            name='letter_detector',
            output='screen',
            parameters=[{
                'topic': LaunchConfiguration('topic'),
                'brightness_threshold': LaunchConfiguration('brightness_threshold'),
                'confidence_threshold': LaunchConfiguration('confidence_threshold'),
            }]
        ),
    ])
