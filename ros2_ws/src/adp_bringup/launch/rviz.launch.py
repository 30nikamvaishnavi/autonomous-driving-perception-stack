from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        Node(package="rqt_image_view", executable="rqt_image_view", arguments=["/perception/overlay"]),
    ])
