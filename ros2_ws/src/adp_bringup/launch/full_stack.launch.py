"""Launch the full stack: CARLA bridge + perception + controller + metrics (+ optional rqt)."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration as LC
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument("scenario", default_value="pedestrian_crossing"),
        DeclareLaunchArgument("weights", default_value="models/best.onnx"),
        DeclareLaunchArgument("midas_onnx", default_value=""),
        DeclareLaunchArgument("use_gt_depth", default_value="false"),
        DeclareLaunchArgument("lane_backend", default_value="classical"),
        DeclareLaunchArgument("duration_s", default_value="30.0"),
        DeclareLaunchArgument("host", default_value="localhost"),
    ]
    return LaunchDescription(args + [
        Node(package="adp_carla_bridge", executable="bridge_node", output="screen",
             parameters=[{"scenario": LC("scenario"), "host": LC("host")}]),
        Node(package="adp_perception", executable="detector_node", output="screen",
             parameters=[{"weights": LC("weights")}]),
        Node(package="adp_perception", executable="tracker_node", output="screen"),
        Node(package="adp_perception", executable="lane_node", output="screen",
             parameters=[{"backend": LC("lane_backend")}]),
        Node(package="adp_perception", executable="depth_node", output="screen",
             parameters=[{"onnx_path": LC("midas_onnx")}]),
        Node(package="adp_perception", executable="fusion_node", output="screen",
             parameters=[{"use_gt_depth": LC("use_gt_depth")}]),
        Node(package="adp_control", executable="controller_node", output="screen"),
        Node(package="adp_control", executable="metrics_node", output="screen",
             parameters=[{"scenario": LC("scenario"), "duration_s": LC("duration_s")}]),
    ])
