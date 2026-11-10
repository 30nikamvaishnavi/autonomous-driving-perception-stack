"""Image -> relative inverse depth (MiDaS) as 32FC1."""
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from adp_perception.ros_utils import f32_to_imgmsg, imgmsg_to_bgr
from adpstack.depth.midas_depth import MiDaSDepth
from adpstack.utils.config import load_config


class DepthNode(Node):
    def __init__(self) -> None:
        super().__init__("depth_node")
        self.declare_parameter("config", "configs/depth.yaml")
        self.declare_parameter("onnx_path", "")
        self.declare_parameter("every_n", 2)
        cfg = load_config(self.get_parameter("config").value)
        onnx = self.get_parameter("onnx_path").value or None
        self.model = MiDaSDepth(cfg["model"], cfg["input_size"], "auto", onnx)
        self.every, self.n = self.get_parameter("every_n").value, 0
        self.pub = self.create_publisher(Image, "/perception/depth_rel", 1)
        self.create_subscription(Image, "/camera/image_raw", self.cb, qos_profile_sensor_data)

    def cb(self, msg: Image) -> None:
        self.n += 1
        if self.n % self.every:
            return
        self.pub.publish(f32_to_imgmsg(self.model(imgmsg_to_bgr(msg)), msg.header))


def main() -> None:
    rclpy.init()
    rclpy.spin(DepthNode())


if __name__ == "__main__":
    main()
