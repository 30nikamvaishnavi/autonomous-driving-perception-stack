"""Image -> DetectionArray using adpstack.detection.Detector (torch / ONNX / TensorRT-EP backends)."""
import rclpy
from adp_msgs.msg import Detection, DetectionArray
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from adp_perception.ros_utils import imgmsg_to_bgr
from adpstack.data.bdd100k_to_yolo import CLASSES
from adpstack.detection.infer import Detector


class DetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("detector_node")
        self.declare_parameter("weights", "best.onnx")
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("providers", ["CUDAExecutionProvider", "CPUExecutionProvider"])
        w = self.get_parameter("weights").value
        prov = list(self.get_parameter("providers").value)
        self.det = Detector(w, imgsz=self.get_parameter("imgsz").value, conf=self.get_parameter("conf").value,
                            device="cuda", providers=prov)
        self.pub = self.create_publisher(DetectionArray, "/perception/detections", 10)
        self.create_subscription(Image, "/camera/image_raw", self.cb, qos_profile_sensor_data)
        self.get_logger().info(f"detector ready: {w} ({self.det.backend})")

    def cb(self, msg: Image) -> None:
        d = self.det(imgmsg_to_bgr(msg))
        out = DetectionArray()
        out.header = msg.header
        for b, c, k in zip(d.xyxy, d.conf, d.cls, strict=True):
            m = Detection()
            m.class_id, m.class_name, m.confidence = int(k), CLASSES[int(k)], float(c)
            m.x1, m.y1, m.x2, m.y2 = (float(v) for v in b)
            out.detections.append(m)
        self.pub.publish(out)


def main() -> None:
    rclpy.init()
    rclpy.spin(DetectorNode())


if __name__ == "__main__":
    main()
