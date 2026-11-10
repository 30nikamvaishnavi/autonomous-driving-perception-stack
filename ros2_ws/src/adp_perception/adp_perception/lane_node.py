"""Image -> LaneInfo (+ lane mask overlay). Backend: classical OpenCV or segmentation model."""
import numpy as np
import rclpy
from adp_msgs.msg import LaneInfo
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from adp_perception.ros_utils import bgr_to_imgmsg, imgmsg_to_bgr
from adpstack.lane.classical import ClassicalLaneDetector
from adpstack.utils.config import load_config
from adpstack.viz.overlay import draw_lane


class LaneNode(Node):
    def __init__(self) -> None:
        super().__init__("lane_node")
        self.declare_parameter("config", "configs/lane.yaml")
        self.declare_parameter("backend", "classical")        # classical | segmentation
        self.declare_parameter("seg_weights", "")
        cfg = load_config(self.get_parameter("config").value)
        self.backend = self.get_parameter("backend").value
        if self.backend == "segmentation":
            from adpstack.lane.segmentation import SegLaneDetector

            self.seg = SegLaneDetector(self.get_parameter("seg_weights").value, cfg)
        else:
            self.det = ClassicalLaneDetector(cfg)
        self.pub = self.create_publisher(LaneInfo, "/perception/lanes", 10)
        self.pub_vis = self.create_publisher(Image, "/perception/lane_overlay", 1)
        self.create_subscription(Image, "/camera/image_raw", self.cb, qos_profile_sensor_data)

    def cb(self, msg: Image) -> None:
        img = imgmsg_to_bgr(msg)
        out = LaneInfo()
        out.header = msg.header
        if self.backend == "segmentation":
            mask = self.seg.ego_lane_mask(self.seg(img))
            out.valid = bool(mask.any())
            xs = np.nonzero(mask[int(0.9 * mask.shape[0])])[0]
            if xs.size:      # lateral offset from mask centre at the bottom of the image (approx, m)
                out.offset_m = float((img.shape[1] / 2 - (xs.min() + xs.max()) / 2) * -0.005)
            out.curvature_m, out.confidence = float("nan"), 0.8 if out.valid else 0.0
        else:
            r = self.det(img)
            mask = r.mask
            out.valid = r.left_fit is not None and r.right_fit is not None
            out.curvature_m = float(r.curvature_m) if r.curvature_m is not None else float("nan")
            out.offset_m = float(r.offset_m) if r.offset_m is not None else 0.0
            out.confidence = float(r.confidence)
            if out.valid:
                out.left_coeffs, out.right_coeffs = (r.left_fit.astype(np.float32).tolist(),
                                                     r.right_fit.astype(np.float32).tolist())
        self.pub.publish(out)
        self.pub_vis.publish(bgr_to_imgmsg(draw_lane(img, mask), msg.header))


def main() -> None:
    rclpy.init()
    rclpy.spin(LaneNode())


if __name__ == "__main__":
    main()
