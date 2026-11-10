"""Tracks + depth -> distance, TTC, risk. Publishes /perception/objects, /perception/risk, overlay image."""
import numpy as np
import rclpy
from adp_msgs.msg import Risk, TrackedObjectArray
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from adp_perception.ros_utils import bgr_to_imgmsg, imgmsg_to_bgr, imgmsg_to_f32
from adpstack.fusion.distance_estimator import DistanceEstimator
from adpstack.fusion.risk import scene_risk
from adpstack.fusion.ttc import TTCEstimator
from adpstack.utils.config import load_config
from adpstack.viz.overlay import draw_hud, draw_tracks


class FusionNode(Node):
    def __init__(self) -> None:
        super().__init__("fusion_node")
        self.declare_parameter("config", "configs/depth.yaml")
        self.declare_parameter("use_gt_depth", False)     # CARLA depth camera (metres) instead of MiDaS
        self.declare_parameter("scale", 1.0)
        self.declare_parameter("shift", 0.0)
        cfg = load_config(self.get_parameter("config").value)
        cfg["scale_shift"] = {"scale": self.get_parameter("scale").value, "shift": self.get_parameter("shift").value}
        self.est, self.ttc = DistanceEstimator(cfg), TTCEstimator()
        self.use_gt = self.get_parameter("use_gt_depth").value
        self.rel = self.gt = self.img = None
        self.pub_obj = self.create_publisher(TrackedObjectArray, "/perception/objects", 10)
        self.pub_risk = self.create_publisher(Risk, "/perception/risk", 10)
        self.pub_vis = self.create_publisher(Image, "/perception/overlay", 1)
        self.create_subscription(Image, "/perception/depth_rel", lambda m: setattr(self, "rel", imgmsg_to_f32(m)), 1)
        self.create_subscription(Image, "/camera/depth", lambda m: setattr(self, "gt", imgmsg_to_f32(m)), 1)
        self.create_subscription(Image, "/camera/image_raw", lambda m: setattr(self, "img", m), qos_profile_sensor_data)
        self.create_subscription(TrackedObjectArray, "/perception/tracks", self.cb, 10)

    def cb(self, msg: TrackedObjectArray) -> None:
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        objs = []
        for o in msg.objects:
            box = np.array([o.x1, o.y1, o.x2, o.y2])
            e = self.est.estimate(box, o.class_name, midas_rel=self.rel,
                                  metric_depth=self.gt if self.use_gt else None)
            if not np.isfinite(e.distance_m):
                continue
            d, v, ttc = self.ttc.update(o.id, e.distance_m, t)
            o.distance_m, o.lateral_m, o.rel_speed_mps, o.ttc_s = float(d), float(e.lateral_m), float(v), float(ttc)
            objs.append({"id": o.id, "name": o.class_name, "xyxy": box, "distance_m": d, "lateral_m": e.lateral_m,
                         "ttc_s": ttc})
        msg.objects = [o for o in msg.objects if np.isfinite(o.distance_m)]
        self.pub_obj.publish(msg)
        r = scene_risk(objs)
        rm = Risk()
        rm.header, rm.level = msg.header, int(r.level)
        rm.track_id = r.track_id if r.track_id is not None else -1
        rm.distance_m, rm.ttc_s = float(r.distance_m), float(r.ttc_s)
        self.pub_risk.publish(rm)
        if self.img is not None and self.pub_vis.get_subscription_count():
            vis = draw_hud(draw_tracks(imgmsg_to_bgr(self.img), objs), int(r.level))
            self.pub_vis.publish(bgr_to_imgmsg(vis, msg.header))


def main() -> None:
    rclpy.init()
    rclpy.spin(FusionNode())


if __name__ == "__main__":
    main()
