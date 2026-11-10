"""DetectionArray -> TrackedObjectArray (ByteTrack IDs; distance filled in later by fusion_node)."""
import numpy as np
import rclpy
from adp_msgs.msg import DetectionArray, TrackedObject, TrackedObjectArray
from rclpy.node import Node

from adpstack.data.bdd100k_to_yolo import CLASSES
from adpstack.tracking.bytetrack_wrapper import ByteTracker
from adpstack.utils.config import load_config


class TrackerNode(Node):
    def __init__(self) -> None:
        super().__init__("tracker_node")
        self.declare_parameter("config", "configs/tracker.yaml")
        self.tracker = ByteTracker(**load_config(self.get_parameter("config").value))
        self.pub = self.create_publisher(TrackedObjectArray, "/perception/tracks", 10)
        self.create_subscription(DetectionArray, "/perception/detections", self.cb, 10)

    def cb(self, msg: DetectionArray) -> None:
        xyxy = np.array([[d.x1, d.y1, d.x2, d.y2] for d in msg.detections], float).reshape(-1, 4)
        conf = np.array([d.confidence for d in msg.detections])
        cls = np.array([d.class_id for d in msg.detections], int)
        out = TrackedObjectArray()
        out.header = msg.header
        for t in self.tracker.update(xyxy, conf, cls):
            o = TrackedObject()
            o.id, o.class_name, o.confidence = int(t.id), CLASSES[t.cls], float(t.conf)
            o.x1, o.y1, o.x2, o.y2 = (float(v) for v in t.xyxy)
            o.distance_m = o.lateral_m = o.rel_speed_mps = float("nan")
            o.ttc_s = float("inf")
            out.objects.append(o)
        self.pub.publish(out)


def main() -> None:
    rclpy.init()
    rclpy.spin(TrackerNode())


if __name__ == "__main__":
    main()
