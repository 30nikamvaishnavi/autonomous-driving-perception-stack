"""AEB + ACC + lane keeping controller. Publishes /vehicle/control_cmd."""
import math

import rclpy
from adp_msgs.msg import LaneInfo, Risk, TrackedObjectArray, VehicleControl
from rclpy.node import Node
from std_msgs.msg import Float32

from adpstack.control.aeb_controller import DrivingController
from adpstack.fusion.risk import Risk as RiskData
from adpstack.fusion.risk import RiskLevel
from adpstack.utils.config import load_config


class ControllerNode(Node):
    def __init__(self) -> None:
        super().__init__("controller_node")
        self.declare_parameter("config", "configs/control.yaml")
        self.declare_parameter("rate_hz", 20.0)
        self.ctrl = DrivingController(load_config(self.get_parameter("config").value))
        self.speed, self.risk, self.lane, self.objs = 0.0, None, None, None
        self.create_subscription(Float32, "/vehicle/speed", lambda m: setattr(self, "speed", m.data), 10)
        self.create_subscription(Risk, "/perception/risk", lambda m: setattr(self, "risk", m), 10)
        self.create_subscription(LaneInfo, "/perception/lanes", lambda m: setattr(self, "lane", m), 10)
        self.create_subscription(TrackedObjectArray, "/perception/objects", lambda m: setattr(self, "objs", m), 10)
        self.pub = self.create_publisher(VehicleControl, "/vehicle/control_cmd", 10)
        self.dt = 1.0 / self.get_parameter("rate_hz").value
        self.create_timer(self.dt, self.step)

    def step(self) -> None:
        r = self.risk
        risk = RiskData(RiskLevel(r.level), r.track_id, r.distance_m, r.ttc_s) if r else \
            RiskData(RiskLevel.SAFE, None, math.inf, math.inf)
        lead_v = None
        if r and self.objs:
            for o in self.objs.objects:
                if o.id == r.track_id and math.isfinite(o.rel_speed_mps):
                    lead_v = max(self.speed + o.rel_speed_mps, 0.0)
        offset = self.lane.offset_m if (self.lane and self.lane.valid) else 0.0
        c = self.ctrl.step(self.speed, risk, lead_v, -offset, 0.0, self.dt)
        m = VehicleControl()
        m.header = r.header if r else m.header
        if not r:
            m.header.stamp = self.get_clock().now().to_msg()
        m.throttle, m.brake, m.steer, m.aeb_active = c.throttle, c.brake, c.steer, c.aeb_active
        self.pub.publish(m)


def main() -> None:
    rclpy.init()
    rclpy.spin(ControllerNode())


if __name__ == "__main__":
    main()
