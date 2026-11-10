"""Per-scenario metrics: collisions, min distance, reaction time, end-to-end latency -> JSON + W&B."""
import json
import math
from pathlib import Path

import rclpy
from adp_msgs.msg import Risk, VehicleControl
from rclpy.node import Node
from std_msgs.msg import Bool, Float32

from adpstack.utils.wandb_logger import WandbLogger


class MetricsNode(Node):
    def __init__(self) -> None:
        super().__init__("metrics_node")
        self.declare_parameter("scenario", "unnamed")
        self.declare_parameter("duration_s", 30.0)
        self.declare_parameter("out_dir", "outputs/scenarios")
        self.declare_parameter("wandb", True)
        self.name = self.get_parameter("scenario").value
        self.t0 = self.get_clock().now()
        self.collisions, self.min_gt = 0, math.inf
        self.first_crit = self.first_brake = None
        self.lat: list[float] = []
        self.create_subscription(Bool, "/vehicle/collision", self.on_col, 10)
        self.create_subscription(Float32, "/scenario/gt_distance", self.on_gt, 10)
        self.create_subscription(Risk, "/perception/risk", self.on_risk, 10)
        self.create_subscription(VehicleControl, "/vehicle/control_cmd", self.on_ctrl, 10)
        self.create_timer(1.0, self.check_done)

    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def on_col(self, m: Bool) -> None:
        self.collisions += int(m.data)

    def on_gt(self, m: Float32) -> None:
        self.min_gt = min(self.min_gt, m.data)

    def on_risk(self, m: Risk) -> None:
        if m.level == Risk.CRITICAL and self.first_crit is None:
            self.first_crit = self.now_s()

    def on_ctrl(self, m: VehicleControl) -> None:
        stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        if stamp > 0:
            self.lat.append((self.now_s() - stamp) * 1000.0)
        if m.brake > 0.5 and self.first_brake is None:
            self.first_brake = self.now_s()

    def check_done(self) -> None:
        if (self.get_clock().now() - self.t0).nanoseconds * 1e-9 < self.get_parameter("duration_s").value:
            return
        reaction = (self.first_brake - self.first_crit) if (self.first_brake and self.first_crit) else None
        res = {"scenario": self.name, "collisions": self.collisions,
               "min_distance_m": None if math.isinf(self.min_gt) else self.min_gt,
               "reaction_time_s": reaction,
               "e2e_latency_ms_mean": sum(self.lat) / len(self.lat) if self.lat else None,
               "e2e_latency_ms_p95": sorted(self.lat)[int(0.95 * len(self.lat))] if self.lat else None}
        out = Path(self.get_parameter("out_dir").value)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{self.name}.json").write_text(json.dumps(res, indent=2))
        wb = WandbLogger(name=f"scenario-{self.name}", config=res, enabled=self.get_parameter("wandb").value,
                         tags=["scenario"])
        wb.log({k: v for k, v in res.items() if isinstance(v, (int, float))})
        wb.finish()
        self.get_logger().info(f"scenario done: {res}")
        raise SystemExit


def main() -> None:
    rclpy.init()
    try:
        rclpy.spin(MetricsNode())
    except SystemExit:
        pass


if __name__ == "__main__":
    main()
