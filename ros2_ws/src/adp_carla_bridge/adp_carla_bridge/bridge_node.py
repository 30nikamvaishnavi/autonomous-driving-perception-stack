"""CARLA <-> ROS2 bridge (synchronous mode).

Publishes : /camera/image_raw (bgr8), /camera/depth (32FC1, metres), /camera/info, /vehicle/speed,
            /vehicle/collision, /scenario/gt_distance
Subscribes: /vehicle/control_cmd (adp_msgs/VehicleControl)
Run on the Ubuntu 22.04 GPU VM with a CARLA 0.9.15 server and the `carla` Python wheel installed.
"""
import math
import queue
import random

import numpy as np
import rclpy
from adp_msgs.msg import VehicleControl
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Float32

from adp_carla_bridge import scenarios


def to_image(arr: np.ndarray, header, encoding: str, step: int) -> Image:
    m = Image()
    m.header, m.height, m.width, m.encoding, m.step = header, arr.shape[0], arr.shape[1], encoding, step
    m.data = np.ascontiguousarray(arr).tobytes()
    return m


class BridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("carla_bridge")
        for k, v in dict(host="localhost", port=2000, town="Town03", fps=20, width=1280, height=720, fov=90.0,
                         scenario="free_drive", seed=7, traffic_vehicles=15, weather="").items():
            self.declare_parameter(k, v)
        import carla

        self.carla = carla
        p = lambda k: self.get_parameter(k).value  # noqa: E731
        self.client = carla.Client(p("host"), p("port"))
        self.client.set_timeout(30.0)
        self.world = self.client.load_world(p("town"))
        st = self.world.get_settings()
        st.synchronous_mode, st.fixed_delta_seconds = True, 1.0 / p("fps")
        self.world.apply_settings(st)
        tm = self.client.get_trafficmanager()
        tm.set_synchronous_mode(True)
        tm.set_random_device_seed(p("seed"))
        rng = random.Random(p("seed"))
        bp = self.world.get_blueprint_library()
        spawns = self.world.get_map().get_spawn_points()
        self.actors = []
        self.ego = self.world.spawn_actor(bp.find("vehicle.tesla.model3"), spawns[0])
        self.actors.append(self.ego)
        if p("scenario") in ("free_drive", "night_rain"):
            for sp in spawns[1:1 + p("traffic_vehicles")]:
                v = self.world.try_spawn_actor(rng.choice(bp.filter("vehicle.*")), sp)
                if v:
                    v.set_autopilot(True, tm.get_port())
                    self.actors.append(v)
        sc = scenarios.setup(carla, self.world, self.client, self.ego, p("scenario"), rng)
        self.actors += sc["actors"]
        self.target, self.on_tick = sc["target"], sc["on_tick"]
        if p("weather"):
            self.world.set_weather(getattr(carla.WeatherParameters, p("weather")))
        self.qs = {}
        tf = carla.Transform(carla.Location(x=1.5, z=1.5))
        for name, bpid in (("rgb", "sensor.camera.rgb"), ("depth", "sensor.camera.depth")):
            b = bp.find(bpid)
            b.set_attribute("image_size_x", str(p("width")))
            b.set_attribute("image_size_y", str(p("height")))
            b.set_attribute("fov", str(p("fov")))
            s = self.world.spawn_actor(b, tf, attach_to=self.ego)
            q: queue.Queue = queue.Queue()
            s.listen(q.put)
            self.qs[name] = q
            self.actors.append(s)
        col = self.world.spawn_actor(bp.find("sensor.other.collision"), carla.Transform(), attach_to=self.ego)
        col.listen(lambda e: self.pub_col.publish(Bool(data=True)))
        self.actors.append(col)
        f = p("width") / (2 * math.tan(math.radians(p("fov")) / 2))
        self.info = CameraInfo(height=p("height"), width=p("width"))
        self.info.k = [f, 0.0, p("width") / 2, 0.0, f, p("height") / 2, 0.0, 0.0, 1.0]
        self.pub_img = self.create_publisher(Image, "/camera/image_raw", 1)
        self.pub_dep = self.create_publisher(Image, "/camera/depth", 1)
        self.pub_info = self.create_publisher(CameraInfo, "/camera/info", 1)
        self.pub_spd = self.create_publisher(Float32, "/vehicle/speed", 10)
        self.pub_col = self.create_publisher(Bool, "/vehicle/collision", 10)
        self.pub_gt = self.create_publisher(Float32, "/scenario/gt_distance", 10)
        self.cmd = VehicleControl()
        self.create_subscription(VehicleControl, "/vehicle/control_cmd", lambda m: setattr(self, "cmd", m), 10)
        self.frame = 0
        self.create_timer(1.0 / p("fps") * 0.5, self.tick)   # drive the sim as fast as the pipeline allows
        self.get_logger().info(f"CARLA bridge up, scenario={p('scenario')}")

    def tick(self) -> None:
        c = self.carla
        c_ = self.cmd
        self.ego.apply_control(c.VehicleControl(throttle=float(np.clip(c_.throttle, 0, 1)),
                                                brake=float(np.clip(c_.brake, 0, 1)),
                                                steer=float(np.clip(c_.steer, -1, 1))))
        self.on_tick(self.frame)
        self.world.tick()
        self.frame += 1
        rgb, dep = self.qs["rgb"].get(timeout=5.0), self.qs["depth"].get(timeout=5.0)
        h, w = rgb.height, rgb.width
        stamp = self.get_clock().now().to_msg()
        from std_msgs.msg import Header

        hdr = Header(stamp=stamp, frame_id="camera")
        bgr = np.frombuffer(rgb.raw_data, np.uint8).reshape(h, w, 4)[:, :, :3]
        raw = np.frombuffer(dep.raw_data, np.uint8).reshape(h, w, 4).astype(np.float32)
        metres = (raw[..., 2] + raw[..., 1] * 256 + raw[..., 0] * 65536) / (256**3 - 1) * 1000.0
        self.pub_img.publish(to_image(bgr, hdr, "bgr8", w * 3))
        self.pub_dep.publish(to_image(metres.astype(np.float32), hdr, "32FC1", w * 4))
        self.info.header = hdr
        self.pub_info.publish(self.info)
        v = self.ego.get_velocity()
        self.pub_spd.publish(Float32(data=float(math.sqrt(v.x**2 + v.y**2 + v.z**2))))
        if self.target is not None:
            d = self.ego.get_location().distance(self.target.get_location())
            self.pub_gt.publish(Float32(data=float(d)))

    def destroy(self) -> None:
        st = self.world.get_settings()
        st.synchronous_mode = False
        self.world.apply_settings(st)
        for a in self.actors:
            try:
                a.destroy()
            except Exception:  # noqa: BLE001
                pass


def main() -> None:
    rclpy.init()
    node = BridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy()


if __name__ == "__main__":
    main()
