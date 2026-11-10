"""Record a synthetic dataset from CARLA (RGB, depth, semantic seg, 2D boxes in YOLO format).

Run on the GPU Linux VM with a CARLA 0.9.15 server running:
    python -m adpstack.data.carla_recorder --config configs/carla.yaml --weather HardRainNight
"""
from __future__ import annotations

import argparse
import queue
from pathlib import Path

import numpy as np

from adpstack.utils.config import load_config
from adpstack.utils.logging import get_logger

log = get_logger(__name__)

# CARLA semantic tag ids (0.9.15): 12 pedestrian, 13 rider, 14 car, 15 truck, 16 bus, 18 motorcycle, 19 bicycle
SEM_TO_CLASS = {14: 0, 15: 1, 16: 2, 18: 3, 19: 4, 12: 5, 13: 6}

WEATHER_PRESETS = {
    "ClearNoon": dict(cloudiness=10, precipitation=0, sun_altitude_angle=70),
    "WetCloudyNoon": dict(cloudiness=80, precipitation=30, precipitation_deposits=60, sun_altitude_angle=60),
    "HardRainNight": dict(cloudiness=100, precipitation=90, precipitation_deposits=90,
                          sun_altitude_angle=-30, wetness=90),
    "ClearNight": dict(cloudiness=0, precipitation=0, sun_altitude_angle=-40),
}


def intrinsics(width: int, height: int, fov_deg: float) -> np.ndarray:
    """Pinhole intrinsic matrix from a horizontal FOV."""
    f = width / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
    return np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]])


def carla_depth_to_meters(bgra: np.ndarray) -> np.ndarray:
    """Decode a CARLA depth image (BGRA uint8) to meters (max 1000 m)."""
    b, g, r = (bgra[..., i].astype(np.float64) for i in range(3))
    return (r + g * 256.0 + b * 256.0**2) / (256.0**3 - 1.0) * 1000.0


def boxes_from_semantic_instance(sem: np.ndarray, inst_ids: np.ndarray, min_area: int = 80
                                 ) -> list[tuple[int, float, float, float, float]]:
    """Build YOLO boxes from semantic-tag + instance-id maps (instance segmentation camera).

    ``sem`` and ``inst_ids`` are HxW int arrays. Returns (cls, cx, cy, w, h) normalized.
    """
    h, w = sem.shape
    out = []
    for tag, cls in SEM_TO_CLASS.items():
        mask_t = sem == tag
        if not mask_t.any():
            continue
        for iid in np.unique(inst_ids[mask_t]):
            ys, xs = np.nonzero(mask_t & (inst_ids == iid))
            if xs.size < min_area:
                continue
            x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
            out.append((cls, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h))
    return out


def main() -> None:  # pragma: no cover - needs a CARLA server
    import carla  # type: ignore

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/carla.yaml")
    ap.add_argument("--weather", default=None)
    ap.add_argument("--frames", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    wname = a.weather or cfg["weather"]
    n_frames = a.frames or cfg["record"]["frames"]
    every = cfg["record"]["every_n"]
    out = Path(cfg["record"]["out"]) / wname
    for sub in ("images", "labels", "depth", "semantic"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    client = carla.Client(cfg["host"], cfg["port"])
    client.set_timeout(20.0)
    world = client.load_world(cfg["town"])
    world.set_weather(carla.WeatherParameters(**WEATHER_PRESETS[wname]))
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / cfg["fps"]
    world.apply_settings(settings)
    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)
    bp = world.get_blueprint_library()
    spawns = world.get_map().get_spawn_points()
    actors = []
    try:
        ego = world.spawn_actor(bp.filter("vehicle.tesla.model3")[0], spawns[0])
        ego.set_autopilot(True, tm.get_port())
        actors.append(ego)
        for sp in spawns[1:1 + cfg["traffic"]["vehicles"]]:
            v = world.try_spawn_actor(np.random.choice(bp.filter("vehicle.*")), sp)
            if v:
                v.set_autopilot(True, tm.get_port())
                actors.append(v)
        cam_cfg = cfg["camera"]
        tf = carla.Transform(carla.Location(x=cam_cfg["x"], z=cam_cfg["z"]))
        qs, sensors = {}, []
        for name, bpid in (("rgb", "sensor.camera.rgb"), ("depth", "sensor.camera.depth"),
                           ("sem", "sensor.camera.instance_segmentation")):
            b = bp.find(bpid)
            b.set_attribute("image_size_x", str(cam_cfg["width"]))
            b.set_attribute("image_size_y", str(cam_cfg["height"]))
            b.set_attribute("fov", str(cam_cfg["fov"]))
            s = world.spawn_actor(b, tf, attach_to=ego)
            q: queue.Queue = queue.Queue()
            s.listen(q.put)
            qs[name] = q
            sensors.append(s)
        actors.extend(sensors)
        import cv2

        saved = 0
        for i in range(n_frames):
            world.tick()
            imgs = {k: q.get(timeout=5.0) for k, q in qs.items()}
            if i % every:
                continue
            H, W = cam_cfg["height"], cam_cfg["width"]
            rgb = np.frombuffer(imgs["rgb"].raw_data, np.uint8).reshape(H, W, 4)[..., :3]
            dep = np.frombuffer(imgs["depth"].raw_data, np.uint8).reshape(H, W, 4)
            ins = np.frombuffer(imgs["sem"].raw_data, np.uint8).reshape(H, W, 4)
            sem_tag = ins[..., 2].astype(np.int32)                       # R = semantic tag
            inst_id = ins[..., 1].astype(np.int32) + 256 * ins[..., 0]   # G,B = instance id
            stem = f"{wname}_{saved:06d}"
            cv2.imwrite(str(out / "images" / f"{stem}.jpg"), rgb)
            np.save(out / "depth" / f"{stem}.npy", carla_depth_to_meters(dep).astype(np.float16))
            cv2.imwrite(str(out / "semantic" / f"{stem}.png"), sem_tag.astype(np.uint8))
            lines = [f"{c} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}"
                     for c, x, y, bw, bh in boxes_from_semantic_instance(sem_tag, inst_id)]
            (out / "labels" / f"{stem}.txt").write_text("\n".join(lines))
            saved += 1
        log.info("saved %d frames to %s", saved, out)
    finally:
        settings.synchronous_mode = False
        world.apply_settings(settings)
        for ac in actors:
            ac.destroy()


if __name__ == "__main__":
    main()
