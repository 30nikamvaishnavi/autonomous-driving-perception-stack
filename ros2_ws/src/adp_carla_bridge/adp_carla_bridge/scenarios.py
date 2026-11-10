"""CARLA scenario setup helpers (pure CARLA API; no ROS imports)."""
from __future__ import annotations

import math
import random

SCENARIOS = ("free_drive", "pedestrian_crossing", "lead_braking", "cut_in", "night_rain")

WEATHER = {
    "night_rain": dict(cloudiness=100, precipitation=90, precipitation_deposits=90, sun_altitude_angle=-30, wetness=90),
    "default": dict(cloudiness=10, precipitation=0, sun_altitude_angle=70),
}


def _ahead(carla, tf, dist: float, lateral: float = 0.0):
    yaw = math.radians(tf.rotation.yaw)
    return carla.Transform(
        carla.Location(tf.location.x + dist * math.cos(yaw) - lateral * math.sin(yaw),
                       tf.location.y + dist * math.sin(yaw) + lateral * math.cos(yaw), tf.location.z + 0.5),
        tf.rotation)


def setup(carla, world, client, ego, scenario: str, rng: random.Random) -> dict:
    """Spawn scenario actors. Returns {'actors': [...], 'target': actor-or-None, 'on_tick': callable}."""
    bp = world.get_blueprint_library()
    tf = ego.get_transform()
    actors, target = [], None
    tm = client.get_trafficmanager()

    def on_tick(frame_idx: int) -> None:
        return None

    if scenario == "pedestrian_crossing":
        walker_bp = rng.choice(bp.filter("walker.pedestrian.*"))
        start = _ahead(carla, tf, 28.0, lateral=-5.0)
        target = world.try_spawn_actor(walker_bp, start)
        if target:
            actors.append(target)
            right = _ahead(carla, tf, 0.0, 1.0).location - _ahead(carla, tf, 0.0, 0.0).location
            fwd_unit = carla.Vector3D(right.x, right.y, 0)

            def on_tick(frame_idx: int) -> None:  # noqa: F811 - starts walking after ~3 s
                if frame_idx == 60:
                    target.apply_control(carla.WalkerControl(direction=fwd_unit, speed=1.4))
    elif scenario == "lead_braking":
        target = world.try_spawn_actor(bp.find("vehicle.audi.tt"), _ahead(carla, tf, 25.0))
        if target:
            actors.append(target)
            tm.ignore_lights_percentage(target, 100)
            target.set_autopilot(True, tm.get_port())
            tm.set_desired_speed(target, 25.0)

            def on_tick(frame_idx: int) -> None:  # noqa: F811 - sudden hard braking at ~5 s
                if frame_idx == 100:
                    target.set_autopilot(False)
                    target.apply_control(carla.VehicleControl(brake=1.0))
    elif scenario == "cut_in":
        target = world.try_spawn_actor(bp.find("vehicle.audi.tt"), _ahead(carla, tf, 18.0, lateral=-3.5))
        if target:
            actors.append(target)
            target.set_autopilot(True, tm.get_port())
            tm.set_desired_speed(target, 18.0)

            def on_tick(frame_idx: int) -> None:  # noqa: F811
                if frame_idx == 80:
                    tm.force_lane_change(target, True)
    elif scenario == "night_rain":
        world.set_weather(carla.WeatherParameters(**WEATHER["night_rain"]))
        target = world.try_spawn_actor(bp.find("vehicle.audi.tt"), _ahead(carla, tf, 30.0))
        if target:
            actors.append(target)
            target.set_autopilot(True, tm.get_port())
            tm.set_desired_speed(target, 20.0)
    return {"actors": actors, "target": target, "on_tick": on_tick}
