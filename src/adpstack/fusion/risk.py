"""Risk assessment from distance/TTC for objects in the ego corridor."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class RiskLevel(IntEnum):
    SAFE = 0
    CAUTION = 1
    WARNING = 2
    CRITICAL = 3


@dataclass
class Risk:
    level: RiskLevel
    track_id: int | None
    distance_m: float
    ttc_s: float


def in_corridor(lateral_m: float, half_width_m: float = 1.8) -> bool:
    return abs(lateral_m) <= half_width_m


def classify(distance_m: float, ttc_s: float, ttc_warn: float = 3.0, ttc_crit: float = 1.8,
             min_dist: float = 4.0) -> RiskLevel:
    if distance_m <= min_dist or ttc_s <= ttc_crit:
        return RiskLevel.CRITICAL
    if ttc_s <= ttc_warn:
        return RiskLevel.WARNING
    if distance_m < 15.0 or ttc_s <= 2 * ttc_warn:
        return RiskLevel.CAUTION
    return RiskLevel.SAFE


def scene_risk(objects: list[dict], **thr: float) -> Risk:
    """Worst risk over objects: each dict needs id, distance_m, lateral_m, ttc_s."""
    worst = Risk(RiskLevel.SAFE, None, float("inf"), float("inf"))
    for o in objects:
        if not in_corridor(o["lateral_m"]):
            continue
        lvl = classify(o["distance_m"], o["ttc_s"], **thr)
        if lvl > worst.level or (lvl == worst.level and o["distance_m"] < worst.distance_m):
            worst = Risk(lvl, o["id"], o["distance_m"], o["ttc_s"])
    return worst
