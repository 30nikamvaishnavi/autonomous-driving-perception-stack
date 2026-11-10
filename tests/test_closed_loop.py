import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from closed_loop_sim import simulate  # noqa: E402


def test_aeb_prevents_collision_with_stationary_obstacle():
    assert all(simulate(True, seed=s)["collision"] is False for s in range(5))
    assert all(simulate(False, seed=s)["collision"] for s in range(3))
