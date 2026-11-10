import numpy as np

from adpstack.export.benchmark import to_markdown
from adpstack.export.to_onnx import max_abs_diff
from adpstack.utils.config import load_config


def test_config_overrides():
    cfg = load_config("configs/yolo.yaml", ["epochs=3", "augment.mosaic=0.0"])
    assert cfg["epochs"] == 3 and cfg["augment"]["mosaic"] == 0.0


def test_markdown_table_speedup():
    mk = lambda m: {"mean_ms": m, "p50_ms": m, "p95_ms": m, "p99_ms": m, "fps": 1000 / m}  # noqa: E731
    md = to_markdown({"a": mk(20.0), "b": mk(5.0)})
    assert "4.00x" in md and md.count("\n") == 3


def test_max_abs_diff():
    assert max_abs_diff(np.array([1.0, 2.0]), np.array([1.0, 2.5])) == 0.5
