"""Export YOLO, segmentation and MiDaS to ONNX (opset 17, static shapes, onnx-simplifier) and verify vs PyTorch."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from adpstack.utils.logging import get_logger

log = get_logger(__name__)
OPSET = 17


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


def simplify_onnx(path: Path) -> None:
    import onnx
    from onnxsim import simplify

    model, ok = simplify(onnx.load(str(path)))
    if ok:
        onnx.save(model, str(path))
    else:
        log.warning("onnx-simplifier could not validate %s; keeping original", path)


def export_yolo(weights: str, imgsz: int = 640, out: str | None = None) -> Path:
    from ultralytics import YOLO

    p = YOLO(weights).export(format="onnx", imgsz=imgsz, opset=OPSET, simplify=True, dynamic=False)
    p = Path(p)
    if out:
        p = p.replace(out)
    log.info("YOLO ONNX -> %s", p)
    return p


def export_torch(model, dummy, out: Path, input_name: str = "input", output_name: str = "output") -> Path:
    import torch

    out.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.onnx.export(model, dummy, str(out), opset_version=OPSET, input_names=[input_name],
                      output_names=[output_name], dynamic_axes=None, do_constant_folding=True)
    simplify_onnx(out)
    return out


def verify(model, dummy, onnx_path: Path, atol: float = 1e-3) -> float:
    """Return max abs diff between PyTorch and ONNX Runtime outputs; raise if above ``atol``."""
    import onnxruntime as ort
    import torch

    with torch.no_grad():
        ref = model(dummy).cpu().numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {sess.get_inputs()[0].name: dummy.cpu().numpy()})[0]
    diff = max_abs_diff(ref, out)
    log.info("%s max|diff| = %.2e", onnx_path.name, diff)
    if diff > atol:
        raise AssertionError(f"ONNX mismatch {diff} > {atol}")
    return diff


def main() -> None:  # pragma: no cover - needs torch & weights
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--yolo", help="path to best.pt")
    ap.add_argument("--seg", help="path to seg state-dict")
    ap.add_argument("--midas", action="store_true")
    ap.add_argument("--out", default="outputs/onnx")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--config", default="configs/lane.yaml")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.yolo:
        export_yolo(a.yolo, a.imgsz)
    if a.seg:
        from adpstack.lane.segmentation import build_model
        from adpstack.utils.config import load_config

        cfg = load_config(a.config)["seg"]
        m = build_model(cfg["encoder"], cfg["classes"], pretrained=False)
        m.load_state_dict(torch.load(a.seg, map_location="cpu"))
        dummy = torch.randn(1, 3, *cfg["imgsz"])
        verify(m, dummy, export_torch(m, dummy, out / "seg.onnx"))
    if a.midas:
        m = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").eval()
        dummy = torch.randn(1, 3, 256, 256)
        verify(m, dummy, export_torch(m, dummy, out / "midas_small.onnx"), atol=5e-3)


if __name__ == "__main__":
    main()
