"""Build TensorRT FP16 / INT8 engines from ONNX. Requires an NVIDIA GPU + TensorRT (Colab T4 / cloud VM).

    python -m adpstack.export.to_tensorrt --onnx best.onnx --precision fp16 --out best_fp16.engine
    python -m adpstack.export.to_tensorrt --onnx best.onnx --precision int8 --calib-dir data/bdd100k/yolo/images/val
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from adpstack.utils.logging import get_logger

log = get_logger(__name__)


def calibration_batches(img_dir: Path, imgsz: int, n: int = 256) -> list[np.ndarray]:
    """Preprocess ``n`` images the same way as inference (letterbox, RGB, /255) for INT8 calibration."""
    from adpstack.detection.infer import letterbox

    out = []
    for p in sorted(img_dir.glob("*.jpg"))[:n]:
        img, _, _ = letterbox(cv2.imread(str(p)), imgsz)
        out.append(np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0))
    return out


def build_engine(onnx_path: str, out_path: str, precision: str = "fp16", calib_batches: list[np.ndarray] | None = None,
                 workspace_gb: int = 4) -> str:  # pragma: no cover - GPU only
    import tensorrt as trt
    import torch

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(Path(onnx_path).read_bytes()):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if precision in ("fp16", "int8"):
        config.set_flag(trt.BuilderFlag.FP16)
    if precision == "int8":
        if not calib_batches:
            raise ValueError("INT8 needs calibration batches (--calib-dir)")
        config.set_flag(trt.BuilderFlag.INT8)

        class Calib(trt.IInt8EntropyCalibrator2):
            def __init__(self) -> None:
                super().__init__()
                self.i = 0
                self.buf = torch.empty_like(torch.from_numpy(calib_batches[0])).cuda()

            def get_batch_size(self) -> int:
                return 1

            def get_batch(self, names):
                if self.i >= len(calib_batches):
                    return None
                self.buf.copy_(torch.from_numpy(calib_batches[self.i]))
                self.i += 1
                return [int(self.buf.data_ptr())]

            def read_calibration_cache(self):
                return None

            def write_calibration_cache(self, cache):
                Path(out_path + ".calib").write_bytes(cache)

        config.int8_calibrator = Calib()
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("TensorRT engine build failed")
    Path(out_path).write_bytes(bytes(engine))
    log.info("built %s (%s)", out_path, precision)
    return out_path


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp16")
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib-dir", default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()
    calib = calibration_batches(Path(a.calib_dir), a.imgsz) if a.calib_dir else None
    build_engine(a.onnx, a.out, a.precision, calib)


if __name__ == "__main__":
    main()
