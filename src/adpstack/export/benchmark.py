"""Latency / FPS benchmark across backends: PyTorch, ONNX Runtime (CPU/DirectML/CUDA), TensorRT FP16/INT8.

Every number is MEASURED on whatever machine runs this; results are tagged with the hardware.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np

from adpstack.utils.logging import get_logger
from adpstack.utils.timing import benchmark_callable

log = get_logger(__name__)


def bench_onnx(path: str, providers: list[str], shape: tuple[int, ...], iters: int = 500, warmup: int = 50) -> dict:
    import onnxruntime as ort

    sess = ort.InferenceSession(path, providers=providers)
    name = sess.get_inputs()[0].name
    x = np.random.rand(*shape).astype(np.float32)
    res = benchmark_callable(lambda: sess.run(None, {name: x}), warmup, iters)
    res["provider"] = sess.get_providers()[0]
    return res


def bench_torch(model, shape: tuple[int, ...], device: str = "cpu", iters: int = 500, warmup: int = 50) -> dict:  # pragma: no cover
    import torch

    model = model.to(device).eval()
    x = torch.rand(*shape, device=device)
    sync = torch.cuda.synchronize if device == "cuda" else None

    def run() -> None:
        with torch.no_grad():
            model(x)

    return benchmark_callable(run, warmup, iters, sync)


def bench_trt(engine_path: str, shape: tuple[int, ...], iters: int = 500, warmup: int = 50) -> dict:  # pragma: no cover
    import tensorrt as trt
    import torch

    rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = rt.deserialize_cuda_engine(Path(engine_path).read_bytes())
    ctx = engine.create_execution_context()
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    bufs = {}
    for n in names:
        shp = tuple(engine.get_tensor_shape(n))
        bufs[n] = torch.rand(*shape, device="cuda") if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT \
            else torch.empty(*shp, device="cuda")
        ctx.set_tensor_address(n, int(bufs[n].data_ptr()))
    stream = torch.cuda.Stream()

    def run() -> None:
        ctx.execute_async_v3(stream.cuda_stream)

    return benchmark_callable(run, warmup, iters, stream.synchronize)


def to_markdown(results: dict[str, dict], baseline: str | None = None) -> str:
    """Markdown table; speed-up is relative to ``baseline`` (default: first row)."""
    base = results[baseline or next(iter(results))]["mean_ms"]
    lines = ["| Backend | mean (ms) | p50 | p95 | p99 | FPS | speed-up |", "|---|---|---|---|---|---|---|"]
    for k, r in results.items():
        lines.append(f"| {k} | {r['mean_ms']:.2f} | {r['p50_ms']:.2f} | {r['p95_ms']:.2f} | {r['p99_ms']:.2f} "
                     f"| {r['fps']:.1f} | {base / r['mean_ms']:.2f}x |")
    return "\n".join(lines)


def bar_chart(results: dict[str, dict], out: Path, title: str = "Latency by backend") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(list(results), [r["mean_ms"] for r in results.values()], color="#2a6fdb")
    ax.set_ylabel("mean latency (ms)")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx")
    ap.add_argument("--engines", nargs="*", default=[], help="label=path pairs, e.g. trt_fp16=a.engine")
    ap.add_argument("--torch-weights", help="YOLO .pt for the PyTorch baseline")
    ap.add_argument("--shape", nargs=4, type=int, default=[1, 3, 640, 640])
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--out", default="docs/figures/benchmark")
    ap.add_argument("--wandb", action="store_true")
    a = ap.parse_args()
    shape = tuple(a.shape)
    res: dict[str, dict] = {}
    if a.torch_weights:
        from ultralytics import YOLO

        net = YOLO(a.torch_weights).model.float()
        res["PyTorch FP32 (CPU)"] = bench_torch(net, shape, "cpu", min(a.iters, 100))
        try:
            import torch

            if torch.cuda.is_available():
                res["PyTorch FP32 (CUDA)"] = bench_torch(net, shape, "cuda", a.iters)
        except ImportError:
            pass
    if a.onnx:
        import onnxruntime as ort

        avail = ort.get_available_providers()
        for label, prov in (("ONNX Runtime (CPU)", "CPUExecutionProvider"),
                            ("ONNX Runtime (DirectML)", "DmlExecutionProvider"),
                            ("ONNX Runtime (CUDA)", "CUDAExecutionProvider")):
            if prov in avail:
                res[label] = bench_onnx(a.onnx, [prov], shape, a.iters)
    for item in a.engines:
        label, path = item.split("=", 1)
        res[label] = bench_trt(path, shape, a.iters)
    meta = {"platform": platform.platform(), "processor": platform.processor(), "shape": shape}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(out) + ".json").write_text(json.dumps({"meta": meta, "results": res}, indent=2))
    md = to_markdown(res)
    Path(str(out) + ".md").write_text(md)
    bar_chart(res, Path(str(out) + ".png"))
    print(md)
    if a.wandb:
        from adpstack.utils.wandb_logger import WandbLogger

        wb = WandbLogger(name="benchmark", tags=["benchmark"], config=meta)
        wb.log_table("latency", ["backend", "mean_ms", "p95_ms", "fps"],
                     [[k, v["mean_ms"], v["p95_ms"], v["fps"]] for k, v in res.items()])
        wb.log_image("latency_chart", str(out) + ".png")
        wb.finish()


if __name__ == "__main__":
    main()
