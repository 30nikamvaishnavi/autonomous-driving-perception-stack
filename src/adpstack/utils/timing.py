"""Latency measurement helpers."""
from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import numpy as np


class Timer:
    """Accumulates per-stage wall-clock timings (ms)."""

    def __init__(self) -> None:
        self.records: dict[str, list[float]] = defaultdict(list)

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.records[name].append((time.perf_counter() - t0) * 1000.0)

    def summary(self) -> dict[str, dict[str, float]]:
        return {k: latency_stats(v) for k, v in self.records.items()}


def latency_stats(samples_ms: list[float]) -> dict[str, float]:
    """mean / p50 / p95 / p99 (ms) and FPS from latency samples."""
    a = np.asarray(samples_ms, dtype=float)
    if a.size == 0:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "fps": 0.0}
    mean = float(a.mean())
    return {
        "mean_ms": mean,
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "fps": 1000.0 / mean if mean > 0 else 0.0,
    }


def benchmark_callable(
    fn: Callable[[], object], warmup: int = 20, iters: int = 200, sync: Callable[[], None] | None = None
) -> dict[str, float]:
    """Time ``fn`` with warmup. ``sync`` (e.g. torch.cuda.synchronize) is called after each run."""
    for _ in range(warmup):
        fn()
    if sync:
        sync()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return latency_stats(samples)
