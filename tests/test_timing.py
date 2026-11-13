from adpstack.utils.timing import Timer, benchmark_callable, latency_stats


def test_timing_helpers_handle_empty_zero_and_recorded_samples():
    assert latency_stats([])["mean_ms"] == 0.0
    assert latency_stats([0.0])["fps"] == 0.0

    timer = Timer()
    assert timer.summary() == {}
    with timer.measure("step"):
        pass
    assert timer.summary()["step"]["mean_ms"] >= 0.0


def test_benchmark_callable_runs_warmup_iterations_and_sync():
    calls = {"fn": 0, "sync": 0}

    def run():
        calls["fn"] += 1

    def sync():
        calls["sync"] += 1

    result = benchmark_callable(run, warmup=2, iters=3, sync=sync)

    assert calls == {"fn": 5, "sync": 4}
    assert result["mean_ms"] >= 0.0
