"""Tests for InferenceLog ring buffer."""
import threading
import pytest
from vimin_core.core.inference_log import InferenceLog, InferenceRecord


def _make_record(**kwargs) -> InferenceRecord:
    defaults = dict(
        model_name="test-model",
        task_type="text-generation",
        inference_time_ms=100.0,
        wall_clock_ms=110.0,
        provider_used="mlx",
    )
    defaults.update(kwargs)
    return InferenceRecord(**defaults)


class TestInferenceLog:
    def test_empty_summary(self):
        log = InferenceLog(max_records=10)
        s = log.get_summary()
        assert s["total_inferences"] == 0

    def test_record_and_retrieve(self):
        log = InferenceLog(max_records=10)
        log.record(_make_record())
        assert log.get_summary()["total_inferences"] == 1

    def test_ring_buffer_eviction(self):
        log = InferenceLog(max_records=3)
        for i in range(5):
            log.record(_make_record(inference_time_ms=float(i * 10)))
        assert log.get_summary()["total_inferences"] == 3

    def test_average_inference_time(self):
        log = InferenceLog(max_records=10)
        log.record(_make_record(inference_time_ms=100.0))
        log.record(_make_record(inference_time_ms=200.0))
        s = log.get_summary()
        assert abs(s["avg_inference_ms"] - 150.0) < 0.01

    def test_min_max_inference(self):
        log = InferenceLog(max_records=10)
        for ms in [50.0, 150.0, 300.0]:
            log.record(_make_record(inference_time_ms=ms))
        s = log.get_summary()
        assert s["min_inference_ms"] == 50.0
        assert s["max_inference_ms"] == 300.0

    def test_provider_distribution(self):
        log = InferenceLog(max_records=10)
        log.record(_make_record(provider_used="mlx"))
        log.record(_make_record(provider_used="mlx"))
        log.record(_make_record(provider_used="llamacpp"))
        s = log.get_summary()
        assert s["provider_distribution"]["mlx"] == 2
        assert s["provider_distribution"]["llamacpp"] == 1

    def test_hot_swap_count(self):
        log = InferenceLog(max_records=10)
        log.record(_make_record(hot_swapped=True))
        log.record(_make_record(hot_swapped=False))
        log.record(_make_record(hot_swapped=True))
        assert log.get_summary()["hot_swap_count"] == 2

    def test_preprocessing_delta(self):
        r = _make_record(inference_time_ms=100.0, wall_clock_ms=130.0)
        assert r.preprocessing_delta_ms == 30.0

    def test_preprocessing_delta_non_negative(self):
        # wall_clock_ms < inference_time_ms should clamp to 0
        r = _make_record(inference_time_ms=200.0, wall_clock_ms=100.0)
        assert r.preprocessing_delta_ms == 0.0

    def test_get_records_all(self):
        log = InferenceLog(max_records=10)
        for _ in range(5):
            log.record(_make_record())
        assert len(log.get_records()) == 5

    def test_get_records_last_n(self):
        log = InferenceLog(max_records=10)
        for i in range(5):
            log.record(_make_record(inference_time_ms=float(i)))
        records = log.get_records(last_n=2)
        assert len(records) == 2
        assert records[-1].inference_time_ms == 4.0

    def test_clear(self):
        log = InferenceLog(max_records=10)
        log.record(_make_record())
        log.clear()
        assert log.get_summary()["total_inferences"] == 0

    def test_to_json(self):
        import json
        log = InferenceLog(max_records=10)
        log.record(_make_record(model_name="qwen"))
        data = json.loads(log.to_json())
        assert len(data) == 1
        assert data[0]["model_name"] == "qwen"

    def test_thread_safety(self):
        log = InferenceLog(max_records=200)
        threads = [
            threading.Thread(target=lambda: [log.record(_make_record()) for _ in range(10)])
            for _ in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert log.get_summary()["total_inferences"] <= 200
