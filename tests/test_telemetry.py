"""Tests for HardwareTelemetry and TelemetryCollector."""
import time
import pytest
from vimin_core.hardware.telemetry import HardwareTelemetry, TelemetryCollector, SystemMetrics


class TestHardwareTelemetry:
    def setup_method(self):
        self.t = HardwareTelemetry()

    def test_get_available_ram_gb_positive(self):
        ram = self.t.get_available_ram_gb()
        assert isinstance(ram, float)
        assert ram > 0.0

    def test_get_cpu_load_percent_range(self):
        cpu = self.t.get_cpu_load_percent()
        assert isinstance(cpu, float)
        assert 0.0 <= cpu <= 100.0

    def test_get_battery_status_returns_tuple(self):
        pct, plugged = self.t.get_battery_status()
        # pct is None on machines with no battery, or a float 0–100
        assert pct is None or (isinstance(pct, float) and 0.0 <= pct <= 100.0)
        assert isinstance(plugged, bool)

    def test_get_npu_availability_returns_bool(self):
        result = self.t.get_npu_availability()
        assert isinstance(result, bool)

    def test_get_thermal_state_returns_numeric_or_none(self):
        temp = self.t.get_thermal_state()
        assert temp is None or isinstance(temp, float)
        if temp is not None:
            # Sanity: temperature between -10 and 150 Celsius
            assert -10.0 < temp < 150.0

    def test_get_thermal_state_caching(self):
        """Second call within 2 s must return the cached value (same object or equal)."""
        t1 = self.t.get_thermal_state()
        t2 = self.t.get_thermal_state()
        assert t1 == t2

    def test_get_system_metrics_returns_dataclass(self):
        m = self.t.get_system_metrics()
        assert isinstance(m, SystemMetrics)
        assert m.available_ram_gb > 0.0
        assert 0.0 <= m.cpu_load_percent <= 100.0

    def test_get_telemetry_summary_keys(self):
        s = self.t.get_telemetry_summary()
        for key in ("available_ram_gb", "cpu_load_percent", "npu_available", "platform", "timestamp"):
            assert key in s, f"Missing key: {key}"

    def test_get_npu_utilization_range(self):
        util = self.t.get_npu_utilization()
        assert isinstance(util, float)
        assert 0.0 <= util <= 100.0


class TestTelemetryCollector:
    def test_not_running_before_start(self):
        c = TelemetryCollector(poll_interval=0.1)
        assert not c.is_running

    def test_starts_and_stops(self):
        c = TelemetryCollector(poll_interval=0.1)
        c.start()
        assert c.is_running
        c.stop()
        assert not c.is_running

    def test_start_idempotent(self):
        c = TelemetryCollector(poll_interval=0.1)
        c.start()
        result = c.start()  # second start should return False (already running)
        assert result is False
        c.stop()

    def test_get_latest_before_poll(self):
        """get_latest() must return valid metrics even before the first poll completes."""
        c = TelemetryCollector(poll_interval=10.0)  # slow poll — won't fire before we call
        c.start()
        try:
            m = c.get_latest()
            assert isinstance(m, SystemMetrics)
            assert m.available_ram_gb > 0.0
        finally:
            c.stop()

    def test_get_latest_after_poll(self):
        c = TelemetryCollector(poll_interval=0.05)
        c.start()
        time.sleep(0.2)  # let it poll at least once
        m = c.get_latest()
        assert isinstance(m, SystemMetrics)
        c.stop()

    def test_is_running_property(self):
        c = TelemetryCollector()
        assert not c.is_running
        c.start()
        assert c.is_running
        c.stop()
        assert not c.is_running

    def test_telemetry_property(self):
        c = TelemetryCollector()
        assert isinstance(c.telemetry, HardwareTelemetry)
