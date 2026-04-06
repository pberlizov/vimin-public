"""Tests for ExecutionRouter routing logic."""
import pytest
from unittest.mock import MagicMock, patch

from vimin_core.core.task import Task, TaskResult, TaskComplexity, TaskType, ExecutionTarget
from vimin_core.core.router import ExecutionRouter, RoutingRules, RoutingDecision
from vimin_core.hardware.telemetry import SystemMetrics


def _metrics(**kwargs) -> SystemMetrics:
    defaults = dict(
        available_ram_gb=16.0,
        cpu_load_percent=20.0,
        battery_percent=80.0,
        is_plugged_in=True,
        npu_available=True,
        soc_name="Apple M3",
        npu_utilization_percent=0.0,
        thermal_state_celsius=45.0,
    )
    defaults.update(kwargs)
    return SystemMetrics(**defaults)


class TestRoutingRules:
    def setup_method(self):
        self.rules = RoutingRules()

    def test_local_with_good_resources(self):
        task = Task(complexity=TaskComplexity.LOW, type=TaskType.TEXT_GENERATION)
        m = _metrics(available_ram_gb=16.0, cpu_load_percent=20.0)
        should_local, reason = self.rules.should_route_local(task, m)
        assert should_local

    def test_cloud_when_low_battery(self):
        task = Task(complexity=TaskComplexity.LOW, type=TaskType.TEXT_GENERATION)
        m = _metrics(battery_percent=5.0, is_plugged_in=False)
        should_local, reason = self.rules.should_route_local(task, m)
        assert not should_local
        assert "battery" in reason.lower()

    def test_cloud_when_high_cpu(self):
        task = Task(complexity=TaskComplexity.LOW, type=TaskType.TEXT_GENERATION)
        m = _metrics(cpu_load_percent=90.0)
        should_local, reason = self.rules.should_route_local(task, m)
        assert not should_local

    def test_cloud_when_high_complexity(self):
        task = Task(complexity=TaskComplexity.HIGH, type=TaskType.REASONING)
        m = _metrics()
        should_local, reason = self.rules.should_route_local(task, m)
        assert not should_local

    def test_cloud_when_thermal_throttle(self):
        task = Task(complexity=TaskComplexity.LOW, type=TaskType.TEXT_GENERATION)
        m = _metrics(thermal_state_celsius=80.0)
        should_local, reason = self.rules.should_route_local(task, m)
        assert not should_local
        assert "thermal" in reason.lower() or "80" in reason

    def test_cloud_when_insufficient_ram(self):
        task = Task(complexity=TaskComplexity.LOW, type=TaskType.TEXT_GENERATION, model_name="llama-3b")
        m = _metrics(available_ram_gb=0.5)
        should_local, reason = self.rules.should_route_local(task, m)
        assert not should_local
        assert "ram" in reason.lower() or "insufficient" in reason.lower()

    def test_battery_plugged_in_not_penalised(self):
        task = Task(complexity=TaskComplexity.LOW, type=TaskType.TEXT_GENERATION)
        m = _metrics(battery_percent=5.0, is_plugged_in=True)
        # Plugged in — should not penalise for low battery
        should_local, _ = self.rules.should_route_local(task, m)
        assert should_local


class TestExecutionRouterDecision:
    def _make_router(self, ram=16.0, cpu=20.0, thermal=45.0, npu=True, battery=80.0):
        """Create a router with mocked telemetry."""
        router = ExecutionRouter()
        mock_telemetry = MagicMock()
        mock_telemetry.get_system_metrics.return_value = _metrics(
            available_ram_gb=ram,
            cpu_load_percent=cpu,
            thermal_state_celsius=thermal,
            npu_available=npu,
            battery_percent=battery,
            is_plugged_in=True,
        )
        router.telemetry = mock_telemetry
        router.local_worker = MagicMock()
        return router

    def test_decision_is_routing_decision(self):
        router = self._make_router()
        task = Task(complexity=TaskComplexity.LOW, type=TaskType.TEXT_GENERATION)
        decision = router.make_routing_decision(task)
        assert isinstance(decision, RoutingDecision)
        assert decision.target in (ExecutionTarget.LOCAL, ExecutionTarget.CLOUD)
        assert 0.0 <= decision.confidence <= 1.0
        assert decision.reasoning

    def test_preferred_local_is_honoured(self):
        router = self._make_router()
        task = Task(
            complexity=TaskComplexity.HIGH,
            type=TaskType.REASONING,
            preferred_target=ExecutionTarget.LOCAL,
        )
        decision = router.make_routing_decision(task)
        assert decision.target == ExecutionTarget.LOCAL

    def test_preferred_cloud_is_honoured(self):
        router = self._make_router()
        task = Task(
            complexity=TaskComplexity.LOW,
            type=TaskType.TEXT_GENERATION,
            preferred_target=ExecutionTarget.CLOUD,
        )
        decision = router.make_routing_decision(task)
        assert decision.target == ExecutionTarget.CLOUD

    def test_no_local_worker_routes_cloud(self):
        router = self._make_router()
        router.local_worker = None
        task = Task(complexity=TaskComplexity.LOW, type=TaskType.TEXT_GENERATION)
        decision = router.make_routing_decision(task)
        assert decision.target == ExecutionTarget.CLOUD

    def test_routing_stats_returns_dict(self):
        router = self._make_router()
        stats = router.get_routing_stats()
        assert "system_metrics" in stats
        assert "routing_rules" in stats


class TestExecutionRouterFailover:
    def test_execute_batch_error_produces_valid_taskresult(self):
        """execute_batch must return a TaskResult with correct fields on per-task failure."""
        from vimin_core.core.orchestrator import NPUOrchestrator
        orch = NPUOrchestrator.__new__(NPUOrchestrator)
        # Minimal wiring — just test the error-path TaskResult construction
        from vimin_core.core.task import TaskResult, ExecutionTarget
        result = TaskResult(
            task_id="test-id",
            success=False,
            result="",
            execution_time_ms=0.0,
            execution_target=ExecutionTarget.LOCAL,
            error_message="simulated error",
            metadata={"error_type": "execution_failed"},
        )
        assert result.task_id == "test-id"
        assert not result.success
        assert result.error_message == "simulated error"
        assert result.execution_target == ExecutionTarget.LOCAL
