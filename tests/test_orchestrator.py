"""Tests for NPUOrchestrator — no real model loading."""
import pytest
from unittest.mock import MagicMock, patch

from vimin_core.core.orchestrator import NPUOrchestrator, OrchestratorConfig, create_orchestrator
from vimin_core.core.task import Task, TaskResult, TaskComplexity, TaskType, ExecutionTarget


class TestOrchestratorConfig:
    def test_defaults(self):
        cfg = OrchestratorConfig()
        assert cfg.model_path is None
        assert cfg.max_concurrent_tasks == 10

    def test_custom(self):
        cfg = OrchestratorConfig(model_path="/some/path", max_concurrent_tasks=4)
        assert cfg.model_path == "/some/path"
        assert cfg.max_concurrent_tasks == 4


class TestNPUOrchestratorInit:
    def test_initialises_without_error(self):
        orch = NPUOrchestrator()
        assert orch is not None

    def test_has_inference_log(self):
        orch = NPUOrchestrator()
        assert orch.inference_log is not None

    def test_telemetry_collector_running(self):
        orch = NPUOrchestrator()
        assert orch.telemetry_collector.is_running

    def test_cleanup_stops_collector(self):
        orch = NPUOrchestrator()
        assert orch.telemetry_collector.is_running
        orch.cleanup()
        assert not orch.telemetry_collector.is_running

    def test_get_backend_status_returns_dict(self):
        orch = NPUOrchestrator()
        status = orch.get_backend_status()
        assert isinstance(status, dict)
        # Must contain at least one key regardless of available backends
        assert len(status) > 0

    def test_get_system_status_returns_dict(self):
        orch = NPUOrchestrator()
        status = orch.get_system_status()
        assert "hardware" in status
        assert "workers" in status
        assert "observability" in status
        assert "telemetry_collector_running" in status

    def test_is_generative_model_loaded_false_by_default(self):
        orch = NPUOrchestrator()
        assert not orch.is_generative_model_loaded()


class TestCreateOrchestrator:
    def test_factory_returns_orchestrator(self):
        orch = create_orchestrator()
        assert isinstance(orch, NPUOrchestrator)


class TestExecuteBatch:
    def test_batch_error_result_has_correct_fields(self):
        """Regression: execute_batch must not pass 'target' or 'error' kwargs to TaskResult."""
        orch = NPUOrchestrator()
        # Patch execute_task to always raise so we exercise the error path
        def _failing_execute(task):
            raise RuntimeError("intentional failure")
        orch.execute_task = _failing_execute

        tasks = [Task(complexity=TaskComplexity.LOW, type=TaskType.TEXT_GENERATION)]
        results = orch.execute_batch(tasks)

        assert len(results) == 1
        r = results[0]
        assert isinstance(r, TaskResult)
        assert not r.success
        assert r.error_message == "intentional failure"
        assert r.execution_target == ExecutionTarget.LOCAL
        # Fields that were previously broken (would have caused TypeError):
        assert not hasattr(r, "target")   # must NOT have stray 'target' field
        assert not hasattr(r, "error")    # must NOT have stray 'error' field

    def test_batch_partial_failure_continues(self):
        """execute_batch must process all tasks even if some fail."""
        orch = NPUOrchestrator()
        call_count = {"n": 0}

        def _sometimes_fail(task):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("first task fails")
            return TaskResult(task_id=task.id, success=True, result="ok")

        orch.execute_task = _sometimes_fail
        tasks = [
            Task(complexity=TaskComplexity.LOW, type=TaskType.TEXT_GENERATION),
            Task(complexity=TaskComplexity.LOW, type=TaskType.TEXT_GENERATION),
        ]
        results = orch.execute_batch(tasks)
        assert len(results) == 2
        assert not results[0].success
        assert results[1].success
