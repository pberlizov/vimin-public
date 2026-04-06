"""Tests for core task schema: Task, TaskResult, TaskBatch, enums."""
import json
import pytest
from vimin_core.core.task import (
    Task, TaskResult, TaskBatch, TaskComplexity, TaskType, ExecutionTarget,
)


# ---------------------------------------------------------------------------
# Task construction & validation
# ---------------------------------------------------------------------------

class TestTaskConstruction:
    def test_defaults(self):
        t = Task()
        assert t.complexity == TaskComplexity.MEDIUM
        assert t.type == TaskType.REASONING
        assert t.priority == 5
        assert t.timeout_seconds == 30.0
        assert t.data == ""
        assert t.id  # UUID auto-generated

    def test_custom_fields(self):
        t = Task(
            complexity=TaskComplexity.LOW,
            type=TaskType.TEXT_GENERATION,
            data="hello",
            priority=3,
            timeout_seconds=10.0,
        )
        assert t.complexity == TaskComplexity.LOW
        assert t.type == TaskType.TEXT_GENERATION
        assert t.data == "hello"
        assert t.priority == 3

    def test_priority_bounds_low(self):
        with pytest.raises(ValueError):
            Task(priority=0)

    def test_priority_bounds_high(self):
        with pytest.raises(ValueError):
            Task(priority=11)

    def test_timeout_positive(self):
        with pytest.raises(ValueError):
            Task(timeout_seconds=-1.0)

    def test_unique_ids(self):
        ids = {Task().id for _ in range(50)}
        assert len(ids) == 50


class TestTaskSerialization:
    def test_to_dict_round_trip(self):
        t = Task(complexity=TaskComplexity.HIGH, type=TaskType.CODE_GENERATION, data="x=1", priority=8)
        d = t.to_dict()
        assert d["complexity"] == "High"
        assert d["type"] == "CODE_GENERATION"
        t2 = Task.from_dict(d)
        assert t2.id == t.id
        assert t2.complexity == TaskComplexity.HIGH
        assert t2.type == TaskType.CODE_GENERATION

    def test_json_round_trip(self):
        t = Task(type=TaskType.SUMMARIZATION, priority=7)
        t2 = Task.from_json(t.to_json())
        assert t2.id == t.id
        assert t2.type == TaskType.SUMMARIZATION
        assert t2.priority == 7

    def test_from_dict_with_preferred_target(self):
        d = Task(preferred_target=ExecutionTarget.LOCAL).to_dict()
        t = Task.from_dict(d)
        assert t.preferred_target == ExecutionTarget.LOCAL

    def test_to_json_is_valid_json(self):
        j = Task().to_json()
        parsed = json.loads(j)
        assert "id" in parsed


# ---------------------------------------------------------------------------
# TaskResult
# ---------------------------------------------------------------------------

class TestTaskResult:
    def test_success_result(self):
        r = TaskResult(task_id="abc", success=True, result="hello")
        assert r.success
        assert r.result == "hello"
        assert r.error_message is None

    def test_failure_result(self):
        r = TaskResult(task_id="abc", success=False, error_message="boom")
        assert not r.success
        assert r.error_message == "boom"

    def test_all_fields(self):
        r = TaskResult(
            task_id="t1",
            success=True,
            result="out",
            error_message=None,
            execution_target=ExecutionTarget.LOCAL,
            execution_time_ms=123.4,
            reasoning="test reasoning",
            latency_ms=10.0,
            memory_usage_mb=200.0,
            metadata={"k": "v"},
        )
        assert r.execution_target == ExecutionTarget.LOCAL
        assert r.execution_time_ms == 123.4
        assert r.metadata == {"k": "v"}

    def test_defaults_are_none(self):
        r = TaskResult(task_id="x", success=True)
        assert r.error_message is None
        assert r.execution_target is None
        assert r.execution_time_ms is None
        assert r.metadata == {}


# ---------------------------------------------------------------------------
# TaskBatch
# ---------------------------------------------------------------------------

class TestTaskBatch:
    def test_add_and_get(self):
        b = TaskBatch()
        t = Task()
        b.add_task(t)
        assert b.get_task_by_id(t.id) is t

    def test_get_missing_returns_none(self):
        b = TaskBatch()
        assert b.get_task_by_id("nonexistent") is None

    def test_to_dict_list_round_trip(self):
        tasks = [Task(priority=i + 1) for i in range(3)]
        b = TaskBatch(tasks)
        b2 = TaskBatch.from_dict_list(b.to_dict_list())
        assert len(b2.tasks) == 3
        assert {t.id for t in b2.tasks} == {t.id for t in tasks}

    def test_json_round_trip(self):
        b = TaskBatch([Task(), Task()])
        b2 = TaskBatch.from_json(b.to_json())
        assert len(b2.tasks) == 2


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TestEnums:
    def test_task_type_values(self):
        assert TaskType.TEXT_GENERATION.value == "TEXT_GENERATION"
        assert TaskType.SPEECH_TO_TEXT.value == "SPEECH_TO_TEXT"

    def test_execution_target_values(self):
        assert ExecutionTarget.LOCAL.value == "Local"
        assert ExecutionTarget.CLOUD.value == "Cloud"
        assert ExecutionTarget.AUTO.value == "Auto"

    def test_complexity_values(self):
        assert TaskComplexity.LOW.value == "Low"
        assert TaskComplexity.MEDIUM.value == "Medium"
        assert TaskComplexity.HIGH.value == "High"
