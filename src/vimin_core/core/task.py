"""
Task Definition Schema

Defines the Task object and related enums for the AI orchestration system.
Provides type hints and validation for task processing.
"""

from enum import Enum
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field
from uuid import uuid4
import json
import time


class TaskComplexity(Enum):
    """Enumeration for task complexity levels"""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class TaskType(Enum):
    """Enumeration for different types of AI tasks"""
    PII_MASKING = "PII_MASKING"
    REASONING = "REASONING"
    TEXT_GENERATION = "TEXT_GENERATION"
    SUMMARIZATION = "SUMMARIZATION"
    CLASSIFICATION = "CLASSIFICATION"
    TRANSLATION = "TRANSLATION"
    CODE_GENERATION = "CODE_GENERATION"
    SENTIMENT_ANALYSIS = "SENTIMENT_ANALYSIS"
    SPEECH_TO_TEXT = "SPEECH_TO_TEXT"
    TEXT_TO_SPEECH = "TEXT_TO_SPEECH"
    VOICE_ACTIVITY_DETECTION = "VOICE_ACTIVITY_DETECTION"


class ExecutionTarget(Enum):
    """Enumeration for execution targets"""
    LOCAL = "Local"
    CLOUD = "Cloud"
    AUTO = "Auto"  # Let router decide


@dataclass
class TaskResult:
    """Data class to hold task execution results"""
    task_id: str
    success: bool
    result: Optional[Any] = None
    error_message: Optional[str] = None
    execution_target: Optional[ExecutionTarget] = None
    execution_time_ms: Optional[float] = None
    reasoning: Optional[str] = None
    latency_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    """
    Task definition for the AI orchestration system.
    
    Attributes:
        id: Unique identifier for the task
        complexity: Complexity level of the task
        type: Type of AI task to perform
        data: Input data for the task (prompt, text, etc.)
        preferred_target: Preferred execution target (optional)
        priority: Task priority (1-10, where 10 is highest)
        timeout_seconds: Maximum allowed execution time
        metadata: Additional task metadata
        created_at: Task creation timestamp
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    complexity: TaskComplexity = TaskComplexity.MEDIUM
    type: TaskType = TaskType.REASONING
    data: Any = ""
    model_name: Optional[str] = None  # Specific model requested/used
    preferred_target: Optional[ExecutionTarget] = None
    priority: int = 5  # 1-10 scale
    timeout_seconds: float = 30.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Validate task parameters after initialization"""
        if not 1 <= self.priority <= 10:
            raise ValueError("Priority must be between 1 and 10")
        if self.timeout_seconds <= 0:
            raise ValueError("Timeout must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary representation"""
        return {
            "id": self.id,
            "complexity": self.complexity.value,
            "type": self.type.value,
            "data": self.data,
            "preferred_target": self.preferred_target.value if self.preferred_target else None,
            "priority": self.priority,
            "timeout_seconds": self.timeout_seconds,
            "metadata": self.metadata,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Create task from dictionary representation"""
        # Handle enum conversions
        if "complexity" in data and isinstance(data["complexity"], str):
            data["complexity"] = TaskComplexity(data["complexity"])
        if "type" in data and isinstance(data["type"], str):
            data["type"] = TaskType(data["type"])
        if "preferred_target" in data and data["preferred_target"]:
            data["preferred_target"] = ExecutionTarget(data["preferred_target"])
        
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert task to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "Task":
        """Create task from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)


class TaskBatch:
    """Container for multiple tasks with batch operations"""
    
    def __init__(self, tasks: Optional[list[Task]] = None):
        self.tasks = tasks or []
    
    def add_task(self, task: Task) -> None:
        """Add a task to the batch"""
        self.tasks.append(task)
    
    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        """Get a task by its ID"""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None
    
    def to_dict_list(self) -> list[Dict[str, Any]]:
        """Convert all tasks to dictionary list"""
        return [task.to_dict() for task in self.tasks]
    
    @classmethod
    def from_dict_list(cls, dict_list: list[Dict[str, Any]]) -> "TaskBatch":
        """Create TaskBatch from dictionary list"""
        tasks = [Task.from_dict(task_dict) for task_dict in dict_list]
        return cls(tasks)
    
    def to_json(self) -> str:
        """Convert batch to JSON string"""
        return json.dumps(self.to_dict_list(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "TaskBatch":
        """Create TaskBatch from JSON string"""
        dict_list = json.loads(json_str)
        return cls.from_dict_list(dict_list)


# Example usage and testing
if __name__ == "__main__":
    # Create a sample task
    task = Task(
        complexity=TaskComplexity.LOW,
        type=TaskType.PII_MASKING,
        data="Please mask PII in this text: John Doe's email is john@example.com",
        priority=7
    )
    
    print("=== Task Schema Test ===")
    print(f"Task ID: {task.id}")
    print(f"Task Complexity: {task.complexity.value}")
    print(f"Task Type: {task.type.value}")
    print(f"Task JSON:\n{task.to_json()}")
    
    # Test batch operations
    batch = TaskBatch([task])
    print(f"\nBatch contains {len(batch.tasks)} tasks")
