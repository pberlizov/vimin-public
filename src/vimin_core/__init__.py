"""
vimin-core — open-source local inference orchestration.

Coordinates up to 10 nodes, broadcast dispatch only.
For larger fleets, per-node targeting, fleet pipelines, and integrations,
see the full vimin distribution at https://vimin.ai.
"""

try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("vimin-core")
except Exception:
    __version__ = "0.0.0"

from vimin_core.core.orchestrator import NPUOrchestrator, create_orchestrator
from vimin_core.core.task import Task, TaskType, TaskComplexity, TaskResult

__all__ = [
    "__version__",
    "NPUOrchestrator",
    "create_orchestrator",
    "Task",
    "TaskType",
    "TaskComplexity",
    "TaskResult",
]
