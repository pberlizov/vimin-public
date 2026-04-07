"""
NPU Orchestrator — vimin-core

Hardware-aware local inference orchestrator. Routes tasks to the best
available local backend (MLX on Apple Silicon, llama-cpp everywhere else,
ONNX for encoder models). No cloud fallback, no cost tracking.
"""

import logging
from typing import Iterator, List, Optional, Dict, Any
from dataclasses import dataclass

from vimin_core.core.task import Task, TaskResult, ExecutionTarget
from vimin_core.hardware.telemetry import HardwareTelemetry, TelemetryCollector
from vimin_core.core.router import ExecutionRouter, RoutingRules
from vimin_core.core.local_worker import LocalWorker
from vimin_core.core.models import ModelRegistry
from vimin_core.core.inference_log import InferenceLog

try:
    from vimin_core.core.backends import ModelDescriptor, InsufficientMemoryError, BackendSelector
    BACKENDS_AVAILABLE = True
except ImportError:
    BACKENDS_AVAILABLE = False
    ModelDescriptor = None          # type: ignore
    InsufficientMemoryError = RuntimeError  # type: ignore
    BackendSelector = None          # type: ignore


@dataclass
class OrchestratorConfig:
    """Configuration for NPU Orchestrator"""
    model_path: Optional[str] = None
    routing_rules: Optional[RoutingRules] = None
    max_concurrent_tasks: int = 10


class NPUOrchestrator:
    """
    Hardware-aware local inference orchestrator.

    Selects the best local backend for the current device and routes tasks
    accordingly. No cloud execution or cost analytics — those are enterprise
    features available in the full vimin distribution.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[OrchestratorConfig] = None,
    ):
        import os
        if os.environ.get("VIMIN_DEMO_MODE") == "1":
            # In demo/test mode, we explicitly fail NPU initialization
            # to force systems (like UserAgent) into their demo fallback paths.
            raise RuntimeError("demo mode — no backend")

        self.logger = logging.getLogger(__name__)

        self.config = config or OrchestratorConfig(model_path=model_path)

        # Hardware telemetry
        self.telemetry = HardwareTelemetry()
        self.telemetry_collector = TelemetryCollector(poll_interval=1.0)
        self.telemetry_collector.start()

        # Routing (local-only: no cloud worker)
        self.router = ExecutionRouter(
            telemetry=self.telemetry,
            rules=self.config.routing_rules,
        )

        # Local worker
        try:
            self.local_worker: Optional[LocalWorker] = LocalWorker(
                model_path=model_path
            )
        except RuntimeError:
            self.local_worker = None

        self.router.set_workers(self.local_worker, None)

        # Observability
        self.inference_log = InferenceLog(max_records=100)

        # Model registry
        self.model_registry = ModelRegistry()

        self.logger.info("NPU Orchestrator (core) initialized")

    # ------------------------------------------------------------------
    # Generative model management
    # ------------------------------------------------------------------

    def load_generative_model(
        self,
        model_id: str,
        task: str = "text-generation",
        path: Optional[str] = None,
        quantization: Optional[str] = None,
        max_context: int = 2048,
        estimated_size_gb: Optional[float] = None,
    ) -> bool:
        """
        Load a generative LLM for local inference.

        Automatically selects the best backend for the current hardware:
          - Apple Silicon with mlx-lm → MLXBackend (ANE)
          - Any platform with llama-cpp-python → LlamaCppBackend (GGUF)

        Raises InsufficientMemoryError if the model would not fit in RAM.
        """
        if not BACKENDS_AVAILABLE:
            raise RuntimeError(
                "Generative backends not available — check vimin_core/core/backends/ imports."
            )

        if self.is_generative_model_loaded():
            loaded = getattr(self.local_worker, "_current_model_name", "unknown") if self.local_worker else "unknown"
            self.logger.info(f"Model '{loaded}' already loaded — skipping. Call unload_generative_model() first.")
            return True

        if quantization is None:
            try:
                import psutil as _psutil
                available_gb = _psutil.virtual_memory().available / (1024 ** 3)
                selector = BackendSelector()
                quantization = selector.recommend_quantization(model_id, available_gb)
                self.logger.info(f"Auto-selected quantization '{quantization}' ({available_gb:.1f} GB available)")
            except Exception as e:
                self.logger.warning(f"Quantization recommendation failed ({e}); using '4bit'")
                quantization = "4bit"

        descriptor = ModelDescriptor(
            model_id=model_id,
            task=task,
            path=path,
            quantization=quantization,
            max_context=max_context,
            estimated_size_gb=estimated_size_gb,
        )

        if self.local_worker is None:
            try:
                self.local_worker = LocalWorker(model_path=None)
            except RuntimeError:
                # ONNX not available — create a minimal shell for generative-only use
                worker = object.__new__(LocalWorker)
                worker.model_path = None
                worker.session = None
                worker.shadow_session = None
                worker.execution_provider = None
                worker.available_providers = []
                worker.model_loaded = False
                worker.tokenizer = None
                worker.is_causal = False
                worker.is_ner = False
                worker._id2label = None
                worker.scanner = None
                worker.provider_options = {}
                worker._cached_inputs = None
                worker._current_model_name = ""
                worker._last_load_time_ms = 0.0
                worker._inference_log = InferenceLog()
                try:
                    from vimin_core.core.backends import BackendSelector as BS
                    worker._backend_selector = BS()
                except Exception:
                    worker._backend_selector = None
                worker._active_backend = None
                self.local_worker = worker  # type: ignore
            self.router.set_workers(self.local_worker, None)

        return self.local_worker.load_generative_model(descriptor)

    def is_generative_model_loaded(self) -> bool:
        return (
            self.local_worker is not None
            and getattr(self.local_worker, "_active_backend", None) is not None
            and self.local_worker._active_backend.is_loaded
        )

    def unload_generative_model(self) -> None:
        if self.local_worker and getattr(self.local_worker, "_active_backend", None):
            self.local_worker._active_backend.unload()
            self.local_worker._active_backend = None
            self.local_worker.model_loaded = False
            self.logger.info("Generative model unloaded")

    def get_backend_status(self) -> Dict[str, Any]:
        if not BACKENDS_AVAILABLE:
            return {"backends_package": False}
        selector = BackendSelector()
        return {"backends_package": True, "available": selector.available_backends()}

    # ------------------------------------------------------------------
    # Task execution
    # ------------------------------------------------------------------

    def execute_task(self, task: Task) -> TaskResult:
        """Execute a task locally with hardware-aware routing."""
        self.logger.info(f"Executing task: {task.type.value} (complexity: {task.complexity.value})")
        try:
            decision = self.router.make_routing_decision(task)
            result = self.router.execute_task_with_failover(task, decision)
            if result and result.metadata and "inference_record" in result.metadata:
                record = result.metadata["inference_record"]
                latest = self.telemetry_collector.get_latest()
                record.thermal_state_celsius = latest.thermal_state_celsius
                self.inference_log.record(record)
            return result
        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            raise

    def stream_execute_task(self, task: Task) -> Iterator[str]:
        """Stream generated tokens as they are produced."""
        if self.is_generative_model_loaded():
            backend = self.local_worker._active_backend
            prompt = task.data if isinstance(task.data, str) else str(task.data)
            max_tokens = task.metadata.get("max_tokens", 256)
            temperature = task.metadata.get("temperature", 0.7)
            stop_sequences = task.metadata.get("stop_sequences", None)
            yield from backend.stream_generate(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                stop_sequences=stop_sequences,
            )
            return
        result = self.execute_task(task)
        if result.success and result.result is not None:
            yield str(result.result)

    def execute_batch(self, tasks: List[Task]) -> List[TaskResult]:
        """Execute multiple tasks sequentially."""
        self.logger.info(f"Executing batch of {len(tasks)} tasks")
        results = []
        for task in tasks:
            try:
                results.append(self.execute_task(task))
            except Exception as e:
                self.logger.error(f"Task {task.id} failed: {e}")
                results.append(TaskResult(
                    task_id=task.id,
                    success=False,
                    result="",
                    execution_time_ms=0.0,
                    execution_target=ExecutionTarget.LOCAL,
                    error_message=str(e),
                    metadata={"error_type": "execution_failed"},
                ))
        return results

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Release all resources: unload any loaded model and stop telemetry polling."""
        self.unload_generative_model()
        self.telemetry_collector.stop()
        self.logger.info("NPUOrchestrator cleaned up")

    def get_system_status(self) -> Dict[str, Any]:
        metrics = self.telemetry.get_system_metrics()
        return {
            "hardware": {
                "platform": f"{self.telemetry.system} {self.telemetry.machine}",
                "available_ram_gb": metrics.available_ram_gb,
                "cpu_load_percent": metrics.cpu_load_percent,
                "battery_percent": metrics.battery_percent,
                "npu_available": metrics.npu_available,
                "thermal_state_celsius": metrics.thermal_state_celsius,
            },
            "workers": {
                "local": {
                    "model_loaded": self.local_worker.model_loaded if self.local_worker else False,
                    "execution_provider": self.local_worker.execution_provider if self.local_worker else None,
                    "available_providers": self.local_worker.available_providers if self.local_worker else [],
                },
            },
            "observability": self.inference_log.get_summary(),
            "telemetry_collector_running": self.telemetry_collector.is_running,
        }


def create_orchestrator(
    model_path: Optional[str] = None,
    **kwargs,
) -> NPUOrchestrator:
    """Convenience factory — creates an NPUOrchestrator with sensible defaults."""
    return NPUOrchestrator(model_path=model_path)
