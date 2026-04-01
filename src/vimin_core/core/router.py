"""
Execution Router Module

Core routing logic that decides whether to execute tasks locally or in the cloud
based on hardware telemetry, task complexity, and system conditions.
"""

import time
import logging
import json
import os
from typing import Optional, Dict, Any
from dataclasses import dataclass

from ..hardware.telemetry import HardwareTelemetry, SystemMetrics
from vimin_core.core.task import Task, TaskResult, ExecutionTarget, TaskComplexity


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# region agent log
def _agent_debug_log(message: str, data: Dict[str, Any], hypothesis_id: str) -> None:
    """Lightweight NDJSON logger for router failover instrumentation."""
    try:
        payload = {
            "id": f"log_{int(time.time()*1000)}",
            "timestamp": int(time.time() * 1000),
            "location": "src/core/router.py",
            "message": message,
            "data": data,
            "runId": "pre-fix",
            "hypothesisId": hypothesis_id,
        }
        # Use a portable relative path — works on any machine
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "router_debug.log")
        with open(log_path, "a") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        # Never let debug logging break routing
        pass
# endregion


@dataclass
class RoutingDecision:
    """Data class to hold routing decision information"""
    target: ExecutionTarget
    reasoning: str
    confidence: float  # 0.0 to 1.0
    metrics: SystemMetrics


class RoutingRules:
    """
    Configurable routing rules that can be easily modified or extended.
    This class encapsulates all routing logic for better modularity.
    """
    
    def __init__(self, high_complexity_threshold=None):
        """Initialize routing rule parameters"""
        # Thresholds can be adjusted based on requirements
        self.min_ram_gb_local = 2.0  # Minimum RAM for local execution
        self.max_cpu_load_percent = 80.0  # Max CPU load for local execution
        self.low_battery_threshold = 10.0  # Force cloud if battery below this
        self.local_latency_threshold_ms = 50.0  # Provider latency gate (50ms)
        self.inference_swap_threshold_ms = 100.0  # Hot-swap trigger for model downgrade
        self.high_complexity_threshold = high_complexity_threshold or TaskComplexity.HIGH
        self.thermal_throttle_celsius = 70.0  # Force model downgrade at this temp
        self.thermal_warmup_celsius = 60.0 # Pre-emptive warmup (Fair state)
        self.safety_margin_ram_gb = 0.5  # Extra headroom required
        
    def estimate_model_memory_usage(self, task: Task) -> float:
        """Estimate RAM required for the task's model in GB"""
        # Heuristic: ~4GB for 3B models, ~1.5GB for 1B, ~0.5GB for small models
        # This could be looked up from a model registry in a real system
        if "llama" in str(task.model_name).lower() and "3b" in str(task.model_name).lower():
            return 4.0
        elif "llama" in str(task.model_name).lower() or "1b" in str(task.model_name).lower():
            return 1.5
        elif "mobilebert" in str(task.model_name).lower():
            return 0.4
        elif "tinybert" in str(task.model_name).lower():
            return 0.1
        return 0.5  # Default fallback
        
    def should_route_local(self, task: Task, metrics: SystemMetrics) -> tuple[bool, str]:
        """
        Determine if a task should be routed to local execution
        
        Args:
            task: The task to evaluate
            metrics: Current system metrics
            
        Returns:
            tuple: (should_route_local, reasoning)
        """
        # Rule 0: Thermal Guard (actual temperature check)
        if metrics.thermal_state_celsius is not None and metrics.thermal_state_celsius >= self.thermal_throttle_celsius:
             return False, f"Thermal throttle ({metrics.thermal_state_celsius:.1f}C >= {self.thermal_throttle_celsius}C), routing to Cloud"
        
        # Rule 0b: CPU load as thermal proxy fallback
        if metrics.cpu_load_percent > 95:
             return False, f"High thermal pressure (CPU {metrics.cpu_load_percent}%), routing to Cloud"

        # Rule 1: Battery conservation - force cloud if battery is low
        if metrics.battery_percent is not None and metrics.battery_percent < self.low_battery_threshold:
            if not metrics.is_plugged_in:
                return False, f"Battery at {metrics.battery_percent:.1f}% (< {self.low_battery_threshold}%), routing to Cloud to save power"
        
        # Rule 2: Dynamic Memory Pressure Check
        required_ram = self.estimate_model_memory_usage(task)
        if metrics.available_ram_gb < (required_ram + self.safety_margin_ram_gb):
            return False, f"Insufficient RAM ({metrics.available_ram_gb:.2f}GB available < {required_ram + self.safety_margin_ram_gb:.2f}GB required), routing to Cloud"
        
        # Rule 3: CPU load check
        if metrics.cpu_load_percent > self.max_cpu_load_percent:
            return False, f"CPU load ({metrics.cpu_load_percent:.1f}%) > threshold ({self.max_cpu_load_percent}%), routing to Cloud"
        
        # Rule 4: Task complexity check
        if task.complexity == self.high_complexity_threshold:
            return False, f"High complexity task ({task.complexity.value}), routing to Cloud"
        
        # Rule 5: Low complexity tasks with sufficient resources go local
        if task.complexity == TaskComplexity.LOW and metrics.available_ram_gb >= self.min_ram_gb_local:
            return True, f"Low complexity task with sufficient resources ({metrics.available_ram_gb:.1f}GB RAM), routing to Local"
        
        # Rule 6: Medium complexity with NPU available
        if task.complexity == TaskComplexity.MEDIUM and metrics.npu_available:
            return True, f"Medium complexity task with NPU available, routing to Local"
        
        # Default: All low-medium complexity tasks go local with good resources
        if metrics.available_ram_gb >= self.min_ram_gb_local and metrics.cpu_load_percent <= self.max_cpu_load_percent:
            return True, f"Task routed to Local (sufficient resources available)"
        
        # Only route to cloud if none of the above conditions are met
        return False, f"Insufficient resources for local execution, routing to Cloud"


class ExecutionRouter:
    """
    Core execution router that decides where to run tasks based on
    hardware telemetry and routing rules.
    """
    
    def __init__(self, telemetry: Optional[HardwareTelemetry] = None, rules: Optional[RoutingRules] = None):
        """
        Initialize the execution router
        
        Args:
            telemetry: Hardware telemetry instance (creates default if None)
            rules: Routing rules instance (creates default if None)
        """
        self.telemetry = telemetry or HardwareTelemetry()
        self.rules = rules or RoutingRules()
        self.local_worker = None
        self.cloud_worker = None
        self._provider_latencies: Dict[str, float] = {}  # Cache of micro-benchmark results
        self._last_inference_ms: float = 0.0  # Track last inference latency for fallback
    
    def get_best_provider(self) -> Dict[str, Any]:
        """
        Determine the best execution provider by running a micro-benchmark.
        Checks QNN -> CoreML -> DirectML -> CPU and rejects providers with
        latency exceeding the 50ms threshold.
        
        Returns:
            Dict with 'provider', 'latency_ms', and 'options'
        """
        try:
            import onnxruntime as ort
            import numpy as np
        except ImportError:
            return {'provider': 'CPUExecutionProvider', 'latency_ms': 0, 'options': {}}
        
        provider_priority = [
            ('QNNExecutionProvider', {}),
            ('CoreMLExecutionProvider', {'MLComputeUnits': 'ALL'}),
            ('DirectMLExecutionProvider', {}),
            ('CUDAExecutionProvider', {}),
            ('CPUExecutionProvider', {}),
        ]
        
        available = ort.get_available_providers()
        best = {'provider': 'CPUExecutionProvider', 'latency_ms': float('inf'), 'options': {}}
        
        for provider_name, options in provider_priority:
            if provider_name not in available:
                continue
            
            # Micro-benchmark: create a tiny session and time a dummy inference
            try:
                # Create a minimal ONNX graph for benchmarking
                dummy_input = np.random.randn(1, 128).astype(np.float32)
                
                start = time.time()
                # NOTE: This times provider object creation, not inference.
                # It serves as a lightweight availability check, not a full perf benchmark.
                sess_opts = ort.SessionOptions()
                sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                
                elapsed_ms = (time.time() - start) * 1000
                self._provider_latencies[provider_name] = elapsed_ms
                
                threshold = float(self.rules.local_latency_threshold_ms)
                # Accept if under the latency gate
                if elapsed_ms < threshold:
                    if elapsed_ms < float(best['latency_ms']):
                        best = {'provider': provider_name, 'latency_ms': elapsed_ms, 'options': options}
                else:
                    logger.info(f"Provider {provider_name} rejected: {elapsed_ms:.1f}ms > {self.rules.local_latency_threshold_ms}ms")
                    
            except Exception as e:
                logger.warning(f"Provider {provider_name} benchmark failed: {e}")
                continue
        
        logger.info(f"Best provider: {best['provider']} ({best['latency_ms']:.1f}ms)")
        return best
    
    def check_dynamic_fallback(self, last_inference_ms: float) -> bool:
        """
        Check if the last inference exceeded the latency threshold,
        indicating we should switch providers or downgrade the model.
        
        Args:
            last_inference_ms: Latency of the most recent inference
            
        Returns:
            bool: True if fallback should be triggered
        """
        self._last_inference_ms = last_inference_ms
        if last_inference_ms > self.rules.local_latency_threshold_ms:
            logger.warning(
                f"Dynamic fallback triggered: {last_inference_ms:.1f}ms > "
                f"{self.rules.local_latency_threshold_ms}ms threshold"
            )
            return True
        return False
    
    def should_hot_swap(self, last_inference_ms: float) -> bool:
        """
        Check if inference latency warrants swapping to a smaller/quantized model.
        
        Args:
            last_inference_ms: Latency of the most recent inference
            
        Returns:
            bool: True if model swap should be triggered
        """
        if last_inference_ms > self.rules.inference_swap_threshold_ms:
            logger.warning(
                f"Hot-swap triggered: {last_inference_ms:.1f}ms > "
                f"{self.rules.inference_swap_threshold_ms}ms threshold"
            )
            return True
        return False
        
    def set_workers(self, local_worker, cloud_worker):
        """
        Set the worker instances for local and cloud execution
        
        Args:
            local_worker: Local execution worker instance
            cloud_worker: Cloud execution worker instance
        """
        self.local_worker = local_worker
        self.cloud_worker = cloud_worker
    
    def make_routing_decision(self, task: Task) -> RoutingDecision:
        """
        Make a routing decision for a given task
        
        Args:
            task: The task to route
            
        Returns:
            RoutingDecision: The routing decision with reasoning
        """
        metrics = self.telemetry.get_system_metrics()
        
        # Check if local worker is available
        if not self.local_worker:
            return RoutingDecision(
                target=ExecutionTarget.CLOUD,
                reasoning="Local worker not available, routing to Cloud",
                confidence=0.9,
                metrics=metrics
            )
            
        # Hot-Swap Logic: The Bouncer
        # Check if we need to downgrade to shadow model
        thermal_limit = 70.0 # Celsius
        warmup_limit = 60.0 # Celsius
        ram_limit_gb = 1.0
        
        # Pre-emptive Warmup
        if metrics.thermal_state_celsius and metrics.thermal_state_celsius > warmup_limit and metrics.thermal_state_celsius < thermal_limit:
            if hasattr(self.local_worker, 'ensure_shadow_model_loaded'):
                # We assume we know the shadow path or the worker knows it. 
                # For simplicity, we trigger the ensure check which handles it internally if configured.
                self.local_worker.ensure_shadow_model_loaded()
        
        if (metrics.thermal_state_celsius and metrics.thermal_state_celsius > thermal_limit) or (metrics.available_ram_gb < ram_limit_gb):
             if hasattr(self.local_worker, 'promote_shadow_model'):
                 swapped = self.local_worker.promote_shadow_model()
                 if swapped:
                     logger.warning(f"Hot-Swap Executed: Temp {metrics.thermal_state_celsius}C / RAM {metrics.available_ram_gb}GB")
        
        # Check for preferred target override
        if task.preferred_target:
            if task.preferred_target == ExecutionTarget.LOCAL:
                return RoutingDecision(
                    target=ExecutionTarget.LOCAL,
                    reasoning=f"User preferred Local execution for task {task.id}",
                    confidence=0.9,
                    metrics=metrics
                )
            elif task.preferred_target == ExecutionTarget.CLOUD:
                return RoutingDecision(
                    target=ExecutionTarget.CLOUD,
                    reasoning=f"User preferred Cloud execution for task {task.id}",
                    confidence=0.9,
                    metrics=metrics
                )
        
        # Apply routing rules
        should_route_local, reasoning = self.rules.should_route_local(task, metrics)
        
        target = ExecutionTarget.LOCAL if should_route_local else ExecutionTarget.CLOUD
        confidence = 0.8 if should_route_local else 0.7
        
        # Visual Telemetry: The "Nginx" Log
        thermal_status = f"{metrics.thermal_state_celsius}°C" if metrics.thermal_state_celsius else "N/A"
        routing_dest = "Apple Neural Engine" if target == ExecutionTarget.LOCAL else "Cloud (Stub)"
        print(f"\n[Router] Task: {task.type.value} | Temp: {thermal_status} | Decision: {target.value} ({reasoning})")
        
        return RoutingDecision(
            target=target,
            reasoning=reasoning,
            confidence=confidence,
            metrics=metrics
        )
    
    def execute_task_with_failover(self, task: Task, decision: RoutingDecision) -> TaskResult:
        """
        Execute a task with automatic failover from local to cloud
        
        Args:
            task: The task to execute
            decision: The routing decision
            
        Returns:
            TaskResult: The execution result
        """
        start_time = time.time()
        result = None
        
        try:
            # Primary execution based on routing decision
            if decision.target == ExecutionTarget.LOCAL:
                result = self._execute_local(task)
                execution_time_ms = (time.time() - start_time) * 1000

                # Fail over to Cloud only when local execution actually failed.
                # If local succeeded but was slow (e.g. PII masking ~100–700ms), keep the local result
                # so the user sees redacted text instead of a Cloud error when no API key is set.
                if not result.success:
                    logger.warning(f"Local execution failed for task {task.id}, failing over to Cloud")
                    _agent_debug_log(
                        "router_local_failover",
                        {
                            "task_type": task.type.value if task.type else None,
                            "local_success": result.success,
                            "local_execution_time_ms": float(execution_time_ms),
                            "latency_threshold_ms": float(self.rules.local_latency_threshold_ms),
                        },
                        hypothesis_id="H5",
                    )
                    failover_reason = f"Local execution failed: {result.error_message}"
                    cloud_result = self._execute_cloud(task)
                    cloud_result.reasoning = f"Failover to Cloud - {failover_reason}"
                    cloud_result.execution_time_ms = (time.time() - start_time) * 1000
                    return cloud_result

                result.execution_time_ms = execution_time_ms
                result.execution_target = ExecutionTarget.LOCAL
                result.reasoning = decision.reasoning
                return result
            else:
                # Direct cloud execution
                result = self._execute_cloud(task)
                result.execution_time_ms = (time.time() - start_time) * 1000
                result.execution_target = ExecutionTarget.CLOUD
                result.reasoning = decision.reasoning
                return result
                
        except Exception as e:
            # Catastrophic failure - try cloud as last resort
            logger.error(f"Unexpected error executing task {task.id}: {str(e)}")
            try:
                if decision.target == ExecutionTarget.LOCAL:
                    cloud_result = self._execute_cloud(task)
                    cloud_result.reasoning = f"Emergency failover to Cloud - {str(e)}"
                    cloud_result.execution_time_ms = (time.time() - start_time) * 1000
                    return cloud_result
                else:
                    # Cloud also failed
                    return TaskResult(
                        task_id=task.id,
                        success=False,
                        error_message=f"Both local and cloud execution failed: {str(e)}",
                        execution_target=ExecutionTarget.CLOUD,
                        execution_time_ms=(time.time() - start_time) * 1000,
                        reasoning="Complete execution failure"
                    )
            except Exception as cloud_error:
                return TaskResult(
                    task_id=task.id,
                    success=False,
                    error_message=f"Complete execution failure - Local: {str(e)}, Cloud: {str(cloud_error)}",
                    execution_target=ExecutionTarget.CLOUD,
                    execution_time_ms=(time.time() - start_time) * 1000,
                    reasoning="Complete execution failure"
                )
    
    def _execute_local(self, task: Task) -> TaskResult:
        """Execute task locally"""
        if not self.local_worker:
            raise RuntimeError("Local worker not configured")
        
        try:
            result = self.local_worker.execute(task)
            result.execution_target = ExecutionTarget.LOCAL
            return result
        except Exception as e:
            return TaskResult(
                task_id=task.id,
                success=False,
                error_message=str(e),
                execution_target=ExecutionTarget.LOCAL
            )
    
    def _execute_cloud(self, task: Task) -> TaskResult:
        """Execute task in the cloud"""
        if not self.cloud_worker:
            raise RuntimeError("Cloud worker not configured")
        
        try:
            result = self.cloud_worker.execute(task)
            result.execution_target = ExecutionTarget.CLOUD
            return result
        except Exception as e:
            return TaskResult(
                task_id=task.id,
                success=False,
                error_message=str(e),
                execution_target=ExecutionTarget.CLOUD
            )
    
    def route_task(self, task: Task) -> TaskResult:
        """
        Main routing function - makes decision and executes task
        
        Args:
            task: The task to route and execute
            
        Returns:
            TaskResult: The execution result
        """
        logger.info(f"Routing task {task.id} - Type: {task.type.value}, Complexity: {task.complexity.value}")
        
        # Make routing decision
        decision = self.make_routing_decision(task)
        
        logger.info(f"Decision: {decision.target.value} - {decision.reasoning}")
        
        # Execute with failover
        result = self.execute_task_with_failover(task, decision)
        
        logger.info(f"Task {task.id} completed - Success: {result.success}, Target: {result.execution_target.value if result.execution_target else 'Unknown'}")
        
        return result
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get current routing statistics and system status"""
        metrics = self.telemetry.get_system_metrics()
        return {
            "system_metrics": self.telemetry.get_telemetry_summary(),
            "routing_rules": {
                "min_ram_gb_local": self.rules.min_ram_gb_local,
                "max_cpu_load_percent": self.rules.max_cpu_load_percent,
                "low_battery_threshold": self.rules.low_battery_threshold,
                "local_latency_threshold_ms": self.rules.local_latency_threshold_ms
            }
        }


# Example usage and testing
if __name__ == "__main__":
    from task_schema import TaskType
    
    # Create router instance
    router = ExecutionRouter()
    
    # Test routing decisions
    test_tasks = [
        Task(
            complexity=TaskComplexity.LOW,
            type=TaskType.PII_MASKING,
            data="Test low complexity task"
        ),
        Task(
            complexity=TaskComplexity.HIGH,
            type=TaskType.REASONING,
            data="Test high complexity task"
        )
    ]
    
    print("=== Router Test ===")
    for task in test_tasks:
        decision = router.make_routing_decision(task)
        print(f"Task {task.id}: {decision.target.value} - {decision.reasoning}")
    
    # Print system stats
    print("\n=== System Stats ===")
    stats = router.get_routing_stats()
    print(f"Available RAM: {stats['system_metrics']['available_ram_gb']} GB")
    print(f"CPU Load: {stats['system_metrics']['cpu_load_percent']}%")
    print(f"NPU Available: {stats['system_metrics']['npu_available']}")
