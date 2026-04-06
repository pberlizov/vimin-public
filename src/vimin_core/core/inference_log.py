"""
Inference Log Module

Provides per-inference observability with a thread-safe ring buffer.
Captures load_time_ms, inference_time_ms, provider_used, thermal_state,
and computes wall-clock vs compute-time delta.
"""

import time
import json
import threading
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class InferenceRecord:
    """Single inference observation record"""
    timestamp: float = field(default_factory=time.time)
    model_name: str = ""
    load_time_ms: float = 0.0
    inference_time_ms: float = 0.0
    wall_clock_ms: float = 0.0  # Total wall-clock time including pre/post processing
    provider_used: str = "unknown"
    thermal_state_celsius: Optional[float] = None
    memory_usage_mb: float = 0.0
    task_type: str = ""
    hot_swapped: bool = False  # True if this inference triggered a model swap
    
    @property
    def preprocessing_delta_ms(self) -> float:
        """
        Delta between wall-clock and pure compute time.
        Identifies bottlenecks in data pre-processing or post-processing.
        """
        return max(0.0, self.wall_clock_ms - self.inference_time_ms)
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['preprocessing_delta_ms'] = self.preprocessing_delta_ms
        return d


class InferenceLog:
    """
    Thread-safe ring buffer for inference records.
    Stores the last N inferences for observability.
    """
    
    def __init__(self, max_records: int = 100):
        self._max = max_records
        self._records: List[InferenceRecord] = []
        self._lock = threading.Lock()
    
    def record(self, rec: InferenceRecord) -> None:
        """Add an inference record (thread-safe)"""
        with self._lock:
            self._records.append(rec)
            if len(self._records) > self._max:
                self._records.pop(0)
    
    @property
    def records(self) -> List[InferenceRecord]:
        """Read-only snapshot of all records (thread-safe)."""
        with self._lock:
            return list(self._records)

    def get_records(self, last_n: Optional[int] = None) -> List[InferenceRecord]:
        """Get recent records (thread-safe)"""
        with self._lock:
            if last_n is None:
                return list(self._records)
            return list(self._records[-last_n:])
    
    def get_summary(self) -> Dict[str, Any]:
        """Get aggregate statistics from the log"""
        with self._lock:
            records = list(self._records)
        
        if not records:
            return {"total_inferences": 0}
        
        inference_times = [r.inference_time_ms for r in records]
        providers = {}
        for r in records:
            providers[r.provider_used] = providers.get(r.provider_used, 0) + 1
        
        hot_swaps = sum(1 for r in records if r.hot_swapped)
        
        return {
            "total_inferences": len(records),
            "avg_inference_ms": sum(inference_times) / len(inference_times),
            "min_inference_ms": min(inference_times),
            "max_inference_ms": max(inference_times),
            "avg_preprocessing_delta_ms": sum(r.preprocessing_delta_ms for r in records) / len(records),
            "provider_distribution": providers,
            "hot_swap_count": hot_swaps,
            "latest_thermal_celsius": records[-1].thermal_state_celsius,
        }
    
    def to_json(self, last_n: Optional[int] = None) -> str:
        """Export records as JSON"""
        records = self.get_records(last_n)
        return json.dumps([r.to_dict() for r in records], indent=2)
    
    def clear(self) -> None:
        """Clear all records"""
        with self._lock:
            self._records.clear()

