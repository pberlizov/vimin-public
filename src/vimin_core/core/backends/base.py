"""
Backend abstraction for generative model execution.

Different hardware platforms require fundamentally different inference stacks.
This module defines the common interface all backends must implement, plus the
ModelDescriptor that describes a model in a hardware-format-agnostic way.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, List, Optional


class InsufficientMemoryError(RuntimeError):
    """Raised before a model load when available RAM is below the estimated requirement."""
    pass


@dataclass
class ModelDescriptor:
    """
    Unified descriptor for a generative model, hardware-format-agnostic.

    The backend uses this to decide how to download, convert, and load the
    model. Callers do not need to know which backend will be chosen.

    Attributes:
        model_id:          HuggingFace repo ID ("meta-llama/Llama-3.2-1B-Instruct")
                           or absolute local path to a weights directory / .gguf file.
        task:              HuggingFace pipeline task string. Controls backend routing.
                           Common values: "text-generation", "automatic-speech-recognition",
                           "fill-mask", "token-classification".
        path:              Optional local override. If set the backend loads from here
                           instead of downloading.
        format:            Hint for the backend: "mlx" | "gguf" | "onnx" | None (auto).
        quantization:      Requested quantization: "4bit" | "8bit" | "fp16" | "fp32" | None (auto).
        max_context:       Maximum token context length. Affects memory use.
        estimated_size_gb: If known, overrides the built-in heuristic for the
                           pre-load memory safety check.
    """
    model_id: str
    task: str = "text-generation"
    path: Optional[str] = None
    format: Optional[str] = None
    quantization: Optional[str] = None
    max_context: int = 2048
    estimated_size_gb: Optional[float] = None


class BaseBackend(ABC):
    """
    Abstract base class for all model execution backends.

    Subclasses handle a specific (platform, format) combination, e.g.
    MLXBackend handles Apple Silicon via mlx-lm, LlamaCppBackend handles
    everything else via GGUF/llama-cpp-python.

    The contract:
      - is_available()      → pure capability check, no I/O
      - estimate_memory_gb() → fast heuristic, no I/O
      - load()              → may download; raises InsufficientMemoryError before download
      - generate()          → blocking, returns full response string
      - stream_generate()   → yields tokens; same stop/temp semantics as generate()
      - unload()            → releases all model memory
    """

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this backend's dependencies are installed and hardware is present."""
        ...

    @abstractmethod
    def estimate_memory_gb(self, descriptor: ModelDescriptor) -> float:
        """
        Estimate peak RAM required to run this model.
        Used for the pre-load safety check — must be fast and require no I/O.
        """
        ...

    @abstractmethod
    def load(self, descriptor: ModelDescriptor) -> bool:
        """
        Download (if needed) and load the model into memory.

        Must raise InsufficientMemoryError before any download begins if
        estimate_memory_gb() exceeds available RAM.

        Returns True on success, False on failure (logs reason internally).
        """
        ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """
        Run synchronous generation.
        Returns the generated text only (prompt is NOT included in the output).
        """
        ...

    @abstractmethod
    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None,
    ) -> Iterator[str]:
        """
        Yield generated text fragments as they are produced.
        Each yielded string is a partial token or a small chunk of text.
        Stops when max_new_tokens is reached, EOS is hit, or a stop sequence appears.
        """
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release the model from memory and clean up any cached state."""
        ...

    @property
    def is_loaded(self) -> bool:
        """True if a model is currently in memory and ready to generate."""
        return False
