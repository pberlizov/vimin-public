"""
BackendSelector — maps (hardware, model descriptor) to the right execution backend.

Selection priority for generative (autoregressive) tasks:
  1. Apple Silicon + mlx-lm installed  →  MLXBackend   (ANE + unified memory)
  2. Any platform + llama-cpp-python   →  LlamaCppBackend (GGUF, CPU/Metal/CUDA)
  3. Neither available                 →  None (caller should surface install instructions)

Encoder / audio tasks (Whisper, BERT) return None unconditionally — they are
handled by the existing ONNX pipeline and do not need a generative backend.
"""

from __future__ import annotations

import logging
import platform
from typing import Optional

from vimin_core.core.backends.base import BaseBackend, ModelDescriptor
from vimin_core.core.backends.mlx_backend import MLXBackend
from vimin_core.core.backends.llamacpp_backend import LlamaCppBackend

logger = logging.getLogger(__name__)

# Tasks that need a generative (autoregressive) backend
_GENERATIVE_TASKS = frozenset({
    "text-generation",
    "text2text-generation",
    "conversational",
    "summarization",
})

# Tasks that the existing ONNX pipeline handles (no generative backend needed)
_ENCODER_TASKS = frozenset({
    "fill-mask",
    "ner",
    "token-classification",
    "feature-extraction",
    "text-classification",
    "automatic-speech-recognition",
    "audio-classification",
})


def _is_apple_silicon() -> bool:
    return (
        platform.system() == "Darwin"
        and platform.machine().lower() in ("arm64", "arm")
    )


class BackendSelector:
    """
    Stateless helper that inspects available backends and hardware to choose
    the best execution path for a given ModelDescriptor.

    Backends are instantiated lazily and cached so availability checks are only
    performed once per process lifetime.
    """

    def __init__(self) -> None:
        self._mlx = MLXBackend()
        self._llamacpp = LlamaCppBackend()

    def needs_generative_backend(self, descriptor: ModelDescriptor) -> bool:
        """
        Return True if this descriptor requires a generative backend.
        Returns False for encoder/audio tasks that the ONNX pipeline already handles.
        """
        if descriptor.task in _ENCODER_TASKS:
            return False
        if descriptor.format in ("onnx",):
            return False
        return descriptor.task in _GENERATIVE_TASKS or descriptor.format in ("mlx", "gguf")

    def select(self, descriptor: ModelDescriptor) -> Optional[BaseBackend]:
        """
        Return the best available backend for this descriptor, or None if the
        task should fall through to the existing ONNX pipeline.

        Never raises; logs the reason if no backend is available.
        """
        if not self.needs_generative_backend(descriptor):
            return None  # ONNX pipeline handles this

        apple_silicon = _is_apple_silicon()

        # Priority 1: Apple Silicon + MLX (native ANE, unified memory, no OOM risk)
        if apple_silicon and self._mlx.is_available():
            logger.info(
                f"BackendSelector: Apple Silicon detected, mlx-lm available → MLXBackend"
            )
            return self._mlx

        # Priority 2: llama-cpp-python (any platform, CPU/Metal/CUDA)
        if self._llamacpp.is_available():
            logger.info(
                f"BackendSelector: llama-cpp-python available → LlamaCppBackend"
            )
            return self._llamacpp

        # Neither available
        logger.error(
            "BackendSelector: no generative backend available for "
            f"'{descriptor.model_id}'.\n"
            f"  {self.install_instructions(descriptor)}"
        )
        return None

    def install_instructions(self, descriptor: ModelDescriptor) -> str:
        """Return a human-readable install command for the best backend on this machine."""
        if _is_apple_silicon():
            return (
                "For Apple Silicon (recommended):\n"
                "  pip install mlx mlx-lm\n\n"
                "Alternative (any platform):\n"
                "  pip install llama-cpp-python"
            )
        return (
            "CPU only:\n"
            "  pip install llama-cpp-python\n\n"
            "macOS Metal:\n"
            '  CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --no-cache-dir\n\n'
            "NVIDIA CUDA:\n"
            '  CMAKE_ARGS="-DLLAMA_CUDA=on"  pip install llama-cpp-python --no-cache-dir'
        )

    def recommend_quantization(self, model_id: str, available_ram_gb: float) -> str:
        """
        Return the highest-quality quantization that fits in available_ram_gb.

        Priority (best → smallest): fp16 → 8bit → 4bit.
        Raises InsufficientMemoryError if the model won't fit even at 4-bit.

        Args:
            model_id:          HuggingFace model ID (used to estimate fp16 size).
            available_ram_gb:  Current free RAM in GB.

        Returns:
            One of "fp16", "8bit", "4bit".
        """
        from vimin_core.core.backends.mlx_backend import _estimate_fp16_gb
        from vimin_core.core.backends.base import InsufficientMemoryError

        fp16_gb = _estimate_fp16_gb(model_id)
        headroom_gb = 1.0  # keep 1 GB free for OS + other apps

        candidates = [
            ("fp16", 1.0),
            ("8bit", 0.5),
            ("4bit", 0.25),
        ]
        for quant, scale in candidates:
            needed = fp16_gb * scale + headroom_gb
            if available_ram_gb >= needed:
                logger.info(
                    f"Quantization recommendation for '{model_id}': {quant} "
                    f"(needs {needed:.1f} GB, {available_ram_gb:.1f} GB available)"
                )
                return quant

        raise InsufficientMemoryError(
            f"'{model_id}' (fp16 ≈ {fp16_gb:.1f} GB) won't fit in "
            f"{available_ram_gb:.1f} GB RAM even at 4-bit quantization. "
            f"Try a smaller model (e.g. Llama-3.2-1B or SmolLM2-360M)."
        )

    def available_backends(self) -> dict[str, bool]:
        """Report which backends are installed (useful for dashboard / health check)."""
        return {
            "mlx": self._mlx.is_available(),
            "llamacpp": self._llamacpp.is_available(),
        }
