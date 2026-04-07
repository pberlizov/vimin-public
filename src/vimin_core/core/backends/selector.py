"""
BackendSelector — maps (hardware, model descriptor) to the right execution backend.

Selection priority for generative (autoregressive) tasks:
  1. Apple Silicon + mlx-lm installed  →  MLXBackend        (ANE + unified memory)
  2. Any platform + llama-cpp-python   →  LlamaCppBackend   (GGUF, CPU/Metal/CUDA)
  3. Neither available                 →  None

Selection priority for ASR (SPEECH_TO_TEXT) tasks:
  1. Apple Silicon + mlx-whisper       →  WhisperBackend    (MLX, ANE-accelerated)
  2. Any platform + faster-whisper     →  FasterWhisperBackend (CTranslate2, CPU/CUDA)
  3. Neither available                 →  None
"""

from __future__ import annotations

import logging
import os
import platform
from typing import Optional

from vimin_core.core.backends.base import BaseBackend, ModelDescriptor
from vimin_core.core.backends.mlx_backend import MLXBackend
from vimin_core.core.backends.llamacpp_backend import LlamaCppBackend
from vimin_core.core.backends.whisper_backend import WhisperBackend
from vimin_core.core.backends.faster_whisper_backend import FasterWhisperBackend
from vimin_core.core.backends.openclaw_backend import OpenClawBackend

logger = logging.getLogger(__name__)

# Tasks that need a generative (autoregressive) backend
_GENERATIVE_TASKS = frozenset({
    "text-generation",
    "text2text-generation",
    "conversational",
    "summarization",
})

# Tasks handled by the Whisper ASR backend
_ASR_TASKS = frozenset({
    "automatic-speech-recognition",
    "audio-classification",
})

# Encoder-only tasks (embeddings, classification) — not currently supported
_ENCODER_TASKS = frozenset({
    "fill-mask",
    "ner",
    "token-classification",
    "feature-extraction",
    "text-classification",
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
        self._whisper = WhisperBackend()
        self._faster_whisper = FasterWhisperBackend()
        self._openclaw = OpenClawBackend()

    def needs_generative_backend(self, descriptor: ModelDescriptor) -> bool:
        """
        Return True if this descriptor requires a generative (autoregressive) backend.
        Returns False for ASR and encoder-only tasks.
        """
        if descriptor.task in _ENCODER_TASKS:
            return False
        if descriptor.task in _ASR_TASKS:
            return False
        if descriptor.format in ("onnx",):
            return False
        return descriptor.task in _GENERATIVE_TASKS or descriptor.format in ("mlx", "gguf")

    def is_asr_task(self, descriptor: ModelDescriptor) -> bool:
        return descriptor.task in _ASR_TASKS

    def select(self, descriptor: ModelDescriptor) -> Optional[BaseBackend]:
        """
        Return the best available backend for this descriptor, or None if no
        backend supports it.

        Never raises; logs the reason if no backend is available.
        """
        # ASR: mlx-whisper (Apple Silicon, ANE-accelerated) → faster-whisper (any platform)
        if self.is_asr_task(descriptor):
            if _is_apple_silicon() and self._whisper.is_available():
                logger.info("BackendSelector: ASR task, Apple Silicon → WhisperBackend (MLX)")
                return self._whisper
            if self._faster_whisper.is_available():
                logger.info("BackendSelector: ASR task → FasterWhisperBackend (CTranslate2)")
                return self._faster_whisper
            if not _is_apple_silicon() and self._whisper.is_available():
                # mlx-whisper on non-Apple is unusual but allow it
                logger.info("BackendSelector: ASR task → WhisperBackend (mlx-whisper)")
                return self._whisper
            logger.error(
                "BackendSelector: no ASR backend available.\n"
                "  Apple Silicon: pip install 'vimin-core[whisper]'\n"
                "  Other platforms: pip install faster-whisper"
            )
            return None

        if not self.needs_generative_backend(descriptor):
            return None  # encoder-only tasks not currently supported

        openclaw_requested = bool(os.environ.get("OPENCLAW_URL")) or descriptor.model_id == "openclaw"
        if openclaw_requested:
            if self._openclaw.is_available():
                logger.info("BackendSelector: OpenClaw requested → OpenClawBackend")
                return self._openclaw
            logger.warning(
                "BackendSelector: OpenClaw requested but gateway is unavailable; "
                "falling back to local backends."
            )

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
            "whisper_mlx": self._whisper.is_available(),
            "whisper_cpu": self._faster_whisper.is_available(),
            "openclaw": self._openclaw.is_available(),
        }
