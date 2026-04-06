"""
Whisper Backend — speech-to-text via mlx-whisper on Apple Silicon.

Uses the mlx-community converted checkpoints which run natively on the ANE/GPU
via Apple's MLX framework. No PyTorch or ONNX runtime required.

Install:
    pip install mlx-whisper

Supported model IDs (pass to ModelDescriptor.model_id):
    openai/whisper-tiny     → mlx-community/whisper-tiny-mlx
    openai/whisper-base     → mlx-community/whisper-base-mlx
    openai/whisper-small    → mlx-community/whisper-small-mlx
    openai/whisper-medium   → mlx-community/whisper-medium-mlx
    openai/whisper-large-v3 → mlx-community/whisper-large-v3-mlx

Or pass the mlx-community ID directly.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

from vimin_core.core.backends.base import BaseBackend, ModelDescriptor

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore
    _NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)

_WHISPER_ALIASES: dict[str, str] = {
    "openai/whisper-tiny":          "mlx-community/whisper-tiny-mlx",
    "openai/whisper-base":          "mlx-community/whisper-base-mlx",
    "openai/whisper-small":         "mlx-community/whisper-small-mlx",
    "openai/whisper-medium":        "mlx-community/whisper-medium-mlx",
    "openai/whisper-large-v3":      "mlx-community/whisper-large-v3-mlx",
    # Turbo: ~50% faster than large-v3 with minimal quality loss
    "openai/whisper-large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}

# Approximate VRAM/RAM usage per model (fp16 on ANE)
_WHISPER_SIZE_GB: dict[str, float] = {
    "tiny":   0.15,
    "base":   0.3,
    "small":  0.6,
    "medium": 1.5,
    "turbo":  1.6,
    "large":  3.0,
}


def _resolve_whisper_id(descriptor: ModelDescriptor) -> str:
    if descriptor.path:
        return descriptor.path
    return _WHISPER_ALIASES.get(descriptor.model_id, descriptor.model_id)


def _size_from_id(model_id: str) -> float:
    name = model_id.lower()
    for key, gb in _WHISPER_SIZE_GB.items():
        if key in name:
            return gb
    return 0.6  # default to base-ish


class WhisperBackend(BaseBackend):
    """
    ASR backend powered by mlx-whisper.

    Accepts audio as a numpy float32 array at 16 kHz, or a file path string.
    Exposes transcribe() rather than generate() — use the backend directly
    for ASR tasks rather than going through the generative pipeline.
    """

    def __init__(self) -> None:
        self._loaded_id: Optional[str] = None

    def is_available(self) -> bool:
        try:
            import mlx_whisper  # noqa: F401
            return True
        except ImportError:
            return False

    def estimate_memory_gb(self, descriptor: ModelDescriptor) -> float:
        if descriptor.estimated_size_gb is not None:
            return descriptor.estimated_size_gb
        return _size_from_id(descriptor.model_id)

    def load(self, descriptor: ModelDescriptor) -> bool:
        """
        For WhisperBackend, 'loading' just validates that the model checkpoint
        is reachable. mlx-whisper caches the model internally between calls.
        """
        if not self.is_available():
            logger.error(
                "WhisperBackend: mlx-whisper not installed.\n"
                "  Install with: pip install mlx-whisper"
            )
            return False
        self._loaded_id = _resolve_whisper_id(descriptor)
        logger.info(f"WhisperBackend: ready — {self._loaded_id}")
        return True

    def transcribe(
        self,
        audio: Union[str, "np.ndarray"],
        language: Optional[str] = None,
        task: str = "transcribe",
    ) -> dict:
        """
        Transcribe audio to text.

        Args:
            audio:    Path to an audio file, or a numpy float32 array at 16 kHz.
            language: BCP-47 language code (e.g. "en"). None = auto-detect.
            task:     "transcribe" or "translate" (translate → English).

        Returns:
            dict with keys:
              text      — full transcription string
              segments  — list of {start, end, text} dicts
              language  — detected or specified language code
        """
        if not self.is_loaded:
            raise RuntimeError("WhisperBackend.transcribe(): no model loaded — call load() first")
        import mlx_whisper
        kwargs = {"path_or_hf_repo": self._loaded_id, "task": task}
        if language:
            kwargs["language"] = language
        return mlx_whisper.transcribe(audio, **kwargs)

    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError(
            "WhisperBackend does not support text generation. Use transcribe() instead."
        )

    def stream_generate(self, prompt: str, **kwargs):
        raise NotImplementedError(
            "WhisperBackend does not support streaming generation. Use transcribe() instead."
        )

    def unload(self) -> None:
        self._loaded_id = None
        logger.info("WhisperBackend: unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._loaded_id is not None
