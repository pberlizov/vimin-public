"""
FasterWhisperBackend — cross-platform speech-to-text via faster-whisper.

Uses CTranslate2 under the hood; runs on CPU, NVIDIA CUDA, and AMD ROCm.
Works on Linux, Windows, and macOS (Intel or Apple Silicon when mlx-whisper
is not installed).

Install:
    pip install faster-whisper

Supported model IDs (same as WhisperBackend — pass the openai/ canonical ID):
    openai/whisper-tiny
    openai/whisper-base
    openai/whisper-small
    openai/whisper-medium
    openai/whisper-large-v3
    openai/whisper-large-v3-turbo

Or pass a Systran HuggingFace ID directly:
    Systran/faster-whisper-small
"""

from __future__ import annotations

import logging
from typing import Optional, Union

from vimin_core.core.backends.base import BaseBackend, ModelDescriptor

logger = logging.getLogger(__name__)

# Map openai/ canonical IDs → faster-whisper size names
_FW_ALIASES: dict[str, str] = {
    "openai/whisper-tiny":           "tiny",
    "openai/whisper-base":           "base",
    "openai/whisper-small":          "small",
    "openai/whisper-medium":         "medium",
    "openai/whisper-large-v2":       "large-v2",
    "openai/whisper-large-v3":       "large-v3",
    "openai/whisper-large-v3-turbo": "large-v3-turbo",
}

_WHISPER_SIZE_GB: dict[str, float] = {
    "tiny":          0.15,
    "base":          0.3,
    "small":         0.6,
    "medium":        1.5,
    "large-v2":      3.0,
    "large-v3":      3.0,
    "large-v3-turbo": 1.6,
}


def _resolve_fw_id(descriptor: ModelDescriptor) -> str:
    if descriptor.path:
        return descriptor.path
    return _FW_ALIASES.get(descriptor.model_id, descriptor.model_id)


def _size_from_id(model_id: str) -> float:
    name = model_id.lower()
    for key, gb in _WHISPER_SIZE_GB.items():
        if key in name:
            return gb
    return 0.6


class FasterWhisperBackend(BaseBackend):
    """
    ASR backend powered by faster-whisper (CTranslate2).

    Cross-platform fallback for SPEECH_TO_TEXT when mlx-whisper is not
    available. Accepts a file path string or a numpy float32 array at 16 kHz.

    Device selection: CUDA if available, otherwise CPU.
    Compute type: int8 on CPU (fastest/smallest), float16 on CUDA.
    """

    def __init__(self) -> None:
        self._model = None
        self._loaded_id: Optional[str] = None

    def is_available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
            return True
        except ImportError:
            return False

    def estimate_memory_gb(self, descriptor: ModelDescriptor) -> float:
        if descriptor.estimated_size_gb is not None:
            return descriptor.estimated_size_gb
        return _size_from_id(descriptor.model_id)

    def load(self, descriptor: ModelDescriptor) -> bool:
        if not self.is_available():
            logger.error(
                "FasterWhisperBackend: faster-whisper not installed.\n"
                "  Install with: pip install faster-whisper"
            )
            return False

        model_id = _resolve_fw_id(descriptor)
        try:
            from faster_whisper import WhisperModel
            device, compute_type = self._best_device()
            logger.info(
                f"FasterWhisperBackend: loading '{model_id}' "
                f"on {device} ({compute_type})"
            )
            self._model = WhisperModel(model_id, device=device, compute_type=compute_type)
            self._loaded_id = model_id
            logger.info(f"FasterWhisperBackend: ready — {model_id}")
            return True
        except Exception as e:
            logger.error(f"FasterWhisperBackend: failed to load '{model_id}': {e}")
            return False

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
            raise RuntimeError(
                "FasterWhisperBackend.transcribe(): no model loaded — call load() first"
            )
        kwargs: dict = {"task": task}
        if language:
            kwargs["language"] = language

        segments, info = self._model.transcribe(audio, **kwargs)
        segment_list = [
            {"start": s.start, "end": s.end, "text": s.text}
            for s in segments
        ]
        return {
            "text": " ".join(s["text"].strip() for s in segment_list),
            "segments": segment_list,
            "language": info.language,
        }

    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError(
            "FasterWhisperBackend does not support text generation. "
            "Use transcribe() instead."
        )

    def stream_generate(self, prompt: str, **kwargs):
        raise NotImplementedError(
            "FasterWhisperBackend does not support streaming generation. "
            "Use transcribe() instead."
        )

    def unload(self) -> None:
        self._model = None
        self._loaded_id = None
        logger.info("FasterWhisperBackend: unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @staticmethod
    def _best_device() -> tuple[str, str]:
        """Return (device, compute_type) for the current machine."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda", "float16"
        except ImportError:
            pass
        return "cpu", "int8"
