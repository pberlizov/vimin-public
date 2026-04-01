"""
MLX Backend — Apple Silicon (M-series) via mlx and mlx-lm.

MLX uses Apple's unified memory architecture: model weights live in the same
physical memory pool as CPU RAM, and the ANE / GPU operate on them in-place.
There is no explicit "upload to device" step and no risk of holding two
copies in memory simultaneously.

Install:
    pip install mlx mlx-lm

Supported models: any checkpoint that mlx-lm can load, including the
pre-quantized 4-bit checkpoints published by the mlx-community org on HF.
We prefer 4-bit variants to keep RAM pressure low on user machines.
"""

from __future__ import annotations

import logging
from typing import Iterator, List, Optional

from vimin_core.core.backends.base import BaseBackend, InsufficientMemoryError, ModelDescriptor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# mlx-community pre-quantized aliases
# When the user passes a raw Meta/Google/Mistral model_id we silently redirect
# to the mlx-community 4-bit checkpoint so no manual conversion is needed.
# All aliases verified public (HTTP 200) as of 2026-03.
# ---------------------------------------------------------------------------
_MLX_COMMUNITY_ALIASES: dict[str, str] = {
    # Llama 3.2
    "meta-llama/Llama-3.2-1B":                   "mlx-community/Llama-3.2-1B-4bit",
    "meta-llama/Llama-3.2-1B-Instruct":          "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "meta-llama/Llama-3.2-3B":                   "mlx-community/Llama-3.2-3B-4bit",
    "meta-llama/Llama-3.2-3B-Instruct":          "mlx-community/Llama-3.2-3B-Instruct-4bit",
    # Llama 3.1
    "meta-llama/Llama-3.1-8B":                   "mlx-community/Meta-Llama-3.1-8B-4bit",
    "meta-llama/Llama-3.1-8B-Instruct":          "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    # Mistral
    "mistralai/Mistral-7B-Instruct-v0.3":        "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "mistralai/Mistral-7B-v0.3":                 "mlx-community/Mistral-7B-v0.3-4bit",
    # Gemma 2
    "google/gemma-2-2b-it":                      "mlx-community/gemma-2-2b-it-4bit",
    "google/gemma-2-9b-it":                      "mlx-community/gemma-2-9b-it-4bit",
    # Phi-3.5
    "microsoft/Phi-3.5-mini-instruct":           "mlx-community/Phi-3.5-mini-instruct-4bit",
    # Qwen 2.5
    "Qwen/Qwen2.5-0.5B-Instruct":               "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    "Qwen/Qwen2.5-1.5B-Instruct":               "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    "Qwen/Qwen2.5-7B-Instruct":                 "mlx-community/Qwen2.5-7B-Instruct-4bit",
    # SmolLM2 — 4-bit variants are gated; use public fp16 checkpoints instead
    "HuggingFaceTB/SmolLM2-360M-Instruct":      "mlx-community/SmolLM2-360M-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct":      "mlx-community/SmolLM2-1.7B-Instruct",
}

# Rough fp16 size in GB, keyed on parameter-count token in the model_id.
# Used as a fallback when estimated_size_gb is not provided.
_SIZE_HINTS: list[tuple[str, float]] = [
    ("360m", 0.7), ("0.5b", 1.0), ("1b", 2.0), ("1.5b", 3.0), ("1.7b", 3.5),
    ("2b", 4.0), ("3b", 6.0), ("4b", 8.0),
    ("7b", 14.0), ("8b", 16.0), ("9b", 18.0),
    ("13b", 26.0), ("70b", 140.0),
]

# How much smaller each quantization is relative to fp16
_QUANT_SCALE: dict[str, float] = {
    "4bit": 0.25, "8bit": 0.5, "fp16": 1.0, "fp32": 2.0,
}


def _estimate_fp16_gb(model_id: str) -> float:
    name = model_id.lower()
    for token, gb in _SIZE_HINTS:
        if token in name:
            return gb
    return 2.0  # safe default


def _resolve_id(descriptor: ModelDescriptor) -> str:
    """
    Return the actual model identifier to pass to mlx_lm.load().
    Priority: descriptor.path > mlx-community alias > original model_id.
    """
    if descriptor.path:
        return descriptor.path
    return _MLX_COMMUNITY_ALIASES.get(descriptor.model_id, descriptor.model_id)


def _cached_local_path(model_id: str) -> Optional[str]:
    """
    Return the local HF cache path for model_id if it is already fully
    downloaded, or None if a network fetch is needed.

    Using a local path bypasses all HF hub freshness-check round-trips,
    cutting cold-start from ~90-600 s (network) to ~2-5 s (disk).
    """
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(model_id, local_files_only=True)
    except Exception:
        return None


class MLXBackend(BaseBackend):
    """
    Generative inference on Apple Silicon via mlx-lm.

    Handles any model that mlx-lm supports, defaulting to 4-bit quantized
    checkpoints from the mlx-community org to minimise RAM usage.
    """

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._loaded_id: Optional[str] = None

    # ------------------------------------------------------------------
    # BaseBackend interface
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        try:
            import mlx.core   # noqa: F401
            import mlx_lm     # noqa: F401
            return True
        except ImportError:
            return False

    def estimate_memory_gb(self, descriptor: ModelDescriptor) -> float:
        if descriptor.estimated_size_gb is not None:
            return descriptor.estimated_size_gb
        fp16_gb = _estimate_fp16_gb(descriptor.model_id)
        quant = descriptor.quantization or "4bit"
        return fp16_gb * _QUANT_SCALE.get(quant, 0.25)

    def load(self, descriptor: ModelDescriptor) -> bool:
        if not self.is_available():
            logger.error(
                "MLXBackend: mlx / mlx-lm not installed.\n"
                "  Install with: pip install mlx mlx-lm"
            )
            return False

        import psutil
        import mlx_lm

        # --- Memory safety check (before any download) ---
        needed_gb = self.estimate_memory_gb(descriptor)
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        headroom_gb = 1.0  # reserve for OS and other apps
        if available_gb < needed_gb + headroom_gb:
            raise InsufficientMemoryError(
                f"'{descriptor.model_id}' needs ~{needed_gb:.1f} GB; "
                f"only {available_gb:.1f} GB available. "
                f"Try a smaller model or close other applications."
            )

        resolved = _resolve_id(descriptor)

        # --- Cache-first load: skip all HF network calls when already downloaded ---
        local_path = _cached_local_path(resolved)
        if local_path:
            load_target = local_path
            logger.info(f"MLXBackend: loading '{resolved}' from local cache…")
        else:
            load_target = resolved
            logger.info(f"MLXBackend: downloading '{resolved}'…")

        # --- Lower OS scheduling priority before model load (heavy GPU work) ---
        try:
            from vimin_core.core.priority import set_inference_priority
            set_inference_priority()
        except Exception:
            pass

        try:
            self._model, self._tokenizer = mlx_lm.load(load_target)
            self._loaded_id = resolved
            logger.info(f"MLXBackend: ready — {resolved}")
            return True
        except Exception as exc:
            logger.error(f"MLXBackend: load failed for '{resolved}': {exc}")
            self._model = None
            self._tokenizer = None
            self._loaded_id = None
            return False

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        if not self.is_loaded:
            raise RuntimeError("MLXBackend.generate(): no model loaded")
        import mlx_lm
        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(temp=temperature)
        return mlx_lm.generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_new_tokens,
            sampler=sampler,
            verbose=False,
        )

    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None,
    ) -> Iterator[str]:
        if not self.is_loaded:
            raise RuntimeError("MLXBackend.stream_generate(): no model loaded")
        import mlx_lm
        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(temp=temperature)

        accumulated = ""
        for response in mlx_lm.stream_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_new_tokens,
            sampler=sampler,
        ):
            token_text = response.text if hasattr(response, "text") else str(response)
            accumulated += token_text

            if stop_sequences:
                for seq in stop_sequences:
                    if seq in accumulated:
                        before = accumulated.split(seq)[0]
                        new_part = before[len(accumulated) - len(token_text):]
                        if new_part:
                            yield new_part
                        return
                yield token_text
            else:
                yield token_text

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._loaded_id = None
        try:
            import mlx.core as mx
            mx.metal.clear_cache()
        except Exception:
            pass
        logger.info("MLXBackend: unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
