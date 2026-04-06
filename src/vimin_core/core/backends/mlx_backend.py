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
    # ------------------------------------------------------------------
    # SmolLM2  (HuggingFace — compact, fast edge models)
    # 4-bit variants are gated; public fp16 checkpoints used instead
    # ------------------------------------------------------------------
    "HuggingFaceTB/SmolLM2-360M-Instruct":           "mlx-community/SmolLM2-360M-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct":           "mlx-community/SmolLM2-1.7B-Instruct",

    # ------------------------------------------------------------------
    # SmolLM3  (HuggingFace — 3B successor with hybrid reasoning + 64K ctx)
    # ------------------------------------------------------------------
    "HuggingFaceTB/SmolLM3-3B":                      "mlx-community/SmolLM3-3B-4bit",

    # ------------------------------------------------------------------
    # Qwen3  (Alibaba — hybrid thinking/non-thinking, 100+ languages)
    # Dense: toggle chain-of-thought at inference time via enable_thinking
    # MoE:  30B-A3B activates only 3B params per token → 3B speed, 30B quality
    # ------------------------------------------------------------------
    "Qwen/Qwen3-0.6B":                               "mlx-community/Qwen3-0.6B-4bit",
    "Qwen/Qwen3-1.7B":                               "mlx-community/Qwen3-1.7B-4bit",
    "Qwen/Qwen3-4B":                                 "mlx-community/Qwen3-4B-4bit",
    "Qwen/Qwen3-8B":                                 "mlx-community/Qwen3-8B-4bit",
    "Qwen/Qwen3-14B":                                "mlx-community/Qwen3-14B-4bit",
    "Qwen/Qwen3-32B":                                "mlx-community/Qwen3-32B-4bit",
    # MoE: 17.2 GB at 4-bit, runs at 3B token speed on 24 GB+ Mac
    "Qwen/Qwen3-30B-A3B":                            "mlx-community/Qwen3-30B-A3B-4bit",

    # ------------------------------------------------------------------
    # Qwen 2.5  (Alibaba — strong multilingual + coding; previous gen)
    # ------------------------------------------------------------------
    "Qwen/Qwen2.5-0.5B-Instruct":                    "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    "Qwen/Qwen2.5-1.5B-Instruct":                    "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    "Qwen/Qwen2.5-3B-Instruct":                      "mlx-community/Qwen2.5-3B-Instruct-4bit",
    "Qwen/Qwen2.5-7B-Instruct":                      "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "Qwen/Qwen2.5-14B-Instruct":                     "mlx-community/Qwen2.5-14B-Instruct-4bit",
    # Qwen 2.5 Coder
    "Qwen/Qwen2.5-Coder-1.5B-Instruct":              "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
    "Qwen/Qwen2.5-Coder-7B-Instruct":                "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
    "Qwen/Qwen2.5-Coder-14B-Instruct":               "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit",

    # ------------------------------------------------------------------
    # DeepSeek-R1 Distill  (original distills — reasoning models from R1)
    # ------------------------------------------------------------------
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B":     "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B":       "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B":      "mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B":      "mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit",
    # R1-0528 Qwen3-8B distill — best open 8B reasoning model as of mid-2025
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B":         "mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit",

    # ------------------------------------------------------------------
    # Llama 3.2  (Meta — efficient small models)
    # ------------------------------------------------------------------
    "meta-llama/Llama-3.2-1B":                       "mlx-community/Llama-3.2-1B-4bit",
    "meta-llama/Llama-3.2-1B-Instruct":              "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "meta-llama/Llama-3.2-3B":                       "mlx-community/Llama-3.2-3B-4bit",
    "meta-llama/Llama-3.2-3B-Instruct":              "mlx-community/Llama-3.2-3B-Instruct-4bit",

    # ------------------------------------------------------------------
    # Llama 3.1  (Meta — strong 8B general purpose)
    # ------------------------------------------------------------------
    "meta-llama/Llama-3.1-8B":                       "mlx-community/Meta-Llama-3.1-8B-4bit",
    "meta-llama/Llama-3.1-8B-Instruct":              "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",

    # ------------------------------------------------------------------
    # Llama 3.3  (Meta — 70B, requires ~35 GB RAM)
    # ------------------------------------------------------------------
    "meta-llama/Llama-3.3-70B-Instruct":             "mlx-community/Llama-3.3-70B-Instruct-4bit",

    # ------------------------------------------------------------------
    # Mistral  (Mistral AI)
    # ------------------------------------------------------------------
    "mistralai/Mistral-7B-v0.3":                     "mlx-community/Mistral-7B-v0.3-4bit",
    "mistralai/Mistral-7B-Instruct-v0.3":            "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "mistralai/Mistral-Nemo-Instruct-2407":          "mlx-community/Mistral-Nemo-Instruct-2407-4bit",

    # ------------------------------------------------------------------
    # Gemma 2  (Google — strong open models)
    # ------------------------------------------------------------------
    "google/gemma-2-2b-it":                          "mlx-community/gemma-2-2b-it-4bit",
    "google/gemma-2-9b-it":                          "mlx-community/gemma-2-9b-it-4bit",
    "google/gemma-2-27b-it":                         "mlx-community/gemma-2-27b-it-4bit",

    # ------------------------------------------------------------------
    # Gemma 3  (Google — newest generation)
    # ------------------------------------------------------------------
    "google/gemma-3-1b-it":                          "mlx-community/gemma-3-1b-it-4bit",
    "google/gemma-3-4b-it":                          "mlx-community/gemma-3-4b-it-4bit",
    "google/gemma-3-12b-it":                         "mlx-community/gemma-3-12b-it-4bit",
    "google/gemma-3-27b-it":                         "mlx-community/gemma-3-27b-it-4bit",

    # ------------------------------------------------------------------
    # Phi  (Microsoft — compact, strong reasoning)
    # ------------------------------------------------------------------
    "microsoft/Phi-3.5-mini-instruct":               "mlx-community/Phi-3.5-mini-instruct-4bit",
    "microsoft/phi-4":                               "mlx-community/phi-4-4bit",
    # Phi-4-mini: 3.8B, 128K context, strong math — instruct and reasoning variants
    "microsoft/Phi-4-mini-instruct":                 "mlx-community/Phi-4-mini-instruct-4bit",
    "microsoft/Phi-4-mini-reasoning":                "mlx-community/Phi-4-mini-reasoning-4bit",
    # Phi-4-reasoning: 14B CoT reasoning — plus variant beats DeepSeek-R1-70B on AIME
    "microsoft/Phi-4-reasoning":                     "mlx-community/Phi-4-reasoning-4bit",
    "microsoft/Phi-4-reasoning-plus":                "mlx-community/Phi-4-reasoning-plus-4bit",

    # ------------------------------------------------------------------
    # Mistral Small 3.2  (Mistral AI — 24B, Apache 2.0)
    # ------------------------------------------------------------------
    "mistralai/Mistral-Small-3.2-24B-Instruct":      "mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit",

    # ------------------------------------------------------------------
    # Devstral Small  (Mistral + All Hands AI — #1 open SWE-Bench agent)
    # 24B, 128K context, agentic software engineering, Apache 2.0
    # ------------------------------------------------------------------
    "mistralai/Devstral-Small-2505":                 "mlx-community/Devstral-Small-2505-4bit",
}

# Rough fp16 size in GB, keyed on parameter-count token in the model_id.
# Sorted longest-first so "12b" matches before "1b", "27b" before "7b", etc.
# Used as a fallback when estimated_size_gb is not provided.
_SIZE_HINTS: list[tuple[str, float]] = sorted([
    ("360m", 0.7), ("0.5b", 1.0), ("0.6b", 1.2), ("1.5b", 3.0), ("1.7b", 3.5), ("1b", 2.0),
    ("2b", 4.0), ("3b", 6.0), ("4b", 8.0),
    ("70b", 140.0), ("7b", 14.0), ("8b", 16.0), ("9b", 18.0),
    ("12b", 24.0), ("13b", 26.0), ("14b", 28.0),
    ("24b", 48.0), ("27b", 54.0), ("30b", 60.0), ("32b", 64.0),
], key=lambda x: len(x[0]), reverse=True)

# Known fixed sizes for models whose names don't contain a clear Nb token.
_KNOWN_SIZES: dict[str, float] = {
    "phi-4-reasoning":            28.0,   # 14B fp16
    "phi-4-mini":                 7.6,    # 3.8B fp16
    "phi-4":                      28.0,   # 14B fp16
    "phi-3.5-mini":               7.6,    # 3.8B fp16
    "mistral-nemo":               24.0,   # 12B fp16
    "devstral":                   48.0,   # 24B fp16
    "mistral-small-3":            48.0,   # 24B fp16
    "smollm3":                    6.0,    # 3B fp16
    "smollm2-360m":               0.7,
    "smollm2-1.7b":               3.5,
    "qwen3-30b-a3b":              17.2,   # MoE: report actual 4-bit size, not fp16 total
}

# How much smaller each quantization is relative to fp16
_QUANT_SCALE: dict[str, float] = {
    "4bit": 0.25, "8bit": 0.5, "fp16": 1.0, "fp32": 2.0,
}


def _estimate_fp16_gb(model_id: str) -> float:
    name = model_id.lower()
    # Check known-size overrides first (models without a clear Nb token)
    for key, gb in _KNOWN_SIZES.items():
        if key in name:
            return gb
    # Then scan size tokens (longest-first to avoid "1b" shadowing "12b")
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

    def _apply_chat_template(self, prompt: str) -> str:
        """
        Wrap a plain-text prompt in the model's chat template if one is available.
        Falls back to the raw prompt string for base (non-instruct) models.
        """
        try:
            tok = self._tokenizer
            if hasattr(tok, "apply_chat_template") and tok.chat_template:
                messages = [{"role": "user", "content": prompt}]
                return tok.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
        except Exception:
            pass
        return prompt

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
        formatted = self._apply_chat_template(prompt)
        return mlx_lm.generate(
            self._model,
            self._tokenizer,
            prompt=formatted,
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
        formatted = self._apply_chat_template(prompt)

        accumulated = ""
        for response in mlx_lm.stream_generate(
            self._model,
            self._tokenizer,
            prompt=formatted,
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
            mx.clear_cache()
        except Exception:
            pass
        logger.info("MLXBackend: unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
