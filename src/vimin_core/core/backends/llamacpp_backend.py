"""
LlamaCpp Backend — CPU / CUDA / Metal via llama-cpp-python.

Works on Linux, Windows, and macOS (including Intel Macs and any machine
without Apple Silicon). Uses GGUF quantized models — a Q4_K_M 7B model
runs in ~4 GB RAM. GPU offload (Metal or CUDA) is enabled automatically
when the binary was compiled with the appropriate flags.

Install — choose the right variant for your hardware:
    CPU only:
        pip install llama-cpp-python

    macOS Metal (Intel or Apple Silicon without MLX):
        CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --no-cache-dir

    NVIDIA CUDA:
        CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --no-cache-dir

    Windows DirectML:
        CMAKE_ARGS="-DLLAMA_CLBLAST=on" pip install llama-cpp-python --no-cache-dir

Pre-built wheels are also available at:
    https://github.com/abetlen/llama-cpp-python/releases
"""

from __future__ import annotations

import logging
import os
import platform
from typing import Iterator, List, Optional

from vimin_core.core.backends.base import BaseBackend, InsufficientMemoryError, ModelDescriptor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GGUF download registry
# Maps HuggingFace model_id → (repo_id, filename) on the Bartowski GGUF hub.
# Bartowski's repos are consistently maintained and well-quantized (Q4_K_M).
# ---------------------------------------------------------------------------
_GGUF_SOURCES: dict[str, tuple[str, str]] = {
    "meta-llama/Llama-3.2-1B":                  ("bartowski/Llama-3.2-1B-GGUF",
                                                   "Llama-3.2-1B-Q4_K_M.gguf"),
    "meta-llama/Llama-3.2-1B-Instruct":         ("bartowski/Llama-3.2-1B-Instruct-GGUF",
                                                   "Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
    "meta-llama/Llama-3.2-3B":                  ("bartowski/Llama-3.2-3B-GGUF",
                                                   "Llama-3.2-3B-Q4_K_M.gguf"),
    "meta-llama/Llama-3.2-3B-Instruct":         ("bartowski/Llama-3.2-3B-Instruct-GGUF",
                                                   "Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
    "meta-llama/Llama-3.1-8B":                  ("bartowski/Meta-Llama-3.1-8B-GGUF",
                                                   "Meta-Llama-3.1-8B-Q4_K_M.gguf"),
    "meta-llama/Llama-3.1-8B-Instruct":         ("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
                                                   "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"),
    "mistralai/Mistral-7B-Instruct-v0.3":       ("bartowski/Mistral-7B-Instruct-v0.3-GGUF",
                                                   "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"),
    "microsoft/Phi-3.5-mini-instruct":          ("bartowski/Phi-3.5-mini-instruct-GGUF",
                                                   "Phi-3.5-mini-instruct-Q4_K_M.gguf"),
    "Qwen/Qwen2.5-1.5B-Instruct":              ("bartowski/Qwen2.5-1.5B-Instruct-GGUF",
                                                   "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"),
    "Qwen/Qwen2.5-7B-Instruct":                ("bartowski/Qwen2.5-7B-Instruct-GGUF",
                                                   "Qwen2.5-7B-Instruct-Q4_K_M.gguf"),
    "HuggingFaceTB/SmolLM2-360M-Instruct":     ("bartowski/SmolLM2-360M-Instruct-GGUF",
                                                   "SmolLM2-360M-Instruct-Q4_K_M.gguf"),
    "HuggingFaceTB/SmolLM2-1.7B-Instruct":     ("bartowski/SmolLM2-1.7B-Instruct-GGUF",
                                                   "SmolLM2-1.7B-Instruct-Q4_K_M.gguf"),
}

_SIZE_HINTS: list[tuple[str, float]] = sorted([
    ("360m", 0.7), ("1b", 2.0), ("1.5b", 3.0), ("1.7b", 3.5),
    ("2b", 4.0), ("3b", 6.0), ("7b", 14.0), ("8b", 16.0),
    ("13b", 26.0), ("70b", 140.0),
], key=lambda x: len(x[0]), reverse=True)

_QUANT_SCALE: dict[str, float] = {
    "q4_k_m": 0.25, "q4_0": 0.22, "q8_0": 0.5,
    "4bit": 0.25, "8bit": 0.5, "f16": 1.0, "fp16": 1.0,
}


def _estimate_fp16_gb(model_id: str) -> float:
    name = model_id.lower()
    for token, gb in _SIZE_HINTS:
        if token in name:
            return gb
    return 2.0


class LlamaCppBackend(BaseBackend):
    """
    Generative inference via llama-cpp-python (GGUF format).

    Automatically downloads Q4_K_M GGUF checkpoints from HuggingFace for
    registered models. Pass descriptor.path to use a local .gguf file.
    GPU layer offload (Metal / CUDA) is requested via n_gpu_layers=-1;
    llama-cpp-python silently ignores this on CPU-only builds.
    """

    def __init__(self) -> None:
        self._llm = None
        self._loaded_path: Optional[str] = None

    # ------------------------------------------------------------------
    # BaseBackend interface
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        try:
            import llama_cpp  # noqa: F401
            return True
        except ImportError:
            return False

    def estimate_memory_gb(self, descriptor: ModelDescriptor) -> float:
        if descriptor.estimated_size_gb is not None:
            return descriptor.estimated_size_gb
        fp16_gb = _estimate_fp16_gb(descriptor.model_id)
        quant = (descriptor.quantization or "q4_k_m").lower()
        return fp16_gb * _QUANT_SCALE.get(quant, 0.25)

    def _resolve_gguf_path(self, descriptor: ModelDescriptor) -> Optional[str]:
        """
        Return path to a local .gguf file. Downloads from HuggingFace if
        not already present. Returns None if resolution fails.
        """
        # 1. Explicit local path
        if descriptor.path and os.path.isfile(descriptor.path):
            return descriptor.path

        # 2. Look up known registry
        source = _GGUF_SOURCES.get(descriptor.model_id)
        if not source:
            logger.error(
                f"LlamaCppBackend: no GGUF source registered for '{descriptor.model_id}'. "
                "Set descriptor.path to a local .gguf file, or add the model to _GGUF_SOURCES."
            )
            return None

        repo_id, filename = source
        try:
            from huggingface_hub import hf_hub_download
            logger.info(f"LlamaCppBackend: downloading {filename} from {repo_id}…")
            local_path = hf_hub_download(repo_id=repo_id, filename=filename)
            logger.info(f"LlamaCppBackend: downloaded to {local_path}")
            return local_path
        except Exception as exc:
            logger.error(f"LlamaCppBackend: download failed — {exc}")
            return None

    def load(self, descriptor: ModelDescriptor) -> bool:
        if not self.is_available():
            logger.error(
                "LlamaCppBackend: llama-cpp-python not installed.\n"
                "  CPU:    pip install llama-cpp-python\n"
                "  Metal:  CMAKE_ARGS=\"-DLLAMA_METAL=on\" pip install llama-cpp-python --no-cache-dir\n"
                "  CUDA:   CMAKE_ARGS=\"-DLLAMA_CUDA=on\"  pip install llama-cpp-python --no-cache-dir"
            )
            return False

        import psutil
        from llama_cpp import Llama

        # --- Memory safety check ---
        needed_gb = self.estimate_memory_gb(descriptor)
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        if available_gb < needed_gb + 0.5:
            raise InsufficientMemoryError(
                f"'{descriptor.model_id}' needs ~{needed_gb:.1f} GB; "
                f"only {available_gb:.1f} GB available."
            )

        gguf_path = self._resolve_gguf_path(descriptor)
        if gguf_path is None:
            return False

        # Request all layers on GPU; llama-cpp silently falls back to CPU
        # if the binary was not compiled with GPU support.
        n_gpu_layers = -1

        logger.info(
            f"LlamaCppBackend: loading '{gguf_path}' "
            f"(n_gpu_layers={n_gpu_layers}, ctx={descriptor.max_context})…"
        )
        try:
            self._llm = Llama(
                model_path=gguf_path,
                n_ctx=descriptor.max_context,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )
            self._loaded_path = gguf_path
            logger.info("LlamaCppBackend: ready")
            return True
        except Exception as exc:
            logger.error(f"LlamaCppBackend: load failed — {exc}")
            self._llm = None
            self._loaded_path = None
            return False

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        if not self.is_loaded:
            raise RuntimeError("LlamaCppBackend.generate(): no model loaded")
        response = self._llm(
            prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            stop=stop_sequences or [],
            echo=False,
        )
        return response["choices"][0]["text"]

    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None,
    ) -> Iterator[str]:
        if not self.is_loaded:
            raise RuntimeError("LlamaCppBackend.stream_generate(): no model loaded")
        stream = self._llm(
            prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            stop=stop_sequences or [],
            echo=False,
            stream=True,
        )
        for chunk in stream:
            yield chunk["choices"][0]["text"]

    def unload(self) -> None:
        self._llm = None
        self._loaded_path = None
        logger.info("LlamaCppBackend: unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None
