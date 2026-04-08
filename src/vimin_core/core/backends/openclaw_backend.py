"""
OpenClaw Backend — local OpenClaw Gateway as an inference engine.

If a node has OpenClaw installed and running, vimin-core can route broadcast
tasks to it instead of (or alongside) MLX or llama-cpp. OpenClaw handles
model management, quantisation, and hardware selection internally — vimin-core
just sends a prompt and receives a completion.

This backend covers the broadcast dispatch use case only. Agent-to-agent
coordination, department lead nodes, fleet pipelines, and multi-turn
orchestration are available in the full vimin distribution.

Prerequisites:
    • OpenClaw daemon must be running on this machine
      (default: http://127.0.0.1:18789)
    • Token is read automatically from ~/.openclaw/openclaw.json
      or set via OPENCLAW_TOKEN environment variable

Usage — start an agent that uses OpenClaw for inference:

    vimin-core start-agent --openclaw

Or point at a non-default gateway:

    vimin-core start-agent --openclaw --openclaw-url http://127.0.0.1:18789
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import pwd
import urllib.error
import urllib.request
from typing import Iterator, List, Optional

from vimin_core.core.backends.base import BaseBackend, ModelDescriptor

logger = logging.getLogger(__name__)

_DEFAULT_URL = "http://127.0.0.1:18789"
_DEFAULT_TOKEN_PATH = pathlib.Path.home() / ".openclaw" / "openclaw.json"
_TOKEN_PATH = _DEFAULT_TOKEN_PATH
def _candidate_config_paths() -> list[pathlib.Path]:
    paths: list[pathlib.Path] = []
    config_path = os.environ.get("OPENCLAW_CONFIG_PATH")
    if config_path:
        paths.append(pathlib.Path(config_path).expanduser())
    paths.append(_TOKEN_PATH)
    if _TOKEN_PATH == _DEFAULT_TOKEN_PATH:
        try:
            real_home = pathlib.Path(pwd.getpwuid(os.getuid()).pw_dir)
            fallback = real_home / ".openclaw" / "openclaw.json"
            if fallback not in paths:
                paths.append(fallback)
        except Exception:
            pass
    return paths


def _load_openclaw_token() -> str:
    """Read the gateway auth token from the active OpenClaw config."""
    env_token = os.environ.get("OPENCLAW_TOKEN", "")
    if env_token:
        return env_token
    for path in _candidate_config_paths():
        try:
            data = json.loads(path.read_text())
            token = data["gateway"]["auth"]["token"]
            if token:
                return token
        except Exception:
            continue
    return ""


class OpenClawBackend(BaseBackend):
    """
    Inference backend that delegates to a locally running OpenClaw Gateway.

    OpenClaw exposes an OpenAI-compatible /v1/chat/completions endpoint.
    vimin-core uses it as a black-box: send a prompt, get a completion.
    No model weights are loaded into vimin-core's process — RAM usage
    is zero from vimin-core's perspective.

    The backend is opt-in: pass ``--openclaw`` to ``vimin-core start-agent``
    to activate it.
    """

    def __init__(
        self,
        url: str = "",
        token: str = "",
        model: str = "openclaw",
    ) -> None:
        resolved_url = url or os.environ.get("OPENCLAW_URL", _DEFAULT_URL)
        self._url = resolved_url.rstrip("/")
        self._token = token or _load_openclaw_token()
        self._model = model
        self._loaded = False

    @property
    def url(self) -> str:
        return self._url

    # ------------------------------------------------------------------
    # BaseBackend interface
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if the OpenClaw Gateway is reachable."""
        try:
            req = urllib.request.Request(
                f"{self._url}/health",
                headers=self._headers(),
            )
            with urllib.request.urlopen(req, timeout=3):
                return True
        except Exception:
            return False

    def estimate_memory_gb(self, descriptor: ModelDescriptor) -> float:
        """OpenClaw manages its own model memory — always 0 from vimin's view."""
        return 0.0

    def load(self, descriptor: ModelDescriptor) -> bool:
        """
        'Loading' for OpenClaw means verifying the gateway is up.
        The model name from the descriptor is used as the OpenAI model field
        if the gateway supports model selection; otherwise it defaults to
        the backend's configured model name.
        """
        if not self.is_available():
            logger.error(
                f"OpenClawBackend: gateway not reachable at {self._url}.\n"
                "  Ensure OpenClaw is running: openclaw gateway start"
            )
            return False
        # OpenClaw's API expects a gateway-owned identifier like "openclaw" or
        # "openclaw/<agentId>", not a raw upstream Hugging Face model name.
        if descriptor.model_id and str(descriptor.model_id).startswith("openclaw"):
            self._model = descriptor.model_id
        else:
            self._model = "openclaw"
        self._loaded = True
        logger.info(f"OpenClawBackend: ready — {self._url}  model={self._model}")
        return True

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        if not self._loaded:
            raise RuntimeError("OpenClawBackend.generate(): call load() first")
        body: dict = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }
        if stop_sequences:
            body["stop"] = stop_sequences

        data = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{self._url}/v1/chat/completions",
            data=data,
            headers={**self._headers(), "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())
                return result["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            body_text = e.read().decode(errors="replace")
            logger.error(f"OpenClawBackend: HTTP {e.code} — {body_text[:200]}")
            raise
        except Exception as exc:
            logger.error(f"OpenClawBackend: request failed — {exc}")
            raise

    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None,
    ) -> Iterator[str]:
        """Stream tokens via SSE from OpenClaw's /v1/chat/completions."""
        if not self._loaded:
            raise RuntimeError("OpenClawBackend.stream_generate(): call load() first")
        body: dict = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if stop_sequences:
            body["stop"] = stop_sequences

        data = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{self._url}/v1/chat/completions",
            data=data,
            headers={**self._headers(), "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                for raw_line in resp:
                    line = raw_line.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        return
                    try:
                        chunk = json.loads(payload)
                        delta = chunk["choices"][0].get("delta", {})
                        token = delta.get("content", "")
                        if token:
                            yield token
                    except (KeyError, json.JSONDecodeError):
                        continue
        except Exception as exc:
            logger.error(f"OpenClawBackend: stream failed — {exc}")
            raise

    def unload(self) -> None:
        self._loaded = False
        logger.info("OpenClawBackend: unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict:
        h = {"Accept": "application/json"}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    def available_models(self) -> list[str]:
        """Return the list of models the gateway reports (if supported)."""
        try:
            req = urllib.request.Request(
                f"{self._url}/v1/models",
                headers=self._headers(),
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                return [m["id"] for m in data.get("data", [])]
        except Exception:
            return []
