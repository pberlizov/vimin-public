#!/usr/bin/env python3
"""
~/.vimin/config.json management for zero-config startup.

vimin start-center  — generates config on first run, writes fleet_token + api_key
vimin start-agent   — reads config, connects to center with matching credentials
"""

from __future__ import annotations

import json
import os
import secrets
import string
import socket
from pathlib import Path
from typing import Optional

_CONFIG_PATH = Path.home() / ".vimin" / "config.json"

_DEFAULTS = {
    "center_url": "http://localhost:8080",
    "fleet_token": None,   # generated on first run
    "api_key": None,       # generated on first run
    "agent_id": None,      # generated on first run, persisted so queued tasks survive restarts
}

_GENERATED_KEYS = {"fleet_token", "api_key", "agent_id"}


def guess_local_ip(fallback: str = "127.0.0.1") -> str:
    """
    Best-effort guess of this machine's LAN IP for demo-friendly URLs.
    Works without external network access by using a UDP connect trick.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # No packets are actually sent for UDP connect.
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            return ip or fallback
        finally:
            s.close()
    except Exception:
        return fallback


def _gen_token(length: int = 32) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def load_config() -> dict:
    """Load ~/.vimin/config.json, returning an empty dict if it doesn't exist."""
    if _CONFIG_PATH.exists():
        try:
            return json.loads(_CONFIG_PATH.read_text())
        except Exception:
            return {}
    return {}


def save_config(cfg: dict) -> None:
    """Write cfg to ~/.vimin/config.json (creates the directory if needed)."""
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_PATH.parent.chmod(0o700)
    _CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    _CONFIG_PATH.chmod(0o600)


def ensure_config() -> dict:
    """
    Load config, generating fleet_token and api_key if they are missing.
    Saves back to disk and returns the (possibly updated) config.
    """
    cfg = load_config()
    changed = False
    for key, default in _DEFAULTS.items():
        if key not in cfg or cfg[key] is None:
            if key in _GENERATED_KEYS:
                cfg[key] = _gen_token()
            else:
                cfg[key] = default
            changed = True
    if changed:
        save_config(cfg)
    return cfg


def get_center_url(override: Optional[str] = None) -> str:
    """Return the effective center URL (env var > override > config > default)."""
    def _is_placeholder(u: Optional[str]) -> bool:
        if not u:
            return False
        return "center_host" in u.lower() or u.strip().upper() == "CENTER_HOST"

    env = os.environ.get("VIMIN_CENTER_URL")
    if env and not _is_placeholder(env):
        return env

    if override and not _is_placeholder(override):
        return override

    cfg_url = load_config().get("center_url", "http://localhost:8080")
    if _is_placeholder(cfg_url):
        return "http://localhost:8080"
    return cfg_url
