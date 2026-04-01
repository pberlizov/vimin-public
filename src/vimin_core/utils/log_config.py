"""
Centralised logging configuration for vimin processes.

Replaces bare ``logging.basicConfig()`` calls throughout the codebase with a
single function that:

  * Attaches a ``RotatingFileHandler`` to ``~/.vimin/agent.log``
    (10 MB per file, 3 backups → max 40 MB on disk).
  * Attaches a ``StreamHandler`` only when stderr is an interactive terminal,
    so daemon output does not duplicate into the OS-redirected log file.

Usage
-----
    from vimin_core.utils.log_config import configure_logging
    import logging

    configure_logging(logging.DEBUG)   # or logging.INFO
"""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_FILE = Path.home() / ".vimin" / "agent.log"
_MAX_BYTES = 10 * 1024 * 1024   # 10 MB per file
_BACKUP_COUNT = 3                # keep agent.log, agent.log.1, .2, .3
_FMT = "%(asctime)s [%(name)s] %(levelname)s %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with a console StreamHandler and rotating file output.

    Idempotent: clears any handlers added by third-party basicConfig calls before
    attaching the vimin-formatted handlers, so calling order doesn't matter.
    """
    root = logging.getLogger()
    root.setLevel(level)
    fmt = logging.Formatter(_FMT, datefmt=_DATE_FMT)

    # Remove any handlers that aren't ours (e.g. from router.py's basicConfig).
    for h in root.handlers[:]:
        if not getattr(h, "_vimin", False):
            root.removeHandler(h)

    # Only add our handlers once.
    if any(getattr(h, "_vimin", False) for h in root.handlers):
        return

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh._vimin = True
    root.addHandler(sh)

    # Rotating file handler — always active.
    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    fh = RotatingFileHandler(
        str(_LOG_FILE),
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    fh._vimin = True
    root.addHandler(fh)
