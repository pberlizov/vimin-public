"""
Inference thread priority management.

Lowers the OS scheduling priority of the calling thread so that model
inference runs as a true background task — it will not starve the user's
foreground applications for CPU time.

Platforms:
  macOS / Linux  — os.nice(+5) raises the nice level of the calling thread.
                   nice 0 = normal, nice 19 = lowest possible.
                   We target nice 5: responsive enough to run at full hardware
                   throughput while yielding when the OS has better things to do.

  Windows        — SetPriorityClass(BELOW_NORMAL_PRIORITY_CLASS) on the current
                   process. This is the coarsest available knob on Windows and
                   affects all threads, so we apply it only once per process.
"""

from __future__ import annotations

import logging
import os
import platform
import threading

logger = logging.getLogger(__name__)

_TARGET_NICE = 5           # nice delta to add on Unix
_WINDOWS_PRIORITY_SET = False
_PRIORITY_LOCK = threading.Lock()


def set_inference_priority() -> None:
    """
    Lower the calling thread's scheduling priority for non-invasive inference.

    Safe to call from any thread; idempotent (calling more than once has no
    extra effect). Errors are logged at DEBUG level and never propagate —
    inference should always proceed even if priority adjustment fails.
    """
    global _WINDOWS_PRIORITY_SET
    system = platform.system()

    try:
        if system in ("Darwin", "Linux"):
            # os.nice() adjusts the current *thread* priority on Linux (via
            # setpriority(PRIO_PROCESS, 0, ...)) and the process on macOS.
            # It raises on each call, so cap at target.
            current_nice = os.nice(0)
            delta = max(0, _TARGET_NICE - current_nice)
            if delta > 0:
                os.nice(delta)
            logger.debug(
                f"Inference priority: nice level {os.nice(0)} "
                f"(was {current_nice}, target {_TARGET_NICE})"
            )

        elif system == "Windows":
            with _PRIORITY_LOCK:
                if not _WINDOWS_PRIORITY_SET:
                    import ctypes
                    BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
                    handle = ctypes.windll.kernel32.GetCurrentProcess()
                    ctypes.windll.kernel32.SetPriorityClass(
                        handle, BELOW_NORMAL_PRIORITY_CLASS
                    )
                    _WINDOWS_PRIORITY_SET = True
                    logger.debug("Windows: set BELOW_NORMAL process priority for inference")

    except Exception as exc:
        logger.debug(f"Could not set inference priority: {exc}")
