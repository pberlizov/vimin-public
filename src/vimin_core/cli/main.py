"""vimin-core CLI"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# ── Brand palette (truecolor ANSI) ────────────────────────────────────────────
_P  = "\033[38;2;139;60;247m"   # #8b3cf7  primary purple
_L  = "\033[38;2;192;132;252m"  # #c084fc  lavender
_W  = "\033[38;2;240;234;255m"  # #f0eaff  off-white
_D  = "\033[38;2;110;90;180m"   # dim purple  (muted accents)
_R  = "\033[0m"                  # reset

# Logo wordmark
_LOGO = f"{_P}  ◈ {_W}vimin{_R}{_D}-core{_R}"

_VIMIN_DIR = Path.home() / ".vimin"
_CENTER_PID = _VIMIN_DIR / "center.pid"
_CENTER_LOG = _VIMIN_DIR / "logs" / "center.log"
_AGENT_PID  = _VIMIN_DIR / "agent.pid"
_AGENT_LOG  = _VIMIN_DIR / "logs" / "agent.log"


def _banner(title: str, fields: list[tuple[str, str]], width: int = 62) -> None:
    """Print a branded box with title and key-value fields."""
    inner = width - 2  # inside the border chars

    print()
    print(_LOGO)
    print()
    print(f"{_P}  ╭{'─' * inner}╮{_R}")

    padding = (inner - len(title)) // 2
    print(f"{_P}  │{_R}{' ' * padding}{_W}{title}{_R}{' ' * (inner - padding - len(title))}{_P}│{_R}")

    print(f"{_P}  ├{'─' * inner}┤{_R}")

    for key, value in fields:
        label = f"{_L}{key}{_R}"
        val   = f"{_W}{value}{_R}"
        raw_len = len(key) + 2 + len(value)   # key + ": " + value
        spaces  = inner - raw_len - 2           # 2 leading spaces before key
        print(f"{_P}  │{_R}  {label}{_D}:{_R} {val}{' ' * max(spaces, 0)}{_P}│{_R}")

    print(f"{_P}  ╰{'─' * inner}╯{_R}")
    print()


def _daemonize(pid_path: Path, log_path: Path) -> None:
    """
    Double-fork daemonization (Unix only).
    The banner must be printed BEFORE calling this.
    The parent process waits for the daemon to write its PID file,
    prints confirmation, then exits. The daemon continues running.
    """
    import time, errno as _errno

    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        pid_path.unlink()
    except FileNotFoundError:
        pass

    try:
        pid = os.fork()
    except AttributeError:
        print(f"  {_P}ERROR{_R}: --daemon is not supported on this platform.\n")
        sys.exit(1)

    if pid > 0:
        # Original process: wait for daemon to write PID file, then exit.
        for _ in range(40):   # up to 4 seconds
            time.sleep(0.1)
            if pid_path.exists():
                daemon_pid = pid_path.read_text().strip()
                print(f"  {_D}Running in background.{_R}")
                print(f"  PID  {_W}{daemon_pid}{_R}   |   Logs  {_W}{log_path}{_R}")
                print(f"  Stop with  {_W}vimin-core stop-center{_R}\n"
                      if "center" in str(pid_path) else
                      f"  Stop with  {_W}vimin-core stop-agent{_R}\n")
                os._exit(0)
        print(f"  {_P}WARNING{_R}: daemon may not have started. Check {log_path}\n")
        os._exit(1)

    # Child 1: become session leader, then fork again
    os.setsid()
    if os.fork() > 0:
        os._exit(0)

    # Daemon process: redirect I/O to log file
    log_fd = open(log_path, "a")
    os.dup2(log_fd.fileno(), sys.stdout.fileno())
    os.dup2(log_fd.fileno(), sys.stderr.fileno())
    log_fd.close()
    devnull = open(os.devnull)
    os.dup2(devnull.fileno(), sys.stdin.fileno())
    devnull.close()

    pid_path.write_text(str(os.getpid()))


def _stop(pid_path: Path, name: str) -> int:
    import signal, time
    if not pid_path.exists():
        print(f"\n  No {name} PID file found at {pid_path}.")
        print(f"  It may not be running, or was started without --daemon.\n")
        return 1
    pid = int(pid_path.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        print(f"\n  {_P}WARNING{_R}: PID {pid} not found — {name} may have already exited.\n")
        pid_path.unlink(missing_ok=True)
        return 0
    for _ in range(30):
        time.sleep(0.1)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
    pid_path.unlink(missing_ok=True)
    print(f"\n  {_D}{name} stopped (PID {pid}).{_R}\n")
    return 0


def _cmd_start_center(args) -> int:
    import logging as _logging
    from vimin_core.cli.config import ensure_config, save_config, guess_local_ip
    from vimin_core.utils.log_config import configure_logging
    from vimin_core.systems.center_node import CenterNode

    configure_logging(_logging.DEBUG if args.debug else _logging.INFO)

    cfg = ensure_config()
    fleet_token = cfg["fleet_token"]
    api_key = cfg["api_key"]

    os.environ.setdefault("VIMIN_FLEET_TOKEN", fleet_token)
    os.environ.setdefault("ORCHESTRATOR_API_KEY", api_key)
    os.environ.setdefault("ORCHESTRATOR_MASTER_KEY", api_key)

    display_host = "localhost" if args.host in ("0.0.0.0", "::") else args.host
    cfg["center_url"] = f"http://{display_host}:{args.port}"
    save_config(cfg)

    _banner(
        "vimin-core  ·  Center Node",
        [
            ("URL",         cfg["center_url"]),
            ("API key",     api_key),
            ("Fleet token", fleet_token),
            ("Node limit",  "10  (upgrade to vimin for more)"),
        ],
    )

    if args.daemon:
        _daemonize(_CENTER_PID, _CENTER_LOG)

    node = CenterNode(host=args.host, port=args.port)

    async def _run():
        await node.start()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await node.stop()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        print(f"\n  {_D}Center stopped.{_R}\n")
    except OSError as e:
        import errno
        if e.errno == errno.EADDRINUSE:
            print(f"\n  {_P}ERROR{_R}: Port {args.port} is already in use.\n")
            print(f"  Another process is already bound to that port.")
            print(f"  To find and stop it:  {_W}lsof -ti:{args.port} | xargs kill{_R}")
            print(f"  Or start on a different port:  {_W}vimin-core start-center --port 8081{_R}\n")
        else:
            raise
    return 0


def _cmd_start_agent(args) -> int:
    import logging as _logging
    from vimin_core.cli.config import ensure_config
    from vimin_core.utils.log_config import configure_logging
    from vimin_core.systems.user_agent import UserAgent

    configure_logging(_logging.DEBUG if args.debug else _logging.INFO)

    cfg = ensure_config()
    api_key = cfg.get("api_key") or os.environ.get("ORCHESTRATOR_API_KEY", "")
    fleet_token = cfg.get("fleet_token") or os.environ.get("VIMIN_FLEET_TOKEN", "")
    center_url = args.center or cfg.get("center_url", "http://localhost:8080")

    openclaw_url = None
    if args.openclaw:
        from vimin_core.core.backends.openclaw_backend import OpenClawBackend, _DEFAULT_URL
        openclaw_url = args.openclaw_url or _DEFAULT_URL
        backend = OpenClawBackend(url=openclaw_url)
        if not backend.is_available():
            print(f"\n  {_P}ERROR{_R}: OpenClaw Gateway not reachable at {openclaw_url}")
            print("  Ensure OpenClaw is running: openclaw gateway start\n")
            return 1

    agent = UserAgent(
        center_node_url=center_url,
        api_key=api_key,
        fleet_token=fleet_token,
        openclaw_url=openclaw_url,
    )

    async def _run():
        await agent.start()
        fields = [
            ("Agent ID", agent.agent_id),
            ("Center",   agent.center_node_url),
        ]
        if openclaw_url:
            fields.append(("OpenClaw", openclaw_url))
        _banner("vimin-core  ·  Inference Agent", fields)

        if args.daemon:
            # Agent ID is only known after start(), so write a custom PID file name.
            pid_path = _VIMIN_DIR / f"agent-{agent.agent_id}.pid"
            log_path = _VIMIN_DIR / "logs" / f"agent-{agent.agent_id}.log"
            _daemonize(pid_path, log_path)

        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await agent.stop()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        print(f"\n  {_D}Agent disconnected.{_R}\n")
    return 0


def _cmd_stop_center(_args) -> int:
    return _stop(_CENTER_PID, "center")


def _cmd_stop_agent(_args) -> int:
    # Stop all agent PIDs found in ~/.vimin/
    pids = list(_VIMIN_DIR.glob("agent-*.pid"))
    if not pids:
        print(f"\n  No running agents found (no PID files in {_VIMIN_DIR}).\n")
        return 1
    code = 0
    for p in pids:
        code |= _stop(p, f"agent ({p.stem})")
    return code


def main():
    parser = argparse.ArgumentParser(
        prog="vimin-core",
        description="vimin-core — open-source local inference orchestration (up to 10 nodes)",
    )
    sub = parser.add_subparsers(dest="command")

    # start-center
    sc = sub.add_parser("start-center", help="Start the orchestration center node")
    sc.add_argument("--host", default="0.0.0.0")
    sc.add_argument("--port", type=int, default=8080)
    sc.add_argument("--daemon", action="store_true", help="Run in the background")
    sc.add_argument("--debug", action="store_true")

    # start-agent
    sa = sub.add_parser("start-agent", help="Start a local inference agent node")
    sa.add_argument("--center", default=os.environ.get("VIMIN_CENTER_URL", "http://localhost:8080"))
    sa.add_argument("--daemon", action="store_true", help="Run in the background")
    sa.add_argument("--debug", action="store_true")
    sa.add_argument("--openclaw", action="store_true",
                    help="Use a local OpenClaw Gateway as the inference engine")
    sa.add_argument("--openclaw-url", default=None,
                    help="OpenClaw Gateway URL (default: http://127.0.0.1:18789)")

    # stop-center / stop-agent
    sub.add_parser("stop-center", help="Stop a daemonized center node")
    sub.add_parser("stop-agent",  help="Stop all daemonized agent nodes")

    args = parser.parse_args()

    if args.command == "start-center":
        return _cmd_start_center(args)
    elif args.command == "start-agent":
        return _cmd_start_agent(args)
    elif args.command == "stop-center":
        return _cmd_stop_center(args)
    elif args.command == "stop-agent":
        return _cmd_stop_agent(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
