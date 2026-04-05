"""vimin-core CLI"""

import argparse
import asyncio
import os
import sys

# ── Brand palette (truecolor ANSI) ────────────────────────────────────────────
_P  = "\033[38;2;139;60;247m"   # #8b3cf7  primary purple
_L  = "\033[38;2;192;132;252m"  # #c084fc  lavender
_W  = "\033[38;2;240;234;255m"  # #f0eaff  off-white
_D  = "\033[38;2;110;90;180m"   # dim purple  (muted accents)
_R  = "\033[0m"                  # reset

# Logo wordmark
_LOGO = f"{_P}  ◈ {_W}vimin{_R}{_D}-core{_R}"


def _banner(title: str, fields: list[tuple[str, str]], width: int = 62) -> None:
    """Print a branded box with title and key-value fields."""
    inner = width - 2  # inside the border chars

    # ── top bar ───────────────────────────────────────────────────────────────
    print()
    print(_LOGO)
    print()
    print(f"{_P}  ╭{'─' * inner}╮{_R}")

    # title line
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
            print(f"  Another vimin-core center (or another process) is already bound to that port.")
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

    # OpenClaw mode: use local OpenClaw Gateway as the inference engine
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
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await agent.stop()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        print(f"\n  {_D}Agent disconnected.{_R}\n")
    return 0


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
    sc.add_argument("--debug", action="store_true")

    # start-agent
    sa = sub.add_parser("start-agent", help="Start a local inference agent node")
    sa.add_argument("--center", default=os.environ.get("VIMIN_CENTER_URL", "http://localhost:8080"))
    sa.add_argument("--debug", action="store_true")
    sa.add_argument("--openclaw", action="store_true",
                    help="Use a local OpenClaw Gateway as the inference engine")
    sa.add_argument("--openclaw-url", default=None,
                    help="OpenClaw Gateway URL (default: http://127.0.0.1:18789)")

    args = parser.parse_args()

    if args.command == "start-center":
        return _cmd_start_center(args)
    elif args.command == "start-agent":
        return _cmd_start_agent(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
