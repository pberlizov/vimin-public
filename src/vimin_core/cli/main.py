"""vimin-core CLI"""

import argparse
import asyncio
import os
import sys


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

    print("=" * 60)
    print("  vimin-core Center Node")
    print("=" * 60)
    print(f"  URL          : {cfg['center_url']}")
    print(f"  API key      : {api_key}")
    print(f"  Fleet token  : {fleet_token}")
    print(f"  Node limit   : 10  (upgrade to vimin for more)")
    print()

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
        print("\n  Stopped.")
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

    agent = UserAgent(
        center_url=center_url,
        api_key=api_key,
        fleet_token=fleet_token,
    )

    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        print("\n  Disconnected.")
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
