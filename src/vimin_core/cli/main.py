"""vimin-core CLI"""

import argparse
import asyncio
import os
import platform
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


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine().lower() in ("arm64", "arm")


def _default_model_for_host() -> str:
    if _is_apple_silicon():
        return "Qwen/Qwen2.5-3B-Instruct"
    return "meta-llama/Llama-3.2-1B-Instruct"


def _write_output(path_str: str, data: dict) -> Path:
    """Write *data* as JSON to *path_str*, creating parent dirs as needed.
    If the file already exists and contains a JSON array, the new record is
    appended. If it contains a single object, both are wrapped in an array.
    Returns the resolved path."""
    import json as _json
    p = Path(path_str).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        try:
            existing = _json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            existing = None
        if isinstance(existing, list):
            existing.append(data)
            p.write_text(_json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            records = [existing, data] if existing is not None else [data]
            p.write_text(_json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        p.write_text(_json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return p
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


def _port_in_use(host: str, port: int) -> bool:
    """Return True if *port* is already bound on *host*."""
    import socket
    # Normalise wildcard binds — we want to probe localhost
    probe = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((probe, port)) == 0


def _check_already_running(pid_path: Path, host: str, port: int, name: str) -> bool:
    """
    Return True (and print a clear error) if a *name* process is already running.

    Checks two independent signals:
      1. A live PID file pointing to a running process.
      2. The target port is already accepting connections.

    Either signal is sufficient — a zombie process with a stale PID file but
    an open port (or vice-versa) is caught by the other check.
    """
    pid_alive = False
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text().strip())
            os.kill(pid, 0)   # raises ProcessLookupError if dead
            pid_alive = True
            print(
                f"\n  {_P}ERROR{_R}: A {name} is already running (PID {pid}).\n"
                f"  Stop it first:  {_W}vimin-core stop-{name.replace(' ', '-')}{_R}\n"
            )
        except (ProcessLookupError, ValueError):
            pid_path.unlink(missing_ok=True)   # stale — clean up

    if not pid_alive and _port_in_use(host, port):
        # Port occupied but no PID file — zombie from a previous session
        import subprocess as _sp
        try:
            pids = _sp.check_output(
                ["lsof", "-ti", f":{port}"], text=True
            ).strip()
        except Exception:
            pids = "unknown"
        print(
            f"\n  {_P}ERROR{_R}: Port {port} is already in use by another process (PID {pids}).\n"
            f"  Kill it first:  {_W}kill {pids}{_R}  then re-run.\n"
        )
        return True

    return pid_alive


def _daemonize(cmd: list, pid_path: Path, log_path: Path) -> None:
    """
    Spawn a background subprocess and record its PID.

    Uses subprocess.Popen with start_new_session=True instead of os.fork().
    On macOS, fork()-without-exec() is unsafe when Apple frameworks
    (Metal, CoreML, Foundation) are in use — the ObjC runtime detects the
    forked child and crashes it the moment it touches any ObjC class.
    A fresh subprocess avoids this entirely.

    The caller must add --foreground to `cmd` so the subprocess runs in foreground
    and does not re-daemonize (which would recurse infinitely).
    """
    import subprocess

    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        pid_path.unlink()
    except FileNotFoundError:
        pass

    with open(log_path, "a") as log_fd:
        proc = subprocess.Popen(
            cmd,
            stdout=log_fd,
            stderr=log_fd,
            stdin=subprocess.DEVNULL,
            start_new_session=True,   # detach from terminal / create new session
            close_fds=True,
        )

    pid_path.write_text(str(proc.pid))
    stop_cmd = "stop-center" if "center" in str(pid_path) else "stop-agent"
    print(f"  {_D}Running in background.{_R}")
    print(f"  PID  {_W}{proc.pid}{_R}   |   Logs  {_W}{log_path}{_R}")
    print(f"  Stop with  {_W}vimin-core {stop_cmd}{_R}\n")


def _stop(pid_path: Path, name: str) -> int:
    import signal, time
    if not pid_path.exists():
        print(f"\n  No {name} PID file found at {pid_path}.")
        print(f"  It may not be running, or was started without this process.\n")
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
    # If the user pre-set ORCHESTRATOR_MASTER_KEY, honour it (allows shared
    # multi-machine secrets). Otherwise fall back to the config api_key so
    # the displayed key and the validated key are always the same.
    if not os.environ.get("ORCHESTRATOR_MASTER_KEY"):
        os.environ["ORCHESTRATOR_MASTER_KEY"] = api_key

    _loopback = {"127.0.0.1", "::1", "localhost"}
    _wildcard  = {"0.0.0.0", "::"}
    display_host = "localhost" if args.host in _loopback | _wildcard else args.host
    cfg["center_url"] = f"http://{display_host}:{args.port}"
    save_config(cfg)

    if args.host not in _loopback:
        print(
            f"\n  {_P}⚠  WARNING{_R}  Center is binding to {_W}{args.host}{_R} "
            f"(reachable from other machines).\n"
            f"  {_D}Use TLS and a firewall rule to restrict access in production.{_R}\n"
            f"  {_D}To restrict to this machine only, omit --host (default: 127.0.0.1).{_R}"
        )

    _banner(
        "vimin-core  ·  Center Node",
        [
            ("URL",         cfg["center_url"]),
            ("API key",     api_key),
            ("Fleet token", fleet_token),
            ("Node limit",  "10  (upgrade to vimin for more)"),
        ],
    )

    if not args.foreground:
        if _check_already_running(_CENTER_PID, args.host, args.port, "center"):
            return 1
        cmd = sys.argv + ["--foreground"]   # subprocess must NOT re-daemonize
        _daemonize(cmd, _CENTER_PID, _CENTER_LOG)
        print(f"  {_D}Outputs dir:  {_W}{_VIMIN_DIR / 'outputs'}{_R}")
        print(f"  {_D}Watch logs:   {_W}tail -f {_CENTER_LOG}{_R}")
        print(f"  {_D}Stop:         {_W}vimin-core stop-center{_R}\n")
        return 0   # parent exits; subprocess runs the server

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
    import uuid as _uuid
    from vimin_core.cli.config import ensure_config
    from vimin_core.utils.log_config import configure_logging
    from vimin_core.systems.user_agent import UserAgent

    configure_logging(_logging.DEBUG if args.debug else _logging.INFO)

    cfg = ensure_config()
    api_key = (os.environ.get("ORCHESTRATOR_MASTER_KEY")
               or os.environ.get("ORCHESTRATOR_API_KEY")
               or cfg.get("api_key", ""))
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

    # Use the ID passed by a daemon parent, then the persisted config ID,
    # then mint a fresh one and save it — so the same device always reconnects
    # with the same agent_id and picks up tasks queued while it was offline.
    agent_id = getattr(args, "agent_id", None) or cfg.get("agent_id") or str(_uuid.uuid4())
    if not cfg.get("agent_id"):
        cfg["agent_id"] = agent_id
        from vimin_core.cli.config import save_config
        save_config(cfg)

    fields = [
        ("Agent ID", agent_id),
        ("Center",   center_url),
    ]
    if openclaw_url:
        fields.append(("OpenClaw", openclaw_url))
    _banner("vimin-core  ·  Inference Agent", fields)

    if not args.foreground:
        pid_path = _VIMIN_DIR / f"agent-{agent_id}.pid"
        log_path = _VIMIN_DIR / "logs" / f"agent-{agent_id}.log"
        cmd = sys.argv + ["--foreground", "--agent-id", agent_id]  # subprocess must NOT re-daemonize
        _daemonize(cmd, pid_path, log_path)
        print(f"  {_D}Watch logs:   {_W}tail -f {log_path}{_R}")
        print(f"  {_D}Stop:         {_W}vimin-core stop-agent{_R}\n")
        return 0   # parent exits; subprocess runs the agent

    agent = UserAgent(
        center_node_url=center_url,
        api_key=api_key,
        fleet_token=fleet_token,
        openclaw_url=openclaw_url,
        agent_id=agent_id,
    )

    default_model = getattr(args, "model", None) or cfg.get("default_model") or _DEFAULT_MODEL

    async def _run():
        import signal as _signal
        loop = asyncio.get_event_loop()
        stop_event = asyncio.Event()
        loop.add_signal_handler(_signal.SIGTERM, stop_event.set)
        loop.add_signal_handler(_signal.SIGINT,  stop_event.set)

        await agent.start()
        # Pre-load the default model immediately so the first task
        # doesn't block on a cold download + load.
        if agent.orchestrator:
            asyncio.create_task(agent._load_model_async(default_model))

        await stop_event.wait()
        await agent.stop()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        print(f"\n  {_D}Agent disconnected.{_R}\n")
    return 0


_DEFAULT_MODEL = _default_model_for_host()


def _cmd_broadcast(args) -> int:
    import json
    import urllib.request
    import urllib.error
    from vimin_core.cli.config import ensure_config

    import datetime as _dt
    cfg = ensure_config()
    api_key   = (os.environ.get("ORCHESTRATOR_MASTER_KEY")
                 or os.environ.get("ORCHESTRATOR_API_KEY")
                 or cfg.get("api_key", ""))
    center    = cfg.get("center_url", "http://localhost:8080")
    model     = args.model or cfg.get("default_model", _DEFAULT_MODEL)
    max_tok   = args.max_tokens
    mode      = args.mode or "return"

    mode_label = (
        f"{_W}return{_R} {_D}(results come back to center){_R}"
        if mode == "return"
        else f"{_W}broadcast{_R} {_D}(results saved on edge device at ~/.vimin/outputs/){_R}"
    )
    print(f"\n{_D}  Mode:{_R}  {mode_label}")

    body: dict = {
        "prompt":     args.prompt,
        "model_id":   model,
        "max_tokens": max_tok,
        "mode":       mode,
    }
    if args.timeout is not None:
        body["timeout"] = args.timeout
    payload = json.dumps(body).encode()

    req = urllib.request.Request(
        f"{center}/api/broadcast",
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )

    # Give the CLI request a bit more headroom than the server timeout.
    cli_timeout = (args.timeout or 60) + 15

    print(f"\n{_D}  Broadcasting to {center} …{_R}", flush=True)

    try:
        with urllib.request.urlopen(req, timeout=cli_timeout) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read())
            error_code = body.get("error", "")
            error_msg  = body.get("message", str(e))
        except Exception:
            error_code, error_msg = "", str(e)
        if error_code == "no_agents":
            print(f"\n  {_P}No agents connected.{_R}\n"
                  f"  Start one with  {_W}vimin-core start-agent{_R}\n")
        elif error_code in ("unauthorized", "forbidden"):
            print(f"\n  {_P}ERROR{_R}: Authentication failed — check your API key.\n")
        else:
            print(f"\n  {_P}ERROR{_R}: Center returned HTTP {e.code}: {error_msg}\n")
        return 1
    except urllib.error.URLError as e:
        print(f"\n  {_P}ERROR{_R}: Cannot connect to center at {center}.\n"
              f"  Start it with  {_W}vimin-core start-center{_R}\n")
        return 1
    except Exception as e:
        print(f"\n  {_P}ERROR{_R}: {e}\n")
        return 1

    results = data.get("results", [])
    if not results:
        print(f"\n  {_P}No results returned.{_R} Are any agents connected?\n"
              f"  Start one with  {_W}vimin-core start-agent{_R}\n")
        return 1

    # Only save to disk if at least one result has real output (not just queued/in-progress)
    has_real_output = any(
        not r.get("queued") and not r.get("in_progress")
        for r in results
    )
    _outputs_dir = _VIMIN_DIR / "outputs"
    if args.output:
        saved = _write_output(args.output, data)
        print(f"  {_D}Response saved to  {_W}{saved}{_R}")
    elif mode == "return" and has_real_output:
        ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        auto_path = _outputs_dir / f"broadcast-{ts}.json"
        saved = _write_output(str(auto_path), data)
        print(f"  {_D}Results auto-saved to  {_W}{saved}{_R}")

    for r in results:
        raw_id   = r.get("agent_id") or ""
        agent_id = raw_id[:8] if raw_id else "unknown"
        latency  = r.get("latency_ms") or 0
        output      = r.get("output")
        error       = r.get("error")
        queued      = r.get("queued", False)
        in_progress = r.get("in_progress", False)
        if queued:
            print(f"\n{_D}  ── agent {agent_id}  queued — will run on reconnect{_R}")
        elif in_progress:
            print(f"\n{_D}  ── agent {agent_id}  running — result stored in task history when done{_R}")
        elif error:
            print(f"\n{_P}  ── agent {agent_id}{_R}  {_D}({latency:.0f} ms){_R}")
            print(f"  {_P}error:{_R} {error}")
        elif isinstance(output, str) and output.startswith("[saved_locally] "):
            saved_path = output[16:]
            print(f"\n{_P}  ── agent {agent_id}{_R}  {_D}({latency:.0f} ms){_R}")
            print(f"  {_D}saved on edge device:{_R}  {_W}{saved_path}{_R}")
        elif output:
            print(f"\n{_P}  ── agent {agent_id}{_R}  {_D}({latency:.0f} ms){_R}")
            for line in output.strip().splitlines():
                print(f"  {line}")
        else:
            print(f"\n{_P}  ── agent {agent_id}{_R}  {_D}({latency:.0f} ms){_R}")
            print(f"  {_D}(no output){_R}")
    print()
    return 0


_PRESETS_DIR = Path(__file__).parent.parent.parent.parent / "presets"


def _cmd_run_pipeline(args) -> int:
    import json
    import urllib.request
    import urllib.error
    from vimin_core.cli.config import ensure_config

    import datetime as _dt
    cfg = ensure_config()
    api_key = (os.environ.get("ORCHESTRATOR_MASTER_KEY")
               or os.environ.get("ORCHESTRATOR_API_KEY")
               or cfg.get("api_key", ""))
    center  = cfg.get("center_url", "http://localhost:8080")
    model   = args.model or cfg.get("default_model", _DEFAULT_MODEL)

    # Load pipeline definition
    pipeline_path = None
    if args.pipeline:
        pipeline_path = Path(args.pipeline)
    elif args.preset:
        pipeline_path = _PRESETS_DIR / f"{args.preset}.json"
        if not pipeline_path.exists():
            available = [p.stem for p in _PRESETS_DIR.glob("*.json")] if _PRESETS_DIR.exists() else []
            print(f"\n  {_P}ERROR{_R}: Preset '{args.preset}' not found.")
            if available:
                print(f"  Available presets: {', '.join(available)}")
            print(f"  Or use --pipeline <path> to specify a custom pipeline file.\n")
            return 1

    if not pipeline_path:
        print(f"\n  {_P}ERROR{_R}: Provide --pipeline <file> or --preset <name>.\n")
        return 1

    try:
        pipeline = json.loads(pipeline_path.read_text())
    except Exception as e:
        print(f"\n  {_P}ERROR{_R}: Could not read pipeline file: {e}\n")
        return 1

    # Optionally inject file content (or path for audio) as {{input}}
    _AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac"}
    if args.file:
        fp = Path(args.file)
        if fp.suffix.lower() in _AUDIO_EXTS:
            # SPEECH_TO_TEXT steps need the file path, not the binary content
            pipeline["input"] = str(fp.resolve())
        else:
            try:
                pipeline["input"] = fp.read_text()
            except Exception as e:
                print(f"\n  {_P}ERROR{_R}: Could not read input file: {e}\n")
                return 1
    elif args.input:
        pipeline["input"] = args.input

    if model:
        pipeline.setdefault("model_id", model)

    mode = args.mode or "return"

    pipeline["mode"] = mode

    name   = pipeline.get("name", pipeline_path.stem)
    nsteps = len(pipeline.get("steps", []))
    mode_label = (
        f"{_W}return{_R} {_D}(results come back to center){_R}"
        if mode == "return"
        else f"{_W}broadcast{_R} {_D}(results saved on edge at ~/.vimin/outputs/){_R}"
    )

    _banner(
        f"vimin-core  ·  Pipeline",
        [
            ("Name",   name),
            ("Steps",  str(nsteps)),
            ("Mode",   mode),
            ("Center", center),
        ],
    )

    payload = json.dumps(pipeline).encode()
    req = urllib.request.Request(
        f"{center}/api/pipeline",
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )

    # Total timeout: 300 s per step + buffer
    total_timeout = 310 * nsteps
    print(f"{_D}  Running pipeline in {mode_label} …{_R}\n", flush=True)

    try:
        with urllib.request.urlopen(req, timeout=total_timeout) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read())
            msg  = body.get("message", str(e))
            code = body.get("error", "")
        except Exception:
            msg, code = str(e), ""
        if code == "no_agents":
            print(f"  {_P}No agents connected.{_R}  Start one: {_W}vimin-core start-agent{_R}\n")
        else:
            print(f"  {_P}ERROR{_R}: {msg}\n")
        return 1
    except urllib.error.URLError:
        print(f"  {_P}ERROR{_R}: Cannot connect to center at {center}.\n"
              f"  Start it: {_W}vimin-core start-center{_R}\n")
        return 1

    _outputs_dir = _VIMIN_DIR / "outputs"
    if args.output:
        saved = _write_output(args.output, data)
        print(f"  {_D}Response saved to  {_W}{saved}{_R}\n")
    elif mode == "return":
        ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        name_slug = pipeline.get("name", pipeline_path.stem).replace(" ", "-")
        auto_path = _outputs_dir / f"pipeline-{name_slug}-{ts}.json"
        saved = _write_output(str(auto_path), data)
        print(f"  {_D}Results auto-saved to  {_W}{saved}{_R}\n")

    steps = data.get("steps", [])
    for s in steps:
        step_num = s.get("step")
        parallel = s.get("parallel", False)
        results  = s.get("results", [])
        label    = f"step {step_num}"
        if parallel:
            label += f" (parallel ×{len(results)})"
        print(f"{_P}  ── {label}{_R}")
        for r in results:
            agent_id = (r.get("agent_id") or "")[:8]
            output   = r.get("output") or ""
            error    = r.get("error")
            if error:
                print(f"  {_D}[{agent_id}]{_R}  {_P}error:{_R} {error}")
            elif isinstance(output, str) and output.startswith("[saved_locally] "):
                saved_path = output[16:]
                print(f"  {_D}[{agent_id}]{_R}  {_D}saved on edge device:{_R}  {_W}{saved_path}{_R}")
            else:
                for line in output.strip().splitlines():
                    print(f"  {_D}[{agent_id}]{_R}  {line}")
        print()

    final = data.get("final_output", "")
    if final and not str(final).startswith("[saved_locally]"):
        print(f"{_P}  ── final output{_R}")
        for line in final.strip().splitlines():
            print(f"  {line}")
        print()

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


def _cmd_clear_tasks(_args) -> int:
    import json
    import urllib.request
    import urllib.error
    from vimin_core.cli.config import ensure_config

    cfg = ensure_config()
    api_key = (os.environ.get("ORCHESTRATOR_MASTER_KEY")
               or os.environ.get("ORCHESTRATOR_API_KEY")
               or cfg.get("api_key", ""))
    center = cfg.get("center_url", "http://localhost:8080")

    req = urllib.request.Request(
        f"{center}/api/tasks/clear",
        data=b"{}",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read())
            msg = body.get("message", str(e))
        except Exception:
            msg = str(e)
        print(f"\n  {_P}ERROR{_R}: {msg}\n")
        return 1
    except urllib.error.URLError:
        print(f"\n  {_P}ERROR{_R}: Cannot connect to center at {center}.\n"
              f"  Start it: {_W}vimin-core start-center{_R}\n")
        return 1

    print(f"\n  {_D}Cleared{_R} {_W}{data.get('cleared_count', 0)}{_R} {_D}queued task(s).{_R}")
    note = data.get("note")
    if note:
        print(f"  {_D}{note}{_R}")
    print()
    return 0


def _cmd_revoke_agent(args) -> int:
    import json
    import urllib.request
    import urllib.error
    from vimin_core.cli.config import ensure_config

    cfg = ensure_config()
    api_key = (os.environ.get("ORCHESTRATOR_MASTER_KEY")
               or os.environ.get("ORCHESTRATOR_API_KEY")
               or cfg.get("api_key", ""))
    center = cfg.get("center_url", "http://localhost:8080")

    req = urllib.request.Request(
        f"{center}/api/agents/{args.agent_id}/revoke",
        data=b"{}",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read())
            msg = body.get("message", str(e))
        except Exception:
            msg = str(e)
        print(f"\n  {_P}ERROR{_R}: {msg}\n")
        return 1
    except urllib.error.URLError:
        print(f"\n  {_P}ERROR{_R}: Cannot connect to center at {center}.\n"
              f"  Start it: {_W}vimin-core start-center{_R}\n")
        return 1

    print(f"\n  {_D}Revoked agent{_R} {_W}{data.get('agent_id', args.agent_id)}{_R}")
    if data.get("revoked_at"):
        print(f"  {_D}Revoked at:{_R} {_W}{data['revoked_at']}{_R}")
    print()
    return 0


def _cmd_list_agents(_args) -> int:
    import json
    import urllib.request
    import urllib.error
    from vimin_core.cli.config import ensure_config

    cfg = ensure_config()
    api_key = (os.environ.get("ORCHESTRATOR_MASTER_KEY")
               or os.environ.get("ORCHESTRATOR_API_KEY")
               or cfg.get("api_key", ""))
    center = cfg.get("center_url", "http://localhost:8080")

    req = urllib.request.Request(
        f"{center}/api/agents",
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )

    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read())
            msg = body.get("message", str(e))
        except Exception:
            msg = str(e)
        print(f"\n  {_P}ERROR{_R}: {msg}\n")
        return 1
    except urllib.error.URLError:
        print(f"\n  {_P}ERROR{_R}: Cannot connect to center at {center}.\n"
              f"  Start it: {_W}vimin-core start-center{_R}\n")
        return 1

    agents = data.get("agents", [])
    _banner(
        "vimin-core  ·  Agents",
        [
            ("Center", center),
            ("Agents", str(len(agents))),
            ("Node limit", str(data.get("node_limit", 10))),
        ],
    )

    if not agents:
        print(f"  {_D}No agents registered.{_R}\n")
        return 0

    for agent in agents:
        agent_id = agent.get("agent_id", "")
        hostname = agent.get("hostname") or "unknown"
        status = agent.get("status") or "unknown"
        platform_name = agent.get("platform") or "unknown"
        loaded_model = agent.get("loaded_model_id") or "none"
        summary = agent.get("task_summary") or {}
        joined = agent.get("first_seen_at") or agent.get("registered_at") or "unknown"
        print(f"{_P}  ── {hostname}{_R}  {_D}[{agent_id[:8]}]{_R}")
        print(f"  {_D}status:{_R} {_W}{status}{_R}   {_D}platform:{_R} {_W}{platform_name}{_R}")
        print(f"  {_D}joined:{_R} {_W}{joined}{_R}")
        print(f"  {_D}model:{_R} {_W}{loaded_model}{_R}")
        print(
            f"  {_D}tasks:{_R} "
            f"{_W}{summary.get('received_total', 0)} total{_R} / "
            f"{_W}{summary.get('queued', 0)} queued{_R} / "
            f"{_W}{summary.get('failed', 0)} failed{_R}"
        )
        print()
    return 0


def _cmd_show_agent(args) -> int:
    import json
    import urllib.request
    import urllib.error
    from vimin_core.cli.config import ensure_config

    cfg = ensure_config()
    api_key = (os.environ.get("ORCHESTRATOR_MASTER_KEY")
               or os.environ.get("ORCHESTRATOR_API_KEY")
               or cfg.get("api_key", ""))
    center = cfg.get("center_url", "http://localhost:8080")

    req = urllib.request.Request(
        f"{center}/api/agents/{args.agent_id}",
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )

    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read())
            msg = body.get("message", str(e))
        except Exception:
            msg = str(e)
        print(f"\n  {_P}ERROR{_R}: {msg}\n")
        return 1
    except urllib.error.URLError:
        print(f"\n  {_P}ERROR{_R}: Cannot connect to center at {center}.\n"
              f"  Start it: {_W}vimin-core start-center{_R}\n")
        return 1

    agent = data.get("agent_info") or {}
    metrics = data.get("latest_metrics") or {}
    summary = data.get("task_summary") or {}

    _banner(
        "vimin-core  ·  Agent Detail",
        [
            ("Agent ID", agent.get("agent_id", args.agent_id)),
            ("Host", agent.get("system_info", {}).get("hostname", "unknown")),
            ("Status", agent.get("status", "unknown")),
            ("Platform", agent.get("system_info", {}).get("platform", "unknown")),
        ],
    )

    print(f"  {_D}first seen:{_R} {_W}{agent.get('first_seen_at', 'unknown')}{_R}")
    print(f"  {_D}last heartbeat:{_R} {_W}{agent.get('last_heartbeat', 'unknown')}{_R}")
    print(f"  {_D}loaded model:{_R} {_W}{agent.get('loaded_model_id') or 'none'}{_R}")
    if agent.get("revoked_at"):
        print(f"  {_D}revoked at:{_R} {_W}{agent['revoked_at']}{_R}")
    print(
        f"  {_D}tasks:{_R} "
        f"{_W}{summary.get('received_total', 0)} total{_R} / "
        f"{_W}{summary.get('queued', 0)} queued{_R} / "
        f"{_W}{summary.get('completed', 0)} completed{_R} / "
        f"{_W}{summary.get('failed', 0)} failed{_R}"
    )
    if metrics:
        print(
            f"  {_D}latest metrics:{_R} "
            f"CPU {_W}{metrics.get('cpu_usage_percent', 0):.1f}%{_R} / "
            f"RAM {_W}{metrics.get('memory_usage_percent', 0):.1f}%{_R} / "
            f"avg latency {_W}{metrics.get('avg_latency_ms', 0):.1f} ms{_R}"
        )
    print()
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="vimin-core",
        description="vimin-core — source-available local inference orchestration (up to 10 nodes)",
    )
    sub = parser.add_subparsers(dest="command")

    # start-center
    sc = sub.add_parser("start-center", help="Start the orchestration center node")
    sc.add_argument("--host", default="127.0.0.1",
                    help="Interface to bind (default: 127.0.0.1 — localhost only). "
                         "Pass 0.0.0.0 to accept connections from other machines.")
    sc.add_argument("--port", type=int, default=8080)
    sc.add_argument("--foreground", action="store_true",
                    help="Run in the foreground (default: daemon)")
    sc.add_argument("--debug", action="store_true")

    # start-agent
    sa = sub.add_parser("start-agent", help="Start a local inference agent node")
    sa.add_argument("--center", default=os.environ.get("VIMIN_CENTER_URL", "http://localhost:8080"))
    sa.add_argument("--model", default=None, metavar="MODEL_ID",
                    help="Pre-load this model on startup instead of waiting for the first task")
    sa.add_argument("--foreground", action="store_true",
                    help="Run in the foreground (default: daemon)")
    sa.add_argument("--debug", action="store_true")
    sa.add_argument("--openclaw", action="store_true",
                    help="Use a local OpenClaw Gateway as the inference engine")
    sa.add_argument("--openclaw-url", default=None,
                    help="OpenClaw Gateway URL (default: http://127.0.0.1:18789)")
    sa.add_argument("--agent-id", default=None, dest="agent_id",
                    help=argparse.SUPPRESS)  # internal: daemon parent pins the UUID

    # broadcast
    bc = sub.add_parser("broadcast", help="Send a prompt to all connected agents")
    bc.add_argument("prompt", help="The prompt to broadcast")
    bc.add_argument("--model", default=None, help=f"Model ID (default: {_DEFAULT_MODEL})")
    bc.add_argument("--max-tokens", type=int, default=300, dest="max_tokens")
    bc.add_argument(
        "--mode", choices=["return", "broadcast"], default=None,
        help=(
            "return — agents send results back to center (default). "
            "broadcast — agents save results locally on the edge device."
        ),
    )
    bc.add_argument("--output", default=None, metavar="FILE",
                    help="Save full JSON response to this file")
    bc.add_argument("--timeout", type=float, default=None, metavar="SECONDS",
                    help="How long to wait for agent results (default: 60s)")

    # run-pipeline
    rp = sub.add_parser("run-pipeline", help="Run a multi-step pipeline on the fleet")
    rp.add_argument("--pipeline", default=None, metavar="FILE",
                    help="Path to a pipeline JSON file")
    rp.add_argument("--preset", default=None, metavar="NAME",
                    help="Built-in preset pipeline name (e.g. summarize-and-questions)")
    rp.add_argument("--file", default=None, metavar="FILE",
                    help="Input file — content replaces {{input}} in pipeline steps")
    rp.add_argument("--input", default=None, metavar="TEXT",
                    help="Input text — replaces {{input}} in pipeline steps")
    rp.add_argument("--model", default=None, help=f"Default model ID (default: {_DEFAULT_MODEL})")
    rp.add_argument(
        "--mode", choices=["return", "broadcast"], default=None,
        help=(
            "return — agents send results back to center (default). "
            "broadcast — agents save results locally on the edge device."
        ),
    )
    rp.add_argument("--output", default=None, metavar="FILE",
                    help="Save full JSON response to this file")

    # stop-center / stop-agent
    sub.add_parser("stop-center", help="Stop a daemonized center node")
    sub.add_parser("stop-agent",  help="Stop all daemonized agent nodes")
    sub.add_parser("clear-tasks", help="Clear queued tasks from the center node")
    sub.add_parser("list-agents", help="List enrolled agents")
    sa_detail = sub.add_parser("show-agent", help="Show one agent in detail")
    sa_detail.add_argument("agent_id", help="Agent ID to inspect")
    ra = sub.add_parser("revoke-agent", help="Revoke an agent and clear its queued work")
    ra.add_argument("agent_id", help="Agent ID to revoke")

    args = parser.parse_args()

    if args.command == "start-center":
        return _cmd_start_center(args)
    elif args.command == "start-agent":
        return _cmd_start_agent(args)
    elif args.command == "broadcast":
        return _cmd_broadcast(args)
    elif args.command == "run-pipeline":
        return _cmd_run_pipeline(args)
    elif args.command == "stop-center":
        return _cmd_stop_center(args)
    elif args.command == "stop-agent":
        return _cmd_stop_agent(args)
    elif args.command == "clear-tasks":
        return _cmd_clear_tasks(args)
    elif args.command == "list-agents":
        return _cmd_list_agents(args)
    elif args.command == "show-agent":
        return _cmd_show_agent(args)
    elif args.command == "revoke-agent":
        return _cmd_revoke_agent(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
