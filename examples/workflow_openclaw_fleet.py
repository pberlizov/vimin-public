"""
Workflow: OpenClaw-backed Fleet Inference
==========================================
Demonstrates using OpenClaw as the inference engine for vimin-core agents.

When a node is started with `vimin-core start-agent --openclaw`, it routes
all broadcast tasks through the local OpenClaw Gateway instead of loading
model weights itself. OpenClaw manages model selection, quantisation, and
hardware scheduling — vimin-core handles fleet coordination.

This script does three things:
  1. Checks which models are available through your local OpenClaw Gateway
  2. Sends a broadcast query to the fleet (works with any backend mix —
     OpenClaw nodes, MLX nodes, and llama-cpp nodes can all coexist)
  3. Optionally runs a direct single-model query against OpenClaw without
     needing a running fleet at all

For agent-to-agent coordination, department routing, and multi-turn
orchestration, see the full vimin distribution.

Requirements:
    pip install vimin-core[mlx]   # for non-OpenClaw nodes in the fleet
    OpenClaw installed and running on at least one agent node

Usage:
    # List models available in your local OpenClaw gateway
    python examples/workflow_openclaw_fleet.py --list-models

    # Run a broadcast query against the fleet (nodes may use any backend)
    python examples/workflow_openclaw_fleet.py --prompt "Explain neural scaling laws."

    # Query OpenClaw directly on this machine (no fleet needed)
    python examples/workflow_openclaw_fleet.py --local --prompt "Summarise the CAP theorem."

    # Specify which OpenClaw model to use for a direct query
    python examples/workflow_openclaw_fleet.py --local --oc-model my-model --prompt "..."
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import sys
import urllib.error
import urllib.request

CENTER_URL    = os.environ.get("VIMIN_CENTER_URL", "http://localhost:8080")
API_KEY       = os.environ.get("ORCHESTRATOR_API_KEY", "")
OPENCLAW_URL  = os.environ.get("OPENCLAW_URL", "http://127.0.0.1:18789")
_TOKEN_PATH   = pathlib.Path.home() / ".openclaw" / "openclaw.json"


# ---------------------------------------------------------------------------
# OpenClaw Gateway helpers (no third-party deps — uses stdlib urllib)
# ---------------------------------------------------------------------------

def _load_openclaw_token() -> str:
    try:
        data = json.loads(_TOKEN_PATH.read_text())
        return data["gateway"]["auth"]["token"]
    except Exception:
        return os.environ.get("OPENCLAW_TOKEN", "")


def _oc_headers(token: str) -> dict:
    h: dict = {"Content-Type": "application/json", "Accept": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def openclaw_list_models(url: str, token: str) -> list[str]:
    req = urllib.request.Request(f"{url}/v1/models", headers=_oc_headers(token))
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            return [m["id"] for m in data.get("data", [])]
    except Exception as e:
        print(f"  ERROR: Could not reach OpenClaw at {url}: {e}")
        return []


def openclaw_generate(
    url: str,
    token: str,
    model: str,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    stream: bool = True,
) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }).encode()

    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=body,
        headers=_oc_headers(token),
        method="POST",
    )

    output_parts: list[str] = []
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            if stream:
                for raw_line in resp:
                    line = raw_line.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                        token_text = chunk["choices"][0].get("delta", {}).get("content", "")
                        if token_text:
                            print(token_text, end="", flush=True)
                            output_parts.append(token_text)
                    except (KeyError, json.JSONDecodeError):
                        continue
                print()  # newline after streaming
            else:
                result = json.loads(resp.read())
                text = result["choices"][0]["message"]["content"]
                print(text)
                return text
    except urllib.error.HTTPError as e:
        print(f"\n  ERROR {e.code}: {e.read().decode(errors='replace')[:200]}")
        sys.exit(1)

    return "".join(output_parts)


# ---------------------------------------------------------------------------
# Fleet broadcast (same API as all other workflow scripts)
# ---------------------------------------------------------------------------

async def fleet_broadcast(prompt: str, center_url: str, api_key: str, max_tokens: int) -> list:
    try:
        import aiohttp
    except ImportError:
        print("aiohttp not installed. Run: pip install aiohttp")
        sys.exit(1)

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"prompt": prompt, "max_tokens": max_tokens}

    print(f"Broadcasting to fleet at {center_url} ...")
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{center_url}/api/broadcast",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            if resp.status != 200:
                print(f"ERROR {resp.status}: {(await resp.text())[:200]}")
                return []
            return (await resp.json()).get("results", [])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="OpenClaw-backed fleet inference demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--list-models", action="store_true",
                        help="List models available in the local OpenClaw Gateway and exit")
    parser.add_argument("--local", action="store_true",
                        help="Query OpenClaw directly on this machine (no fleet needed)")
    parser.add_argument("--prompt", help="Prompt to send")
    parser.add_argument("--oc-model", default="",
                        help="OpenClaw model name for --local mode (default: first available)")
    parser.add_argument("--oc-url", default=OPENCLAW_URL,
                        help=f"OpenClaw Gateway URL (default: {OPENCLAW_URL})")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--center", default=CENTER_URL)
    parser.add_argument("--api-key", default=API_KEY)
    args = parser.parse_args()

    token = _load_openclaw_token()

    # ── List models ──────────────────────────────────────────────────────────
    if args.list_models:
        print(f"\n=== OpenClaw Models at {args.oc_url} ===\n")
        models = openclaw_list_models(args.oc_url, token)
        if models:
            for m in models:
                print(f"  • {m}")
        else:
            print("  (no models found — is OpenClaw running?)")
            print(f"\n  Start the gateway:  openclaw gateway start")
            print(f"  Start a vimin agent: vimin-core start-agent --openclaw")
        return

    # ── Require a prompt for anything else ───────────────────────────────────
    if not args.prompt:
        if not sys.stdin.isatty():
            args.prompt = sys.stdin.read().strip()
        else:
            parser.print_help()
            sys.exit(1)

    # ── Local OpenClaw query (single machine, no fleet) ──────────────────────
    if args.local:
        models = openclaw_list_models(args.oc_url, token)
        if not models:
            print(f"ERROR: OpenClaw gateway not reachable at {args.oc_url}")
            print("  Ensure it is running: openclaw gateway start")
            sys.exit(1)

        model = args.oc_model or models[0]
        print(f"\n=== Local OpenClaw Inference ===")
        print(f"  Gateway : {args.oc_url}")
        print(f"  Model   : {model}")
        print(f"  Prompt  : {args.prompt[:80]}{'...' if len(args.prompt) > 80 else ''}\n")
        print("─" * 60)
        openclaw_generate(
            url=args.oc_url,
            token=token,
            model=model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
        )
        print("─" * 60)
        print(f"\nTip: to use OpenClaw for all nodes in your fleet, start agents with:")
        print(f"     vimin-core start-agent --openclaw --openclaw-url {args.oc_url}")
        return

    # ── Fleet broadcast (nodes may use any backend, including OpenClaw) ───────
    print(f"\n=== Fleet Broadcast (OpenClaw nodes welcome) ===")
    print(f"  Prompt : {args.prompt[:80]}{'...' if len(args.prompt) > 80 else ''}\n")
    print("Nodes running `vimin-core start-agent --openclaw` will use OpenClaw")
    print("for inference; other nodes use their configured MLX or llama-cpp backend.\n")

    results = asyncio.run(fleet_broadcast(args.prompt, args.center, args.api_key, args.max_tokens))

    print(f"\n=== Results from {len(results)} node(s) ===\n")
    for r in results:
        print(f"── Node: {r.get('agent_id', 'unknown')} ──")
        print((r.get("output") or "(timeout/no output)").strip())
        print()


if __name__ == "__main__":
    main()
