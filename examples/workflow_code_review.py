"""
Workflow: Parallel Code Review
================================
Broadcasts a code snippet or file to all connected nodes for review.
Each node independently analyses the code and returns findings.
Aggregated output gives a broad view of bugs, style issues, and risks.

Works well with code-focused models like Qwen2.5-Coder or Devstral.

Requirements:
    pip install vimin-core[mlx]

Usage:
    python examples/workflow_code_review.py --file src/main.py
    python examples/workflow_code_review.py --file src/main.py --model Qwen/Qwen2.5-Coder-7B-Instruct
    git diff HEAD~1 | python examples/workflow_code_review.py
"""

import argparse
import asyncio
import os
import sys

CENTER_URL = os.environ.get("VIMIN_CENTER_URL", "http://localhost:8080")
API_KEY    = os.environ.get("ORCHESTRATOR_API_KEY", "")
MODEL      = "Qwen/Qwen2.5-Coder-7B-Instruct"

REVIEW_PROMPT = """\
Review the following code. Be concise. Cover:
1. Bugs or logic errors (if any)
2. Security concerns (injection, auth, secrets, etc.)
3. Performance issues
4. Style / readability suggestions
5. Overall verdict: Ready / Needs work / Reject

Code:
```
{code}
```
"""


async def broadcast(code: str, center_url: str, api_key: str, model: str) -> list:
    try:
        import aiohttp
    except ImportError:
        print("aiohttp not installed.")
        sys.exit(1)

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "prompt": REVIEW_PROMPT.format(code=code[:8000]),
        "model_id": model,
        "max_tokens": 500,
    }

    print(f"Sending to {center_url} for review ...")
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{center_url}/api/broadcast",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=180),
        ) as resp:
            if resp.status != 200:
                print(f"ERROR {resp.status}: {(await resp.text())[:200]}")
                return []
            return (await resp.json()).get("results", [])


def main():
    parser = argparse.ArgumentParser(description="Broadcast code to all nodes for review")
    parser.add_argument("--file", help="Source file to review")
    parser.add_argument("--center", default=CENTER_URL)
    parser.add_argument("--api-key", default=API_KEY)
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            code = f.read()
    elif not sys.stdin.isatty():
        code = sys.stdin.read()
    else:
        parser.print_help()
        sys.exit(1)

    print(f"\n=== Code Review — {len(code.splitlines())} lines ===\n")
    results = asyncio.run(broadcast(code, args.center, args.api_key, args.model))

    print(f"\n=== Reviews from {len(results)} node(s) ===\n")
    for r in results:
        print(f"── Node: {r.get('agent_id', 'unknown')} ──")
        print((r.get("output") or "(timeout/no output)").strip())
        print()


if __name__ == "__main__":
    main()
