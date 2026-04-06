"""
Workflow: Customer Support Ticket Triage
=========================================
Sends a batch of support tickets (or a single ticket) to all connected
nodes for parallel classification and prioritisation. Each node
independently labels the ticket, assigns a priority, and suggests a
routing department. Useful for building a local, private support queue
without sending customer data to a cloud API.

Each node produces an independent classification — you can use the
majority vote or the most detailed response depending on your use case.

Requirements:
    pip install vimin-core[mlx]

Usage:
    # Single ticket via stdin
    echo "My payment failed but I was charged twice." | \\
        python examples/workflow_support_triage.py

    # Batch: one ticket per line in a file
    python examples/workflow_support_triage.py --batch tickets.txt

    # Save triage results as JSON
    python examples/workflow_support_triage.py --batch tickets.txt --output triage.json
"""

import argparse
import asyncio
import json
import os
import sys

CENTER_URL = os.environ.get("VIMIN_CENTER_URL", "http://localhost:8080")
API_KEY    = os.environ.get("ORCHESTRATOR_API_KEY", "")
MODEL      = "meta-llama/Llama-3.2-3B-Instruct"

TRIAGE_PROMPT = """\
You are a support ticket classifier. Analyse the ticket below and return a \
JSON object (and nothing else) with these fields:

{{
  "category": "<one of: Billing, Technical, Account, Shipping, General>",
  "priority": "<one of: Critical, High, Medium, Low>",
  "department": "<team that should handle this>",
  "sentiment": "<Positive | Neutral | Frustrated | Angry>",
  "summary": "<one sentence summary of the issue>",
  "suggested_action": "<brief recommended first response or action>"
}}

Ticket:
{ticket}
"""


async def triage_ticket(
    session,
    ticket_id: str,
    ticket: str,
    center_url: str,
    api_key: str,
    model: str,
) -> dict:
    import aiohttp

    prompt = TRIAGE_PROMPT.format(ticket=ticket[:3000])
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"prompt": prompt, "model_id": model, "max_tokens": 200}

    try:
        async with session.post(
            f"{center_url}/api/broadcast",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            if resp.status != 200:
                return {"id": ticket_id, "ticket": ticket, "error": f"HTTP {resp.status}", "results": []}
            data = await resp.json()
            return {"id": ticket_id, "ticket": ticket, "results": data.get("results", [])}
    except Exception as e:
        return {"id": ticket_id, "ticket": ticket, "error": str(e), "results": []}


def parse_triage(output: str) -> dict:
    """Try to extract the JSON block from the model output."""
    import re
    match = re.search(r'\{.*\}', output, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"raw": output.strip()}


async def run_batch(
    tickets: list[tuple[str, str]],
    center_url: str,
    api_key: str,
    model: str,
    concurrency: int = 4,
) -> list[dict]:
    try:
        import aiohttp
    except ImportError:
        print("aiohttp not installed. Run: pip install aiohttp")
        sys.exit(1)

    semaphore = asyncio.Semaphore(concurrency)

    async def guarded(ticket_id, ticket):
        async with semaphore:
            print(f"  → Ticket {ticket_id} ...")
            return await triage_ticket(session, ticket_id, ticket, center_url, api_key, model)

    async with aiohttp.ClientSession() as session:
        tasks = [guarded(tid, t) for tid, t in tickets]
        return await asyncio.gather(*tasks)


def main():
    parser = argparse.ArgumentParser(description="Parallel support ticket triage")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--batch", help="File with one ticket per line")
    parser.add_argument("--output", help="Write JSON results to this file")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--center", default=CENTER_URL)
    parser.add_argument("--api-key", default=API_KEY)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    if args.batch:
        with open(args.batch) as f:
            lines = [l.strip() for l in f if l.strip()]
        tickets = [(str(i + 1), line) for i, line in enumerate(lines)]
    elif not sys.stdin.isatty():
        text = sys.stdin.read().strip()
        tickets = [("1", text)]
    else:
        parser.print_help()
        sys.exit(1)

    print(f"\n=== Support Ticket Triage ===")
    print(f"  Tickets : {len(tickets)}")
    print(f"  Model   : {args.model}\n")

    all_results = asyncio.run(
        run_batch(tickets, args.center, args.api_key, args.model, args.concurrency)
    )

    print(f"\n=== Triage Results ===\n")
    for entry in all_results:
        print(f"── Ticket #{entry['id']} ──")
        print(f"  \"{entry['ticket'][:80]}{'...' if len(entry['ticket']) > 80 else ''}\"")
        if "error" in entry:
            print(f"  ERROR: {entry['error']}")
        else:
            for r in entry["results"]:
                triage = parse_triage(r.get("output", ""))
                node = r.get("agent_id", "?")
                if "raw" in triage:
                    print(f"  [Node {node}] {triage['raw'][:120]}")
                else:
                    print(f"  [Node {node}] {triage.get('priority','?')} priority | "
                          f"{triage.get('category','?')} | "
                          f"{triage.get('sentiment','?')} | "
                          f"{triage.get('department','?')}")
                    print(f"    → {triage.get('suggested_action', '')}")
        print()

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
