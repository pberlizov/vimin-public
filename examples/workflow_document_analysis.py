"""
Workflow: Parallel Document Analysis
=====================================
Broadcasts a document to all connected nodes simultaneously. Each node
analyses the document from a different angle defined by a system prompt.
The center node collects all results and prints a consolidated report.

This is useful when you want multiple independent reads of the same text —
e.g. legal review, financial analysis, and risk scoring in one pass.

Requirements:
    pip install vimin-core[mlx]

Usage:
    # Analyse a file
    python examples/workflow_document_analysis.py --file report.txt

    # Pipe text directly
    echo "Q3 revenue grew 12%..." | python examples/workflow_document_analysis.py
"""

import argparse
import asyncio
import json
import os
import sys

CENTER_URL = os.environ.get("VIMIN_CENTER_URL", "http://localhost:8080")
API_KEY    = os.environ.get("ORCHESTRATOR_API_KEY", "")
MODEL      = "meta-llama/Llama-3.2-3B-Instruct"

# Each node receives the same document but a different analytical lens.
# In vimin-core, all nodes get the same prompt (broadcast-only).
# Configure each node's system prompt / persona via its local model setup,
# or use the prompt prefix below to embed multiple perspectives in one call.
ANALYSIS_PROMPT = """\
You are an expert analyst. Read the following document and provide:
1. A 2-sentence executive summary
2. Key facts or figures (bullet points)
3. Any risks, concerns, or action items
4. A sentiment rating: Positive / Neutral / Negative

Document:
{document}
"""


async def broadcast(document: str, center_url: str, api_key: str, model: str) -> list:
    try:
        import aiohttp
    except ImportError:
        print("aiohttp not installed. Run: pip install aiohttp")
        sys.exit(1)

    prompt = ANALYSIS_PROMPT.format(document=document[:6000])  # ~4K tokens

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"prompt": prompt, "model_id": model, "max_tokens": 400}

    print(f"Broadcasting to {center_url} ...")
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
    parser = argparse.ArgumentParser(description="Broadcast a document to all nodes for analysis")
    parser.add_argument("--file", help="Path to text file to analyse")
    parser.add_argument("--center", default=CENTER_URL)
    parser.add_argument("--api-key", default=API_KEY)
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            document = f.read()
    elif not sys.stdin.isatty():
        document = sys.stdin.read()
    else:
        print("Provide a document via --file or stdin.")
        parser.print_help()
        sys.exit(1)

    print(f"\n=== Document Analysis — {len(document)} chars ===\n")
    results = asyncio.run(broadcast(document, args.center, args.api_key, args.model))

    print(f"\n=== Results from {len(results)} node(s) ===\n")
    for r in results:
        print(f"── Node: {r.get('agent_id', 'unknown')} ──")
        print((r.get("output") or "(timeout/no output)").strip())
        print()


if __name__ == "__main__":
    main()
