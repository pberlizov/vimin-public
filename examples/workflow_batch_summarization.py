"""
Workflow: Batch Document Summarization
=======================================
Reads a folder of text files (or a list of URLs / file paths) and
distributes them across all connected nodes for parallel summarization.
Each node processes one document at a time; the center node collects
results and writes a summary report.

Useful for: processing research papers, log batches, news articles,
customer feedback dumps, or any large set of documents that would take
too long to summarise sequentially on a single machine.

Requirements:
    pip install vimin-core[mlx]

Usage:
    # Summarise every .txt file in a folder
    python examples/workflow_batch_summarization.py --dir docs/

    # Summarise a specific list of files
    python examples/workflow_batch_summarization.py --files a.txt b.txt c.txt

    # Write results to a JSON report
    python examples/workflow_batch_summarization.py --dir docs/ --output report.json
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

CENTER_URL = os.environ.get("VIMIN_CENTER_URL", "http://localhost:8080")
API_KEY    = os.environ.get("ORCHESTRATOR_API_KEY", "")
MODEL      = "meta-llama/Llama-3.1-8B-Instruct"

SUMMARY_PROMPT = """\
Summarise the following document in 3–5 bullet points. Be concise and factual.
Capture the main topic, key findings or arguments, and any notable conclusions.

Document title: {title}

Document:
{text}
"""


async def summarise_one(
    session,
    title: str,
    text: str,
    center_url: str,
    api_key: str,
    model: str,
) -> dict:
    """Send a single document to the fleet and return the aggregated result."""
    import aiohttp

    prompt = SUMMARY_PROMPT.format(title=title, text=text[:6000])
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"prompt": prompt, "model_id": model, "max_tokens": 300}

    async with session.post(
        f"{center_url}/api/broadcast",
        json=payload,
        headers=headers,
        timeout=aiohttp.ClientTimeout(total=180),
    ) as resp:
        if resp.status != 200:
            return {"title": title, "error": f"HTTP {resp.status}", "results": []}
        data = await resp.json()
        return {"title": title, "results": data.get("results", [])}


async def batch_summarise(
    documents: list[tuple[str, str]],  # [(title, text), ...]
    center_url: str,
    api_key: str,
    model: str,
    concurrency: int = 4,
) -> list[dict]:
    """
    Process documents with bounded concurrency so we don't flood the fleet
    with more simultaneous broadcasts than it can handle gracefully.
    """
    try:
        import aiohttp
    except ImportError:
        print("aiohttp not installed. Run: pip install aiohttp")
        sys.exit(1)

    semaphore = asyncio.Semaphore(concurrency)
    results = []

    async def guarded(title, text):
        async with semaphore:
            print(f"  → {title} ...")
            return await summarise_one(session, title, text, center_url, api_key, model)

    async with aiohttp.ClientSession() as session:
        tasks = [guarded(title, text) for title, text in documents]
        results = await asyncio.gather(*tasks)

    return list(results)


def load_documents(paths: list[Path]) -> list[tuple[str, str]]:
    docs = []
    for p in paths:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
            docs.append((p.name, text.strip()))
        except Exception as e:
            print(f"  WARNING: Could not read {p}: {e}")
    return docs


def main():
    parser = argparse.ArgumentParser(description="Batch parallel document summarization")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dir", help="Directory of text files to summarise")
    source.add_argument("--files", nargs="+", help="Specific files to summarise")
    parser.add_argument("--glob", default="*.txt", help="File pattern when using --dir (default: *.txt)")
    parser.add_argument("--output", help="Write JSON report to this file")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--center", default=CENTER_URL)
    parser.add_argument("--api-key", default=API_KEY)
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Max simultaneous broadcasts (default: 4)")
    args = parser.parse_args()

    if args.dir:
        paths = sorted(Path(args.dir).glob(args.glob))
    else:
        paths = [Path(f) for f in args.files]

    if not paths:
        print("No files found.")
        sys.exit(1)

    print(f"\n=== Batch Summarization ===")
    print(f"  Files     : {len(paths)}")
    print(f"  Model     : {args.model}")
    print(f"  Concurrency: {args.concurrency} simultaneous broadcasts\n")

    documents = load_documents(paths)
    if not documents:
        print("No readable documents.")
        sys.exit(1)

    all_results = asyncio.run(
        batch_summarise(documents, args.center, args.api_key, args.model, args.concurrency)
    )

    print(f"\n=== Summaries ({len(all_results)} documents) ===\n")
    for entry in all_results:
        print(f"── {entry['title']} ──")
        if "error" in entry:
            print(f"  ERROR: {entry['error']}")
        else:
            for r in entry["results"]:
                print(f"  [Node {r.get('agent_id', '?')}]")
                print(f"  {(r.get('output') or '(timeout/no output)').strip()}")
        print()

    if args.output:
        Path(args.output).write_text(json.dumps(all_results, indent=2))
        print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
