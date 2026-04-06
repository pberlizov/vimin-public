"""
Workflow: Competitive Research Synthesis
=========================================
Takes a set of text snippets (competitor announcements, product pages,
press releases, earnings call transcripts, etc.) and sends them to the
fleet for parallel analysis from different strategic lenses:

  • Node A → Feature / capability comparison
  • Node B → Pricing and business model signals
  • Node C → Weaknesses and risks
  • Node D → Strategic positioning and messaging

In vimin-core broadcast mode, all nodes receive the same prompt. Configure
each agent's system prompt for a specialised persona, or use this script
which embeds multiple analytical perspectives in a single prompt.

Requirements:
    pip install vimin-core[mlx]

Usage:
    # Analyse text passed directly
    python examples/workflow_competitive_research.py --text "Acme Corp today announced..."

    # From a file (press release, article, etc.)
    python examples/workflow_competitive_research.py --file competitor_release.txt

    # Combine multiple files into one analysis
    python examples/workflow_competitive_research.py --files a.txt b.txt c.txt
"""

import argparse
import asyncio
import os
import sys

CENTER_URL = os.environ.get("VIMIN_CENTER_URL", "http://localhost:8080")
API_KEY    = os.environ.get("ORCHESTRATOR_API_KEY", "")
MODEL      = "meta-llama/Llama-3.1-8B-Instruct"

RESEARCH_PROMPT = """\
You are a strategic analyst. Analyse the following competitive intelligence \
text and provide a structured report covering:

## 1. Key Capabilities / Features Announced
What new products, features, or capabilities are described? Be specific.

## 2. Business Model & Pricing Signals
Any pricing, packaging, target market, or monetisation information. Infer \
from context if not stated explicitly.

## 3. Weaknesses or Risks
Gaps, limitations, technical debt, regulatory risks, or anything that could \
be a competitive vulnerability.

## 4. Strategic Positioning
How is the competitor positioning itself? What narrative or messaging are \
they pushing? Who are they targeting?

## 5. Key Takeaways
3 bullet points: what matters most from a competitive standpoint.

Source material:
{text}
"""


async def broadcast_research(text: str, center_url: str, api_key: str, model: str) -> list:
    try:
        import aiohttp
    except ImportError:
        print("aiohttp not installed. Run: pip install aiohttp")
        sys.exit(1)

    prompt = RESEARCH_PROMPT.format(text=text[:8000])
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"prompt": prompt, "model_id": model, "max_tokens": 700}

    print(f"Sending to {center_url} for analysis ...")
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{center_url}/api/broadcast",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=240),
        ) as resp:
            if resp.status != 200:
                print(f"ERROR {resp.status}: {(await resp.text())[:200]}")
                return []
            return (await resp.json()).get("results", [])


def main():
    parser = argparse.ArgumentParser(description="Parallel competitive research synthesis")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", help="Text to analyse")
    source.add_argument("--file", help="Single file to analyse")
    source.add_argument("--files", nargs="+", help="Multiple files to combine and analyse")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--center", default=CENTER_URL)
    parser.add_argument("--api-key", default=API_KEY)
    args = parser.parse_args()

    if args.text:
        text = args.text
    elif args.file:
        with open(args.file) as f:
            text = f.read().strip()
    elif args.files:
        parts = []
        for path in args.files:
            with open(path) as f:
                parts.append(f"=== {path} ===\n{f.read().strip()}")
        text = "\n\n".join(parts)
    elif not sys.stdin.isatty():
        text = sys.stdin.read().strip()
    else:
        parser.print_help()
        sys.exit(1)

    print(f"\n=== Competitive Research Synthesis ===")
    print(f"  Model : {args.model}")
    print(f"  Input : {len(text)} chars\n")

    results = asyncio.run(broadcast_research(text, args.center, args.api_key, args.model))

    print(f"\n=== Analysis from {len(results)} node(s) ===\n")
    for r in results:
        print(f"── Node: {r.get('agent_id', 'unknown')} ──")
        print((r.get("output") or "(timeout/no output)").strip())
        print()


if __name__ == "__main__":
    main()
