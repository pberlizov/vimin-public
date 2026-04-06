"""
Workflow: Parallel Multi-Language Translation / Localisation
=============================================================
Broadcasts a piece of text to all connected nodes simultaneously.
Each node translates or localises the content independently.

In broadcast mode, all nodes receive the same prompt. To get translations
into different languages, either:
  (a) Configure each agent with a different system prompt locally, or
  (b) Use this script which requests multiple languages in a single prompt
      and collects independent renderings from each node.

Useful for: marketing copy, product descriptions, support articles,
legal notices — any content that needs rapid multi-language output
without a cloud translation API.

Requirements:
    pip install vimin-core[mlx]

Usage:
    python examples/workflow_multi_language.py --text "Your product description here."
    python examples/workflow_multi_language.py --file copy.txt --languages "French,German,Japanese,Spanish"
"""

import argparse
import asyncio
import os
import sys

CENTER_URL = os.environ.get("VIMIN_CENTER_URL", "http://localhost:8080")
API_KEY    = os.environ.get("ORCHESTRATOR_API_KEY", "")
MODEL      = "Qwen/Qwen3-8B"  # Strong multilingual model

TRANSLATION_PROMPT = """\
Translate the following text into {languages}.

Return each translation labelled clearly, e.g.:
**French:** ...
**German:** ...

Preserve the tone and intent of the original. Do not add commentary.

Original text:
{text}
"""


async def broadcast(text: str, languages: str, center_url: str, api_key: str, model: str) -> list:
    try:
        import aiohttp
    except ImportError:
        print("aiohttp not installed.")
        sys.exit(1)

    prompt = TRANSLATION_PROMPT.format(languages=languages, text=text)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"prompt": prompt, "model_id": model, "max_tokens": 800}

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
    parser = argparse.ArgumentParser(description="Parallel multi-language translation")
    parser.add_argument("--text", help="Text to translate")
    parser.add_argument("--file", help="File containing text to translate")
    parser.add_argument("--languages", default="French, German, Spanish, Japanese, Portuguese",
                        help="Comma-separated list of target languages")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--center", default=CENTER_URL)
    parser.add_argument("--api-key", default=API_KEY)
    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            text = f.read().strip()
    elif args.text:
        text = args.text
    elif not sys.stdin.isatty():
        text = sys.stdin.read().strip()
    else:
        parser.print_help()
        sys.exit(1)

    print(f"\n=== Multi-Language Translation ===")
    print(f"  Languages : {args.languages}")
    print(f"  Model     : {args.model}")
    print(f"  Text      : {text[:120]}{'...' if len(text) > 120 else ''}\n")

    results = asyncio.run(broadcast(text, args.languages, args.center, args.api_key, args.model))

    print(f"\n=== Translations from {len(results)} node(s) ===\n")
    for r in results:
        print(f"── Node: {r.get('agent_id', 'unknown')} ──")
        print((r.get("output") or "(timeout/no output)").strip())
        print()


if __name__ == "__main__":
    main()
