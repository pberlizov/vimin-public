"""
Workflow: On-device PII Detection and Redaction
================================================
Sends text to all connected nodes for parallel PII (Personally Identifiable
Information) detection. Each node independently identifies and redacts
sensitive data — names, emails, phone numbers, addresses, SSNs, etc.
The center node collects results; you pick which redaction to use or
take a consensus view.

Because everything runs locally, no data leaves your network. This is
useful for pre-processing documents before uploading to external services,
sanitising logs, or complying with GDPR / HIPAA data-minimisation requirements.

Requirements:
    pip install vimin-core[mlx]

Usage:
    # Pipe text directly
    echo "Contact John Smith at john@example.com or call 555-123-4567." | \\
        python examples/workflow_pii_redaction.py

    # From a file
    python examples/workflow_pii_redaction.py --file customer_email.txt

    # Write redacted output to a file
    python examples/workflow_pii_redaction.py --file raw.txt --output redacted.txt
"""

import argparse
import asyncio
import os
import sys

CENTER_URL = os.environ.get("VIMIN_CENTER_URL", "http://localhost:8080")
API_KEY    = os.environ.get("ORCHESTRATOR_API_KEY", "")
MODEL      = "meta-llama/Llama-3.1-8B-Instruct"

REDACTION_PROMPT = """\
You are a privacy compliance tool. Your task is to detect and redact all PII \
(Personally Identifiable Information) from the text below.

Replace each piece of PII with a labelled placeholder in square brackets, for example:
  - Person names → [NAME]
  - Email addresses → [EMAIL]
  - Phone numbers → [PHONE]
  - Physical addresses → [ADDRESS]
  - Social Security / national ID numbers → [ID_NUMBER]
  - Credit card / bank account numbers → [FINANCIAL_ID]
  - Dates of birth → [DATE_OF_BIRTH]
  - IP addresses → [IP_ADDRESS]
  - Any other unique identifiers → [PII]

Return ONLY the redacted text. Do not add commentary, explanations, or a preamble.

Text to redact:
{text}
"""


async def broadcast_redaction(text: str, center_url: str, api_key: str, model: str) -> list:
    try:
        import aiohttp
    except ImportError:
        print("aiohttp not installed. Run: pip install aiohttp")
        sys.exit(1)

    prompt = REDACTION_PROMPT.format(text=text[:8000])
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"prompt": prompt, "model_id": model, "max_tokens": len(text.split()) + 200}

    print(f"Broadcasting to {center_url} for redaction ...")
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


def pick_consensus(results: list) -> str:
    """
    Simple consensus: return the result from the node that produced the
    most redactions (most conservative / safest from a privacy standpoint).
    """
    if not results:
        return ""
    valid = [r for r in results if r.get("output")]
    if not valid:
        return ""
    return max(valid, key=lambda r: r["output"].count("["))["output"].strip()


def main():
    parser = argparse.ArgumentParser(description="On-device PII redaction via fleet broadcast")
    parser.add_argument("--file", help="Text file to redact")
    parser.add_argument("--output", help="Write redacted text to this file")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--center", default=CENTER_URL)
    parser.add_argument("--api-key", default=API_KEY)
    parser.add_argument("--show-all", action="store_true",
                        help="Show all node outputs instead of consensus only")
    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            text = f.read().strip()
    elif not sys.stdin.isatty():
        text = sys.stdin.read().strip()
    else:
        parser.print_help()
        sys.exit(1)

    print(f"\n=== PII Redaction ===")
    print(f"  Model  : {args.model}")
    print(f"  Length : {len(text)} chars\n")

    results = asyncio.run(broadcast_redaction(text, args.center, args.api_key, args.model))

    if not results:
        print("No results returned.")
        sys.exit(1)

    if args.show_all:
        print(f"\n=== All Node Outputs ({len(results)} node(s)) ===\n")
        for r in results:
            print(f"── Node: {r.get('agent_id', 'unknown')} ──")
            print((r.get("output") or "(timeout/no output)").strip())
            print()
    else:
        redacted = pick_consensus(results)
        print(f"\n=== Redacted Output (consensus from {len(results)} node(s)) ===\n")
        print(redacted)
        print()
        if args.output:
            with open(args.output, "w") as f:
                f.write(redacted)
            print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
