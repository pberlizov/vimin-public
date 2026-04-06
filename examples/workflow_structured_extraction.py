"""
Workflow: Structured Data Extraction
======================================
Extracts structured fields from unstructured text (invoices, contracts,
job postings, product listings, medical notes, etc.) using the fleet.
Each node returns a JSON object; the script validates and merges results,
preferring the most complete response.

Useful for building local data pipelines that process sensitive documents
without sending them to a cloud API.

Requirements:
    pip install vimin-core[mlx]

Usage:
    # Extract from a single invoice
    python examples/workflow_structured_extraction.py \\
        --file invoice.txt --schema invoice

    # Extract from a job posting
    python examples/workflow_structured_extraction.py \\
        --file job_post.txt --schema job

    # Use a custom schema (JSON file describing the fields)
    python examples/workflow_structured_extraction.py \\
        --file contract.txt --schema-file my_schema.json

    # Batch: process a folder
    python examples/workflow_structured_extraction.py \\
        --dir invoices/ --schema invoice --output extracted.jsonl
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

CENTER_URL = os.environ.get("VIMIN_CENTER_URL", "http://localhost:8080")
API_KEY    = os.environ.get("ORCHESTRATOR_API_KEY", "")
MODEL      = "Qwen/Qwen2.5-7B-Instruct"

# Pre-built schemas for common document types
BUILT_IN_SCHEMAS = {
    "invoice": {
        "vendor_name": "string",
        "invoice_number": "string",
        "invoice_date": "YYYY-MM-DD string",
        "due_date": "YYYY-MM-DD string",
        "total_amount": "number",
        "currency": "3-letter ISO code",
        "line_items": "array of {description, quantity, unit_price, total}",
        "payment_terms": "string",
        "billing_address": "string",
    },
    "job": {
        "job_title": "string",
        "company": "string",
        "location": "string or 'Remote'",
        "salary_range": "string or null",
        "employment_type": "Full-time | Part-time | Contract | Internship",
        "experience_required": "string (e.g. '3-5 years')",
        "key_skills": "array of strings",
        "application_deadline": "YYYY-MM-DD or null",
    },
    "contract": {
        "parties": "array of party names",
        "effective_date": "YYYY-MM-DD string",
        "expiry_date": "YYYY-MM-DD or null",
        "governing_law": "jurisdiction string",
        "payment_terms": "string or null",
        "termination_notice": "string (e.g. '30 days')",
        "key_obligations": "array of brief strings",
    },
    "receipt": {
        "merchant": "string",
        "date": "YYYY-MM-DD string",
        "total": "number",
        "currency": "3-letter ISO code",
        "items": "array of {description, price}",
        "payment_method": "string",
    },
}

EXTRACTION_PROMPT = """\
Extract structured data from the document below. Return ONLY valid JSON \
matching this schema (use null for missing fields):

{schema_json}

Document:
{text}
"""


def build_prompt(text: str, schema: dict) -> str:
    schema_json = json.dumps(schema, indent=2)
    return EXTRACTION_PROMPT.format(schema_json=schema_json, text=text[:7000])


def parse_json_output(output: str) -> dict | None:
    import re
    # Try direct parse first
    try:
        return json.loads(output.strip())
    except json.JSONDecodeError:
        pass
    # Look for a JSON block
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', output, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r'\{.*\}', output, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def best_result(results: list) -> dict | None:
    """Return the parsed result with the most non-null fields."""
    best = None
    best_count = -1
    for r in results:
        parsed = parse_json_output(r.get("output") or "")
        if parsed:
            non_null = sum(1 for v in parsed.values() if v is not None)
            if non_null > best_count:
                best = parsed
                best_count = non_null
    return best


async def extract_one(
    session,
    filename: str,
    text: str,
    schema: dict,
    center_url: str,
    api_key: str,
    model: str,
) -> dict:
    import aiohttp

    prompt = build_prompt(text, schema)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"prompt": prompt, "model_id": model, "max_tokens": 400}

    async with session.post(
        f"{center_url}/api/broadcast",
        json=payload,
        headers=headers,
        timeout=aiohttp.ClientTimeout(total=180),
    ) as resp:
        if resp.status != 200:
            return {"file": filename, "error": f"HTTP {resp.status}", "extracted": None}
        results = (await resp.json()).get("results", [])
        extracted = best_result(results)
        return {"file": filename, "extracted": extracted, "raw_results": len(results)}


async def run_batch(
    docs: list[tuple[str, str]],
    schema: dict,
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

    async def guarded(filename, text):
        async with semaphore:
            print(f"  → {filename} ...")
            return await extract_one(session, filename, text, schema, center_url, api_key, model)

    async with aiohttp.ClientSession() as session:
        return await asyncio.gather(*[guarded(fn, t) for fn, t in docs])


def main():
    parser = argparse.ArgumentParser(description="Parallel structured data extraction")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--file", help="Single document to extract from")
    source.add_argument("--dir", help="Directory of documents to process")
    parser.add_argument("--glob", default="*.txt", help="File glob when using --dir")
    schema_group = parser.add_mutually_exclusive_group(required=True)
    schema_group.add_argument("--schema",
                              choices=list(BUILT_IN_SCHEMAS.keys()),
                              help="Built-in document schema")
    schema_group.add_argument("--schema-file", help="Path to custom JSON schema file")
    parser.add_argument("--output", help="Write results to JSONL file")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--center", default=CENTER_URL)
    parser.add_argument("--api-key", default=API_KEY)
    args = parser.parse_args()

    # Load schema
    if args.schema:
        schema = BUILT_IN_SCHEMAS[args.schema]
        schema_name = args.schema
    else:
        with open(args.schema_file) as f:
            schema = json.load(f)
        schema_name = Path(args.schema_file).stem

    # Load documents
    if args.file:
        with open(args.file) as f:
            docs = [(Path(args.file).name, f.read().strip())]
    elif args.dir:
        paths = sorted(Path(args.dir).glob(args.glob))
        docs = [(p.name, p.read_text(errors="replace").strip()) for p in paths]
    elif not sys.stdin.isatty():
        docs = [("stdin", sys.stdin.read().strip())]
    else:
        parser.print_help()
        sys.exit(1)

    print(f"\n=== Structured Extraction — {schema_name} schema ===")
    print(f"  Documents : {len(docs)}")
    print(f"  Model     : {args.model}\n")

    results = asyncio.run(
        run_batch(docs, schema, args.center, args.api_key, args.model)
    )

    print(f"\n=== Results ===\n")
    output_lines = []
    for entry in results:
        print(f"── {entry['file']} ──")
        if "error" in entry:
            print(f"  ERROR: {entry['error']}")
        elif entry["extracted"]:
            print(json.dumps(entry["extracted"], indent=2))
            output_lines.append(json.dumps({"file": entry["file"], **entry["extracted"]}))
        else:
            print("  (Could not parse structured output)")
        print()

    if args.output and output_lines:
        with open(args.output, "w") as f:
            f.write("\n".join(output_lines) + "\n")
        print(f"Written {len(output_lines)} records to {args.output}")


if __name__ == "__main__":
    main()
