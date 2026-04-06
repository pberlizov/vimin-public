"""
Workflow: Local RAG (Retrieval-Augmented Generation)
======================================================
Builds a simple keyword-based index from a folder of local documents, retrieves
the most relevant chunks for a query, then sends them to the fleet for answer
generation. Everything runs on-device — no cloud embedding API, no vector
database service, no data leaving your network.

The retrieval step uses TF-IDF scoring (scipy/sklearn) if available, falling
back to simple word-overlap scoring so it works even with no extra dependencies.

Useful for: internal knowledge bases, local wikis, document archives, research
paper collections, or any corpus that must not be sent to external services.

Requirements:
    pip install vimin-core[mlx]
    # Optional (better retrieval):
    pip install scikit-learn

Usage:
    # Index a folder and ask a question
    python examples/workflow_local_rag.py --dir docs/ --query "What is our refund policy?"

    # Index markdown files
    python examples/workflow_local_rag.py --dir wiki/ --glob "*.md" \\
        --query "How do I set up the development environment?"

    # Adjust how many chunks to retrieve
    python examples/workflow_local_rag.py --dir docs/ --query "..." --top-k 5

    # Use a different model for generation
    python examples/workflow_local_rag.py --dir docs/ --query "..." \\
        --model Qwen/Qwen3-8B
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from pathlib import Path

CENTER_URL = os.environ.get("VIMIN_CENTER_URL", "http://localhost:8080")
API_KEY    = os.environ.get("ORCHESTRATOR_API_KEY", "")
MODEL      = "meta-llama/Llama-3.1-8B-Instruct"
CHUNK_SIZE = 400   # words per chunk
CHUNK_OVERLAP = 50  # word overlap between adjacent chunks

RAG_PROMPT = """\
Answer the question below using ONLY the provided context. If the answer \
cannot be found in the context, say so clearly — do not guess.

Question:
{query}

Context (retrieved from local documents):
---
{context}
---

Answer:"""


# ---------------------------------------------------------------------------
# Document loading & chunking
# ---------------------------------------------------------------------------

def load_documents(directory: Path, glob: str) -> list[tuple[str, str]]:
    """Return list of (source_name, text) for every file matching the glob."""
    docs = []
    for path in sorted(directory.glob(glob)):
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                docs.append((str(path.relative_to(directory)), text))
        except Exception as e:
            print(f"  WARNING: Could not read {path}: {e}", file=sys.stderr)
    return docs


def chunk_text(source: str, text: str, chunk_size: int, overlap: int) -> list[dict]:
    """Split text into overlapping word windows."""
    words = text.split()
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size]
        if len(chunk_words) < 10:
            continue
        chunks.append({
            "source": source,
            "start": i,
            "text": " ".join(chunk_words),
        })
    return chunks


def build_index(docs: list[tuple[str, str]], chunk_size: int, overlap: int) -> list[dict]:
    chunks = []
    for source, text in docs:
        chunks.extend(chunk_text(source, text, chunk_size, overlap))
    return chunks


# ---------------------------------------------------------------------------
# Retrieval — TF-IDF preferred, word-overlap fallback
# ---------------------------------------------------------------------------

def retrieve_tfidf(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    corpus = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in top_indices if scores[i] > 0]


def retrieve_overlap(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    """Simple word-overlap scorer — no extra deps needed."""
    query_words = set(re.findall(r"\w+", query.lower()))
    scored = []
    for c in chunks:
        chunk_words = set(re.findall(r"\w+", c["text"].lower()))
        overlap = len(query_words & chunk_words)
        if overlap > 0:
            scored.append((overlap, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]


def retrieve(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    try:
        return retrieve_tfidf(query, chunks, top_k)
    except ImportError:
        return retrieve_overlap(query, chunks, top_k)


def format_context(retrieved: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(retrieved, 1):
        parts.append(f"[{i}] Source: {chunk['source']}\n{chunk['text']}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Fleet broadcast
# ---------------------------------------------------------------------------

async def broadcast_rag(prompt: str, center_url: str, api_key: str, model: str, max_tokens: int) -> list:
    try:
        import aiohttp
    except ImportError:
        print("aiohttp not installed. Run: pip install aiohttp")
        sys.exit(1)

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"prompt": prompt, "model_id": model, "max_tokens": max_tokens}

    print(f"Sending to fleet at {center_url} ...")
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Local RAG: retrieve from docs, generate on fleet")
    parser.add_argument("--dir", required=True, help="Directory containing source documents")
    parser.add_argument("--glob", default="*.txt", help="File pattern to index (default: *.txt)")
    parser.add_argument("--query", help="Question to answer")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--center", default=CENTER_URL)
    parser.add_argument("--api-key", default=API_KEY)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--show-context", action="store_true",
                        help="Print retrieved chunks before sending to fleet")
    args = parser.parse_args()

    if args.query:
        query = args.query
    elif not sys.stdin.isatty():
        query = sys.stdin.read().strip()
    else:
        parser.print_help()
        sys.exit(1)

    directory = Path(args.dir)
    if not directory.is_dir():
        print(f"ERROR: {args.dir} is not a directory.")
        sys.exit(1)

    print(f"\n=== Local RAG ===")
    print(f"  Directory : {args.dir} (glob: {args.glob})")
    print(f"  Query     : {query}")
    print(f"  Model     : {args.model}\n")

    # Build index
    print("  Loading documents ...")
    docs = load_documents(directory, args.glob)
    if not docs:
        print(f"  No documents found matching {args.glob} in {args.dir}")
        sys.exit(1)
    print(f"  Loaded {len(docs)} documents")

    chunks = build_index(docs, args.chunk_size, CHUNK_OVERLAP)
    print(f"  Built index: {len(chunks)} chunks\n")

    # Retrieve
    retrieved = retrieve(query, chunks, args.top_k)
    if not retrieved:
        print("  No relevant chunks found.")
        sys.exit(0)

    print(f"  Retrieved {len(retrieved)} chunk(s) from: {', '.join(set(c['source'] for c in retrieved))}")

    if args.show_context:
        print("\n── Retrieved Context ──")
        print(format_context(retrieved))
        print("──────────────────────\n")

    # Generate
    context = format_context(retrieved)
    prompt = RAG_PROMPT.format(query=query, context=context)

    results = asyncio.run(broadcast_rag(prompt, args.center, args.api_key, args.model, args.max_tokens))

    print(f"\n=== Answers from {len(results)} node(s) ===\n")
    for r in results:
        print(f"── Node: {r.get('agent_id', 'unknown')} ──")
        print((r.get("output") or "(timeout/no output)").strip())
        print()


if __name__ == "__main__":
    main()
