"""
Workflow: On-device Voice Transcription + Analysis
===================================================
Records audio from the microphone, transcribes it locally via Whisper,
then broadcasts the transcript to all connected nodes for analysis.

Each node returns its own analysis — useful for getting independent
perspectives, summaries in different styles, or parallel workloads.

Requirements:
    pip install vimin-core[mlx,whisper] sounddevice numpy

Usage:
    python examples/workflow_voice_transcription.py
    python examples/workflow_voice_transcription.py --seconds 10 --center http://192.168.1.10:8080
"""

import argparse
import asyncio
import json
import os
import sys
import numpy as np

CENTER_URL = os.environ.get("VIMIN_CENTER_URL", "http://localhost:8080")
API_KEY    = os.environ.get("ORCHESTRATOR_API_KEY", "")
WHISPER_MODEL = "openai/whisper-base"
TEXT_MODEL    = "meta-llama/Llama-3.2-3B-Instruct"


def record_audio(seconds: int = 5, sample_rate: int = 16000) -> np.ndarray:
    """Record from the default microphone and return a float32 array."""
    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice not installed. Run: pip install sounddevice")
        sys.exit(1)

    print(f"  Recording {seconds}s of audio...")
    audio = sd.rec(int(seconds * sample_rate), samplerate=sample_rate,
                   channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()


def transcribe_locally(audio: np.ndarray, model_id: str = WHISPER_MODEL) -> str:
    """Transcribe audio using the local WhisperBackend."""
    from vimin_core.core.backends.whisper_backend import WhisperBackend
    from vimin_core.core.backends.base import ModelDescriptor

    backend = WhisperBackend()
    desc = ModelDescriptor(model_id=model_id, task="automatic-speech-recognition")
    if not backend.load(desc):
        print("  ERROR: Whisper backend failed to load.")
        sys.exit(1)

    print(f"  Transcribing with {model_id}...")
    result = backend.transcribe(audio, language="en")
    backend.unload()
    text = result.get("text", "").strip()
    print(f"  Transcript: {text!r}")
    return text


async def broadcast_for_analysis(transcript: str, center_url: str, api_key: str) -> list:
    """Broadcast the transcript to all nodes and collect analyses."""
    try:
        import aiohttp
    except ImportError:
        print("aiohttp not installed. Run: pip install aiohttp")
        sys.exit(1)

    prompt = (
        "The following is a transcription of a spoken recording. "
        "Provide a concise analysis: key topics, any action items, and a one-sentence summary.\n\n"
        f"Transcript:\n{transcript}"
    )

    payload = {
        "prompt": prompt,
        "model_id": TEXT_MODEL,
        "max_tokens": 200,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    print(f"\n  Broadcasting to {center_url}/api/broadcast ...")
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{center_url}/api/broadcast",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                print(f"  ERROR {resp.status}: {text[:200]}")
                return []
            return (await resp.json()).get("results", [])


def main():
    parser = argparse.ArgumentParser(description="Voice transcription + fleet analysis")
    parser.add_argument("--seconds", type=int, default=5, help="Recording duration")
    parser.add_argument("--center", default=CENTER_URL)
    parser.add_argument("--api-key", default=API_KEY)
    parser.add_argument("--whisper", default=WHISPER_MODEL)
    parser.add_argument("--model", default=TEXT_MODEL)
    args = parser.parse_args()

    global TEXT_MODEL
    TEXT_MODEL = args.model

    print("=== Voice Transcription + Fleet Analysis ===\n")
    audio = record_audio(seconds=args.seconds)
    transcript = transcribe_locally(audio, model_id=args.whisper)

    if not transcript:
        print("  No speech detected. Exiting.")
        return

    results = asyncio.run(broadcast_for_analysis(transcript, args.center, args.api_key))

    print(f"\n=== Results from {len(results)} node(s) ===\n")
    for r in results:
        print(f"[{r.get('agent_id', 'unknown')}]")
        print(r.get("output", "(no output)"))
        print()


if __name__ == "__main__":
    main()
