"""
Workflow: Meeting Transcription → Minutes + Action Items
=========================================================
Records a meeting (or loads an audio file), transcribes it locally
via Whisper, then sends the transcript to the fleet for parallel extraction:
  • Node A → Executive summary
  • Node B → Action items with owners (if names are mentioned)
  • Node C → Key decisions made
  • Node D → Follow-up questions or unresolved points

In vimin-core (broadcast mode), all nodes receive the same prompt.
Configure each agent's system prompt locally for specialised output,
or run this script which sends a single comprehensive extraction prompt.

Requirements:
    pip install vimin-core[mlx,whisper] sounddevice numpy

Usage:
    # Record live
    python examples/workflow_meeting_minutes.py --seconds 300

    # From an audio file (WAV/MP3/etc — requires ffmpeg)
    python examples/workflow_meeting_minutes.py --audio meeting.wav

    # From an existing transcript
    python examples/workflow_meeting_minutes.py --transcript transcript.txt
"""

import argparse
import asyncio
import os
import sys
import numpy as np

CENTER_URL    = os.environ.get("VIMIN_CENTER_URL", "http://localhost:8080")
API_KEY       = os.environ.get("ORCHESTRATOR_API_KEY", "")
WHISPER_MODEL = "openai/whisper-small"
TEXT_MODEL    = "meta-llama/Llama-3.1-8B-Instruct"

MINUTES_PROMPT = """\
You are a professional meeting secretary. Given the following meeting transcript, extract:

## Summary
2-3 sentences capturing what the meeting was about.

## Key Decisions
Bullet list of decisions made during the meeting.

## Action Items
Bullet list of tasks, each formatted as: [Owner if mentioned] — Task — Deadline if mentioned

## Open Questions
Any questions raised that were not resolved.

Transcript:
{transcript}
"""


def load_audio_file(path: str) -> np.ndarray:
    """Load an audio file and return float32 array at 16 kHz."""
    try:
        import librosa
        audio, _ = librosa.load(path, sr=16000, mono=True)
        return audio.astype(np.float32)
    except ImportError:
        pass
    # Fallback: use ffmpeg via subprocess
    import subprocess, tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    subprocess.run(
        ["ffmpeg", "-i", path, "-ar", "16000", "-ac", "1", "-f", "wav", tmp_path, "-y", "-loglevel", "quiet"],
        check=True,
    )
    with open(tmp_path, "rb") as f:
        import wave
        with wave.open(f) as wf:
            frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    os.unlink(tmp_path)
    return audio


def record_audio(seconds: int, sample_rate: int = 16000) -> np.ndarray:
    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice not installed. Run: pip install sounddevice")
        sys.exit(1)
    print(f"  Recording {seconds}s ... (Ctrl+C to stop early)")
    try:
        audio = sd.rec(int(seconds * sample_rate), samplerate=sample_rate,
                       channels=1, dtype="float32")
        sd.wait()
    except KeyboardInterrupt:
        sd.stop()
        print("  Recording stopped early.")
    return audio.flatten()


def transcribe(audio: np.ndarray, model_id: str) -> str:
    from vimin_core.core.backends.whisper_backend import WhisperBackend
    from vimin_core.core.backends.base import ModelDescriptor

    backend = WhisperBackend()
    desc = ModelDescriptor(model_id=model_id, task="automatic-speech-recognition")
    backend.load(desc)
    print(f"  Transcribing with {model_id} ...")
    result = backend.transcribe(audio, language="en")
    backend.unload()
    return result.get("text", "").strip()


async def broadcast_for_minutes(transcript: str, center_url: str, api_key: str, model: str) -> list:
    try:
        import aiohttp
    except ImportError:
        print("aiohttp not installed.")
        sys.exit(1)

    prompt = MINUTES_PROMPT.format(transcript=transcript[:10000])
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"prompt": prompt, "model_id": model, "max_tokens": 600}

    print(f"  Sending transcript to {center_url} ...")
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{center_url}/api/broadcast",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            if resp.status != 200:
                print(f"  ERROR {resp.status}: {(await resp.text())[:200]}")
                return []
            return (await resp.json()).get("results", [])


def main():
    parser = argparse.ArgumentParser(description="Meeting transcription and minutes")
    parser.add_argument("--seconds", type=int, default=60, help="Recording duration in seconds")
    parser.add_argument("--audio", help="Path to audio file instead of recording")
    parser.add_argument("--transcript", help="Path to existing transcript text file")
    parser.add_argument("--whisper", default=WHISPER_MODEL)
    parser.add_argument("--model", default=TEXT_MODEL)
    parser.add_argument("--center", default=CENTER_URL)
    parser.add_argument("--api-key", default=API_KEY)
    args = parser.parse_args()

    print("=== Meeting Minutes Generator ===\n")

    if args.transcript:
        with open(args.transcript) as f:
            transcript = f.read().strip()
        print(f"  Loaded transcript: {len(transcript)} chars")
    elif args.audio:
        print(f"  Loading audio: {args.audio}")
        audio = load_audio_file(args.audio)
        transcript = transcribe(audio, args.whisper)
    else:
        audio = record_audio(seconds=args.seconds)
        transcript = transcribe(audio, args.whisper)

    if not transcript:
        print("  Empty transcript — nothing to process.")
        return

    print(f"\n  Transcript ({len(transcript)} chars):\n  {transcript[:200]}...\n")

    results = asyncio.run(broadcast_for_minutes(transcript, args.center, args.api_key, args.model))

    print(f"\n=== Minutes from {len(results)} node(s) ===\n")
    for r in results:
        print(f"── Node: {r.get('agent_id', 'unknown')} ──")
        print((r.get("output") or "(timeout/no output)").strip())
        print()


if __name__ == "__main__":
    main()
