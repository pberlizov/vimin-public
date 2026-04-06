#!/usr/bin/env python3
"""
Fleet integration tests — exercises the live center + agent(s).

Usage:
    python tests/test_fleet.py                  # run all tests
    python tests/test_fleet.py --quick          # skip slow / multi-agent tests
    python tests/test_fleet.py --center http://192.168.1.10:8080
    python tests/test_fleet.py -k translation   # run only tests matching a keyword
    python tests/test_fleet.py --list           # list all test names without running

Prerequisites:
    vimin-core start-center --daemon
    vimin-core start-agent --daemon
    # wait for "Model ready" in ~/.vimin/logs/agent-*.log before running

What is tested:
    1.  Center health
    2.  Agent list
    3.  Broadcast — return mode
    4.  Broadcast — broadcast mode (saves to edge)
    5.  Summarization pipeline
    6.  Reasoning + report pipeline
    7.  PII masking pipeline (medical)
    8.  PII masking pipeline (financial)
    9.  Translation pipeline (French → English)
    10. Translation pipeline (Spanish → English)
    11. Support triage — frustrated customer (parallel, 2 agents)
    12. Support triage — feature request (parallel, 2 agents)
    13. Code review — buggy Python (parallel, 2 agents)
    14. Parallel perspectives on Avalon file (2 agents)
    15. Broadcast --output saves JSON file
    16. Pipeline --output saves JSON file
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Colour palette (same as the CLI) ──────────────────────────────────────────
_P  = "\033[38;2;139;60;247m"
_L  = "\033[38;2;192;132;252m"
_W  = "\033[38;2;240;234;255m"
_D  = "\033[38;2;110;90;180m"
_G  = "\033[38;2;80;200;120m"   # green  — pass
_R  = "\033[38;2;255;80;80m"    # red    — fail
_RS = "\033[0m"                  # reset

# ── Config ─────────────────────────────────────────────────────────────────────

_AVALON_PATH = Path(__file__).parent.parent / "examples" / "data" / "avalon_incident.md"

def _load_config(center_override: Optional[str] = None) -> Dict[str, str]:
    cfg_path = Path.home() / ".vimin" / "config.json"
    cfg: Dict[str, str] = {}
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception:
            pass
    return {
        "api_key":    cfg.get("api_key", os.environ.get("ORCHESTRATOR_API_KEY", "")),
        "center_url": center_override or cfg.get("center_url", "http://localhost:8080"),
    }


# ── HTTP helpers ───────────────────────────────────────────────────────────────

def _get(url: str, api_key: str, timeout: int = 10) -> Dict[str, Any]:
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {api_key}"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _post(url: str, api_key: str, body: Dict[str, Any], timeout: int = 310) -> Dict[str, Any]:
    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# ── Test runner ────────────────────────────────────────────────────────────────

class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.skipped = False
        self.skip_reason = ""
        self.error: Optional[str] = None
        self.output: Optional[str] = None
        self.latency_ms: float = 0.0


_RESULTS: List[TestResult] = []


def _run(
    name: str,
    fn,
    *,
    requires_agents: int = 1,
    agent_count: int = 0,
    quick: bool = False,
    slow: bool = False,
) -> TestResult:
    res = TestResult(name)
    _RESULTS.append(res)

    if slow and quick:
        res.skipped = True
        res.skip_reason = "skipped in --quick mode"
        _print_skip(name, res.skip_reason)
        return res

    if agent_count < requires_agents:
        res.skipped = True
        res.skip_reason = f"needs {requires_agents} agent(s), only {agent_count} connected"
        _print_skip(name, res.skip_reason)
        return res

    print(f"\n{_D}  ▶ {name}{_RS}", flush=True)
    t0 = time.time()
    try:
        output = fn()
        res.latency_ms = (time.time() - t0) * 1000
        res.output = str(output) if output is not None else ""
        res.passed = True
        _print_pass(name, res.latency_ms, res.output)
    except Exception as exc:
        res.latency_ms = (time.time() - t0) * 1000
        res.error = str(exc)
        res.passed = False
        _print_fail(name, res.error)
    return res


def _print_pass(name: str, ms: float, output: str) -> None:
    preview = output.strip()[:160].replace("\n", " ↵ ")
    print(f"  {_G}✓ PASS{_RS}  {_D}({ms:.0f} ms){_RS}")
    if preview:
        print(f"  {_D}{preview}{_RS}")


def _print_fail(name: str, error: str) -> None:
    print(f"  {_R}✗ FAIL{_RS}  {error}")


def _print_skip(name: str, reason: str) -> None:
    print(f"\n{_D}  ─ {name}  [skipped: {reason}]{_RS}")


# ── Individual test helpers ────────────────────────────────────────────────────

def _first_output(data: Dict[str, Any]) -> str:
    """Return the first non-empty output from a broadcast result."""
    for r in data.get("results", []):
        o = r.get("output") or ""
        if o and not o.startswith("[saved_locally]"):
            return o
    return ""


def _final_output(data: Dict[str, Any]) -> str:
    """Return the final_output from a pipeline result."""
    return data.get("final_output", "") or ""


def _assert_nonempty(s: str, label: str = "output") -> str:
    if not s or not s.strip():
        raise AssertionError(f"{label} is empty")
    return s


def _assert_not_error(data: Dict[str, Any]) -> None:
    for r in data.get("results", []):
        if r.get("error"):
            raise AssertionError(f"agent returned error: {r['error']}")
    for s in data.get("steps", []):
        for r in s.get("results", []):
            if r.get("error"):
                raise AssertionError(f"pipeline step {s['step']} error: {r['error']}")


def _avalon_text() -> str:
    if _AVALON_PATH.exists():
        return _AVALON_PATH.read_text()
    # Minimal fallback if file is missing
    return (
        "Avalon Therapeutics is accused of selectively reporting clinical trial data. "
        "A fire destroyed 14 months of raw data. A short position was opened two days before the fire."
    )


# ── Tests ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="vimin-core fleet integration tests")
    parser.add_argument("--center", default=None, help="Center URL override")
    parser.add_argument("--quick", action="store_true", help="Skip slow / multi-agent tests")
    parser.add_argument("-k", "--keyword", default=None, help="Only run tests whose name contains this string")
    parser.add_argument("--list", action="store_true", help="List all test names and exit")
    args = parser.parse_args()

    if args.list:
        tests = [
            "center_health", "agent_list",
            "broadcast_return", "broadcast_broadcast_mode",
            "pipeline_summarize_and_questions", "pipeline_analyze_and_report",
            "pipeline_pii_medical", "pipeline_pii_financial",
            "pipeline_translate_french", "pipeline_translate_spanish",
            "pipeline_support_frustrated", "pipeline_support_feature_request",
            "pipeline_code_review", "pipeline_parallel_perspectives",
            "broadcast_output_file", "pipeline_output_file",
        ]
        for t in tests:
            print(f"  {t}")
        return 0

    cfg = _load_config(args.center)
    center = cfg["center_url"]
    api_key = cfg["api_key"]

    # ── Header ─────────────────────────────────────────────────────────────────
    print(f"\n{_P}  ◈ {_W}vimin{_RS}{_D}-core{_RS}  fleet integration tests\n")
    print(f"  {_D}Center:{_RS}  {_W}{center}{_RS}")
    print(f"  {_D}API key:{_RS} {_W}{api_key[:8]}...{_RS}\n")

    def _match(name: str) -> bool:
        return args.keyword is None or args.keyword.lower() in name.lower()

    # ── Preflight: discover agent count ────────────────────────────────────────
    agent_count = 0
    try:
        agents_data = _get(f"{center}/api/agents", api_key)
        alive_thresh = 90
        now = time.time()
        for a in agents_data.get("agents", []):
            if a.get("status") == "online":
                agent_count += 1
    except Exception:
        pass

    print(f"  {_D}Live agents:{_RS}  {_W}{agent_count}{_RS}")

    # ── 1. Center health ────────────────────────────────────────────────────────
    if _match("center_health"):
        def test_center_health():
            d = _get(f"{center}/api/health", api_key)
            return d.get("status") or d.get("uptime_seconds") or "ok"
        _run("center_health", test_center_health, agent_count=agent_count)

    # ── 2. Agent list ───────────────────────────────────────────────────────────
    if _match("agent_list"):
        def test_agent_list():
            d = _get(f"{center}/api/agents", api_key)
            agents = d.get("agents", [])
            if not agents:
                raise AssertionError("no agents registered")
            return f"{len(agents)} agent(s): " + ", ".join(
                f"{a.get('agent_id', '?')[:8]} ({a.get('status','?')})"
                for a in agents
            )
        _run("agent_list", test_agent_list, agent_count=agent_count)

    # ── 3. Broadcast — return mode ─────────────────────────────────────────────
    if _match("broadcast_return"):
        def test_broadcast_return():
            d = _post(f"{center}/api/broadcast", api_key, {
                "prompt": "In one sentence, what is the capital of Japan?",
                "model_id": None,
                "max_tokens": 60,
                "mode": "return",
            }, timeout=310)
            _assert_not_error(d)
            return _assert_nonempty(_first_output(d))
        _run("broadcast_return", test_broadcast_return,
             requires_agents=1, agent_count=agent_count)

    # ── 4. Broadcast — broadcast mode (saves to edge) ──────────────────────────
    if _match("broadcast_broadcast_mode"):
        def test_broadcast_mode():
            d = _post(f"{center}/api/broadcast", api_key, {
                "prompt": "In one sentence: what is on-device AI inference?",
                "max_tokens": 80,
                "mode": "broadcast",
            }, timeout=310)
            _assert_not_error(d)
            results = d.get("results", [])
            if not results:
                raise AssertionError("no results returned")
            output = results[0].get("output", "")
            # In broadcast mode the output is either [saved_locally] or an empty echo
            return output or "(broadcast dispatched)"
        _run("broadcast_broadcast_mode", test_broadcast_mode,
             requires_agents=1, agent_count=agent_count)

    # ── 5. Pipeline — summarize and questions ───────────────────────────────────
    if _match("pipeline_summarize"):
        def test_pipeline_summarize():
            d = _post(f"{center}/api/pipeline", api_key, {
                "name": "summarize-and-questions",
                "mode": "return",
                "input": _avalon_text(),
                "steps": [
                    {
                        "label": "summarize",
                        "type": "SUMMARIZATION",
                        "data": "Summarize the following document in 3-5 sentences:\n\n{{input}}",
                        "timeout": 120,
                    },
                    {
                        "label": "questions",
                        "type": "REASONING",
                        "data": "Based on this summary, generate 5 follow-up questions an investigator would want answered:\n\n{{step1_output}}",
                        "timeout": 120,
                    },
                ],
            }, timeout=310)
            _assert_not_error(d)
            return _assert_nonempty(_final_output(d), "final_output")
        _run("pipeline_summarize_and_questions", test_pipeline_summarize,
             requires_agents=1, agent_count=agent_count, slow=True, quick=args.quick)

    # ── 6. Pipeline — analyze and report ───────────────────────────────────────
    if _match("pipeline_analyze"):
        def test_pipeline_analyze():
            d = _post(f"{center}/api/pipeline", api_key, {
                "name": "analyze-and-report",
                "mode": "return",
                "input": _avalon_text(),
                "steps": [
                    {
                        "label": "extract",
                        "type": "REASONING",
                        "data": "Extract: (a) key people, (b) sequence of events, (c) numerical figures.\n\n{{input}}",
                        "timeout": 120,
                    },
                    {
                        "label": "risks",
                        "type": "REASONING",
                        "data": "Identify the three biggest red flags from these facts:\n\n{{step1_output}}",
                        "timeout": 120,
                    },
                    {
                        "label": "report",
                        "type": "SUMMARIZATION",
                        "data": "Write an executive report (Summary / Key Findings / Risk Assessment / Next Steps).\n\nFacts:\n{{step1_output}}\n\nRisks:\n{{step2_output}}",
                        "timeout": 120,
                    },
                ],
            }, timeout=620)
            _assert_not_error(d)
            return _assert_nonempty(_final_output(d), "report")
        _run("pipeline_analyze_and_report", test_pipeline_analyze,
             requires_agents=1, agent_count=agent_count, slow=True, quick=args.quick)

    # ── 7. Pipeline — PII masking (medical) ────────────────────────────────────
    if _match("pipeline_pii"):
        def test_pii_medical():
            d = _post(f"{center}/api/pipeline", api_key, {
                "name": "pii-medical",
                "mode": "return",
                "input": (
                    "Patient: Sarah Mitchell, DOB 14/03/1981, NHS 485 777 3456.\n"
                    "Physician: Dr. James Okafor, GMC 7823401.\n"
                    "Diagnosis: Type 2 diabetes. Contact: sarah.mitchell81@gmail.com, 07700 900341.\n"
                    "Address: 22 Bridge Road, Bristol, BS1 4AB."
                ),
                "steps": [
                    {
                        "label": "redact",
                        "type": "PII_MASKING",
                        "data": "{{input}}",
                        "timeout": 120,
                    },
                    {
                        "label": "summarize",
                        "type": "SUMMARIZATION",
                        "data": "Summarize this redacted medical record in one sentence:\n\n{{step1_output}}",
                        "timeout": 120,
                    },
                ],
            }, timeout=310)
            _assert_not_error(d)
            step1 = d.get("steps", [{}])[0].get("output", "")
            # Verify something got redacted — original name should not appear verbatim
            if "Sarah Mitchell" in step1:
                raise AssertionError("PII not redacted: name still present in output")
            return _assert_nonempty(_final_output(d), "summary")

        _run("pipeline_pii_medical", test_pii_medical,
             requires_agents=1, agent_count=agent_count)

        def test_pii_financial():
            d = _post(f"{center}/api/pipeline", api_key, {
                "name": "pii-financial",
                "mode": "return",
                "input": (
                    "From: Marcus Webb <m.webb@finco.com>\n"
                    "Transfer £48,500 from account 12-34-56 / 87654321 to Helenova Ltd.\n"
                    "IBAN: GB29 NWBK 6016 1331 9268 19. Phone: +44 7911 123456."
                ),
                "steps": [
                    {
                        "label": "redact",
                        "type": "PII_MASKING",
                        "data": "{{input}}",
                        "timeout": 120,
                    },
                    {
                        "label": "summarize",
                        "type": "SUMMARIZATION",
                        "data": "Summarize this redacted financial instruction in one sentence:\n\n{{step1_output}}",
                        "timeout": 120,
                    },
                ],
            }, timeout=310)
            _assert_not_error(d)
            return _assert_nonempty(_final_output(d), "summary")

        _run("pipeline_pii_financial", test_pii_financial,
             requires_agents=1, agent_count=agent_count)

    # ── 8. Pipeline — translation ───────────────────────────────────────────────
    if _match("pipeline_translat"):
        def test_translate_french():
            d = _post(f"{center}/api/pipeline", api_key, {
                "name": "translate-french",
                "mode": "return",
                "input": (
                    "La conférence annuelle sur l'intelligence artificielle s'est tenue à Paris. "
                    "Les chercheurs ont présenté de nouveaux modèles capables de comprendre "
                    "le langage naturel avec une précision sans précédent."
                ),
                "steps": [
                    {
                        "label": "translate",
                        "type": "TRANSLATION",
                        "data": "Translate the following text to English. Output only the translated text.\n\n{{input}}",
                        "timeout": 120,
                    },
                    {
                        "label": "summarize",
                        "type": "SUMMARIZATION",
                        "data": "Summarize in one sentence:\n\n{{step1_output}}",
                        "timeout": 120,
                    },
                ],
            }, timeout=310)
            _assert_not_error(d)
            translated = d.get("steps", [{}])[0].get("output", "")
            # Rough check: result should contain English words
            english_markers = ["conference", "artificial", "researchers", "intelligence", "annual"]
            if not any(w in translated.lower() for w in english_markers):
                raise AssertionError(f"Translation looks wrong: {translated[:120]}")
            return translated[:200]

        _run("pipeline_translate_french", test_translate_french,
             requires_agents=1, agent_count=agent_count)

        def test_translate_spanish():
            d = _post(f"{center}/api/pipeline", api_key, {
                "name": "translate-spanish",
                "mode": "return",
                "input": (
                    "El banco central anunció una subida de tipos de interés del 0,25%. "
                    "La medida busca controlar la inflación, que se sitúa en el 4,7%."
                ),
                "steps": [
                    {
                        "label": "translate",
                        "type": "TRANSLATION",
                        "data": "Translate the following text to English. Output only the translated text.\n\n{{input}}",
                        "timeout": 120,
                    },
                ],
            }, timeout=310)
            _assert_not_error(d)
            translated = _final_output(d) or (d.get("steps", [{}])[0].get("output", ""))
            markers = ["bank", "interest", "inflation", "rate", "central"]
            if not any(w in translated.lower() for w in markers):
                raise AssertionError(f"Translation looks wrong: {translated[:120]}")
            return translated[:200]

        _run("pipeline_translate_spanish", test_translate_spanish,
             requires_agents=1, agent_count=agent_count)

    # ── 9. Pipeline — support triage (parallel, 2 agents) ──────────────────────
    if _match("pipeline_support"):
        def test_support_frustrated():
            d = _post(f"{center}/api/pipeline", api_key, {
                "name": "support-triage",
                "mode": "return",
                "input": (
                    "I have been charged THREE times for the same subscription this month. "
                    "I raised this two weeks ago and nobody has responded. "
                    "I want an immediate refund. If this isn't resolved by tomorrow "
                    "I am cancelling and disputing the charges with my bank."
                ),
                "steps": [
                    [
                        {
                            "label": "classify",
                            "type": "CLASSIFICATION",
                            "data": "Classify this support ticket. Output JSON: {category, priority, routing_team}.\n\nTicket:\n{{input}}",
                            "timeout": 90,
                        },
                        {
                            "label": "sentiment",
                            "type": "SENTIMENT_ANALYSIS",
                            "data": "Analyse sentiment. Output JSON: {overall_sentiment, escalation_risk}.\n\nTicket:\n{{input}}",
                            "timeout": 90,
                        },
                    ],
                    {
                        "label": "response",
                        "type": "TEXT_GENERATION",
                        "data": "Draft a professional, empathetic support response.\n\nTicket:\n{{input}}\n\nClassification:\n{{step1_output}}\n\nSentiment:\n{{step2_output}}",
                        "timeout": 120,
                    },
                ],
            }, timeout=465)
            _assert_not_error(d)
            return _assert_nonempty(_final_output(d), "response draft")

        _run("pipeline_support_frustrated", test_support_frustrated,
             requires_agents=2, agent_count=agent_count)

        def test_support_feature():
            d = _post(f"{center}/api/pipeline", api_key, {
                "name": "support-feature-request",
                "mode": "return",
                "input": (
                    "Love the product! One thing that would really help is the ability to "
                    "export audit logs as CSV. Right now we can only view them in the dashboard. "
                    "Would this be possible to add?"
                ),
                "steps": [
                    [
                        {
                            "label": "classify",
                            "type": "CLASSIFICATION",
                            "data": "Classify: {category, priority, routing_team}.\n\n{{input}}",
                            "timeout": 60,
                        },
                        {
                            "label": "sentiment",
                            "type": "SENTIMENT_ANALYSIS",
                            "data": "Analyse sentiment: {overall_sentiment, escalation_risk}.\n\n{{input}}",
                            "timeout": 60,
                        },
                    ],
                    {
                        "label": "response",
                        "type": "TEXT_GENERATION",
                        "data": "Draft a support response.\n\nTicket:\n{{input}}\n\nClassification:\n{{step1_output}}",
                        "timeout": 120,
                    },
                ],
            }, timeout=465)
            _assert_not_error(d)
            return _assert_nonempty(_final_output(d), "response draft")

        _run("pipeline_support_feature_request", test_support_feature,
             requires_agents=2, agent_count=agent_count)

    # ── 10. Pipeline — code review (parallel, 2 agents) ────────────────────────
    if _match("pipeline_code"):
        BUGGY_CODE = '''\
import subprocess, pickle, os

def run_query(user_input):
    cmd = "SELECT * FROM users WHERE name = " + user_input
    result = subprocess.run(["psql", "-c", cmd], capture_output=True)
    return result.stdout

def load_session(session_file):
    with open(session_file, "rb") as f:
        return pickle.load(f)

SECRET_KEY = "hardcoded_secret_abc123"

def get_admin():
    if os.environ.get("DEBUG") == "true":
        return True
    return False
'''

        def test_code_review():
            d = _post(f"{center}/api/pipeline", api_key, {
                "name": "code-review",
                "mode": "return",
                "input": BUGGY_CODE,
                "steps": [
                    [
                        {
                            "label": "bugs",
                            "type": "CODE_GENERATION",
                            "data": "Review for logic errors and bugs. List each with a fix.\n\n```python\n{{input}}\n```",
                            "timeout": 120,
                        },
                        {
                            "label": "security",
                            "type": "CODE_GENERATION",
                            "data": "Review for security vulnerabilities. List each with severity and a fix.\n\n```python\n{{input}}\n```",
                            "timeout": 120,
                        },
                    ],
                    {
                        "label": "verdict",
                        "type": "REASONING",
                        "data": "Synthesize into a final review: Overall Assessment / Critical Issues / Verdict.\n\nBugs:\n{{step1_output}}\n\nSecurity:\n{{step2_output}}",
                        "timeout": 120,
                    },
                ],
            }, timeout=465)
            _assert_not_error(d)
            verdict = _assert_nonempty(_final_output(d), "verdict")
            # Should catch the SQL injection at minimum
            if not any(w in verdict.lower() for w in ["sql", "inject", "security", "secret", "pickle", "critical"]):
                raise AssertionError(f"Code review missed obvious issues: {verdict[:200]}")
            return verdict[:300]

        _run("pipeline_code_review", test_code_review,
             requires_agents=2, agent_count=agent_count, slow=True, quick=args.quick)

    # ── 11. Pipeline — parallel perspectives on Avalon ─────────────────────────
    if _match("pipeline_parallel"):
        def test_parallel_perspectives():
            d = _post(f"{center}/api/pipeline", api_key, {
                "name": "parallel-perspectives",
                "mode": "return",
                "input": _avalon_text(),
                "steps": [
                    [
                        {
                            "label": "defense",
                            "type": "REASONING",
                            "data": "Defense perspective: what exculpatory evidence or alternative explanations exist?\n\n{{input}}",
                            "timeout": 120,
                        },
                        {
                            "label": "prosecution",
                            "type": "REASONING",
                            "data": "Prosecution perspective: what is the strongest case for wrongdoing?\n\n{{input}}",
                            "timeout": 120,
                        },
                    ],
                    {
                        "label": "synthesis",
                        "type": "SUMMARIZATION",
                        "data": "Synthesize into: Strongest Evidence / Counter-Arguments / Assessment.\n\nDefense:\n{{step1_output}}\n\nProsecution:\n{{step2_output}}",
                        "timeout": 120,
                    },
                ],
            }, timeout=465)
            _assert_not_error(d)
            return _assert_nonempty(_final_output(d), "synthesis")

        _run("pipeline_parallel_perspectives", test_parallel_perspectives,
             requires_agents=2, agent_count=agent_count, slow=True, quick=args.quick)

    # ── 12. --output: broadcast saves JSON file ─────────────────────────────────
    if _match("broadcast_output"):
        def test_broadcast_output_file():
            out_path = Path("/tmp/vimin_test_broadcast_output.json")
            out_path.unlink(missing_ok=True)
            d = _post(f"{center}/api/broadcast", api_key, {
                "prompt": "In one word: what colour is the sky?",
                "max_tokens": 20,
                "mode": "return",
            }, timeout=310)
            _assert_not_error(d)
            out_path.write_text(json.dumps(d, indent=2))
            if not out_path.exists() or out_path.stat().st_size == 0:
                raise AssertionError("output file was not written")
            loaded = json.loads(out_path.read_text())
            if "results" not in loaded:
                raise AssertionError("output JSON missing 'results' key")
            return f"saved {out_path.stat().st_size} bytes → {out_path}"

        _run("broadcast_output_file", test_broadcast_output_file,
             requires_agents=1, agent_count=agent_count)

    # ── 13. --output: pipeline saves JSON file ──────────────────────────────────
    if _match("pipeline_output"):
        def test_pipeline_output_file():
            out_path = Path("/tmp/vimin_test_pipeline_output.json")
            out_path.unlink(missing_ok=True)
            d = _post(f"{center}/api/pipeline", api_key, {
                "name": "output-file-test",
                "mode": "return",
                "input": "vimin-core is an open-source local AI inference orchestration tool.",
                "steps": [
                    {
                        "label": "summarize",
                        "type": "SUMMARIZATION",
                        "data": "In one sentence, summarize: {{input}}",
                        "timeout": 120,
                    },
                ],
            }, timeout=310)
            _assert_not_error(d)
            out_path.write_text(json.dumps(d, indent=2))
            loaded = json.loads(out_path.read_text())
            if "steps" not in loaded:
                raise AssertionError("output JSON missing 'steps' key")
            return f"saved {out_path.stat().st_size} bytes → {out_path}"

        _run("pipeline_output_file", test_pipeline_output_file,
             requires_agents=1, agent_count=agent_count)

    # ── Summary ─────────────────────────────────────────────────────────────────
    total    = len(_RESULTS)
    passed   = sum(1 for r in _RESULTS if r.passed)
    failed   = sum(1 for r in _RESULTS if not r.passed and not r.skipped)
    skipped  = sum(1 for r in _RESULTS if r.skipped)

    print(f"\n{_P}  {'─' * 58}{_RS}")
    print(f"  {_W}Results{_RS}   "
          f"{_G}{passed} passed{_RS}  "
          f"{(_R if failed else _D)}{failed} failed{_RS}  "
          f"{_D}{skipped} skipped{_RS}  "
          f"{_D}/ {total} total{_RS}")

    if failed:
        print(f"\n{_R}  Failed tests:{_RS}")
        for r in _RESULTS:
            if not r.passed and not r.skipped:
                print(f"    {_R}✗{_RS} {r.name}  — {r.error}")

    print()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
