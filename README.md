<img src="vimin_logo.png" alt="vimin" width="200"/>

# vimin-core

Open-source local AI inference orchestration. Run open-source LLMs across up to **10 machines** on your network — no cloud, no API keys, no data leaving your premises.

## What it does

vimin-core lets you coordinate a small fleet of machines (laptops, desktops, Mac minis, servers) to run local AI inference together. You start a **center node** on one machine to act as the orchestration hub, then connect **agent nodes** on each machine that will run models. Any broadcast query you send to the center node goes to all connected agents simultaneously.

**Use cases:**
- Parallel inference across multiple machines for higher throughput
- Running different models on different machines and comparing outputs
- Offline AI workflows in air-gapped or privacy-sensitive environments
- Local AI demos on a small cluster without cloud dependencies
- Development and experimentation with multi-node inference pipelines

**Limits in vimin-core (open-source):**
- Maximum 10 nodes
- Broadcast dispatch only — every query goes to all nodes simultaneously
- No per-node targeting, fleet pipelines, or workflow orchestration

For larger fleets, per-node routing, and production features, see [vimin](https://vimin.ai).

---

## Quickstart

### 1. Install

```bash
# Base install (networking only — add a backend for inference)
pip install -e .

# Apple Silicon text models (recommended for M-series Macs)
pip install -e ".[mlx]"

# Apple Silicon voice / speech-to-text (Whisper)
pip install -e ".[whisper]"

# Any platform — CPU, CUDA, or Apple Metal via GGUF
pip install -e ".[llamacpp]"

# Everything
pip install -e ".[all]"
```

### 2. Start the center node

Run this once on the machine that will act as the hub:

```bash
vimin-core start-center
```

```
  ◈ vimin-core

  ╭────────────────────────────────────────────────╮
  │           vimin-core  ·  Center Node           │
  ├────────────────────────────────────────────────┤
  │  URL:          http://localhost:8080           │
  │  API key:      <generated-key>                 │
  │  Fleet token:  <generated-token>               │
  │  Node limit:   10  (upgrade to vimin for more) │
  ╰────────────────────────────────────────────────╯
```

> **zsh users:** quote the extras specifier to avoid glob expansion:
> `pip install 'vimin-core[mlx]'`

Custom host/port:
```bash
vimin-core start-center --host 192.168.1.10 --port 9000
```

The generated API key and fleet token are saved to `~/.vimin/config.json` and reused on subsequent starts.

### 3. Connect agent nodes

On each machine that will run models:

```bash
VIMIN_CENTER_URL=http://<center-ip>:8080 vimin-core start-agent
```

Or pass the URL directly:
```bash
vimin-core start-agent --center http://192.168.1.10:8080
```

The agent registers with the center and waits for tasks.

### 4. Send a broadcast query

From any machine with network access to the center:

```bash
curl -X POST http://<center-ip>:8080/api/broadcast \
  -H "Authorization: Bearer <api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Summarize the key benefits of local AI inference.",
    "model_id": "meta-llama/Llama-3.2-3B-Instruct",
    "max_tokens": 200
  }'
```

All online agents receive the prompt, run inference locally, and return results. The center node aggregates and returns all responses.

---

## Supported Models

vimin-core ships with built-in aliases for the models below — pass the canonical HuggingFace ID and the right 4-bit MLX checkpoint is loaded automatically. Any other `mlx-community/` checkpoint also works by passing it directly.

### Text — Apple Silicon (MLX backend)

4-bit quantised checkpoints load from the `mlx-community` org automatically. No manual conversion needed. Install with `pip install -e ".[mlx]"`.

**Compact (≤ 2 GB RAM — fits on any modern Mac)**

| Model | Params | RAM (4-bit) | Notes |
|-------|--------|-------------|-------|
| `HuggingFaceTB/SmolLM2-360M-Instruct` | 360M | ~0.7 GB | Fastest; good for simple tasks |
| `Qwen/Qwen2.5-0.5B-Instruct` | 500M | ~1 GB | Strong for size; multilingual |
| `Qwen/Qwen3-0.6B` | 600M | ~0.8 GB | Qwen3 generation; thinking mode support |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 1.5B | ~1 GB | Reasoning model; shows thinking steps |
| `meta-llama/Llama-3.2-1B-Instruct` | 1B | ~1 GB | Meta's efficient small model |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | ~1 GB | Multilingual; strong instruction following |
| `Qwen/Qwen3-1.7B` | 1.7B | ~1.5 GB | Qwen3; fast with reasoning support |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 1.7B | ~1.5 GB | Compact general purpose |

**Mid-range (2–6 GB RAM — 8 GB+ Mac recommended)**

| Model | Params | RAM (4-bit) | Notes |
|-------|--------|-------------|-------|
| `google/gemma-3-1b-it` | 1B | ~1 GB | Google's newest generation |
| `google/gemma-2-2b-it` | 2B | ~2 GB | Reliable; good reasoning |
| `google/gemma-3-4b-it` | 4B | ~3 GB | Gemma 3; strong all-round |
| `Qwen/Qwen3-4B` | 4B | ~3 GB | Qwen3 with hybrid thinking mode |
| `meta-llama/Llama-3.2-3B-Instruct` | 3B | ~2 GB | Meta's best small instruct |
| `Qwen/Qwen2.5-3B-Instruct` | 3B | ~2 GB | Multilingual; fast |
| `HuggingFaceTB/SmolLM3-3B` | 3B | ~2 GB | SmolLM3; efficient on-device model |
| `microsoft/Phi-3.5-mini-instruct` | 3.8B | ~3 GB | Microsoft; strong reasoning |
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 1.5B | ~1 GB | Code-optimised |

**Standard (6–10 GB RAM — 16 GB Mac recommended)**

| Model | Params | RAM (4-bit) | Notes |
|-------|--------|-------------|-------|
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | 7B | ~5 GB | Best reasoning at 7B |
| `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | 8B | ~6 GB | Reasoning; Llama architecture |
| `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` | 8B | ~6 GB | DeepSeek R1 May 2025; Qwen3 base |
| `Qwen/Qwen3-8B` | 8B | ~6 GB | Qwen3 flagship 8B; best multilingual |
| `Qwen/Qwen2.5-7B-Instruct` | 7B | ~5 GB | Strong multilingual |
| `Qwen/Qwen2.5-Coder-7B-Instruct` | 7B | ~5 GB | Top open-source code model |
| `mistralai/Mistral-7B-Instruct-v0.3` | 7B | ~5 GB | Reliable general purpose |
| `meta-llama/Llama-3.1-8B-Instruct` | 8B | ~6 GB | Meta's flagship open model |
| `microsoft/Phi-4-mini-instruct` | 7.6B | ~6 GB | Microsoft's compact powerhouse |
| `microsoft/Phi-4-mini-reasoning` | 7.6B | ~6 GB | Phi-4-mini fine-tuned for math/logic |
| `google/gemma-2-9b-it` | 9B | ~7 GB | Google; strong instruction following |
| `google/gemma-3-12b-it` | 12B | ~9 GB | Gemma 3 mid-range |

**Large (12–40 GB RAM — Mac Studio / Pro / server)**

| Model | Params | RAM (4-bit) | Notes |
|-------|--------|-------------|-------|
| `mistralai/Mistral-Nemo-Instruct-2407` | 12B | ~9 GB | Mistral; strong multilingual |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | 14B | ~10 GB | Best reasoning per dollar |
| `Qwen/Qwen3-14B` | 14B | ~10 GB | Qwen3 14B; near-frontier reasoning |
| `Qwen/Qwen2.5-14B-Instruct` | 14B | ~10 GB | Multilingual flagship |
| `Qwen/Qwen2.5-Coder-14B-Instruct` | 14B | ~10 GB | Best open-source code model |
| `microsoft/phi-4` | 14B | ~10 GB | Microsoft's strongest 14B model |
| `microsoft/phi-4-reasoning` | 14B | ~10 GB | Phi-4 fine-tuned for deep reasoning |
| `microsoft/phi-4-reasoning-plus` | 14B | ~10 GB | Phi-4-reasoning with RLVR polish |
| `mistralai/Devstral-Small-2505` | 24B | ~14 GB | Best open-source coding agent model |
| `Qwen/Qwen3-30B-A3B` | 30B MoE | ~17 GB | MoE: 3B active params, 30B knowledge |
| `Qwen/Qwen3-32B` | 32B | ~24 GB | Qwen3 flagship; frontier-class |
| `google/gemma-2-27b-it` | 27B | ~20 GB | Google; near-frontier quality |
| `google/gemma-3-27b-it` | 27B | ~20 GB | Gemma 3 flagship |
| `meta-llama/Llama-3.3-70B-Instruct` | 70B | ~42 GB | Frontier-class open model |

### Voice — Apple Silicon (Whisper backend)

Install with `pip install -e ".[whisper]"`. Pass `openai/whisper-*` IDs and the optimised MLX checkpoint is used automatically.

| Model | RAM | Speed | Best for |
|-------|-----|-------|----------|
| `openai/whisper-tiny` | ~0.2 GB | Fastest | Real-time on constrained hardware |
| `openai/whisper-base` | ~0.3 GB | Very fast | Good default for most tasks |
| `openai/whisper-small` | ~0.6 GB | Fast | Better accuracy, still lightweight |
| `openai/whisper-medium` | ~1.5 GB | Moderate | High accuracy, 16 GB+ Mac |
| `openai/whisper-large-v3-turbo` | ~1.6 GB | Fast | Near-large quality, 2× faster |
| `openai/whisper-large-v3` | ~3 GB | Slower | Best accuracy available |

### Any Platform (llama-cpp backend)

Runs GGUF models on CPU, Apple Metal, or NVIDIA CUDA. Install with `pip install -e ".[llamacpp]"`. Download `.gguf` files from HuggingFace and pass the local path:

```json
{ "model_id": "local-model", "path": "/path/to/model.gguf" }
```

Recommended Q4_K_M checkpoints:
- `bartowski/Llama-3.2-3B-Instruct-GGUF`
- `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF`
- `bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF`
- `bartowski/Phi-3.5-mini-instruct-GGUF`
- `bartowski/Qwen2.5-7B-Instruct-GGUF`

For Metal acceleration (macOS):
```bash
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --no-cache-dir
```

For CUDA (Linux/Windows):
```bash
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --no-cache-dir
```

---

## Example Workflows

Pre-built workflow scripts live in the [`examples/`](examples/) directory. Each is a self-contained script that connects to your fleet via the broadcast API.

| Script | What it does |
|--------|-------------|
| [`workflow_voice_transcription.py`](examples/workflow_voice_transcription.py) | Record microphone → Whisper transcription → fleet analysis |
| [`workflow_meeting_minutes.py`](examples/workflow_meeting_minutes.py) | Audio/transcript → Whisper → extract summary, decisions, action items |
| [`workflow_document_analysis.py`](examples/workflow_document_analysis.py) | File or stdin → parallel doc analysis (summary, facts, risks, sentiment) |
| [`workflow_code_review.py`](examples/workflow_code_review.py) | Source file or `git diff` → parallel code review (bugs, security, verdict) |
| [`workflow_multi_language.py`](examples/workflow_multi_language.py) | Text → simultaneous translation into multiple languages |
| [`workflow_batch_summarization.py`](examples/workflow_batch_summarization.py) | Folder of documents → distributed parallel summarization with JSON report |
| [`workflow_pii_redaction.py`](examples/workflow_pii_redaction.py) | Text or file → on-device PII detection and redaction (GDPR/HIPAA prep) |
| [`workflow_support_triage.py`](examples/workflow_support_triage.py) | Support tickets → parallel classification, priority, routing, sentiment |
| [`workflow_competitive_research.py`](examples/workflow_competitive_research.py) | Competitor text → strategic analysis (features, pricing, risks, positioning) |
| [`workflow_structured_extraction.py`](examples/workflow_structured_extraction.py) | Documents → JSON extraction using built-in schemas (invoice, job, contract) |
| [`workflow_local_rag.py`](examples/workflow_local_rag.py) | Local RAG: index a doc folder, retrieve relevant chunks, generate on fleet |
| [`workflow_openclaw_fleet.py`](examples/workflow_openclaw_fleet.py) | OpenClaw-backed inference: list models, direct query, or fleet broadcast |

All scripts read `VIMIN_CENTER_URL` and `ORCHESTRATOR_API_KEY` from the environment, or accept `--center` and `--api-key` CLI arguments.

---

## API Reference

All endpoints require `Authorization: Bearer <api-key>`.

### `POST /api/broadcast`

Send a prompt to all online agents simultaneously.

```json
{
  "prompt": "Your prompt here",
  "model_id": "meta-llama/Llama-3.2-3B-Instruct",
  "max_tokens": 256,
  "temperature": 0.7
}
```

Response:
```json
{
  "broadcast_id": "abc123",
  "results": [
    { "agent_id": "node-1", "output": "...", "latency_ms": 1240 },
    { "agent_id": "node-2", "output": "...", "latency_ms": 980 }
  ]
}
```

### `GET /api/agents`

List all registered agents and their status.

### `GET /api/health`

Health check — returns center node uptime and node count.

---

## Configuration

Settings are stored in `~/.vimin/config.json`:

```json
{
  "api_key": "auto-generated",
  "fleet_token": "auto-generated",
  "center_url": "http://localhost:8080"
}
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VIMIN_CENTER_URL` | `http://localhost:8080` | Center node URL (used by agents) |
| `ORCHESTRATOR_API_KEY` | from config | API key for authenticating requests |
| `VIMIN_FLEET_TOKEN` | from config | Token for agent registration |

---

## Hardware Requirements

**Center node:** Any machine with Python 3.10+ and network access. Minimal CPU/RAM needed — it only routes tasks.

**Agent nodes:**

| Backend | Minimum RAM | Recommended |
|---------|-------------|-------------|
| MLX (Apple Silicon) | 8 GB unified | 16 GB+ for 7B+ models |
| llama-cpp (CPU) | 8 GB | 16 GB+ for 7B+ models |
| llama-cpp (CUDA) | GPU VRAM ≥ model size | 8 GB+ VRAM |
| ONNX encoders | 4 GB | 8 GB |

---

## Project structure

```
vimin-core/
├── src/vimin_core/
│   ├── cli/          # Command-line interface
│   ├── core/         # Inference orchestrator, backends, task types
│   │   └── backends/ # MLX, llama-cpp, ONNX backend implementations
│   ├── hardware/     # Hardware detection and telemetry
│   ├── systems/      # Center node, agent node, database
│   └── utils/        # Logging
├── pyproject.toml
└── README.md
```

---

## License

vimin-core is released under the [Business Source License 1.1](LICENSE).

**Free to use** for personal, research, academic, and internal non-commercial purposes, and for commercial evaluation on up to **5 devices**.

**A commercial license is required** if you:
- Deploy across more than 5 devices in production
- Offer vimin-core as a hosted or managed service to third parties
- Embed it in commercial software you distribute to customers
- Use it as the basis for a competing inference orchestration product

The license converts to **Apache 2.0** on **2030-04-01**.

For commercial licensing: [hello@vimin.ai](mailto:hello@vimin.ai)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to report bugs, add model aliases, build new backends, and submit pull requests.

---

## vimin (full distribution)

vimin-core is the open-source foundation. The full [vimin](https://vimin.ai) distribution adds:

- Unlimited nodes
- Per-node task targeting
- Fleet pipelines and workflow orchestration
- OpenClaw integration for device management
- Advanced dashboard and analytics
- Priority support
