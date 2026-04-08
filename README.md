<img src="vimin_logo.png" alt="vimin" width="200"/>

# vimin-core

Source-available local AI inference orchestration for up to **10 machines**. Run open-source LLMs and speech models without a cloud service, with local credentials and local execution by default.

## What it does

vimin-core lets you coordinate a fleet of machines (laptops, desktops, Mac minis, servers) to run local AI inference together. You start a **center node** on one machine as the orchestration hub, then connect **agent nodes** on each machine that will run models.

**Two ways to use it**

- **Broadcast**: send a prompt to all connected agents at once and collect every response.
- **Pipelines**: run multi-step workflows where each step uses a different task type and feeds into the next.

**Task types:**

| Type | What runs it |
|------|-------------|
| `TEXT_GENERATION`, `SUMMARIZATION`, `REASONING`, `TRANSLATION`, `CODE_GENERATION`, `CLASSIFICATION`, `SENTIMENT_ANALYSIS` | The loaded LLM (MLX or llama-cpp) |
| `PII_MASKING` | ONNX NER model, regex scrubber, or LLM fallback. Data stays on the device. |
| `SPEECH_TO_TEXT` | Whisper. `mlx-whisper` on Apple Silicon, `faster-whisper` on other platforms. |

**Use cases:**
- Parallel inference across multiple machines for higher throughput
- Multi-step document pipelines (translate → redact PII → summarize)
- Meeting transcription → action item extraction
- Code review, support ticket triage, competitive research
- Offline AI workflows in air-gapped or privacy-sensitive environments
- Comparing outputs from different models side-by-side

**Limits in vimin-core**
- Maximum 10 nodes
- No per-node targeting. Pipelines use basic center-driven scheduling.
- No role-based access control or compliance-grade audit reporting
- No enterprise dashboard

More about the advanced version is on the website: [viminlabs.com](https://viminlabs.com).

---

## Quickstart

### 1. Install

```bash
# Apple Silicon text models (recommended for M-series Macs)
pip install 'vimin-core[mlx]'

# Apple Silicon voice / speech-to-text (Whisper)
pip install 'vimin-core[whisper]'

# Any platform: CPU, CUDA, or Apple Metal via GGUF
pip install 'vimin-core[llamacpp]'

# Everything
pip install 'vimin-core[all]'
```

### 2. Start the center node

```bash
vimin-core start-center
```

The center runs as a **background daemon** by default. To run in the foreground instead (e.g. to watch logs live):

```bash
vimin-core start-center --foreground
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

  Running in background.
  PID  1234  |  Logs  ~/.vimin/logs/center.log
  Stop with  vimin-core stop-center
```

By default the center binds to `127.0.0.1` (this machine only). To accept connections from other machines:

```bash
vimin-core start-center --host 0.0.0.0
```

A warning is printed when binding to a non-loopback interface. Use TLS and a firewall rule to protect the port in production.

The generated API key and fleet token are saved to `~/.vimin/config.json` and reused on subsequent starts. To use a custom key across all machines in your fleet:

```bash
export ORCHESTRATOR_MASTER_KEY="your-shared-secret"
vimin-core start-center --host 0.0.0.0
```

Set the same `ORCHESTRATOR_MASTER_KEY` on every agent machine.

Watch center logs:
```bash
tail -f ~/.vimin/logs/center.log
```

### 3. Connect agent nodes

On the same machine (or any machine with network access to the center):

```bash
# Same machine
vimin-core start-agent

# Remote machine: pass the center's LAN IP
vimin-core start-agent --center http://192.168.1.10:8080

# Or via environment variable
VIMIN_CENTER_URL=http://192.168.1.10:8080 vimin-core start-agent
```

Agents also run as background daemons by default. Watch agent logs:

```bash
tail -f ~/.vimin/logs/agent-*.log
```

**Agent ID persistence:** Each agent gets a stable ID on first run and saves it to `~/.vimin/config.json`. If it disconnects and reconnects, queued tasks can still be delivered to the same machine.

**Graceful shutdown:** When you run `vimin-core stop-agent`, the agent sends a goodbye heartbeat to the center before exiting. The node slot is freed immediately rather than waiting for a heartbeat timeout.

### 4. Broadcast a prompt

```bash
vimin-core broadcast "What is the capital of Japan?" --mode return
```

`--mode return` sends results back to your terminal and auto-saves them to `~/.vimin/outputs/broadcast-YYYYMMDD-HHMMSS.json`. `--mode broadcast` runs inference and saves results on the edge device only.

**Offline queuing:** If an agent is offline when a broadcast goes out, the task stays queued at the center. When that agent reconnects, the queued task is dispatched automatically. The result is written to the agent log and the center audit log.

To find offline task results:
```bash
# Agent log (contains the model's output)
tail -100 ~/.vimin/logs/agent-*.log

# Center audit log (structured JSONL records of all completed tasks)
tail -20 ~/.vimin/audit.jsonl
```

### 5. Run a pipeline

```bash
# Translate Spanish → English, then summarize
vimin-core run-pipeline \
  --preset translate-and-summarize \
  --input "El banco central anunció una subida de tipos de interés del 0,25%." \
  --mode return

# Redact PII from a document, then summarize locally
vimin-core run-pipeline \
  --preset pii-redact-then-summarize \
  --file patient_record.txt \
  --mode broadcast

# Full investigative report, saved to a JSON file
vimin-core run-pipeline \
  --preset analyze-and-report \
  --file case_file.md \
  --mode return \
  --output ~/results/report.json
```

---

## Built-in Presets

| Preset | Steps | What it does |
|--------|-------|-------------|
| `translate-and-summarize` | `TRANSLATION` → `SUMMARIZATION` | Translate any language to English, then summarize |
| `pii-redact-then-summarize` | `PII_MASKING` → `SUMMARIZATION` | Redact PII on-device, then summarize the clean text |
| `summarize-and-questions` | `SUMMARIZATION` → `REASONING` | Summarize a document, then generate follow-up questions |
| `analyze-and-report` | `REASONING` → `REASONING` → `SUMMARIZATION` | Extract facts, identify risks, produce an executive summary |
| `code-review` | parallel [`CODE_GENERATION`, `CODE_GENERATION`] → `REASONING` | Bug hunt and security review in parallel, then a combined verdict |
| `support-triage` | parallel [`CLASSIFICATION`, `SENTIMENT_ANALYSIS`] → `TEXT_GENERATION` | Classify and score sentiment in parallel, then draft a response |
| `transcribe-and-analyze` | `SPEECH_TO_TEXT` → `TEXT_GENERATION` | Transcribe audio, then analyze the content |
| `meeting-minutes` | `SPEECH_TO_TEXT` → `SUMMARIZATION` → `CLASSIFICATION` | Full meeting minutes: transcript → summary → action items |
| `parallel-perspectives` | grouped [`REASONING`, `REASONING`] → `SUMMARIZATION` | Two reasoning tasks run together, then a final summarization step combines them |

Pass a file or inline text with `--file` or `--input`. Audio files (`.wav`, `.mp3`, `.m4a`, etc.) are automatically routed as paths for `SPEECH_TO_TEXT` steps.

Custom pipelines: write a JSON file and pass it with `--pipeline`:

```json
{
  "name": "My pipeline",
  "steps": [
    {
      "type": "TRANSLATION",
      "data": "Translate to English: {{input}}",
      "timeout": 180
    },
    {
      "type": "SUMMARIZATION",
      "data": "Summarize in 3 sentences: {{step1_output}}"
    }
  ]
}
```

```bash
vimin-core run-pipeline --pipeline my_pipeline.json --input "..." --mode return
```

### 6. Clear queued tasks

```bash
vimin-core clear-tasks
```

This clears the center node's queued task list and pending dispatch commands. It does **not** interrupt tasks that are already running on agents.

### 7. Revoke an agent

```bash
vimin-core revoke-agent <agent-id>
```

Revoking an agent clears its queued work, prevents future reconnects with its old identity, and marks it as revoked in the center's agent list.

### 8. Inspect agents

```bash
vimin-core list-agents
vimin-core show-agent <agent-id>
```

Use these to inspect enrolled agents, their status, joined time, loaded model, and task counts from the center node.

---

## Supported Models

vimin-core ships with built-in aliases for the models below. Pass the canonical HuggingFace ID and the matching 4-bit MLX checkpoint is loaded automatically. Any other `mlx-community/` checkpoint also works if you pass it directly.

### Text: Apple Silicon (MLX backend)

4-bit quantised checkpoints load from the `mlx-community` org automatically. No manual conversion needed. Install with `pip install 'vimin-core[mlx]'`.

**Compact (≤ 2 GB RAM, fits on any modern Mac)**

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

**Mid-range (2–6 GB RAM, 8 GB+ Mac recommended)**

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

**Standard (6–10 GB RAM, 16 GB Mac recommended)**

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

**Large (12–40 GB RAM, Mac Studio / Pro / server)**

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

### Voice: Speech-to-Text (Whisper)

Install with `pip install 'vimin-core[whisper]'`. The right backend is chosen automatically:

- **Apple Silicon**: `mlx-whisper` (ANE-accelerated, fastest)
- **Linux / Windows / Intel Mac**: `faster-whisper` (CTranslate2, CPU or CUDA)

Pass `openai/whisper-*` IDs on any platform:

| Model | RAM | Speed | Best for |
|-------|-----|-------|----------|
| `openai/whisper-tiny` | ~0.2 GB | Fastest | Real-time on constrained hardware |
| `openai/whisper-base` | ~0.3 GB | Very fast | Good default for most tasks |
| `openai/whisper-small` | ~0.6 GB | Fast | Better accuracy, still lightweight |
| `openai/whisper-medium` | ~1.5 GB | Moderate | High accuracy |
| `openai/whisper-large-v3-turbo` | ~1.6 GB | Fast | Near-large quality, 2× faster |
| `openai/whisper-large-v3` | ~3 GB | Slower | Best accuracy available |

### Any Platform (llama-cpp backend)

Runs GGUF models on CPU, Apple Metal, or NVIDIA CUDA. Install with `pip install 'vimin-core[llamacpp]'`. Download `.gguf` files from HuggingFace and pass the local path:

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

## API Reference

All endpoints require `Authorization: Bearer <api-key>`.

### `POST /api/broadcast`

Send a prompt to all online agents simultaneously.

```bash
curl -X POST http://localhost:8080/api/broadcast \
  -H "Authorization: Bearer <api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Your prompt here",
    "model_id": "meta-llama/Llama-3.2-3B-Instruct",
    "max_tokens": 256,
    "mode": "return",
    "timeout": 60
  }'
```

`mode`: `"return"` (default) sends results to the caller; `"broadcast"` saves results on each agent at `~/.vimin/outputs/`.

Response:
```json
{
  "broadcast_id": "bcast_abc123",
  "results": [
    { "agent_id": "node-1", "output": "Tokyo.", "latency_ms": 1240 },
    { "agent_id": "node-2", "output": "Tokyo.", "latency_ms": 980 }
  ]
}
```

### `POST /api/pipeline`

Run a multi-step pipeline. Steps execute sequentially; an array of steps executes in parallel across available agents.

```bash
curl -X POST http://localhost:8080/api/pipeline \
  -H "Authorization: Bearer <api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Translate and Summarize",
    "input": "El banco central anunció...",
    "model_id": "mlx-community/Qwen2.5-3B-Instruct-4bit",
    "mode": "return",
    "steps": [
      { "type": "TRANSLATION", "data": "Translate to English: {{input}}" },
      { "type": "SUMMARIZATION", "data": "Summarize: {{step1_output}}" }
    ]
  }'
```

Use `{{input}}` and `{{stepN_output}}` as placeholders. Each step can override `model_id`, `timeout`, and `metadata` (e.g. `max_tokens`).

### `GET /api/agents`

List all registered agents and their status.

### `GET /api/health`

Health check. Returns center uptime and node count.

---

## Configuration

Settings are stored in `~/.vimin/config.json`:

```json
{
  "api_key": "auto-generated",
  "fleet_token": "auto-generated",
  "agent_id": "auto-generated",
  "center_url": "http://localhost:8080"
}
```

`agent_id` is generated once and reused across restarts so the center can match a reconnecting agent to its queued tasks.

After an agent first connects, a `pinned_center_url` key is added automatically. If the center URL changes on a subsequent run, the agent prints a warning. Delete that key to reset.

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ORCHESTRATOR_MASTER_KEY` | from config | Shared secret for center + agents. Set the same value on all machines. Takes priority over config. |
| `VIMIN_CENTER_URL` | from config | Center node URL (used by agents) |
| `ORCHESTRATOR_API_KEY` | from config | Alternative API key (lower priority than `ORCHESTRATOR_MASTER_KEY`) |
| `VIMIN_FLEET_TOKEN` | from config | Token for agent registration |

---

## Security

- The center binds to `127.0.0.1` (localhost only) by default. Pass `--host 0.0.0.0` to expose it to the network; a warning is printed when you do.
- The agent prints a warning if connecting to a non-localhost center over plain HTTP. Use HTTPS for connections across untrusted networks.
- The agent pins the center URL on first registration and warns if it changes, preventing silent redirections.
- Task data is never executed as code. It is passed only to inference backends (MLX, llama-cpp, ONNX, Whisper).
- The fleet token (`VIMIN_FLEET_TOKEN`) restricts which agents can register with your center.
- Each enrolled agent also receives a per-agent secret on first registration. Future heartbeats, command polling, and reconnects must present that secret, preventing one enrolled node from impersonating another by reusing only the shared fleet credential.
- The node limit of 10 is enforced at the center; registration is rejected beyond this.

---

## Hardware Requirements

**Center node:** Any machine with Python 3.10+ and network access. It only routes tasks, so CPU and RAM needs are modest.

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
│   │   └── backends/ # MLX, llama-cpp, ONNX, Whisper backend implementations
│   ├── hardware/     # Hardware detection and telemetry
│   ├── systems/      # Center node, agent node, database
│   └── utils/        # Logging
├── presets/          # Built-in pipeline JSON files
├── pyproject.toml
└── README.md
```

---

## License

vimin-core is released under the [Business Source License 1.1](LICENSE).

**Free to use** for personal, research, academic, and internal non-commercial purposes, and for commercial evaluation on up to **10 connected nodes**.

**A commercial license is required** if you:
- Deploy across **more than 10 nodes** in production
- Offer vimin-core as a hosted or managed service to third parties
- Embed it in commercial software you distribute to customers
- Use it as the basis for a competing inference orchestration product

The license converts to the **Apache License 2.0** on **April 6, 2030**.

For commercial licensing: [pberlizov@college.harvard.edu](mailto:pberlizov@college.harvard.edu)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to report bugs, add model aliases, build new backends, and submit pull requests.

---

## vimin

vimin-core is the source-available foundation. The more advanced version of vimin is described on the website: [viminlabs.com](https://viminlabs.com).

That version adds:

- Unlimited nodes
- Per-node task targeting and tag-based routing
- Fleet pipelines with advanced workflow orchestration
- OpenClaw integration for device management
- Manual approval for new agent enrollments
- Role-based access control and audit logging
- Advanced dashboard and analytics
- Priority support

[viminlabs.com](https://viminlabs.com)
