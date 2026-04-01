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

# Apple Silicon (recommended for M-series Macs)
pip install -e ".[mlx]"

# Any platform — CPU, CUDA, or Apple Metal via GGUF
pip install -e ".[llamacpp]"

# ONNX encoder models (Whisper, BERT, embeddings)
pip install -e ".[onnx]"

# Everything
pip install -e ".[all]"
```

### 2. Start the center node

Run this once on the machine that will act as the hub:

```bash
vimin-core start-center
```

```
============================================================
  vimin-core Center Node
============================================================
  URL          : http://0.0.0.0:8080
  API key      : <generated-key>
  Fleet token  : <generated-token>
  Node limit   : 10  (upgrade to vimin for more)
```

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

vimin-core works with any model that the installed backends can load. The following are tested and recommended.

### Apple Silicon (MLX backend)

Optimised for M-series Macs using Apple's unified memory. 4-bit quantized checkpoints load automatically — no manual conversion needed.

| Model | Size | RAM needed (4-bit) |
|-------|------|--------------------|
| `HuggingFaceTB/SmolLM2-360M-Instruct` | 360M | ~1 GB |
| `Qwen/Qwen2.5-0.5B-Instruct` | 500M | ~1 GB |
| `meta-llama/Llama-3.2-1B-Instruct` | 1B | ~1.5 GB |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | ~2 GB |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 1.7B | ~2 GB |
| `google/gemma-2-2b-it` | 2B | ~3 GB |
| `meta-llama/Llama-3.2-3B-Instruct` | 3B | ~3 GB |
| `microsoft/Phi-3.5-mini-instruct` | 3.8B | ~4 GB |
| `Qwen/Qwen2.5-7B-Instruct` | 7B | ~5 GB |
| `mistralai/Mistral-7B-Instruct-v0.3` | 7B | ~5 GB |
| `meta-llama/Llama-3.1-8B-Instruct` | 8B | ~6 GB |
| `google/gemma-2-9b-it` | 9B | ~7 GB |

For models not in this list, pass any `mlx-community/` checkpoint ID directly:

```json
{ "model_id": "mlx-community/Mistral-Nemo-Instruct-2407-4bit" }
```

### Any Platform (llama-cpp backend)

Runs GGUF models on CPU, Apple Metal, or NVIDIA CUDA. Download `.gguf` files from HuggingFace and pass the local path:

```json
{ "model_id": "local-gguf", "path": "/path/to/model.gguf" }
```

Recommended GGUF checkpoints (4-bit, Q4_K_M):
- [bartowski/Llama-3.2-3B-Instruct-GGUF](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF)
- [bartowski/Meta-Llama-3.1-8B-Instruct-GGUF](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF)
- [TheBloke/Mistral-7B-Instruct-v0.2-GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF)
- [bartowski/Phi-3.5-mini-instruct-GGUF](https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF)

For Metal acceleration on macOS:
```bash
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --no-cache-dir
```

For CUDA:
```bash
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --no-cache-dir
```

### ONNX encoder models

For speech recognition, embeddings, and classification tasks (ONNX backend):

| Model | Task |
|-------|------|
| `openai/whisper-tiny` | Speech-to-text |
| `openai/whisper-base` | Speech-to-text |
| `openai/whisper-small` | Speech-to-text |
| `BAAI/bge-small-en-v1.5` | Text embeddings |
| `dslim/bert-base-NER` | Named entity recognition |

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

Apache 2.0 — see [LICENSE](LICENSE).

---

## vimin (full distribution)

vimin-core is the open-source foundation. The full [vimin](https://vimin.ai) distribution adds:

- Unlimited nodes
- Per-node task targeting
- Fleet pipelines and workflow orchestration
- OpenClaw integration for device management
- Advanced dashboard and analytics
- Priority support
