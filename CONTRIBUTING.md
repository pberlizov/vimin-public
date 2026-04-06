# Contributing to vimin-core

Thank you for your interest in contributing. vimin-core is the open-source foundation for local, on-device AI inference orchestration. Contributions — bug fixes, new model aliases, backend improvements, documentation, and workflow examples — are welcome.

---

## Before you start

**Check existing issues and PRs first.** Someone may already be working on the same thing.

**License:** vimin-core is released under [BSL 1.1](LICENSE). By submitting a pull request you agree that your contribution will be licensed under the same terms.

**Scope:** vimin-core covers the open-source tier — up to 10 nodes, broadcast dispatch, and local inference backends. Features that belong in the commercial tier (per-node routing, fleet pipelines, OpenClaw agent coordination) should not be added here.

---

## Getting started

### 1. Fork and clone

```bash
git clone https://github.com/your-fork/vimin-core.git
cd vimin-core
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install in editable mode with dev dependencies

```bash
pip install -e ".[mlx,whisper,llamacpp,dev]"
```

The `dev` extra includes `pytest`, `ruff`, and `mypy`. If you don't have Apple Silicon, omit `mlx` and `whisper`.

### 4. Verify the install

```bash
python -c "import vimin_core; print(vimin_core.__version__)"
```

---

## Project layout

```
vimin-core/
├── src/vimin_core/
│   ├── cli/             # Command-line entry points (start-center, start-agent)
│   ├── core/
│   │   ├── backends/    # Inference backends: MLX, llama-cpp, Whisper, OpenClaw
│   │   ├── orchestrator.py
│   │   ├── router.py
│   │   └── task.py
│   ├── hardware/        # CPU/GPU/ANE detection and telemetry
│   ├── systems/         # CenterNode, UserAgent, database
│   └── utils/           # Logging helpers
├── examples/            # Standalone workflow scripts
├── tests/               # pytest test suite
├── pyproject.toml
└── LICENSE
```

---

## How to contribute

### Reporting bugs

Open a GitHub issue with:
- Python version and platform (`python --version`, `uname -a` or Windows version)
- vimin-core version (`pip show vimin-core`)
- Minimal reproduction steps
- Full traceback

### Requesting features

Open an issue describing the use case, not just the solution. If it fits the open-source scope (see above), we'll discuss implementation before you write code.

### Submitting a pull request

1. **Open an issue first** for non-trivial changes so we can align before you invest time.
2. Create a branch from `main`:
   ```bash
   git checkout -b feat/my-feature
   ```
3. Make your changes (see guidelines below).
4. Run the test suite and linter before pushing.
5. Open a PR against `main` with a clear description of what changed and why.

---

## Code guidelines

### Style

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
ruff check src/ tests/
ruff format src/ tests/
```

The CI gate will block PRs that fail ruff checks.

### Type annotations

New functions and methods should include type annotations. We use `mypy` in non-strict mode:

```bash
mypy src/vimin_core --ignore-missing-imports
```

### Dependencies

- **No new required dependencies without discussion.** vimin-core's base install is intentionally lightweight.
- Optional features go in extras in `pyproject.toml` (e.g., `mlx`, `whisper`, `llamacpp`).
- Backends should use stdlib where possible for their core logic (see `OpenClawBackend` which uses `urllib.request`).

### Imports

Use absolute imports (`from vimin_core.core.backends.base import ...`) throughout. Do not use relative imports or try/except fallback import patterns.

### Logging

Use `logging.getLogger(__name__)` — never `print()` inside library code. `print()` is fine in CLI entry points and example scripts.

---

## Adding a new model alias

Model aliases live in `src/vimin_core/core/backends/mlx_backend.py` in the `_MLX_COMMUNITY_ALIASES` dict.

**Requirements before adding an alias:**
- The `mlx-community/<checkpoint>` checkpoint must exist on HuggingFace.
- The checkpoint must be a 4-bit quantised MLX format (not PyTorch or ONNX).
- Multimodal models (vision + text) require `mlx-vlm` and belong in a separate `VLMBackend` (not yet implemented — open an issue to discuss).

**How to add:**

```python
# In _MLX_COMMUNITY_ALIASES:
"org/Model-Name-Instruct": "mlx-community/Model-Name-Instruct-4bit",
```

If the model name doesn't contain a clear `Nb` token (e.g., `phi-4`, `devstral`, `mistral-nemo`), add an entry to `_KNOWN_SIZES` with the 4-bit checkpoint size in GB:

```python
_KNOWN_SIZES: dict[str, float] = {
    ...
    "my-model": 12.5,  # 4-bit size in GB
}
```

Also update the model table in `README.md`.

---

## Adding a new inference backend

A backend is a class that inherits from `BaseBackend` in `src/vimin_core/core/backends/base.py`.

**Required methods:**
```python
def is_available(self) -> bool: ...
def estimate_memory_gb(self, descriptor: ModelDescriptor) -> float: ...
def load(self, descriptor: ModelDescriptor) -> bool: ...
def generate(self, prompt: str, max_new_tokens: int, temperature: float, stop_sequences) -> str: ...
def unload(self) -> None: ...
```

**Optional but recommended:**
```python
def stream_generate(self, ...) -> Iterator[str]: ...
```

After implementing the backend:
1. Export it from `src/vimin_core/core/backends/__init__.py`.
2. Wire it into `BackendSelector.select()` in `src/vimin_core/core/backends/selector.py`.
3. Add an optional dependency in `pyproject.toml` if it requires a new package.
4. Add tests in `tests/`.

---

## Adding a workflow example

Workflow scripts live in `examples/`. Each script should:

- Be self-contained and runnable with a single `python examples/workflow_*.py` command.
- Read `VIMIN_CENTER_URL` and `ORCHESTRATOR_API_KEY` from environment, with sensible defaults.
- Accept `--center` and `--api-key` CLI overrides.
- Print a clear header showing configuration before running.
- Include a module docstring with requirements and usage examples.
- Not import from `vimin_core` internals — only use the HTTP broadcast API.

After adding a script, add a row to the workflow table in `README.md`.

---

## Running tests

```bash
pytest tests/ -v
```

Tests that require Apple Silicon (MLX) are skipped automatically on other platforms. Tests that require a running center node are marked `@pytest.mark.integration` and skipped by default:

```bash
# Run integration tests (requires a running center node at localhost:8080)
pytest tests/ -v -m integration
```

---

## Release process

Releases are made by the maintainers. If you think a fix or feature is ready to ship, comment on the PR or issue.

---

## Questions?

Open an issue or start a GitHub Discussion. For commercial licensing enquiries: [pberlizov@college.harvard.edu](mailto:pberlizov@college.harvard.edu).
