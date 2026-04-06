"""Tests for MLX backend alias resolution and memory estimation — no model loading."""
import pytest
from vimin_core.core.backends.mlx_backend import (
    MLXBackend, _estimate_fp16_gb, _resolve_id,
    _MLX_COMMUNITY_ALIASES, _SIZE_HINTS, _KNOWN_SIZES,
)
from vimin_core.core.backends.base import ModelDescriptor


class TestEstimateFp16Gb:
    """Test the RAM estimation heuristic for known and unknown models."""

    def test_known_alias_via_size_hint(self):
        # 7B model should estimate around 14 GB fp16
        gb = _estimate_fp16_gb("meta-llama/Llama-3.1-8B-Instruct")
        assert 10.0 < gb < 25.0

    def test_small_model(self):
        gb = _estimate_fp16_gb("HuggingFaceTB/SmolLM2-360M-Instruct")
        assert gb < 5.0

    def test_large_model(self):
        gb = _estimate_fp16_gb("meta-llama/Llama-3.3-70B-Instruct")
        assert gb > 30.0

    def test_moe_model_uses_known_sizes(self):
        # Qwen3-30B-A3B is a MoE; its fp16 equivalent should be ~17 GB, not 60 GB
        gb = _estimate_fp16_gb("Qwen/Qwen3-30B-A3B")
        assert gb < 30.0

    def test_phi4_uses_known_sizes(self):
        gb = _estimate_fp16_gb("microsoft/phi-4")
        assert 5.0 < gb < 40.0

    def test_unknown_model_returns_fallback(self):
        # Model with no recognisable size token — should return the default fallback
        gb = _estimate_fp16_gb("some/unknown-model-with-no-size")
        assert gb > 0.0

    def test_size_hints_sorted_longest_first(self):
        """Ensure '12b' is matched before '2b' to prevent token shadowing."""
        keys = [hint for hint, _ in _SIZE_HINTS]
        for i in range(len(keys) - 1):
            assert len(keys[i]) >= len(keys[i + 1]), (
                f"SIZE_HINTS not sorted longest-first: {keys[i]!r} followed by {keys[i+1]!r}"
            )

    def test_no_shadowing_12b_vs_2b(self):
        """'Llama-3-12B' should not match '2b' and give wrong size."""
        gb_12b = _estimate_fp16_gb("org/Llama-3-12B-Instruct")
        gb_2b = _estimate_fp16_gb("org/Llama-3-2B-Instruct")
        assert gb_12b > gb_2b

    def test_no_shadowing_27b_vs_7b(self):
        gb_27b = _estimate_fp16_gb("google/gemma-2-27b-it")
        gb_7b = _estimate_fp16_gb("org/model-7b")
        assert gb_27b > gb_7b


class TestMLXCommunityAliases:
    """Verify alias table is consistent and well-formed."""

    def test_aliases_not_empty(self):
        assert len(_MLX_COMMUNITY_ALIASES) > 10

    def test_all_values_are_mlx_community(self):
        for hf_id, mlx_id in _MLX_COMMUNITY_ALIASES.items():
            assert mlx_id.startswith("mlx-community/"), (
                f"Alias for '{hf_id}' does not start with 'mlx-community/': {mlx_id}"
            )

    def test_no_self_references(self):
        for hf_id, mlx_id in _MLX_COMMUNITY_ALIASES.items():
            assert hf_id != mlx_id, f"Alias maps to itself: {hf_id}"

    def test_key_known_models_present(self):
        expected = [
            "meta-llama/Llama-3.2-3B-Instruct",
            "Qwen/Qwen3-8B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "microsoft/Phi-4-mini-instruct",
        ]
        for model_id in expected:
            assert model_id in _MLX_COMMUNITY_ALIASES, f"Missing alias: {model_id}"


class TestMLXBackendIsAvailable:
    def test_returns_bool(self):
        b = MLXBackend()
        result = b.is_available()
        assert isinstance(result, bool)

    def test_estimate_memory_gb_returns_positive_for_known_model(self):
        b = MLXBackend()
        d = ModelDescriptor(model_id="meta-llama/Llama-3.2-3B-Instruct")
        gb = b.estimate_memory_gb(d)
        assert gb > 0.0

    def test_estimate_memory_with_explicit_size(self):
        b = MLXBackend()
        d = ModelDescriptor(model_id="any/model", estimated_size_gb=7.5)
        gb = b.estimate_memory_gb(d)
        assert gb == 7.5

    def test_resolve_alias_known_model(self):
        d = ModelDescriptor(model_id="Qwen/Qwen3-8B")
        resolved = _resolve_id(d)
        assert resolved.startswith("mlx-community/")

    def test_resolve_alias_unknown_passthrough(self):
        d = ModelDescriptor(model_id="mlx-community/my-custom-model-4bit")
        resolved = _resolve_id(d)
        assert resolved == "mlx-community/my-custom-model-4bit"

    def test_resolve_path_override(self):
        d = ModelDescriptor(model_id="Qwen/Qwen3-8B", path="/local/weights")
        resolved = _resolve_id(d)
        assert resolved == "/local/weights"
