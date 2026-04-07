"""Tests for BackendSelector routing logic."""
import platform
import pytest
from unittest.mock import patch, MagicMock

from vimin_core.core.backends.base import ModelDescriptor
from vimin_core.core.backends.selector import BackendSelector, _GENERATIVE_TASKS, _ASR_TASKS, _ENCODER_TASKS


class TestBackendSelectorRouting:
    """Test descriptor classification — no real model loading."""

    def setup_method(self):
        self.selector = BackendSelector()

    def test_generative_task_detected(self):
        d = ModelDescriptor(model_id="x", task="text-generation")
        assert self.selector.needs_generative_backend(d)

    def test_asr_task_not_generative(self):
        d = ModelDescriptor(model_id="x", task="automatic-speech-recognition")
        assert not self.selector.needs_generative_backend(d)

    def test_encoder_task_not_generative(self):
        d = ModelDescriptor(model_id="x", task="fill-mask")
        assert not self.selector.needs_generative_backend(d)

    def test_onnx_format_not_generative(self):
        d = ModelDescriptor(model_id="x", task="text-generation", format="onnx")
        assert not self.selector.needs_generative_backend(d)

    def test_mlx_format_is_generative(self):
        d = ModelDescriptor(model_id="x", task="text-generation", format="mlx")
        assert self.selector.needs_generative_backend(d)

    def test_gguf_format_is_generative(self):
        d = ModelDescriptor(model_id="x", task="text-generation", format="gguf")
        assert self.selector.needs_generative_backend(d)

    def test_is_asr_task(self):
        d = ModelDescriptor(model_id="x", task="automatic-speech-recognition")
        assert self.selector.is_asr_task(d)
        d2 = ModelDescriptor(model_id="x", task="audio-classification")
        assert self.selector.is_asr_task(d2)

    def test_is_not_asr_task(self):
        d = ModelDescriptor(model_id="x", task="text-generation")
        assert not self.selector.is_asr_task(d)

    def test_available_backends_returns_dict(self):
        backends = self.selector.available_backends()
        assert isinstance(backends, dict)
        assert set(backends.keys()) == {"mlx", "llamacpp", "whisper_mlx", "whisper_cpu", "openclaw"}
        for v in backends.values():
            assert isinstance(v, bool)

    def test_install_instructions_returns_string(self):
        d = ModelDescriptor(model_id="x", task="text-generation")
        s = self.selector.install_instructions(d)
        assert isinstance(s, str)
        assert len(s) > 10


class TestBackendSelectorMemoryRecommendation:
    def setup_method(self):
        self.selector = BackendSelector()

    def test_recommend_returns_valid_quantization(self):
        # With ample RAM, should recommend something
        result = self.selector.recommend_quantization("meta-llama/Llama-3.2-1B-Instruct", 32.0)
        assert result in ("fp16", "8bit", "4bit")

    def test_recommend_4bit_for_tight_ram(self):
        # 1.7 GB available — fp16 needs 3.0 GB (2.0*1.0+1.0), 8bit needs 2.0 GB (2.0*0.5+1.0),
        # so only 4bit fits (needs 1.5 GB: 2.0*0.25+1.0)
        result = self.selector.recommend_quantization("meta-llama/Llama-3.2-1B-Instruct", 1.7)
        assert result == "4bit"

    def test_recommend_insufficient_memory_raises(self):
        from vimin_core.core.backends.base import InsufficientMemoryError
        # 0.1 GB available — should raise for any model
        with pytest.raises(InsufficientMemoryError):
            self.selector.recommend_quantization("meta-llama/Llama-3.3-70B-Instruct", 0.1)

    def test_recommend_fp16_with_lots_of_ram(self):
        # 200 GB available — small model should get fp16
        result = self.selector.recommend_quantization("Qwen/Qwen3-0.6B", 200.0)
        assert result == "fp16"


class TestBackendSelectorSelect:
    """Test select() routing — mock backends to avoid hardware dependency."""

    def _make_selector_with_mocks(self, mlx_available=False, llamacpp_available=False, whisper_available=False):
        selector = BackendSelector()
        selector._mlx = MagicMock()
        selector._mlx.is_available.return_value = mlx_available
        selector._llamacpp = MagicMock()
        selector._llamacpp.is_available.return_value = llamacpp_available
        selector._whisper = MagicMock()
        selector._whisper.is_available.return_value = whisper_available
        selector._openclaw = MagicMock()
        selector._openclaw.is_available.return_value = False
        return selector

    def test_asr_routes_to_whisper_when_available(self):
        selector = self._make_selector_with_mocks(whisper_available=True)
        d = ModelDescriptor(model_id="openai/whisper-base", task="automatic-speech-recognition")
        backend = selector.select(d)
        assert backend is selector._whisper

    def test_asr_returns_none_when_whisper_unavailable(self):
        selector = self._make_selector_with_mocks(whisper_available=False)
        d = ModelDescriptor(model_id="openai/whisper-base", task="automatic-speech-recognition")
        backend = selector.select(d)
        assert backend is None

    def test_encoder_returns_none(self):
        selector = self._make_selector_with_mocks(mlx_available=True)
        d = ModelDescriptor(model_id="bert-base", task="fill-mask")
        assert selector.select(d) is None

    @patch("vimin_core.core.backends.selector._is_apple_silicon", return_value=False)
    def test_no_apple_silicon_falls_through_to_llamacpp(self, _):
        selector = self._make_selector_with_mocks(mlx_available=True, llamacpp_available=True)
        d = ModelDescriptor(model_id="any/model", task="text-generation")
        backend = selector.select(d)
        assert backend is selector._llamacpp

    @patch("vimin_core.core.backends.selector._is_apple_silicon", return_value=True)
    def test_apple_silicon_prefers_mlx(self, _):
        selector = self._make_selector_with_mocks(mlx_available=True, llamacpp_available=True)
        d = ModelDescriptor(model_id="any/model", task="text-generation")
        backend = selector.select(d)
        assert backend is selector._mlx

    @patch("vimin_core.core.backends.selector._is_apple_silicon", return_value=True)
    def test_apple_silicon_falls_through_when_mlx_unavailable(self, _):
        selector = self._make_selector_with_mocks(mlx_available=False, llamacpp_available=True)
        d = ModelDescriptor(model_id="any/model", task="text-generation")
        backend = selector.select(d)
        assert backend is selector._llamacpp

    @patch("vimin_core.core.backends.selector._is_apple_silicon", return_value=False)
    def test_no_backends_available_returns_none(self, _):
        selector = self._make_selector_with_mocks(mlx_available=False, llamacpp_available=False)
        d = ModelDescriptor(model_id="any/model", task="text-generation")
        assert selector.select(d) is None

    @patch.dict("os.environ", {"OPENCLAW_URL": "http://127.0.0.1:18789"}, clear=False)
    def test_openclaw_requested_prefers_openclaw_backend(self):
        selector = self._make_selector_with_mocks(mlx_available=True, llamacpp_available=True)
        selector._openclaw.is_available.return_value = True
        d = ModelDescriptor(model_id="any/model", task="text-generation")
        backend = selector.select(d)
        assert backend is selector._openclaw

    @patch.dict("os.environ", {"OPENCLAW_URL": "http://127.0.0.1:18789"}, clear=False)
    @patch("vimin_core.core.backends.selector._is_apple_silicon", return_value=True)
    def test_openclaw_unavailable_falls_back_to_local_backend(self, _):
        selector = self._make_selector_with_mocks(mlx_available=True, llamacpp_available=True)
        selector._openclaw.is_available.return_value = False
        d = ModelDescriptor(model_id="any/model", task="text-generation")
        backend = selector.select(d)
        assert backend is selector._mlx
