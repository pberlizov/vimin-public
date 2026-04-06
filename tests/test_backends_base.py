"""Tests for backend base classes and ModelDescriptor."""
import pytest
from vimin_core.core.backends.base import (
    BaseBackend, ModelDescriptor, InsufficientMemoryError,
)


class TestModelDescriptor:
    def test_defaults(self):
        d = ModelDescriptor(model_id="org/model")
        assert d.model_id == "org/model"
        assert d.task == "text-generation"
        assert d.path is None
        assert d.quantization is None
        assert d.max_context == 2048
        assert d.estimated_size_gb is None
        assert d.format is None

    def test_custom(self):
        d = ModelDescriptor(
            model_id="org/model",
            task="automatic-speech-recognition",
            quantization="4bit",
            max_context=4096,
            estimated_size_gb=5.0,
            format="mlx",
        )
        assert d.task == "automatic-speech-recognition"
        assert d.quantization == "4bit"
        assert d.max_context == 4096
        assert d.estimated_size_gb == 5.0
        assert d.format == "mlx"


class TestInsufficientMemoryError:
    def test_is_exception(self):
        e = InsufficientMemoryError("not enough RAM")
        assert isinstance(e, Exception)
        assert "not enough RAM" in str(e)


class ConcreteBackend(BaseBackend):
    """Minimal concrete implementation for testing the abstract interface."""

    def is_available(self) -> bool:
        return True

    def estimate_memory_gb(self, descriptor: ModelDescriptor) -> float:
        return 1.0

    def load(self, descriptor: ModelDescriptor) -> bool:
        self._loaded = True
        return True

    def generate(self, prompt, max_new_tokens=256, temperature=0.7, stop_sequences=None):
        return f"response to: {prompt}"

    def stream_generate(self, prompt, max_new_tokens=256, temperature=0.7, stop_sequences=None):
        for word in f"response to: {prompt}".split():
            yield word + " "

    def unload(self) -> None:
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return getattr(self, "_loaded", False)


class TestBaseBackend:
    def test_concrete_backend_lifecycle(self):
        b = ConcreteBackend()
        d = ModelDescriptor(model_id="test/model")
        assert b.is_available()
        assert not b.is_loaded
        assert b.load(d)
        assert b.is_loaded
        assert b.generate("hello").startswith("response")
        b.unload()
        assert not b.is_loaded

    def test_estimate_memory_gb(self):
        b = ConcreteBackend()
        d = ModelDescriptor(model_id="test/model")
        assert b.estimate_memory_gb(d) == 1.0

    def test_stream_generate(self):
        b = ConcreteBackend()
        d = ModelDescriptor(model_id="test/model")
        b.load(d)
        tokens = list(b.stream_generate("hello"))
        assert len(tokens) >= 1
        full = "".join(tokens)
        assert "response" in full
