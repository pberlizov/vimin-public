"""Tests for OpenClawBackend — all network calls mocked."""
import json
import pytest
from unittest.mock import patch, MagicMock
from io import BytesIO

from vimin_core.core.backends.openclaw_backend import OpenClawBackend, _load_openclaw_token
from vimin_core.core.backends.base import ModelDescriptor


class TestLoadOpenclawToken:
    def test_returns_empty_string_when_no_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENCLAW_TOKEN", "")
        monkeypatch.setattr(
            "vimin_core.core.backends.openclaw_backend._TOKEN_PATH",
            tmp_path / "does_not_exist.json",
        )
        token = _load_openclaw_token()
        assert token == ""

    def test_reads_env_var_fallback(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENCLAW_TOKEN", "test-token-123")
        monkeypatch.setattr(
            "vimin_core.core.backends.openclaw_backend._TOKEN_PATH",
            tmp_path / "does_not_exist.json",
        )
        token = _load_openclaw_token()
        assert token == "test-token-123"

    def test_reads_from_json_file(self, tmp_path, monkeypatch):
        config_file = tmp_path / "openclaw.json"
        config_file.write_text(json.dumps({"gateway": {"auth": {"token": "file-token"}}}))
        monkeypatch.setattr(
            "vimin_core.core.backends.openclaw_backend._TOKEN_PATH",
            config_file,
        )
        token = _load_openclaw_token()
        assert token == "file-token"


def _make_urllib_response(body: bytes, status: int = 200):
    """Create a minimal mock for urllib.request.urlopen context manager."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.__iter__ = MagicMock(return_value=iter([]))
    return mock_resp


class TestOpenClawBackendAvailability:
    def test_not_available_when_gateway_down(self):
        import urllib.error
        b = OpenClawBackend(url="http://127.0.0.1:19999", token="")
        with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            assert not b.is_available()

    def test_available_when_gateway_up(self):
        b = OpenClawBackend(url="http://127.0.0.1:18789", token="tok")
        mock_resp = _make_urllib_response(b'{"status":"ok"}')
        with patch("urllib.request.urlopen", return_value=mock_resp):
            assert b.is_available()

    def test_estimate_memory_always_zero(self):
        b = OpenClawBackend()
        d = ModelDescriptor(model_id="any/model")
        assert b.estimate_memory_gb(d) == 0.0


class TestOpenClawBackendLoad:
    def test_load_fails_when_gateway_down(self):
        b = OpenClawBackend(url="http://127.0.0.1:19999", token="")
        d = ModelDescriptor(model_id="any/model")
        with patch("urllib.request.urlopen", side_effect=OSError("refused")):
            assert not b.load(d)
        assert not b.is_loaded

    def test_load_succeeds_when_gateway_up(self):
        b = OpenClawBackend(url="http://127.0.0.1:18789", token="tok")
        d = ModelDescriptor(model_id="openclaw")
        mock_resp = _make_urllib_response(b'{"status":"ok"}')
        with patch("urllib.request.urlopen", return_value=mock_resp):
            assert b.load(d)
        assert b.is_loaded

    def test_unload(self):
        b = OpenClawBackend(url="http://127.0.0.1:18789", token="tok")
        d = ModelDescriptor(model_id="openclaw")
        mock_resp = _make_urllib_response(b'{"status":"ok"}')
        with patch("urllib.request.urlopen", return_value=mock_resp):
            b.load(d)
        assert b.is_loaded
        b.unload()
        assert not b.is_loaded


class TestOpenClawBackendGenerate:
    def _loaded_backend(self) -> OpenClawBackend:
        b = OpenClawBackend(url="http://127.0.0.1:18789", token="tok")
        mock_resp = _make_urllib_response(b'{"status":"ok"}')
        with patch("urllib.request.urlopen", return_value=mock_resp):
            d = ModelDescriptor(model_id="openclaw")
            b.load(d)
        return b

    def test_generate_returns_content(self):
        b = self._loaded_backend()
        completion = {"choices": [{"message": {"content": "hello world"}}]}
        mock_resp = _make_urllib_response(json.dumps(completion).encode())
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = b.generate("say hello")
        assert result == "hello world"

    def test_generate_raises_when_not_loaded(self):
        b = OpenClawBackend()
        with pytest.raises(RuntimeError, match="call load"):
            b.generate("hello")

    def test_available_models_returns_list(self):
        b = OpenClawBackend(url="http://127.0.0.1:18789", token="tok")
        data = {"data": [{"id": "model-a"}, {"id": "model-b"}]}
        mock_resp = _make_urllib_response(json.dumps(data).encode())
        with patch("urllib.request.urlopen", return_value=mock_resp):
            models = b.available_models()
        assert models == ["model-a", "model-b"]

    def test_available_models_returns_empty_on_error(self):
        b = OpenClawBackend(url="http://127.0.0.1:18789", token="tok")
        with patch("urllib.request.urlopen", side_effect=OSError("refused")):
            models = b.available_models()
        assert models == []
