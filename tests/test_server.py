"""Unit / integration tests for the FastAPI translation server.

Run with:
    pytest tests/test_server.py -v
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers to mock the model so tests don't require a trained checkpoint
# ---------------------------------------------------------------------------

def _make_mock_tokenizer():
    tok = MagicMock()
    # tokenizer(text, ...) returns a dict-like that has .to(device)
    encoded = MagicMock()
    encoded.to = MagicMock(return_value=encoded)
    tok.return_value = encoded
    tok.decode.return_value = "Hello, how are you?"
    tok.batch_decode.return_value = ["Hello!", "Good morning."]
    return tok


def _make_mock_model():
    model = MagicMock()
    model.to = MagicMock(return_value=model)
    model.generate.return_value = [MagicMock()]
    return model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client(tmp_path):
    """Return a TestClient with the model mocked out so no files are needed.

    We patch at the from_pretrained level so the lifespan startup succeeds
    without requiring a real checkpoint on disk.
    """
    mock_tok = _make_mock_tokenizer()
    mock_mdl = _make_mock_model()

    with (
        patch("app.server.MODEL_DIR", str(tmp_path)),
        patch("app.server.AutoTokenizer.from_pretrained", return_value=mock_tok),
        patch(
            "app.server.AutoModelForSeq2SeqLM.from_pretrained", return_value=mock_mdl
        ),
    ):
        from app.server import app

        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert "device" in data

    def test_health_model_unloaded(self, client):
        import app.server as server_module

        with (
            patch.object(server_module, "_tokenizer", None),
            patch.object(server_module, "_model", None),
        ):
            resp = client.get("/health")
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# /translate
# ---------------------------------------------------------------------------

class TestTranslate:
    def test_translate_success(self, client):
        resp = client.post("/translate", json={"text": "Mirëdita, si jeni?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "translation" in data
        assert data["source_length"] == len("Mirëdita, si jeni?")
        assert "model" in data

    def test_translate_empty_text_rejected(self, client):
        resp = client.post("/translate", json={"text": ""})
        assert resp.status_code == 422  # Pydantic min_length=1

    def test_translate_text_too_long_rejected(self, client):
        resp = client.post("/translate", json={"text": "a" * 5001})
        assert resp.status_code == 422

    def test_translate_invalid_num_beams(self, client):
        resp = client.post("/translate", json={"text": "test", "num_beams": 0})
        assert resp.status_code == 422

    def test_translate_model_not_loaded(self, client):
        import app.server as server_module

        with (
            patch.object(server_module, "_tokenizer", None),
            patch.object(server_module, "_model", None),
        ):
            resp = client.post("/translate", json={"text": "test"})
        assert resp.status_code == 503

    def test_translate_inference_error(self, client):
        import app.server as server_module

        mock_mdl = _make_mock_model()
        mock_mdl.generate.side_effect = RuntimeError("CUDA error")

        with patch.object(server_module, "_model", mock_mdl):
            resp = client.post("/translate", json={"text": "test"})
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# /translate/batch
# ---------------------------------------------------------------------------

class TestTranslateBatch:
    def test_batch_translate_success(self, client):
        resp = client.post(
            "/translate/batch",
            json={"texts": ["Mirëdita", "Miremëngjes"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "translations" in data
        assert data["count"] == len(data["translations"])

    def test_batch_empty_list_rejected(self, client):
        resp = client.post("/translate/batch", json={"texts": []})
        assert resp.status_code == 422

    def test_batch_too_many_items_rejected(self, client):
        resp = client.post("/translate/batch", json={"texts": ["x"] * 33})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Static / index
# ---------------------------------------------------------------------------

class TestStaticRoutes:
    def test_openapi_schema_available(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert schema["info"]["title"] == "Albanian Translator"
