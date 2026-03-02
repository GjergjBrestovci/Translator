"""Unit tests for the translate CLI module (app/translate.py)."""

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_tokenizer(decoded: str = "hello world"):
    tok = MagicMock()
    encoded = MagicMock()
    encoded.to = MagicMock(return_value=encoded)
    tok.return_value = encoded
    tok.decode.return_value = decoded
    return tok


def _make_mock_model():
    model = MagicMock()
    model.generate.return_value = [MagicMock()]
    return model


# ---------------------------------------------------------------------------
# translate()
# ---------------------------------------------------------------------------

class TestTranslateFunction:
    def test_returns_decoded_string(self):
        from app.translate import translate

        tok = _make_mock_tokenizer("Good morning")
        mdl = _make_mock_model()

        result = translate("Miremëngjes", tok, mdl)
        assert result == "Good morning"

    def test_calls_generate_with_inputs(self):
        from app.translate import translate

        tok = _make_mock_tokenizer()
        mdl = _make_mock_model()
        encoded = tok.return_value

        translate("test", tok, mdl, max_length=128)

        mdl.generate.assert_called_once()
        call_kwargs = mdl.generate.call_args[1]
        assert call_kwargs["max_new_tokens"] == 128

    def test_uses_device(self):
        """Encoded inputs must be moved to DEVICE."""
        from app.translate import translate

        tok = _make_mock_tokenizer()
        mdl = _make_mock_model()
        encoded = tok.return_value

        translate("test", tok, mdl)

        encoded.to.assert_called_once()


# ---------------------------------------------------------------------------
# load_model()
# ---------------------------------------------------------------------------

class TestLoadModel:
    def test_returns_tokenizer_and_model(self, tmp_path):
        """load_model should call from_pretrained and return both objects."""
        from app import translate as translate_module

        mock_tok = MagicMock()
        mock_mdl = MagicMock()
        mock_mdl.to = MagicMock(return_value=mock_mdl)

        with (
            patch(
                "app.translate.AutoTokenizer.from_pretrained", return_value=mock_tok
            ),
            patch(
                "app.translate.AutoModelForSeq2SeqLM.from_pretrained",
                return_value=mock_mdl,
            ),
        ):
            tok, model = translate_module.load_model(str(tmp_path))

        assert tok is mock_tok
        assert model is mock_mdl
        mock_mdl.eval.assert_called_once()
        mock_mdl.to.assert_called_once()


# ---------------------------------------------------------------------------
# main() CLI
# ---------------------------------------------------------------------------

class TestMain:
    def test_exits_if_model_dir_missing(self, tmp_path):
        import sys
        from app.translate import main

        nonexistent = str(tmp_path / "no_model")
        with patch.object(sys, "argv", ["translate", "--model-dir", nonexistent]):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 1

    def test_translates_positional_args(self, tmp_path, capsys):
        """CLI with positional text args should print translation and exit."""
        import sys
        from app.translate import main

        mock_tok = _make_mock_tokenizer("Hello there")
        mock_mdl = _make_mock_model()
        mock_mdl.to = MagicMock(return_value=mock_mdl)

        # Create a dummy model directory so the path check passes
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        with (
            patch.object(sys, "argv", ["translate", "--model-dir", str(model_dir), "Mirëdita"]),
            patch("app.translate.AutoTokenizer.from_pretrained", return_value=mock_tok),
            patch(
                "app.translate.AutoModelForSeq2SeqLM.from_pretrained",
                return_value=mock_mdl,
            ),
        ):
            main()

        captured = capsys.readouterr()
        assert "Hello there" in captured.out
