"""
Tests für T-MKIT-023: Validierung, Error Handling, Cleanup, VERSION.

Prüft:
- accuracy Enum-Validierung (ConvertRequest + ExtractRequest)
- chunk_size ge=1 Validierung
- VERSION == "2.0.0"
- High-Accuracy ohne API-Key → Warning in Meta
- output_format ist "Reserved for future use" (nicht entfernt, aber nicht funktional genutzt)
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

# models.py liegt direkt in mcp/ — sys.path anpassen
_mcp_path = str(Path(__file__).parent.parent)
if _mcp_path not in sys.path:
    sys.path.insert(0, _mcp_path)

from models import ConvertRequest, ExtractRequest

from conftest import load_server_module, run_async, PNG_100x100


# Einmal laden; alle Tests in diesem Modul teilen diese Instanz
_server = load_server_module(use_real_pil=False)
convert_auto = _server.convert_auto


# ---------------------------------------------------------------------------
# Tests: accuracy Enum-Validierung
# ---------------------------------------------------------------------------


class TestAccuracyValidation:
    """accuracy muss 'standard' oder 'high' sein — alles andere ist ein Fehler."""

    def test_accuracy_invalid_value_raises_convert_request(self):
        """accuracy='ultra' → ValidationError in ConvertRequest."""
        with pytest.raises(ValidationError) as exc_info:
            ConvertRequest(path="/data/test.pdf", accuracy="ultra")
        errors = exc_info.value.errors()
        assert any("accuracy" in str(e) for e in errors)

    def test_accuracy_invalid_value_raises_extract_request(self):
        """accuracy='ultra' → ValidationError in ExtractRequest."""
        with pytest.raises(ValidationError) as exc_info:
            ExtractRequest(path="/data/test.pdf", accuracy="ultra")
        errors = exc_info.value.errors()
        assert any("accuracy" in str(e) for e in errors)

    def test_accuracy_standard_valid(self):
        """accuracy='standard' ist gültig."""
        req = ConvertRequest(path="/data/test.pdf", accuracy="standard")
        assert req.accuracy == "standard"

    def test_accuracy_high_valid(self):
        """accuracy='high' ist gültig."""
        req = ConvertRequest(path="/data/test.pdf", accuracy="high")
        assert req.accuracy == "high"

    def test_accuracy_default_is_standard(self):
        """Default-Wert von accuracy ist 'standard'."""
        req = ConvertRequest(path="/data/test.pdf")
        assert req.accuracy == "standard"

    def test_accuracy_empty_string_raises(self):
        """accuracy='' → ValidationError."""
        with pytest.raises(ValidationError):
            ConvertRequest(path="/data/test.pdf", accuracy="")

    def test_accuracy_extract_request_standard_valid(self):
        """ExtractRequest akzeptiert accuracy='standard'."""
        req = ExtractRequest(path="/data/test.pdf", accuracy="standard")
        assert req.accuracy == "standard"

    def test_accuracy_extract_request_high_valid(self):
        """ExtractRequest akzeptiert accuracy='high'."""
        req = ExtractRequest(path="/data/test.pdf", accuracy="high")
        assert req.accuracy == "high"


# ---------------------------------------------------------------------------
# Tests: chunk_size Validierung
# ---------------------------------------------------------------------------


class TestChunkSizeValidation:
    """chunk_size muss >= 1 sein."""

    def test_chunk_size_zero_raises(self):
        """chunk_size=0 → ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ConvertRequest(path="/data/test.pdf", chunk_size=0)
        errors = exc_info.value.errors()
        assert any("chunk_size" in str(e) for e in errors)

    def test_chunk_size_negative_raises(self):
        """chunk_size=-1 → ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ConvertRequest(path="/data/test.pdf", chunk_size=-1)
        errors = exc_info.value.errors()
        assert any("chunk_size" in str(e) for e in errors)

    def test_chunk_size_one_valid(self):
        """chunk_size=1 ist gültig (Mindestgröße)."""
        req = ConvertRequest(path="/data/test.pdf", chunk_size=1)
        assert req.chunk_size == 1

    def test_chunk_size_default_is_512(self):
        """Default chunk_size ist 512."""
        req = ConvertRequest(path="/data/test.pdf")
        assert req.chunk_size == 512

    def test_chunk_size_large_valid(self):
        """chunk_size=4096 ist gültig."""
        req = ConvertRequest(path="/data/test.pdf", chunk_size=4096)
        assert req.chunk_size == 4096


# ---------------------------------------------------------------------------
# Tests: VERSION
# ---------------------------------------------------------------------------


class TestVersion:
    """VERSION muss auf 2.0.0 gesetzt sein."""

    def test_version_is_2_0_0(self):
        """VERSION == '2.0.0'."""
        assert _server.VERSION == "2.0.0"


# ---------------------------------------------------------------------------
# Tests: High-Accuracy ohne API-Key → Warning in Meta
# ---------------------------------------------------------------------------


class TestHighAccuracyNoKeyWarning:
    """Wenn accuracy='high' aber kein MISTRAL_API_KEY → Warning in Meta, Degradierung auf 'standard'."""

    def _make_kwargs(self, **overrides) -> dict:
        defaults = dict(
            file_data=PNG_100x100,
            filename="test.png",
            source="test.png",
            source_type="base64",
            input_meta={},
            language="de",
            accuracy="high",
        )
        defaults.update(overrides)
        return defaults

    def test_high_accuracy_no_key_warning_in_meta(self):
        """
        accuracy='high' + kein API-Key → accuracy_warning in Meta + accuracy_mode='standard'.

        Das Warning und die Degradierung auf 'standard' werden gesetzt, BEVOR die
        Vision-Verarbeitung beginnt. Da kein API-Key gesetzt ist, schlägt die Vision
        fehl — aber accuracy_mode und accuracy_warning in Meta sind korrekt gesetzt.
        """
        with patch.object(_server, "MISTRAL_API_KEY", ""):
            result = run_async(convert_auto(**self._make_kwargs()))

        # Nach Degradierung: accuracy_mode muss 'standard' sein (nicht 'high')
        assert result.meta.accuracy_mode == "standard"
        # Warning muss in Meta als extra-Feld vorhanden sein
        warning = result.meta.model_extra.get("accuracy_warning")
        assert warning is not None
        assert "MISTRAL_API_KEY" in warning

    def test_high_accuracy_with_key_no_warning(self):
        """
        accuracy='high' + gültiger API-Key → kein accuracy_warning, accuracy_mode='high'.
        """
        vision_resp = {
            "choices": [{"message": {"content": "# High Accuracy Result"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        dual_pass_resp = {
            "choices": [{"message": {"content": "# Dual Pass Result"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

        with patch.object(_server, "MISTRAL_API_KEY", "test-key"), \
             patch.object(_server, "call_mistral_vision_api", new=AsyncMock(
                 side_effect=[vision_resp, dual_pass_resp]
             )):
            result = run_async(convert_auto(**self._make_kwargs()))

        assert result.success is True
        assert result.meta.accuracy_mode == "high"
        # Kein Warning
        warning = getattr(result.meta, "accuracy_warning", None) or \
                  result.meta.model_extra.get("accuracy_warning")
        assert warning is None


# ---------------------------------------------------------------------------
# Tests: output_format und password — Reserved for future use
# ---------------------------------------------------------------------------


class TestReservedFields:
    """output_format und password sind als 'Reserved for future use' markiert — müssen weiterhin akzeptiert werden."""

    def test_output_format_still_accepted(self):
        """output_format ist noch im Modell (Reserved) und wird akzeptiert."""
        req = ConvertRequest(path="/data/test.pdf", output_format="markdown")
        assert req.output_format == "markdown"

    def test_password_still_accepted(self):
        """password ist noch im Modell (Reserved) und wird akzeptiert."""
        req = ConvertRequest(path="/data/test.pdf", password="secret")
        assert req.password == "secret"
