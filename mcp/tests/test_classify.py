"""
Tests für T-MKIT-013: Dokumenten-Klassifizierung via LLM.

Alle Tests laufen ohne Docker-Container und ohne echte API-Calls.
Alle externen Abhängigkeiten werden per unittest.mock gemockt.
"""

import asyncio
import json
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conftest import load_server_module, run_async


# Einmal laden; alle Tests in diesem Modul teilen diese Instanz
_server = load_server_module(use_real_pil=False)
classify_document = _server.classify_document


def _make_mistral_response(doc_type: str, confidence: float) -> dict:
    """Erzeugt eine minimale Mistral-API-Antwort mit JSON im Content."""
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps({"type": doc_type, "confidence": confidence})
                }
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestClassifyDocument:
    """Tests für die classify_document() Funktion."""

    def test_classify_document_returns_type(self):
        """AC-013-2: document_type wird korrekt aus der API-Antwort extrahiert."""
        api_response = _make_mistral_response("invoice", 0.92)

        with patch.object(_server, "MISTRAL_API_KEY", "test-key"), \
             patch.object(_server, "call_mistral_vision_api", new=AsyncMock(return_value=api_response)):
            result = run_async(classify_document("Rechnung Nr. 1234\nBetrag: 100 EUR"))

        assert result["document_type"] == "invoice"

    def test_classify_document_returns_confidence(self):
        """AC-013-3: document_type_confidence liegt zwischen 0.0 und 1.0."""
        api_response = _make_mistral_response("contract", 0.85)

        with patch.object(_server, "MISTRAL_API_KEY", "test-key"), \
             patch.object(_server, "call_mistral_vision_api", new=AsyncMock(return_value=api_response)):
            result = run_async(classify_document("Dieser Vertrag wird zwischen ..."))

        assert "document_type_confidence" in result
        confidence = result["document_type_confidence"]
        assert 0.0 <= confidence <= 1.0
        assert abs(confidence - 0.85) < 1e-9

    def test_classify_custom_categories(self):
        """AC-013-6: Custom Kategorien werden an den Prompt übergeben und respektiert."""
        custom_categories = ["receipt", "memo", "report"]
        api_response = _make_mistral_response("memo", 0.78)

        with patch.object(_server, "MISTRAL_API_KEY", "test-key"), \
             patch.object(_server, "call_mistral_vision_api", new=AsyncMock(return_value=api_response)) as mock_api:
            result = run_async(classify_document(
                "Kurze Mitteilung an alle Mitarbeiter ...",
                categories=custom_categories,
            ))

        assert result["document_type"] == "memo"
        # Sicherstellen, dass die Custom-Kategorien im Payload des API-Calls stehen
        call_payload = mock_api.call_args[0][0]
        user_content = call_payload["messages"][1]["content"]
        for cat in custom_categories:
            assert cat in user_content

    def test_classify_default_off(self):
        """AC-013-5: Kein API-Call wenn MISTRAL_API_KEY nicht gesetzt (Feature safety)."""
        with patch.object(_server, "MISTRAL_API_KEY", ""), \
             patch.object(_server, "call_mistral_vision_api", new=AsyncMock()) as mock_api:
            result = run_async(classify_document("Irgendein Dokument"))
            mock_api.assert_not_called()

        assert result["document_type"] == "other"
        assert result["document_type_confidence"] == 0.0

    def test_classify_invalid_json_response(self):
        """Graceful Fallback bei ungültigem LLM-Output (kein JSON)."""
        bad_response = {
            "choices": [
                {
                    "message": {
                        "content": "Das ist keine JSON-Antwort."
                    }
                }
            ]
        }

        with patch.object(_server, "MISTRAL_API_KEY", "test-key"), \
             patch.object(_server, "call_mistral_vision_api", new=AsyncMock(return_value=bad_response)):
            result = run_async(classify_document("Beliebiger Text"))

        assert result["document_type"] == "other"
        assert result["document_type_confidence"] == 0.0

    def test_classify_api_error(self):
        """Graceful Handling bei API-Fehler (Exception)."""
        with patch.object(_server, "MISTRAL_API_KEY", "test-key"), \
             patch.object(_server, "call_mistral_vision_api", new=AsyncMock(side_effect=Exception("Connection refused"))):
            result = run_async(classify_document("Beliebiger Text"))

        assert result["document_type"] == "other"
        assert result["document_type_confidence"] == 0.0

    def test_classify_unknown_type_falls_back_to_other(self):
        """Wenn LLM einen nicht erlaubten Typ zurückgibt → Fallback 'other'."""
        api_response = _make_mistral_response("classified_ad", 0.9)

        with patch.object(_server, "MISTRAL_API_KEY", "test-key"), \
             patch.object(_server, "call_mistral_vision_api", new=AsyncMock(return_value=api_response)):
            result = run_async(classify_document("Stellenausschreibung ..."))

        # "classified_ad" ist kein erlaubter Default-Typ → muss zu "other" normalisiert werden
        assert result["document_type"] == "other"

    def test_classify_confidence_clamped(self):
        """Konfidenz-Werte > 1.0 werden auf 1.0 begrenzt."""
        api_response = _make_mistral_response("invoice", 1.5)

        with patch.object(_server, "MISTRAL_API_KEY", "test-key"), \
             patch.object(_server, "call_mistral_vision_api", new=AsyncMock(return_value=api_response)):
            result = run_async(classify_document("Rechnung ..."))

        assert result["document_type_confidence"] <= 1.0

    def test_classify_uses_default_categories_when_none(self):
        """Wenn categories=None, werden DEFAULT_CLASSIFY_CATEGORIES verwendet."""
        api_response = _make_mistral_response("cv", 0.88)

        with patch.object(_server, "MISTRAL_API_KEY", "test-key"), \
             patch.object(_server, "call_mistral_vision_api", new=AsyncMock(return_value=api_response)) as mock_api:
            result = run_async(classify_document("Max Mustermann\nLebenslauf ..."))

        assert result["document_type"] == "cv"
        # Payload soll die Default-Kategorien enthalten
        call_payload = mock_api.call_args[0][0]
        user_content = call_payload["messages"][1]["content"]
        for cat in _server.DEFAULT_CLASSIFY_CATEGORIES:
            assert cat in user_content

    def test_classify_json_with_markdown_codeblock(self):
        """LLM-Antwort mit Markdown-Code-Block wird korrekt geparst."""
        json_in_codeblock = '```json\n{"type": "letter", "confidence": 0.75}\n```'
        wrapped_response = {
            "choices": [{"message": {"content": json_in_codeblock}}]
        }

        with patch.object(_server, "MISTRAL_API_KEY", "test-key"), \
             patch.object(_server, "call_mistral_vision_api", new=AsyncMock(return_value=wrapped_response)):
            result = run_async(classify_document("Sehr geehrte Damen und Herren ..."))

        assert result["document_type"] == "letter"
        assert abs(result["document_type_confidence"] - 0.75) < 1e-9
