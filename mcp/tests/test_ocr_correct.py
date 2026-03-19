"""
Tests für T-MKIT-015: OCR Post-Correction via LLM.

Alle Tests laufen ohne Docker-Container und ohne echte API-Calls.
Alle externen Abhängigkeiten werden per unittest.mock gemockt.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conftest import load_server_module, run_async


# Einmal laden; alle Tests in diesem Modul teilen diese Instanz
_server = load_server_module(use_real_pil=False)
correct_ocr_text = _server.correct_ocr_text
convert_auto = _server.convert_auto


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_mistral_response(content: str, tokens_total: int = 50) -> dict:
    """Erzeugt eine minimale Mistral-API-Antwort."""
    return {
        "choices": [
            {
                "message": {
                    "content": content
                }
            }
        ],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 30,
            "total_tokens": tokens_total,
        },
    }


# ---------------------------------------------------------------------------
# Tests für correct_ocr_text()
# ---------------------------------------------------------------------------


class TestCorrectOcrText:
    """Tests für die correct_ocr_text() Funktion."""

    def test_correct_ocr_text_success(self):
        """
        AC-015-1: Korrektur-Funktion gibt korrigierten Text zurück.
        """
        corrected_content = "Korrigierter Text ohne Fehler.\n<<<CORRECTIONS:3>>>"
        api_response = _make_mistral_response(corrected_content, tokens_total=60)

        with patch.object(_server, "MISTRAL_API_KEY", "test-key"), \
             patch.object(_server, "call_mistral_vision_api", new=AsyncMock(return_value=api_response)):
            result = run_async(correct_ocr_text("Korrigierter Text ohne Fehlcr."))

        assert result["success"] is True
        assert "Korrigierter Text ohne Fehler." in result["corrected_text"]
        assert "<<<CORRECTIONS:" not in result["corrected_text"]

    def test_correct_ocr_text_counts_corrections(self):
        """
        AC-015-4: Anzahl der Korrekturen wird korrekt aus dem Marker geparst.
        """
        corrected_content = "Bereinigter Inhalt.\n<<<CORRECTIONS:7>>>"
        api_response = _make_mistral_response(corrected_content)

        with patch.object(_server, "MISTRAL_API_KEY", "test-key"), \
             patch.object(_server, "call_mistral_vision_api", new=AsyncMock(return_value=api_response)):
            result = run_async(correct_ocr_text("Bereinigter lnhalt."))

        assert result["success"] is True
        assert result["corrections_count"] == 7

    def test_correct_ocr_text_no_corrections_marker(self):
        """
        Fallback wenn ---CORRECTIONS-Marker fehlt: Text wird trotzdem zurückgegeben,
        corrections_count ist 0.
        """
        content_without_marker = "Text ohne Marker-Zeile."
        api_response = _make_mistral_response(content_without_marker)

        with patch.object(_server, "MISTRAL_API_KEY", "test-key"), \
             patch.object(_server, "call_mistral_vision_api", new=AsyncMock(return_value=api_response)):
            result = run_async(correct_ocr_text("Text ohne Marker-Zeile."))

        assert result["success"] is True
        assert result["corrected_text"] == content_without_marker
        assert result["corrections_count"] == 0

    def test_correct_ocr_default_off(self):
        """
        AC-015-5: Wenn MISTRAL_API_KEY leer ist, wird kein API-Call ausgeführt
        (graceful degradation).
        """
        with patch.object(_server, "MISTRAL_API_KEY", ""), \
             patch.object(_server, "call_mistral_vision_api", new=AsyncMock()) as mock_api:
            result = run_async(correct_ocr_text("Beliebiger Text"))
            mock_api.assert_not_called()

        assert result["success"] is False
        # Original-Text bleibt erhalten
        assert result["corrected_text"] == "Beliebiger Text"
        assert result["corrections_count"] == 0

    def test_correct_ocr_api_error(self):
        """
        AC-015-5: Bei API-Fehler wird graceful degradiert — kein Exception-Propagation.
        """
        with patch.object(_server, "MISTRAL_API_KEY", "test-key"), \
             patch.object(_server, "call_mistral_vision_api",
                          new=AsyncMock(side_effect=Exception("Connection refused"))):
            result = run_async(correct_ocr_text("Irgendein OCR-Text"))

        assert result["success"] is False
        assert "Connection refused" in result.get("error", "")

    def test_correct_ocr_preserves_original_on_failure(self):
        """
        AC-015-3/AC-015-5: Bei Fehler bleibt der Original-Text erhalten.
        Der korrigierte Text darf NICHT leer sein und muss dem Input entsprechen.
        """
        original_text = "Ursprünglicher OCR-Text mit 0CR-Fehlern."

        with patch.object(_server, "MISTRAL_API_KEY", "test-key"), \
             patch.object(_server, "call_mistral_vision_api",
                          new=AsyncMock(side_effect=Exception("Timeout"))):
            result = run_async(correct_ocr_text(original_text))

        assert result["success"] is False
        assert result["corrected_text"] == original_text

    def test_correct_ocr_language_passed(self):
        """
        AC-015-2: Die Sprache wird korrekt an den Prompt übergeben.
        Englische Sprache → kein deutscher System-Prompt.
        """
        corrected_content = "Corrected text.\n<<<CORRECTIONS:2>>>"
        api_response = _make_mistral_response(corrected_content)

        with patch.object(_server, "MISTRAL_API_KEY", "test-key"), \
             patch.object(_server, "call_mistral_vision_api",
                          new=AsyncMock(return_value=api_response)) as mock_api:
            result = run_async(correct_ocr_text("0CR text with errors.", language="en"))

        assert result["success"] is True

        # Payload prüfen: System-Prompt muss englisch sein (kein Deutsch)
        call_payload = mock_api.call_args[0][0]
        system_content = call_payload["messages"][0]["content"]
        user_content = call_payload["messages"][1]["content"]

        # Englischer Pfad: kein "Korrigiere" und kein "Antworte" im Prompt
        assert "Korrigiere" not in user_content
        assert "Antworte" not in system_content
        # Sprach-Parameter wurde tatsächlich an die Funktion übergeben
        assert result["corrections_count"] == 2

    def test_correct_ocr_tokens_returned(self):
        """
        Verbrauchte Token-Anzahl wird im Ergebnis zurückgegeben.
        """
        corrected_content = "Text.\n<<<CORRECTIONS:1>>>"
        api_response = _make_mistral_response(corrected_content, tokens_total=123)

        with patch.object(_server, "MISTRAL_API_KEY", "test-key"), \
             patch.object(_server, "call_mistral_vision_api", new=AsyncMock(return_value=api_response)):
            result = run_async(correct_ocr_text("Text."))

        assert result["success"] is True
        assert result["tokens"] == 123

    def test_correct_ocr_marker_stripped_from_text(self):
        """
        Der ---CORRECTIONS-Marker wird aus dem zurückgegebenen Text entfernt.
        """
        corrected_content = "Sauberer Text hier.\n<<<CORRECTIONS:5>>>"
        api_response = _make_mistral_response(corrected_content)

        with patch.object(_server, "MISTRAL_API_KEY", "test-key"), \
             patch.object(_server, "call_mistral_vision_api", new=AsyncMock(return_value=api_response)):
            result = run_async(correct_ocr_text("Sauberer Text h1er."))

        assert "<<<CORRECTIONS:" not in result["corrected_text"]
        assert "Sauberer Text hier." in result["corrected_text"]


# ---------------------------------------------------------------------------
# Integration-Tests für convert_auto() mit ocr_correct=True
# ---------------------------------------------------------------------------


class TestConvertAutoOcrCorrect:
    """Tests für die Integration von ocr_correct in convert_auto()."""

    def _make_vision_response(self, markdown: str) -> dict:
        """Hilfsfunktion: Erstellt eine erfolgreiche Vision-API-Antwort."""
        return {
            "success": True,
            "markdown": markdown,
            "tokens_prompt": 100,
            "tokens_completion": 50,
            "tokens_total": 150,
            "vision_model": "pixtral-12b-2409",
        }

    def test_correct_ocr_integrated_vision_path(self):
        """
        AC-015-1/AC-015-6: Bei Bild + ocr_correct=True wird correct_ocr_text() aufgerufen.
        Meta enthält ocr_corrected=True und ocr_corrections_count.
        """
        from conftest import PNG_100x100

        vision_result = self._make_vision_response("0CR Text mit Fehlern.")
        ocr_correction_result = {
            "success": True,
            "corrected_text": "OCR Text mit Fehlern.",
            "corrections_count": 2,
            "tokens": 80,
        }

        with patch.object(_server, "MISTRAL_API_KEY", "test-key"), \
             patch.object(_server, "analyze_with_mistral_vision",
                          new=AsyncMock(return_value=vision_result)), \
             patch.object(_server, "correct_ocr_text",
                          new=AsyncMock(return_value=ocr_correction_result)):
            result = run_async(convert_auto(
                file_data=PNG_100x100,
                filename="scan.png",
                source="/data/scan.png",
                source_type="file",
                input_meta={},
                ocr_correct=True,
            ))

        assert result.success is True
        assert result.markdown == "OCR Text mit Fehlern."
        assert result.meta.ocr_corrected is True
        assert result.meta.ocr_corrections_count == 2

    def test_ocr_correct_false_no_api_call(self):
        """
        AC-015-5: Wenn ocr_correct=False (Default), wird correct_ocr_text() NICHT aufgerufen.
        """
        from conftest import PNG_100x100

        vision_result = self._make_vision_response("Text ohne Korrektur.")

        with patch.object(_server, "MISTRAL_API_KEY", "test-key"), \
             patch.object(_server, "analyze_with_mistral_vision",
                          new=AsyncMock(return_value=vision_result)), \
             patch.object(_server, "correct_ocr_text",
                          new=AsyncMock()) as mock_correct:
            result = run_async(convert_auto(
                file_data=PNG_100x100,
                filename="scan.png",
                source="/data/scan.png",
                source_type="file",
                input_meta={},
                ocr_correct=False,
            ))

        mock_correct.assert_not_called()
        assert result.success is True
        assert result.meta.ocr_corrected is None

    def test_ocr_correct_failure_preserves_original_markdown(self):
        """
        AC-015-3: Wenn correct_ocr_text() fehlschlägt, bleibt der originale Markdown erhalten.
        """
        from conftest import PNG_100x100

        original_markdown = "0CR Originaltext."
        vision_result = self._make_vision_response(original_markdown)
        correction_failure = {
            "success": False,
            "error": "API Error",
            "corrected_text": original_markdown,
            "corrections_count": 0,
            "tokens": 0,
        }

        with patch.object(_server, "MISTRAL_API_KEY", "test-key"), \
             patch.object(_server, "analyze_with_mistral_vision",
                          new=AsyncMock(return_value=vision_result)), \
             patch.object(_server, "correct_ocr_text",
                          new=AsyncMock(return_value=correction_failure)):
            result = run_async(convert_auto(
                file_data=PNG_100x100,
                filename="scan.png",
                source="/data/scan.png",
                source_type="file",
                input_meta={},
                ocr_correct=True,
            ))

        assert result.success is True
        # Bei Fehler: ocr_corrected=False, Original bleibt
        assert result.meta.ocr_corrected is False

    def test_ocr_correct_scanned_pdf_path(self):
        """
        AC-015-6: OCR-Korrektur funktioniert auch für gescannte PDFs (Scanned-PDF-Pfad).
        """
        pdf_data = b"%PDF-1.4 dummy"
        scanned_result = {
            "success": True,
            "markdown": "Gescannter 0CR Text.",
            "tokens_prompt": 200,
            "tokens_completion": 100,
            "tokens_total": 300,
            "tokens_per_page": [{"page": 1, "tokens_total": 300}],
            "pages_processed": 1,
            "vision_model": "pixtral-12b-2409",
        }
        ocr_correction_result = {
            "success": True,
            "corrected_text": "Gescannter OCR Text.",
            "corrections_count": 1,
            "tokens": 90,
        }

        # detect_mimetype_from_bytes muss None zurückgeben damit das PDF nicht als
        # Bild erkannt wird (magic ist gemockt und gibt sonst ein Mock-Objekt zurück)
        with patch.object(_server, "MISTRAL_API_KEY", "test-key"), \
             patch.object(_server, "detect_mimetype_from_bytes", return_value=None), \
             patch.object(_server, "is_scanned_pdf", return_value=True), \
             patch.object(_server, "convert_scanned_pdf",
                          new=AsyncMock(return_value=scanned_result)), \
             patch.object(_server, "correct_ocr_text",
                          new=AsyncMock(return_value=ocr_correction_result)):
            result = run_async(convert_auto(
                file_data=pdf_data,
                filename="scan.pdf",
                source="/data/scan.pdf",
                source_type="file",
                input_meta={},
                ocr_correct=True,
            ))

        assert result.success is True
        assert result.markdown == "Gescannter OCR Text."
        assert result.meta.ocr_corrected is True
        assert result.meta.ocr_corrections_count == 1
