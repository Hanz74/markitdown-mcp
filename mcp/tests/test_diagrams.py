"""
Tests für T-MKIT-004: Diagramm/Chart-Erkennung + Mermaid/Datentabellen-Konvertierung.

Alle Tests laufen ohne Docker-Container und ohne echte API-Calls.
Alle externen Abhängigkeiten werden per unittest.mock gemockt.
"""

import asyncio
import io
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conftest import (
    load_server_module, run_async,
    PNG_200x200 as _PNG_200x200,
    PNG_300x200 as _PNG_300x200,
    PNG_400x300 as _PNG_400x300,
)


# Einmal laden; alle Tests teilen diese Instanz
# use_real_pil=True: echtes PIL für Image.open (Bildklassifizierung nutzt PIL)
_server = load_server_module(use_real_pil=True)


# ---------------------------------------------------------------------------
# Helper: Vision-API-Ergebnis-Factories
# ---------------------------------------------------------------------------

def _vision_success(text: str, tokens: int = 100) -> dict:
    return {
        "success": True,
        "markdown": text,
        "tokens_prompt": tokens // 2,
        "tokens_completion": tokens // 2,
        "tokens_total": tokens,
    }


def _vision_failure(error: str = "API nicht erreichbar") -> dict:
    return {
        "success": False,
        "error": error,
        "error_code": "API_ERROR",
    }


# ---------------------------------------------------------------------------
# Tests: classify_image_type (AC-004-1)
# ---------------------------------------------------------------------------

class TestClassifyImageType:
    """Tests für classify_image_type — Bildklassifizierung via Vision-API."""

    def test_classify_image_photo(self) -> None:
        """Foto-Erkennung: Vision antwortet mit 'photo' → Rückgabe 'photo'."""
        async def _run():
            with patch.object(_server, "analyze_with_mistral_vision",
                              new=AsyncMock(return_value=_vision_success("photo"))):
                return await _server.classify_image_type(_PNG_200x200, "image/png")

        result = run_async(_run())
        assert result == "photo"

    def test_classify_image_chart(self) -> None:
        """Chart-Erkennung: Vision antwortet mit 'chart' → Rückgabe 'chart'."""
        async def _run():
            with patch.object(_server, "analyze_with_mistral_vision",
                              new=AsyncMock(return_value=_vision_success("chart"))):
                return await _server.classify_image_type(_PNG_300x200, "image/png")

        result = run_async(_run())
        assert result == "chart"

    def test_classify_image_diagram(self) -> None:
        """Diagramm-Erkennung: Vision antwortet mit 'diagram' → Rückgabe 'diagram'."""
        async def _run():
            with patch.object(_server, "analyze_with_mistral_vision",
                              new=AsyncMock(return_value=_vision_success("diagram"))):
                return await _server.classify_image_type(_PNG_400x300, "image/png")

        result = run_async(_run())
        assert result == "diagram"

    def test_classify_image_text_scan(self) -> None:
        """Text-Scan-Erkennung: Vision antwortet mit 'text_scan' → Rückgabe 'text_scan'."""
        async def _run():
            with patch.object(_server, "analyze_with_mistral_vision",
                              new=AsyncMock(return_value=_vision_success("text_scan"))):
                return await _server.classify_image_type(_PNG_200x200, "image/png")

        result = run_async(_run())
        assert result == "text_scan"

    def test_classify_image_decorative(self) -> None:
        """Decorative-Erkennung: Vision antwortet mit 'decorative' → Rückgabe 'decorative'."""
        async def _run():
            with patch.object(_server, "analyze_with_mistral_vision",
                              new=AsyncMock(return_value=_vision_success("decorative"))):
                return await _server.classify_image_type(_PNG_200x200, "image/png")

        result = run_async(_run())
        assert result == "decorative"

    def test_classify_image_fallback(self) -> None:
        """Fallback bei API-Fehler: Rückgabe 'photo' ohne Exception."""
        async def _run():
            with patch.object(_server, "analyze_with_mistral_vision",
                              new=AsyncMock(return_value=_vision_failure())):
                return await _server.classify_image_type(_PNG_200x200, "image/png")

        result = run_async(_run())
        assert result == "photo"

    def test_classify_image_unknown_response_fallback(self) -> None:
        """Unbekannte API-Antwort → Fallback 'photo'."""
        async def _run():
            with patch.object(_server, "analyze_with_mistral_vision",
                              new=AsyncMock(return_value=_vision_success("unknown_category"))):
                return await _server.classify_image_type(_PNG_200x200, "image/png")

        result = run_async(_run())
        assert result == "photo"

    def test_classify_uses_correct_prompt(self) -> None:
        """classify_image_type sendet den Klassifizierungs-Prompt an die Vision-API."""
        captured_calls: list[dict] = []

        async def fake_vision(image_data, mimetype, prompt, language="de"):
            captured_calls.append({"prompt": prompt, "language": language})
            return _vision_success("chart")

        async def _run():
            with patch.object(_server, "analyze_with_mistral_vision", new=fake_vision):
                await _server.classify_image_type(_PNG_200x200, "image/png")

        run_async(_run())

        assert len(captured_calls) == 1
        prompt = captured_calls[0]["prompt"]
        assert "Classify" in prompt or "Klassifiziere" in prompt
        assert "photo" in prompt
        assert "chart" in prompt
        assert "diagram" in prompt
        assert "text_scan" in prompt
        assert "decorative" in prompt

    def test_classify_handles_mixed_case_response(self) -> None:
        """Vision-Antwort in Großbuchstaben wird korrekt normalisiert."""
        async def _run():
            with patch.object(_server, "analyze_with_mistral_vision",
                              new=AsyncMock(return_value=_vision_success("CHART"))):
                return await _server.classify_image_type(_PNG_200x200, "image/png")

        result = run_async(_run())
        assert result == "chart"


# ---------------------------------------------------------------------------
# Tests: convert_diagram_to_mermaid (AC-004-2)
# ---------------------------------------------------------------------------

class TestConvertDiagramToMermaid:
    """Tests für convert_diagram_to_mermaid — Flowchart → Mermaid-Syntax."""

    def test_convert_diagram_to_mermaid_returns_mermaid_block(self) -> None:
        """Mermaid-Output enthält ```mermaid ... ``` Block."""
        mermaid_code = "```mermaid\ngraph TD\n  A[Start] --> B[End]\n```"

        async def _run():
            with patch.object(_server, "analyze_with_mistral_vision",
                              new=AsyncMock(return_value=_vision_success(mermaid_code))):
                return await _server.convert_diagram_to_mermaid(_PNG_300x200, "image/png")

        result = run_async(_run())
        assert "```mermaid" in result
        assert "graph TD" in result
        assert "```" in result

    def test_convert_diagram_uses_mermaid_prompt(self) -> None:
        """convert_diagram_to_mermaid sendet Mermaid-spezifischen Prompt."""
        captured_calls: list[dict] = []

        async def fake_vision(image_data, mimetype, prompt, language="de"):
            captured_calls.append({"prompt": prompt})
            return _vision_success("```mermaid\ngraph TD\n  A --> B\n```")

        async def _run():
            with patch.object(_server, "analyze_with_mistral_vision", new=fake_vision):
                await _server.convert_diagram_to_mermaid(_PNG_300x200, "image/png")

        run_async(_run())

        assert len(captured_calls) == 1
        prompt = captured_calls[0]["prompt"]
        assert "Mermaid" in prompt or "mermaid" in prompt
        assert "graph TD" in prompt or "Flowchart" in prompt or "Diagramm" in prompt

    def test_convert_diagram_fallback_on_api_error(self) -> None:
        """Bei API-Fehler wird ein Fallback-Text zurückgegeben (kein Crash)."""
        async def _run():
            with patch.object(_server, "analyze_with_mistral_vision",
                              new=AsyncMock(return_value=_vision_failure("Timeout"))):
                return await _server.convert_diagram_to_mermaid(_PNG_300x200, "image/png")

        result = run_async(_run())
        assert isinstance(result, str)
        assert len(result) > 0
        # Kein Mermaid-Code bei Fehler
        assert "nicht verfügbar" in result or "error" in result.lower() or result.startswith("[")

    def test_convert_diagram_sequence_diagram(self) -> None:
        """Sequenzdiagramme werden als sequenceDiagram zurückgegeben."""
        mermaid_code = "```mermaid\nsequenceDiagram\n  A->>B: Hello\n```"

        async def _run():
            with patch.object(_server, "analyze_with_mistral_vision",
                              new=AsyncMock(return_value=_vision_success(mermaid_code))):
                return await _server.convert_diagram_to_mermaid(_PNG_400x300, "image/png")

        result = run_async(_run())
        assert "sequenceDiagram" in result


# ---------------------------------------------------------------------------
# Tests: extract_chart_data (AC-004-3)
# ---------------------------------------------------------------------------

class TestExtractChartData:
    """Tests für extract_chart_data — Chart-Daten → Markdown-Tabelle."""

    def test_extract_chart_data_returns_markdown_table(self) -> None:
        """Chart-Extraktion gibt Markdown-Tabelle zurück."""
        table_md = "| Monat | Umsatz |\n|-------|--------|\n| Jan | 1000 |\n| Feb | 1500 |"

        async def _run():
            with patch.object(_server, "analyze_with_mistral_vision",
                              new=AsyncMock(return_value=_vision_success(table_md))):
                return await _server.extract_chart_data(_PNG_300x200, "image/png")

        result = run_async(_run())
        assert "| Monat |" in result
        assert "| Jan |" in result
        assert "|" in result

    def test_extract_chart_data_uses_table_prompt(self) -> None:
        """extract_chart_data sendet Daten-Extraktions-Prompt."""
        captured_calls: list[dict] = []

        async def fake_vision(image_data, mimetype, prompt, language="de"):
            captured_calls.append({"prompt": prompt})
            return _vision_success("| X | Y |\n|---|---|\n| 1 | 2 |")

        async def _run():
            with patch.object(_server, "analyze_with_mistral_vision", new=fake_vision):
                await _server.extract_chart_data(_PNG_300x200, "image/png")

        run_async(_run())

        assert len(captured_calls) == 1
        prompt = captured_calls[0]["prompt"]
        # Prompt soll Tabellen-Extraktion und Achsenbeschriftungen erwähnen
        assert "Tabelle" in prompt or "table" in prompt.lower()
        assert "Achse" in prompt or "Daten" in prompt or "data" in prompt.lower()

    def test_extract_chart_data_fallback_on_api_error(self) -> None:
        """Bei API-Fehler wird ein Fallback-Text zurückgegeben (kein Crash)."""
        async def _run():
            with patch.object(_server, "analyze_with_mistral_vision",
                              new=AsyncMock(return_value=_vision_failure("API_ERROR"))):
                return await _server.extract_chart_data(_PNG_300x200, "image/png")

        result = run_async(_run())
        assert isinstance(result, str)
        assert len(result) > 0
        assert "nicht verfügbar" in result or result.startswith("[")

    def test_extract_chart_data_bar_chart(self) -> None:
        """Balkendiagramm-Daten werden als Tabelle extrahiert."""
        table_md = "| Kategorie | Wert |\n|-----------|------|\n| A | 42 |\n| B | 87 |"

        async def _run():
            with patch.object(_server, "analyze_with_mistral_vision",
                              new=AsyncMock(return_value=_vision_success(table_md))):
                return await _server.extract_chart_data(_PNG_400x300, "image/png")

        result = run_async(_run())
        assert "| Kategorie |" in result or "| A |" in result


# ---------------------------------------------------------------------------
# Tests: Integration in describe_embedded_images (AC-004-4)
# ---------------------------------------------------------------------------

class TestIntegrationDiagramInDescribe:
    """Integrationstests für describe_embedded_images mit typ-spezifischer Verarbeitung."""

    def test_diagram_image_uses_mermaid_conversion(self) -> None:
        """Bild mit Typ 'diagram' wird zu Mermaid konvertiert (nicht generisch beschrieben)."""
        mermaid_output = "```mermaid\ngraph TD\n  A --> B\n```"
        images = [{"name": "flowchart.png", "data": _PNG_300x200}]

        async def _run():
            with patch.object(_server, "classify_image_type",
                              new=AsyncMock(return_value="diagram")), \
                 patch.object(_server, "convert_diagram_to_mermaid",
                              new=AsyncMock(return_value=mermaid_output)) as mock_mermaid, \
                 patch.object(_server, "extract_chart_data",
                              new=AsyncMock(return_value="")) as mock_chart, \
                 patch.object(_server, "analyze_with_mistral_vision",
                              new=AsyncMock(return_value=_vision_success("generic"))) as mock_vision, \
                 patch.object(_server, "detect_mimetype_from_bytes", return_value="image/png"):
                results = await _server.describe_embedded_images(images, language="de")
                return results, mock_mermaid, mock_chart, mock_vision

        results, mock_mermaid, mock_chart, mock_vision = run_async(_run())

        assert len(results) == 1
        assert results[0]["name"] == "flowchart.png"
        assert "```mermaid" in results[0]["description"]
        assert results[0]["image_type"] == "diagram"
        mock_mermaid.assert_called_once()
        mock_chart.assert_not_called()
        mock_vision.assert_not_called()

    def test_chart_image_uses_data_extraction(self) -> None:
        """Bild mit Typ 'chart' wird als Datentabelle extrahiert (nicht generisch beschrieben)."""
        table_output = "| X | Y |\n|---|---|\n| 1 | 10 |"
        images = [{"name": "barchart.png", "data": _PNG_400x300}]

        async def _run():
            with patch.object(_server, "classify_image_type",
                              new=AsyncMock(return_value="chart")), \
                 patch.object(_server, "convert_diagram_to_mermaid",
                              new=AsyncMock(return_value="")) as mock_mermaid, \
                 patch.object(_server, "extract_chart_data",
                              new=AsyncMock(return_value=table_output)) as mock_chart, \
                 patch.object(_server, "analyze_with_mistral_vision",
                              new=AsyncMock(return_value=_vision_success("generic"))) as mock_vision, \
                 patch.object(_server, "detect_mimetype_from_bytes", return_value="image/png"):
                results = await _server.describe_embedded_images(images, language="de")
                return results, mock_mermaid, mock_chart, mock_vision

        results, mock_mermaid, mock_chart, mock_vision = run_async(_run())

        assert len(results) == 1
        assert results[0]["name"] == "barchart.png"
        assert "| X |" in results[0]["description"]
        assert results[0]["image_type"] == "chart"
        mock_chart.assert_called_once()
        mock_mermaid.assert_not_called()
        mock_vision.assert_not_called()

    def test_photo_image_uses_generic_vision(self) -> None:
        """Bild mit Typ 'photo' wird mit generischer Vision-Beschreibung verarbeitet."""
        images = [{"name": "photo.png", "data": _PNG_200x200}]

        async def _run():
            with patch.object(_server, "classify_image_type",
                              new=AsyncMock(return_value="photo")), \
                 patch.object(_server, "convert_diagram_to_mermaid",
                              new=AsyncMock(return_value="")) as mock_mermaid, \
                 patch.object(_server, "extract_chart_data",
                              new=AsyncMock(return_value="")) as mock_chart, \
                 patch.object(_server, "analyze_with_mistral_vision",
                              new=AsyncMock(return_value=_vision_success("Ein Foto einer Landschaft"))) as mock_vision, \
                 patch.object(_server, "detect_mimetype_from_bytes", return_value="image/png"):
                results = await _server.describe_embedded_images(images, language="de")
                return results, mock_mermaid, mock_chart, mock_vision

        results, mock_mermaid, mock_chart, mock_vision = run_async(_run())

        assert len(results) == 1
        assert results[0]["name"] == "photo.png"
        assert results[0]["description"] == "Ein Foto einer Landschaft"
        assert results[0]["image_type"] == "photo"
        mock_vision.assert_called_once()
        mock_mermaid.assert_not_called()
        mock_chart.assert_not_called()

    def test_text_scan_image_uses_generic_vision(self) -> None:
        """Bild mit Typ 'text_scan' wird mit generischer Vision-Beschreibung verarbeitet."""
        images = [{"name": "scan.png", "data": _PNG_200x200}]

        async def _run():
            with patch.object(_server, "classify_image_type",
                              new=AsyncMock(return_value="text_scan")), \
                 patch.object(_server, "analyze_with_mistral_vision",
                              new=AsyncMock(return_value=_vision_success("Gescannter Text..."))) as mock_vision, \
                 patch.object(_server, "detect_mimetype_from_bytes", return_value="image/png"):
                results = await _server.describe_embedded_images(images, language="de")
                return results, mock_vision

        results, mock_vision = run_async(_run())

        assert len(results) == 1
        assert results[0]["image_type"] == "text_scan"
        mock_vision.assert_called_once()

    def test_decorative_image_is_skipped(self) -> None:
        """Bild mit Typ 'decorative' wird übersprungen und erscheint nicht in den Ergebnissen."""
        images = [
            {"name": "decorative.png", "data": _PNG_200x200},
            {"name": "photo.png", "data": _PNG_300x200},
        ]

        async def _run():
            classify_responses = ["decorative", "photo"]
            classify_iter = iter(classify_responses)

            async def fake_classify(data, mimetype):
                return next(classify_iter)

            with patch.object(_server, "classify_image_type", new=fake_classify), \
                 patch.object(_server, "analyze_with_mistral_vision",
                              new=AsyncMock(return_value=_vision_success("Ein Foto"))), \
                 patch.object(_server, "detect_mimetype_from_bytes", return_value="image/png"):
                return await _server.describe_embedded_images(images, language="de")

        results = run_async(_run())

        # Nur das Foto soll im Ergebnis sein, nicht das dekorative Bild
        assert len(results) == 1
        assert results[0]["name"] == "photo.png"
        names = [r["name"] for r in results]
        assert "decorative.png" not in names

    def test_image_type_in_result_dict(self) -> None:
        """Das Ergebnis-Dict enthält das Feld 'image_type' (AC-004-4)."""
        images = [{"name": "chart.png", "data": _PNG_300x200}]

        async def _run():
            with patch.object(_server, "classify_image_type",
                              new=AsyncMock(return_value="chart")), \
                 patch.object(_server, "extract_chart_data",
                              new=AsyncMock(return_value="| Col | Val |\n|-----|-----|\n| A | 1 |")), \
                 patch.object(_server, "detect_mimetype_from_bytes", return_value="image/png"):
                return await _server.describe_embedded_images(images, language="de")

        results = run_async(_run())

        assert len(results) == 1
        assert "image_type" in results[0]
        assert results[0]["image_type"] == "chart"

    def test_mixed_images_all_processed_correctly(self) -> None:
        """Mehrere Bilder verschiedener Typen werden korrekt verarbeitet."""
        images = [
            {"name": "diagram.png", "data": _PNG_200x200},
            {"name": "chart.png", "data": _PNG_300x200},
            {"name": "photo.jpg", "data": _PNG_400x300},
        ]
        type_map = {
            "diagram.png": "diagram",
            "chart.png": "chart",
            "photo.jpg": "photo",
        }

        async def _run():
            call_count = {"n": 0}
            img_names = ["diagram.png", "chart.png", "photo.jpg"]

            async def fake_classify(data, mimetype):
                name = img_names[call_count["n"]]
                call_count["n"] += 1
                return type_map[name]

            with patch.object(_server, "classify_image_type", new=fake_classify), \
                 patch.object(_server, "convert_diagram_to_mermaid",
                              new=AsyncMock(return_value="```mermaid\ngraph TD\n  A-->B\n```")), \
                 patch.object(_server, "extract_chart_data",
                              new=AsyncMock(return_value="| X | Y |\n|---|---|\n| 1 | 2 |")), \
                 patch.object(_server, "analyze_with_mistral_vision",
                              new=AsyncMock(return_value=_vision_success("Foto-Beschreibung"))), \
                 patch.object(_server, "detect_mimetype_from_bytes", return_value="image/png"):
                return await _server.describe_embedded_images(images, language="de")

        results = run_async(_run())

        assert len(results) == 3
        result_map = {r["name"]: r for r in results}

        assert result_map["diagram.png"]["image_type"] == "diagram"
        assert "```mermaid" in result_map["diagram.png"]["description"]

        assert result_map["chart.png"]["image_type"] == "chart"
        assert "| X |" in result_map["chart.png"]["description"]

        assert result_map["photo.jpg"]["image_type"] == "photo"
        assert result_map["photo.jpg"]["description"] == "Foto-Beschreibung"
