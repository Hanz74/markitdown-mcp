"""
Unit-Tests für Scanned PDF Erkennung und Vision-Routing.

Tests laufen ohne Docker-Container und ohne echte PDF-Dateien (Mocking).
Alle externen Abhängigkeiten (subprocess, pdf2image, Vision-API) werden gemockt.
"""

import sys
import asyncio
import types
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

from conftest import load_server_module, run_async


# Einmal laden; alle Tests in diesem Modul teilen diese Instanz
# use_real_pil=False: dieses Modul braucht PIL nur als Mock (pdf2image-Bilder)
_server = load_server_module(use_real_pil=False)
is_scanned_pdf = _server.is_scanned_pdf
convert_scanned_pdf = _server.convert_scanned_pdf


# =============================================================================
# TestIsScannedPdf
# =============================================================================

class TestIsScannedPdf:
    """Tests für die Scan-Erkennungsfunktion is_scanned_pdf()."""

    def _make_pdf(self, tmp_path: Path) -> Path:
        """Erstellt eine Dummy-PDF-Datei für Tests."""
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 dummy")
        return pdf

    def test_is_scanned_pdf_with_text_returns_false(self, tmp_path):
        """
        PDF mit reichlich Text (> SCAN_THRESHOLD_CHARS/Seite) → False.
        AC-002-1
        """
        pdf = self._make_pdf(tmp_path)
        long_text = "A" * 500  # 500 Zeichen auf 1 Seite → klar über dem Standard-Threshold

        pdftotext_result = MagicMock()
        pdftotext_result.returncode = 0
        pdftotext_result.stdout = long_text

        pdfinfo_result = MagicMock()
        pdfinfo_result.returncode = 0
        pdfinfo_result.stdout = "Pages: 1\nCreator: Test\n"

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [pdftotext_result, pdfinfo_result]
            with patch.object(_server, "SCAN_THRESHOLD_CHARS", 50):
                result = is_scanned_pdf(pdf)

        assert result is False

    def test_is_scanned_pdf_with_scan_returns_true(self, tmp_path):
        """
        PDF mit wenig Text (< SCAN_THRESHOLD_CHARS/Seite) → True.
        AC-002-1
        """
        pdf = self._make_pdf(tmp_path)
        sparse_text = "ab"  # nur 2 Zeichen auf 3 Seiten → weit unter Threshold

        pdftotext_result = MagicMock()
        pdftotext_result.returncode = 0
        pdftotext_result.stdout = sparse_text

        pdfinfo_result = MagicMock()
        pdfinfo_result.returncode = 0
        pdfinfo_result.stdout = "Pages: 3\nCreator: Scanner\n"

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [pdftotext_result, pdfinfo_result]
            with patch.object(_server, "SCAN_THRESHOLD_CHARS", 50):
                result = is_scanned_pdf(pdf)

        assert result is True

    def test_pdftotext_not_found_returns_false(self, tmp_path):
        """Wenn pdftotext nicht installiert ist (FileNotFoundError) → False."""
        pdf = self._make_pdf(tmp_path)

        with patch("subprocess.run", side_effect=FileNotFoundError("pdftotext not found")):
            result = is_scanned_pdf(pdf)

        assert result is False

    def test_pdftotext_returncode_nonzero_returns_false(self, tmp_path):
        """Wenn pdftotext mit Fehler-Exit-Code endet → False."""
        pdf = self._make_pdf(tmp_path)

        failed_result = MagicMock()
        failed_result.returncode = 1
        failed_result.stdout = ""
        failed_result.stderr = "Error opening document"

        with patch("subprocess.run", return_value=failed_result):
            result = is_scanned_pdf(pdf)

        assert result is False

    def test_pdfinfo_failure_defaults_to_one_page(self, tmp_path):
        """Wenn pdfinfo fehlschlägt, wird 1 Seite angenommen."""
        pdf = self._make_pdf(tmp_path)
        short_text = "x" * 30  # 30 Zeichen / 1 Seite → unter Threshold von 50

        pdftotext_result = MagicMock()
        pdftotext_result.returncode = 0
        pdftotext_result.stdout = short_text

        # pdfinfo schlägt fehl
        pdfinfo_result = MagicMock()
        pdfinfo_result.returncode = 1
        pdfinfo_result.stdout = ""

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [pdftotext_result, pdfinfo_result]
            with patch.object(_server, "SCAN_THRESHOLD_CHARS", 50):
                result = is_scanned_pdf(pdf)

        # 30 Zeichen / 1 Seite = 30 < 50 → True
        assert result is True

    def test_threshold_configurable(self, tmp_path):
        """
        SCAN_THRESHOLD_CHARS wird respektiert — gleicher Text, anderer Threshold.
        AC-002-5
        """
        pdf = self._make_pdf(tmp_path)
        text = "A" * 100  # 100 Zeichen auf 1 Seite

        def make_subprocess_side_effect(text_content, pages=1):
            pdftotext_result = MagicMock()
            pdftotext_result.returncode = 0
            pdftotext_result.stdout = text_content
            pdfinfo_result = MagicMock()
            pdfinfo_result.returncode = 0
            pdfinfo_result.stdout = f"Pages: {pages}\n"
            return [pdftotext_result, pdfinfo_result]

        # Threshold 50 → 100 Zeichen/Seite → NICHT Scan
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = make_subprocess_side_effect(text)
            with patch.object(_server, "SCAN_THRESHOLD_CHARS", 50):
                result_low_threshold = is_scanned_pdf(pdf)

        # Threshold 200 → 100 Zeichen/Seite < 200 → Scan
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = make_subprocess_side_effect(text)
            with patch.object(_server, "SCAN_THRESHOLD_CHARS", 200):
                result_high_threshold = is_scanned_pdf(pdf)

        assert result_low_threshold is False
        assert result_high_threshold is True

    def test_timeout_returns_false(self, tmp_path):
        """TimeoutExpired von subprocess → False (kein Absturz)."""
        import subprocess as sp
        pdf = self._make_pdf(tmp_path)

        with patch("subprocess.run", side_effect=sp.TimeoutExpired("pdftotext", 30)):
            result = is_scanned_pdf(pdf)

        assert result is False


# =============================================================================
# TestConvertScannedPdf
# =============================================================================

class TestConvertScannedPdf:
    """Tests für convert_scanned_pdf() — Vision-basierte Seiten-Verarbeitung."""

    def _make_mock_image(self):
        """Erstellt ein Mock-PIL-Image das save() unterstützt."""
        mock_img = MagicMock()
        mock_img.width = 800
        mock_img.height = 1200
        mock_img.save = MagicMock()
        return mock_img

    def test_convert_scanned_pdf_merges_pages(self, tmp_path):
        """
        Multi-Page-Scan → Markdown enthält ## Seite N für jede Seite.
        AC-002-3
        """
        pdf = tmp_path / "scan.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        mock_pages = [self._make_mock_image(), self._make_mock_image()]

        vision_response_p1 = {
            "success": True,
            "markdown": "Seite 1 Inhalt",
            "tokens_prompt": 100,
            "tokens_completion": 50,
            "tokens_total": 150,
            "vision_model": "pixtral-12b-2409",
        }
        vision_response_p2 = {
            "success": True,
            "markdown": "Seite 2 Inhalt",
            "tokens_prompt": 120,
            "tokens_completion": 60,
            "tokens_total": 180,
            "vision_model": "pixtral-12b-2409",
        }

        with patch.object(_server, "PDF2IMAGE_AVAILABLE", True):
            with patch.object(_server, "MISTRAL_API_KEY", "test-key"):
                with patch.object(_server, "convert_from_path", return_value=mock_pages):
                    with patch.object(
                        _server, "analyze_with_mistral_vision",
                        new=AsyncMock(side_effect=[vision_response_p1, vision_response_p2]),
                    ):
                        import io
                        with patch("io.BytesIO") as mock_bytesio:
                            mock_buffer = MagicMock()
                            mock_buffer.getvalue.return_value = b"fake-png-data"
                            mock_bytesio.return_value = mock_buffer
                            result = run_async(convert_scanned_pdf(pdf))

        assert result["success"] is True
        assert "## Seite 1" in result["markdown"]
        assert "## Seite 2" in result["markdown"]
        assert "Seite 1 Inhalt" in result["markdown"]
        assert "Seite 2 Inhalt" in result["markdown"]

    def test_scanned_pdf_meta_contains_scanned_true(self, tmp_path):
        """
        Ergebnis-Dict enthält scanned=True und Token-Info pro Seite.
        AC-002-4
        """
        pdf = tmp_path / "scan.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        mock_pages = [self._make_mock_image()]

        vision_response = {
            "success": True,
            "markdown": "Inhalt",
            "tokens_prompt": 80,
            "tokens_completion": 40,
            "tokens_total": 120,
            "vision_model": "pixtral-12b-2409",
        }

        with patch.object(_server, "PDF2IMAGE_AVAILABLE", True):
            with patch.object(_server, "MISTRAL_API_KEY", "test-key"):
                with patch.object(_server, "convert_from_path", return_value=mock_pages):
                    with patch.object(
                        _server, "analyze_with_mistral_vision",
                        new=AsyncMock(return_value=vision_response),
                    ):
                        import io
                        with patch("io.BytesIO") as mock_bytesio:
                            mock_buffer = MagicMock()
                            mock_buffer.getvalue.return_value = b"fake-png-data"
                            mock_bytesio.return_value = mock_buffer
                            result = run_async(convert_scanned_pdf(pdf))

        assert result["scanned"] is True
        assert "tokens_per_page" in result
        assert isinstance(result["tokens_per_page"], list)
        assert len(result["tokens_per_page"]) == 1
        assert result["tokens_per_page"][0]["page"] == 1
        assert result["tokens_per_page"][0]["tokens_total"] == 120
        assert result["tokens_total"] == 120
        assert result["pages_processed"] == 1

    def test_pdf2image_not_available_returns_error(self, tmp_path):
        """Wenn pdf2image nicht installiert → Fehler-Dict."""
        pdf = tmp_path / "scan.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        with patch.object(_server, "PDF2IMAGE_AVAILABLE", False):
            result = run_async(convert_scanned_pdf(pdf))

        assert result["success"] is False
        assert "pdf2image" in result["error"].lower()

    def test_no_api_key_returns_error(self, tmp_path):
        """Wenn MISTRAL_API_KEY fehlt → Fehler-Dict."""
        pdf = tmp_path / "scan.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        with patch.object(_server, "PDF2IMAGE_AVAILABLE", True):
            with patch.object(_server, "MISTRAL_API_KEY", ""):
                result = run_async(convert_scanned_pdf(pdf))

        assert result["success"] is False
        assert "MISTRAL_API_KEY" in result["error"]

    def test_failed_page_gets_placeholder_in_markdown(self, tmp_path):
        """
        Wenn Vision-API für eine Seite fehlschlägt, erscheint ein Fehler-Placeholder
        im zusammengeführten Markdown.
        """
        pdf = tmp_path / "scan.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        mock_pages = [self._make_mock_image(), self._make_mock_image()]

        vision_ok = {
            "success": True,
            "markdown": "Seite 1 OK",
            "tokens_prompt": 100,
            "tokens_completion": 50,
            "tokens_total": 150,
            "vision_model": "pixtral-12b-2409",
        }
        vision_fail = {
            "success": False,
            "error": "API Timeout",
            "error_code": "TIMEOUT",
        }

        with patch.object(_server, "PDF2IMAGE_AVAILABLE", True):
            with patch.object(_server, "MISTRAL_API_KEY", "test-key"):
                with patch.object(_server, "convert_from_path", return_value=mock_pages):
                    with patch.object(
                        _server, "analyze_with_mistral_vision",
                        new=AsyncMock(side_effect=[vision_ok, vision_fail]),
                    ):
                        import io
                        with patch("io.BytesIO") as mock_bytesio:
                            mock_buffer = MagicMock()
                            mock_buffer.getvalue.return_value = b"fake-png-data"
                            mock_bytesio.return_value = mock_buffer
                            result = run_async(convert_scanned_pdf(pdf))

        assert result["success"] is True
        assert "## Seite 1" in result["markdown"]
        assert "Seite 1 OK" in result["markdown"]
        assert "## Seite 2" in result["markdown"]
        # Seite 2 muss Fehler-Placeholder enthalten
        assert "konnte nicht verarbeitet werden" in result["markdown"]
        # Nur 1 Seite erfolgreich verarbeitet
        assert result["pages_processed"] == 1

    def test_pdf2image_exception_returns_error(self, tmp_path):
        """Wenn convert_from_path eine Exception wirft → Fehler-Dict."""
        pdf = tmp_path / "scan.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        with patch.object(_server, "PDF2IMAGE_AVAILABLE", True):
            with patch.object(_server, "MISTRAL_API_KEY", "test-key"):
                with patch.object(
                    _server, "convert_from_path",
                    side_effect=Exception("Poppler not found"),
                ):
                    result = run_async(convert_scanned_pdf(pdf))

        assert result["success"] is False
        assert "PDF-Rendering fehlgeschlagen" in result["error"]

    def test_tokens_aggregated_across_pages(self, tmp_path):
        """
        Token-Verbrauch wird korrekt über alle Seiten summiert.
        AC-002-4
        """
        pdf = tmp_path / "scan.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        mock_pages = [self._make_mock_image(), self._make_mock_image(), self._make_mock_image()]

        def make_vision_response(page_num):
            return {
                "success": True,
                "markdown": f"Inhalt Seite {page_num}",
                "tokens_prompt": 100,
                "tokens_completion": 50,
                "tokens_total": 150,
                "vision_model": "pixtral-12b-2409",
            }

        with patch.object(_server, "PDF2IMAGE_AVAILABLE", True):
            with patch.object(_server, "MISTRAL_API_KEY", "test-key"):
                with patch.object(_server, "convert_from_path", return_value=mock_pages):
                    with patch.object(
                        _server, "analyze_with_mistral_vision",
                        new=AsyncMock(side_effect=[
                            make_vision_response(1),
                            make_vision_response(2),
                            make_vision_response(3),
                        ]),
                    ):
                        import io
                        with patch("io.BytesIO") as mock_bytesio:
                            mock_buffer = MagicMock()
                            mock_buffer.getvalue.return_value = b"fake-png-data"
                            mock_bytesio.return_value = mock_buffer
                            result = run_async(convert_scanned_pdf(pdf))

        assert result["success"] is True
        assert result["pages_processed"] == 3
        assert result["tokens_total"] == 450  # 3 × 150
        assert result["tokens_prompt"] == 300  # 3 × 100
        assert result["tokens_completion"] == 150  # 3 × 50
        assert len(result["tokens_per_page"]) == 3
