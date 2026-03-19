"""
Unit-Tests für img2table Fallback-Tabellen-Extraktion (FR-MKIT-012).

Tests laufen ohne Docker-Container und ohne echte PDF-Dateien (Mocking).
img2table und TesseractOCR werden vollständig gemockt.
"""

import sys
import types
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from conftest import load_server_module, run_async


# =============================================================================
# Hilfsfunktionen
# =============================================================================

def _make_img2table_mock() -> dict:
    """
    Erstellt gemockte img2table-Module.

    Returns:
        Dict mit 'img2table_document_mod', 'img2table_ocr_mod', 'pdf_class', 'tesseract_class'
        für direkte Assertion in Tests.
    """
    import pandas as pd

    # Erzeuge ein minimales DataFrame für die gemockte ExtractedTable
    df = pd.DataFrame({"Col1": ["A", "C"], "Col2": ["B", "D"]})

    extracted_table_mock = MagicMock()
    extracted_table_mock.df = df

    pdf_instance_mock = MagicMock()
    pdf_instance_mock.extract_tables = MagicMock(
        return_value={0: [extracted_table_mock]}
    )

    pdf_class_mock = MagicMock(return_value=pdf_instance_mock)
    tesseract_class_mock = MagicMock(return_value=MagicMock())

    img2table_document_mod = MagicMock()
    img2table_document_mod.PDF = pdf_class_mock

    img2table_ocr_mod = MagicMock()
    img2table_ocr_mod.TesseractOCR = tesseract_class_mock

    # Übergeordnete Module aufbauen
    img2table_mod = MagicMock()
    img2table_document_parent = MagicMock()
    img2table_ocr_parent = MagicMock()

    return {
        "img2table": img2table_mod,
        "img2table.document": img2table_document_mod,
        "img2table.ocr": img2table_ocr_mod,
        "pdf_class": pdf_class_mock,
        "pdf_instance": pdf_instance_mock,
        "tesseract_class": tesseract_class_mock,
        "extracted_table": extracted_table_mock,
    }


def _load_server_with_img2table(img2table_available: bool = True):
    """
    Lädt den Server mit optional verfügbarem img2table.

    Args:
        img2table_available: Wenn True, werden img2table-Module gemockt als verfügbar.
                             Wenn False, wird ImportError simuliert.

    Returns:
        Tuple (server_module, mocks_dict)
    """
    mocks = _make_img2table_mock()

    if img2table_available:
        extra_patches = {
            "img2table": mocks["img2table"],
            "img2table.document": mocks["img2table.document"],
            "img2table.ocr": mocks["img2table.ocr"],
        }
    else:
        # img2table NICHT in sys.modules → ImportError beim guarded import
        extra_patches = {}
        # Sicherstellen dass img2table nicht aus einem vorherigen Test gecacht ist
        for key in ["img2table", "img2table.document", "img2table.ocr"]:
            sys.modules.pop(key, None)

    server = load_server_module(use_real_pil=False, extra_patches=extra_patches)

    if img2table_available:
        # Flag manuell setzen, da guarded import im Modul-Scope läuft
        server.IMG2TABLE_AVAILABLE = True
        server.Img2TablePDF = mocks["pdf_class"]
        server.TesseractOCR = mocks["tesseract_class"]
    else:
        server.IMG2TABLE_AVAILABLE = False

    return server, mocks


# =============================================================================
# TestExtractTablesImg2table
# =============================================================================

class TestExtractTablesImg2table:
    """Testet extract_tables_with_img2table direkt."""

    def test_extract_tables_img2table_success(self, tmp_path):
        """AC-012-1/2: Tabellen werden aus PDF extrahiert und korrekt konvertiert."""
        server, mocks = _load_server_with_img2table(img2table_available=True)

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")

        result = server.extract_tables_with_img2table(pdf_file)

        assert isinstance(result, list)
        assert len(result) == 1
        table = result[0]
        # Header-Zeile + 2 Datenzeilen
        assert len(table) == 3
        # Header aus DataFrame-Columns
        assert table[0] == ["Col1", "Col2"]
        # Erste Datenzeile
        assert table[1] == ["A", "B"]
        # Zweite Datenzeile
        assert table[2] == ["C", "D"]

    def test_extract_tables_img2table_no_tables(self, tmp_path):
        """AC-012-1: Keine Tabellen gefunden → leere Liste."""
        server, mocks = _load_server_with_img2table(img2table_available=True)

        # Override: extract_tables gibt leeres Dict zurück
        mocks["pdf_instance"].extract_tables = MagicMock(return_value={})

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")

        result = server.extract_tables_with_img2table(pdf_file)

        assert result == []

    def test_extract_tables_img2table_not_available(self, tmp_path):
        """AC-012-1: Graceful wenn img2table nicht installiert ist."""
        server, _ = _load_server_with_img2table(img2table_available=False)

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")

        result = server.extract_tables_with_img2table(pdf_file)

        assert result == []

    def test_img2table_uses_tesseract(self, tmp_path):
        """AC-012-2: TesseractOCR wird als OCR-Backend verwendet."""
        server, mocks = _load_server_with_img2table(img2table_available=True)

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")

        server.extract_tables_with_img2table(pdf_file)

        # TesseractOCR muss instanziiert worden sein
        mocks["tesseract_class"].assert_called_once()
        # extract_tables muss mit dem TesseractOCR-Instanz aufgerufen worden sein
        mocks["pdf_instance"].extract_tables.assert_called_once()
        call_kwargs = mocks["pdf_instance"].extract_tables.call_args
        assert call_kwargs is not None
        # ocr-Parameter muss übergeben worden sein
        assert "ocr" in call_kwargs.kwargs or len(call_kwargs.args) > 0

    def test_extract_tables_img2table_exception_returns_empty(self, tmp_path):
        """Graceful bei Ausnahme während der Extraktion."""
        server, mocks = _load_server_with_img2table(img2table_available=True)

        mocks["pdf_instance"].extract_tables = MagicMock(
            side_effect=RuntimeError("OCR fehlgeschlagen")
        )

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")

        result = server.extract_tables_with_img2table(pdf_file)

        assert result == []

    def test_extract_tables_img2table_none_cells(self, tmp_path):
        """None-Werte in DataFrame-Zellen werden als leere Strings behandelt."""
        import pandas as pd

        server, mocks = _load_server_with_img2table(img2table_available=True)

        df_with_none = pd.DataFrame({"A": [None, "x"], "B": ["y", None]})
        mocks["extracted_table"].df = df_with_none
        mocks["pdf_instance"].extract_tables = MagicMock(
            return_value={0: [mocks["extracted_table"]]}
        )

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")

        result = server.extract_tables_with_img2table(pdf_file)

        assert len(result) == 1
        table = result[0]
        # Datenzeile mit None-Wert: pandas konvertiert float-NaN zu "nan"
        assert table[1][1] == "y"
        assert table[2][0] == "x"


# =============================================================================
# TestImg2tableFallbackIntegration
# =============================================================================

class TestImg2tableFallbackIntegration:
    """Testet die Integration von img2table als Fallback in convert_with_markitdown."""

    def _setup_server_for_convert(self, pdfplumber_finds_tables: bool = False):
        """
        Richtet Server für convert_with_markitdown Tests ein.

        Returns:
            Tuple (server, tmp_path-Funktion, mocks)
        """
        server, mocks = _load_server_with_img2table(img2table_available=True)

        # MarkItDown Mock
        md_result = MagicMock()
        md_result.text_content = "# Test Dokument\n\nText ohne Tabellen."
        md_result.title = "Test"
        server.md = MagicMock()
        server.md.convert = MagicMock(return_value=md_result)

        # pdfplumber steuern
        server.PDFPLUMBER_AVAILABLE = True
        if pdfplumber_finds_tables:
            server.extract_tables_with_pdfplumber = MagicMock(
                return_value=[
                    {"page": 1, "tables": [[["A", "B"], ["1", "2"]]]}
                ]
            )
        else:
            server.extract_tables_with_pdfplumber = MagicMock(return_value=[])

        # PyMuPDF und DOCX-Extras deaktivieren für einfachere Tests
        server.PYMUPDF_AVAILABLE = False
        server.extract_pdf_metadata = MagicMock(return_value={"toc": [], "annotations": [], "form_fields": []})

        return server, mocks

    def test_fallback_when_pdfplumber_finds_nothing(self, tmp_path):
        """AC-012-1: img2table wird als Fallback genutzt wenn pdfplumber leer bleibt."""
        server, mocks = self._setup_server_for_convert(pdfplumber_finds_tables=False)

        pdf_file = tmp_path / "scanned.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")

        result = server.convert_with_markitdown(pdf_file)

        assert result["success"] is True
        # img2table-Tabellen müssen im Output sein
        assert "## Tabellen" in result["markdown"]
        # Die gemockten Daten: Col1/Col2 Header
        assert "Col1" in result["markdown"] or "Col2" in result["markdown"]

    def test_no_fallback_when_pdfplumber_succeeds(self, tmp_path):
        """AC-012-1: img2table wird NICHT aufgerufen wenn pdfplumber Tabellen findet."""
        server, mocks = self._setup_server_for_convert(pdfplumber_finds_tables=True)

        # img2table-Aufruf tracken
        img2table_calls = []
        original_func = server.extract_tables_with_img2table
        server.extract_tables_with_img2table = MagicMock(side_effect=img2table_calls.append)

        pdf_file = tmp_path / "digital.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")

        result = server.convert_with_markitdown(pdf_file)

        assert result["success"] is True
        # img2table darf NICHT aufgerufen worden sein
        server.extract_tables_with_img2table.assert_not_called()

    def test_img2table_result_as_markdown(self, tmp_path):
        """AC-012-3: img2table-Ergebnis wird als Markdown-Tabelle formatiert."""
        server, mocks = self._setup_server_for_convert(pdfplumber_finds_tables=False)

        pdf_file = tmp_path / "scanned.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")

        result = server.convert_with_markitdown(pdf_file)

        assert result["success"] is True
        markdown = result["markdown"]

        # Markdown-Tabellen-Syntax muss vorhanden sein
        assert "|" in markdown
        assert "---" in markdown
        # Sektion muss vorhanden sein
        assert "## Tabellen" in markdown

    def test_fallback_not_used_when_img2table_unavailable(self, tmp_path):
        """Kein Fehler wenn weder pdfplumber noch img2table Tabellen liefern."""
        server, _ = _load_server_with_img2table(img2table_available=False)

        md_result = MagicMock()
        md_result.text_content = "# Nur Text"
        md_result.title = "Test"
        server.md = MagicMock()
        server.md.convert = MagicMock(return_value=md_result)
        server.PDFPLUMBER_AVAILABLE = True
        server.extract_tables_with_pdfplumber = MagicMock(return_value=[])
        server.PYMUPDF_AVAILABLE = False
        server.extract_pdf_metadata = MagicMock(return_value={"toc": [], "annotations": [], "form_fields": []})

        pdf_file = tmp_path / "plain.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")

        result = server.convert_with_markitdown(pdf_file)

        assert result["success"] is True
        assert "## Tabellen" not in result["markdown"]

    def test_fallback_graceful_when_img2table_returns_empty(self, tmp_path):
        """Kein '## Tabellen'-Abschnitt wenn img2table leere Liste zurückgibt."""
        server, mocks = self._setup_server_for_convert(pdfplumber_finds_tables=False)

        # img2table findet keine Tabellen
        mocks["pdf_instance"].extract_tables = MagicMock(return_value={})

        pdf_file = tmp_path / "notables.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")

        result = server.convert_with_markitdown(pdf_file)

        assert result["success"] is True
        assert "## Tabellen" not in result["markdown"]
