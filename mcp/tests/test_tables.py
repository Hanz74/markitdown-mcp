"""
Unit-Tests für pdfplumber Tabellen-Extraktion und Cross-Page Merger.

Tests laufen ohne Docker-Container und ohne echte PDF-Dateien (Mocking).
"""

import sys
import types
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

from conftest import load_server_module


# Einmal laden, alle Tests nutzen dieselbe Instanz
_server = load_server_module(use_real_pil=False)
tables_to_markdown = _server.tables_to_markdown
merge_cross_page_tables = _server.merge_cross_page_tables
extract_tables_with_pdfplumber = _server.extract_tables_with_pdfplumber


# =============================================================================
# test_tables_to_markdown
# =============================================================================

class TestTablesToMarkdown:
    """Testet die Konvertierung von Zeilen-Listen zu Markdown-Tabellen."""

    def test_simple_table(self):
        """Einfache 2-Spalten-Tabelle mit Header und einer Datenzeile."""
        tables = [
            [["Name", "Alter"], ["Alice", "30"], ["Bob", "25"]]
        ]
        result = tables_to_markdown(tables)

        assert "| Name | Alter |" in result
        assert "| --- | --- |" in result
        assert "| Alice | 30 |" in result
        assert "| Bob | 25 |" in result

    def test_none_cells_become_empty_string(self):
        """None-Werte in Zellen werden als leere Strings dargestellt."""
        tables = [
            [["A", "B"], [None, "value"], ["x", None]]
        ]
        result = tables_to_markdown(tables)

        assert "|  | value |" in result
        assert "| x |  |" in result

    def test_multiple_tables_separated_by_blank_line(self):
        """Mehrere Tabellen werden durch Leerzeilen getrennt."""
        tables = [
            [["H1", "H2"], ["r1c1", "r1c2"]],
            [["X", "Y", "Z"], ["a", "b", "c"]],
        ]
        result = tables_to_markdown(tables)

        # Beide Tabellen müssen im Output sein
        assert "| H1 | H2 |" in result
        assert "| X | Y | Z |" in result
        # Getrennt durch Leerzeile (zwei Newlines zwischen Tabellen)
        assert "\n\n" in result

    def test_empty_tables_list_returns_empty_string(self):
        """Leere Eingabe ergibt leeren String."""
        assert tables_to_markdown([]) == ""

    def test_table_with_only_header(self):
        """Tabelle mit nur einer Header-Zeile (keine Datenzeilen)."""
        tables = [
            [["Col1", "Col2", "Col3"]]
        ]
        result = tables_to_markdown(tables)

        assert "| Col1 | Col2 | Col3 |" in result
        assert "| --- | --- | --- |" in result

    def test_row_shorter_than_header_gets_padded(self):
        """Zeilen die kürzer als der Header sind werden mit leeren Strings aufgefüllt."""
        tables = [
            [["A", "B", "C"], ["only_one"]]
        ]
        result = tables_to_markdown(tables)
        # Zeile muss 3 Spalten haben (mit Padding)
        assert "| only_one |  |  |" in result


# =============================================================================
# test_merge_cross_page_tables
# =============================================================================

class TestMergeCrossPageTables:
    """Testet den Cross-Page Table Merger."""

    def test_merge_simple_same_column_count(self):
        """
        Zwei Tabellen mit gleicher Spaltenanzahl werden zusammengeführt.
        AC-001-2
        """
        page_tables = [
            {
                "page": 1,
                "tables": [
                    [["Name", "Wert"], ["Alice", "1"], ["Bob", "2"]]
                ]
            },
            {
                "page": 2,
                "tables": [
                    [["Charlie", "3"], ["Dave", "4"]]
                ]
            },
        ]
        result = merge_cross_page_tables(page_tables)

        assert len(result) == 1
        merged = result[0]
        # Alle 5 Zeilen vorhanden (1 Header + 4 Daten)
        assert len(merged) == 5
        assert merged[0] == ["Name", "Wert"]
        assert ["Charlie", "3"] in merged
        assert ["Dave", "4"] in merged

    def test_merge_header_deduplication(self):
        """
        Wenn die Folgeseite denselben Header wiederholt, wird er entfernt.
        AC-001-3
        """
        page_tables = [
            {
                "page": 1,
                "tables": [
                    [["Name", "Score"], ["Alice", "95"]]
                ]
            },
            {
                "page": 2,
                "tables": [
                    # Header identisch → soll dedupliziert werden
                    [["Name", "Score"], ["Bob", "88"], ["Carol", "72"]]
                ]
            },
        ]
        result = merge_cross_page_tables(page_tables)

        assert len(result) == 1
        merged = result[0]
        # Header genau einmal vorhanden
        assert merged.count(["Name", "Score"]) == 1
        # Daten beider Seiten vorhanden
        assert ["Alice", "95"] in merged
        assert ["Bob", "88"] in merged
        assert ["Carol", "72"] in merged

    def test_independent_tables_not_merged(self):
        """
        Tabellen mit unterschiedlicher Spaltenanzahl bleiben getrennt.
        AC-001-4
        """
        page_tables = [
            {
                "page": 1,
                "tables": [
                    [["A", "B"], ["1", "2"]]
                ]
            },
            {
                "page": 2,
                "tables": [
                    # Andere Spaltenanzahl (3 statt 2)
                    [["X", "Y", "Z"], ["a", "b", "c"]]
                ]
            },
        ]
        result = merge_cross_page_tables(page_tables)

        assert len(result) == 2
        assert result[0] == [["A", "B"], ["1", "2"]]
        assert result[1] == [["X", "Y", "Z"], ["a", "b", "c"]]

    def test_multiple_tables_per_page(self):
        """
        Mehrere Tabellen auf einer Seite: nur Tabellen mit gleicher Spaltenanzahl
        werden zusammengeführt, unabhängige bleiben getrennt.
        """
        page_tables = [
            {
                "page": 1,
                "tables": [
                    [["Produkt", "Preis"], ["Widget", "9.99"]],
                    [["Kategorie", "Anzahl", "Lager"], ["A", "10", "Berlin"]],
                ]
            },
            {
                "page": 2,
                "tables": [
                    # Gleiche Spaltenanzahl wie Tabelle 1 auf Seite 1 (2 Spalten)
                    [["Gadget", "14.99"]],
                ]
            },
        ]
        result = merge_cross_page_tables(page_tables)

        # Tabelle 1 (2 Spalten) + Tabelle 2 (3 Spalten) auf Seite 1 → 2 Tabellen
        # Dann Folgeseite (2 Spalten) merged mit letzter Tabelle die 2 Spalten hat
        # Reihenfolge: [Produkt/Preis], [Kategorie/Anzahl/Lager-merged-mit-Gadget?]
        # Nein: Tabelle 2 (3 Spalten) ≠ Tabelle 3 (2 Spalten) → separate
        # Aber Tabelle 3 hat gleiche Spaltenanzahl wie Tabelle 2 nach Tabelle 2...
        # Erwartung: 3 getrennte Tabellen (2-Spalten, 3-Spalten, 2-Spalten)
        assert len(result) == 3

    def test_empty_input_returns_empty_list(self):
        """Leere Eingabe ergibt leere Liste. AC-001-5 (edge case)"""
        assert merge_cross_page_tables([]) == []

    def test_single_page_no_merge_needed(self):
        """Einzelne Seite mit einer Tabelle bleibt unverändert."""
        page_tables = [
            {
                "page": 1,
                "tables": [
                    [["Header"], ["Row1"], ["Row2"]]
                ]
            }
        ]
        result = merge_cross_page_tables(page_tables)

        assert len(result) == 1
        assert result[0] == [["Header"], ["Row1"], ["Row2"]]


# =============================================================================
# test_extract_tables_with_pdfplumber (mit Mocking)
# =============================================================================

class TestExtractTablesWithPdfplumber:
    """Testet die pdfplumber-Integration (gemockt)."""

    def test_returns_empty_list_when_pdfplumber_not_available(self, tmp_path):
        """Wenn pdfplumber nicht importierbar ist, leere Liste zurückgeben."""
        dummy_pdf = tmp_path / "test.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4 dummy")

        with patch.object(_server, "PDFPLUMBER_AVAILABLE", False):
            result = extract_tables_with_pdfplumber(dummy_pdf)

        assert result == []

    def test_extracts_tables_from_pages(self, tmp_path):
        """Tabellen werden seitenweise korrekt zurückgegeben."""
        dummy_pdf = tmp_path / "test.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4 dummy")

        mock_page1 = MagicMock()
        mock_page1.extract_tables.return_value = [
            [["Col1", "Col2"], ["A", "B"]]
        ]

        mock_page2 = MagicMock()
        mock_page2.extract_tables.return_value = [
            [["C", "D"], ["E", "F"]]
        ]

        mock_pdf = MagicMock()
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdf.pages = [mock_page1, mock_page2]

        with patch.object(_server, "PDFPLUMBER_AVAILABLE", True):
            with patch("pdfplumber.open", return_value=mock_pdf):
                result = extract_tables_with_pdfplumber(dummy_pdf)

        assert len(result) == 2
        assert result[0]["page"] == 1
        assert result[0]["tables"] == [[["Col1", "Col2"], ["A", "B"]]]
        assert result[1]["page"] == 2
        assert result[1]["tables"] == [[["C", "D"], ["E", "F"]]]

    def test_pages_without_tables_not_included(self, tmp_path):
        """Seiten ohne Tabellen erscheinen nicht im Ergebnis."""
        dummy_pdf = tmp_path / "test.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4 dummy")

        mock_page_with = MagicMock()
        mock_page_with.extract_tables.return_value = [[["X", "Y"]]]

        mock_page_without = MagicMock()
        mock_page_without.extract_tables.return_value = []

        mock_pdf = MagicMock()
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdf.pages = [mock_page_without, mock_page_with]

        with patch.object(_server, "PDFPLUMBER_AVAILABLE", True):
            with patch("pdfplumber.open", return_value=mock_pdf):
                result = extract_tables_with_pdfplumber(dummy_pdf)

        assert len(result) == 1
        assert result[0]["page"] == 2

    def test_returns_empty_on_pdfplumber_exception(self, tmp_path):
        """Bei Exception von pdfplumber wird eine leere Liste zurückgegeben."""
        dummy_pdf = tmp_path / "test.pdf"
        dummy_pdf.write_bytes(b"not a real pdf")

        with patch.object(_server, "PDFPLUMBER_AVAILABLE", True):
            with patch("pdfplumber.open", side_effect=Exception("Corrupt PDF")):
                result = extract_tables_with_pdfplumber(dummy_pdf)

        assert result == []


# =============================================================================
# test_empty_tables (edge cases)
# =============================================================================

class TestEdgeCases:
    """Edge Cases für leere und fehlerhafte Eingaben."""

    def test_tables_to_markdown_with_empty_table_in_list(self):
        """Eine leere Tabelle in der Liste wird übersprungen."""
        tables = [[], [["A", "B"], ["1", "2"]]]
        result = tables_to_markdown(tables)

        assert "| A | B |" in result
        assert "| 1 | 2 |" in result

    def test_merge_cross_page_skips_empty_tables(self):
        """Leere Tabellen im page_tables Dict werden korrekt behandelt."""
        page_tables = [
            {"page": 1, "tables": [[]]},
            {"page": 2, "tables": [[["A", "B"], ["1", "2"]]]},
        ]
        result = merge_cross_page_tables(page_tables)
        # Leere Tabelle + nicht-leere Tabelle
        assert len(result) == 2

    def test_tables_to_markdown_single_column(self):
        """Einzelspaltige Tabelle wird korrekt formatiert."""
        tables = [
            [["Item"], ["Apfel"], ["Birne"]]
        ]
        result = tables_to_markdown(tables)

        assert "| Item |" in result
        assert "| --- |" in result
        assert "| Apfel |" in result
        assert "| Birne |" in result

    def test_merge_no_tables_in_page_entry(self):
        """page_tables mit leerer tables-Liste wird korrekt verarbeitet."""
        page_tables = [
            {"page": 1, "tables": []},
        ]
        result = merge_cross_page_tables(page_tables)
        assert result == []
