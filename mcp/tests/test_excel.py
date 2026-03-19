"""
Unit-Tests für erweiterte Excel-Konvertierung (FR-MKIT-007).

AC-007-1: Jedes Worksheet als eigene ## Sheet: <Name> Sektion
AC-007-2: Charts als Datentabellen extrahiert
AC-007-3: Zellen mit Formeln optional annotiert
AC-007-4: Merged Cells korrekt aufgelöst

Tests laufen ohne Docker-Container und ohne echte XLSX-Dateien (Mocking).
"""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import pytest

from conftest import load_server_module, run_async


# =============================================================================
# Modul-Setup mit openpyxl-Mock
# =============================================================================

def _make_openpyxl_mock():
    """Erstellt einen vollständigen openpyxl-Mock der sicher importiert werden kann."""
    openpyxl_mod = MagicMock()
    openpyxl_utils_mod = MagicMock()
    openpyxl_utils_mod.get_column_letter = lambda i: chr(ord("A") + i - 1)
    openpyxl_mod.utils = openpyxl_utils_mod

    # openpyxl.load_workbook gibt einen Workbook-Mock zurück
    openpyxl_mod.load_workbook = MagicMock(return_value=MagicMock())

    return openpyxl_mod, openpyxl_utils_mod


_openpyxl_mock, _openpyxl_utils_mock = _make_openpyxl_mock()

_server = load_server_module(
    use_real_pil=False,
    extra_patches={
        "openpyxl": _openpyxl_mock,
        "openpyxl.utils": _openpyxl_utils_mock,
    },
)

convert_excel_enhanced = _server.convert_excel_enhanced
convert_with_markitdown = _server.convert_with_markitdown
convert_auto = _server.convert_auto


# =============================================================================
# Hilfsfunktionen für Mock-Workbooks
# =============================================================================

def _make_cell(value=None, formula=None):
    """Erstellt eine Mock-Zelle mit value und optionaler Formel."""
    cell = MagicMock()
    cell.value = value
    return cell, MagicMock(value=formula if formula is not None else value)


def _make_row(*cells):
    """Gibt (value_cells, formula_cells) als Row-Tupel zurück."""
    val_cells = [c[0] for c in cells]
    form_cells = [c[1] for c in cells]
    return val_cells, form_cells


def _make_worksheet(name: str, rows_val, rows_form, merged_ranges=None, charts=None):
    """
    Erstellt einen Mock-Worksheet.

    Args:
        name: Sheet-Name
        rows_val: Liste von Zeilen (jede Zeile: Liste von MagicMock-Zellen mit .value)
        rows_form: Liste von Zeilen für Formel-Workbook
        merged_ranges: Liste von Mock-MergeRange-Objekten
        charts: Liste von Chart-Mocks (optional)
    """
    ws = MagicMock()
    ws.title = name

    # iter_rows() gibt Tupel von Zellen zurück — side_effect für frischen Iterator je Aufruf
    _rows_val_tuples = [tuple(row) for row in rows_val]
    _rows_form_tuples = [tuple(row) for row in rows_form]
    ws.iter_rows = MagicMock(side_effect=lambda: iter(_rows_val_tuples))

    ws_form = MagicMock()
    ws_form.title = name
    ws_form.iter_rows = MagicMock(side_effect=lambda: iter(_rows_form_tuples))

    # Merged cells
    merge_mock = MagicMock()
    merge_mock.ranges = merged_ranges or []
    ws.merged_cells = merge_mock
    ws_form.merged_cells = merge_mock

    # cell(row, col) für Merged-Cell-Auflösung
    def _cell_value(row, col):
        """Gibt eine Zelle mit dem value aus rows_val zurück."""
        try:
            raw_val = rows_val[row - 1][col - 1]
            c = MagicMock()
            c.value = raw_val.value if hasattr(raw_val, "value") else raw_val
            return c
        except (IndexError, AttributeError):
            return MagicMock(value=None)

    def _cell_formula(row, col):
        """Gibt eine Zelle mit dem value aus rows_form zurück."""
        try:
            raw_val = rows_form[row - 1][col - 1]
            c = MagicMock()
            c.value = raw_val.value if hasattr(raw_val, "value") else raw_val
            return c
        except (IndexError, AttributeError):
            return MagicMock(value=None)

    ws.cell = _cell_value
    ws_form.cell = _cell_formula

    # Charts
    ws._charts = charts or []

    return ws, ws_form


def _make_workbook(sheets: list[tuple]):
    """
    Erstellt Mock-Workbooks für values (data_only=True) und formulas (data_only=False).

    Args:
        sheets: Liste von (name, ws_values, ws_formulas) Tupeln

    Returns:
        Tuple (wb_values, wb_formulas) als Mock-Workbook-Objekte.

    WICHTIG: In convert_excel_enhanced wird geladen als:
        wb_formulas = load_workbook(..., data_only=False)   # 1. Aufruf
        wb_values   = load_workbook(..., data_only=True)    # 2. Aufruf
    Daher muss in Tests der patch als side_effect=[wb_form, wb_val] übergeben werden.
    """
    wb_val = MagicMock()
    wb_form = MagicMock()

    sheet_names = [s[0] for s in sheets]
    wb_val.sheetnames = sheet_names
    wb_form.sheetnames = sheet_names

    val_sheet_map = {s[0]: s[1] for s in sheets}
    form_sheet_map = {s[0]: s[2] for s in sheets}

    wb_val.__getitem__ = MagicMock(side_effect=lambda name: val_sheet_map[name])
    wb_form.__getitem__ = MagicMock(side_effect=lambda name: form_sheet_map[name])

    wb_val.close = MagicMock()
    wb_form.close = MagicMock()

    return wb_val, wb_form


def _load_wb_side_effect(wb_val, wb_form):
    """
    Gibt die richtige load_workbook side_effect-Liste zurück.

    convert_excel_enhanced ruft load_workbook in dieser Reihenfolge auf:
    1. data_only=False → wb_formulas  (wb_form)
    2. data_only=True  → wb_values    (wb_val)
    """
    return [wb_form, wb_val]


# =============================================================================
# TestMultiSheetOutput (AC-007-1)
# =============================================================================

class TestMultiSheetOutput:
    """Testet dass jedes Worksheet als eigene ## Sheet: <Name> Sektion ausgegeben wird."""

    def _make_simple_sheet(self, name: str, data: list[list]):
        """Hilfsmethode: Sheet mit einfachen Stringwerten."""
        rows_val = []
        rows_form = []
        for row_data in data:
            row_v = []
            row_f = []
            for val in row_data:
                cv = MagicMock()
                cv.value = val
                cf = MagicMock()
                cf.value = val
                row_v.append(cv)
                row_f.append(cf)
            rows_val.append(row_v)
            rows_form.append(row_f)

        ws_v, ws_f = _make_worksheet(name, rows_val, rows_form)
        return ws_v, ws_f

    def test_single_sheet_has_heading(self, tmp_path):
        """Ein Sheet erzeugt eine ## Sheet: Überschrift."""
        ws_v, ws_f = self._make_simple_sheet("Tabelle1", [["A", "B"], ["1", "2"]])
        wb_val, wb_form = _make_workbook([("Tabelle1", ws_v, ws_f)])

        with patch.object(_server, "OPENPYXL_AVAILABLE", True), \
             patch("openpyxl.load_workbook", side_effect=_load_wb_side_effect(wb_val, wb_form)):
            result = convert_excel_enhanced(tmp_path / "test.xlsx")

        assert result["success"] is True
        assert "## Sheet: Tabelle1" in result["markdown"]

    def test_multi_sheet_all_headings_present(self, tmp_path):
        """Mehrere Sheets erzeugen je eine ## Sheet: Überschrift."""
        ws1_v, ws1_f = self._make_simple_sheet("Sheet1", [["Name"], ["Alice"]])
        ws2_v, ws2_f = self._make_simple_sheet("Sheet2", [["City"], ["Berlin"]])
        ws3_v, ws3_f = self._make_simple_sheet("Summary", [["Total"], ["42"]])

        wb_val, wb_form = _make_workbook([
            ("Sheet1", ws1_v, ws1_f),
            ("Sheet2", ws2_v, ws2_f),
            ("Summary", ws3_v, ws3_f),
        ])

        with patch.object(_server, "OPENPYXL_AVAILABLE", True), \
             patch("openpyxl.load_workbook", side_effect=_load_wb_side_effect(wb_val, wb_form)):
            result = convert_excel_enhanced(tmp_path / "test.xlsx")

        assert result["success"] is True
        assert "## Sheet: Sheet1" in result["markdown"]
        assert "## Sheet: Sheet2" in result["markdown"]
        assert "## Sheet: Summary" in result["markdown"]
        assert result["sheets_count"] == 3

    def test_sheet_data_appears_as_markdown_table(self, tmp_path):
        """Sheet-Daten erscheinen als Markdown-Tabelle mit Header-Trennzeile."""
        ws_v, ws_f = self._make_simple_sheet(
            "Data", [["Name", "Value"], ["Alice", "100"], ["Bob", "200"]]
        )
        wb_val, wb_form = _make_workbook([("Data", ws_v, ws_f)])

        with patch.object(_server, "OPENPYXL_AVAILABLE", True), \
             patch("openpyxl.load_workbook", side_effect=_load_wb_side_effect(wb_val, wb_form)):
            result = convert_excel_enhanced(tmp_path / "test.xlsx")

        md = result["markdown"]
        assert "| Name | Value |" in md
        assert "| --- | --- |" in md
        assert "| Alice | 100 |" in md
        assert "| Bob | 200 |" in md

    def test_sheets_count_returned(self, tmp_path):
        """sheets_count gibt die korrekte Anzahl der Sheets zurück."""
        ws1_v, ws1_f = self._make_simple_sheet("A", [["X"], ["1"]])
        ws2_v, ws2_f = self._make_simple_sheet("B", [["Y"], ["2"]])
        wb_val, wb_form = _make_workbook([("A", ws1_v, ws1_f), ("B", ws2_v, ws2_f)])

        with patch.object(_server, "OPENPYXL_AVAILABLE", True), \
             patch("openpyxl.load_workbook", side_effect=_load_wb_side_effect(wb_val, wb_form)):
            result = convert_excel_enhanced(tmp_path / "test.xlsx")

        assert result["sheets_count"] == 2


# =============================================================================
# TestMergedCells (AC-007-4)
# =============================================================================

class TestMergedCells:
    """Testet die korrekte Auflösung von Merged Cells."""

    def test_merged_cell_value_propagated(self, tmp_path):
        """Der Wert der Hauptzelle wird für alle Zellen im Merge-Bereich übernommen."""
        # Zelle (1,1) ist gemergt mit (1,2) — beide sollen "Merged" anzeigen

        # rows_val: Zeile 1 hat 2 Zellen
        c1_v = MagicMock()
        c1_v.value = "Merged"
        c2_v = MagicMock()
        c2_v.value = None  # Merged-Zelle ohne eigenen Wert

        c1_f = MagicMock()
        c1_f.value = "Merged"
        c2_f = MagicMock()
        c2_f.value = None

        rows_val = [[c1_v, c2_v]]
        rows_form = [[c1_f, c2_f]]

        # Merge-Range: Zeile 1, Spalten 1-2
        merge_range = MagicMock()
        merge_range.min_row = 1
        merge_range.max_row = 1
        merge_range.min_col = 1
        merge_range.max_col = 2

        ws_v, ws_f = _make_worksheet(
            "MergeSheet", rows_val, rows_form, merged_ranges=[merge_range]
        )
        wb_val, wb_form = _make_workbook([("MergeSheet", ws_v, ws_f)])

        with patch.object(_server, "OPENPYXL_AVAILABLE", True), \
             patch("openpyxl.load_workbook", side_effect=_load_wb_side_effect(wb_val, wb_form)):
            result = convert_excel_enhanced(tmp_path / "test.xlsx")

        assert result["success"] is True
        md = result["markdown"]
        # Beide Spalten sollten "Merged" zeigen
        assert "Merged" in md


# =============================================================================
# TestFormulas (AC-007-3)
# =============================================================================

class TestFormulas:
    """Testet die Formel-Annotierung."""

    def _make_formula_sheet(self, name, raw_value, formula_str):
        """Sheet mit einer Datenzelle die eine Formel enthält."""
        # Header-Zeile
        h_v = MagicMock()
        h_v.value = "Result"
        h_f = MagicMock()
        h_f.value = "Result"

        # Datenzelle mit Formel
        d_v = MagicMock()
        d_v.value = raw_value
        d_f = MagicMock()
        d_f.value = formula_str

        rows_val = [[h_v], [d_v]]
        rows_form = [[h_f], [d_f]]

        return _make_worksheet(name, rows_val, rows_form)

    def test_formulas_shown_when_enabled(self, tmp_path):
        """Wenn show_formulas=True, wird Formel-Annotation zur Zelle hinzugefügt."""
        ws_v, ws_f = self._make_formula_sheet("Sheet1", 42, "=SUM(A1:A10)")
        wb_val, wb_form = _make_workbook([("Sheet1", ws_v, ws_f)])

        with patch.object(_server, "OPENPYXL_AVAILABLE", True), \
             patch("openpyxl.load_workbook", side_effect=_load_wb_side_effect(wb_val, wb_form)):
            result = convert_excel_enhanced(tmp_path / "test.xlsx", show_formulas=True)

        assert result["success"] is True
        assert "[=SUM(A1:A10)]" in result["markdown"]
        assert "42" in result["markdown"]

    def test_formulas_hidden_by_default(self, tmp_path):
        """Ohne show_formulas=True werden keine Formel-Annotierungen angezeigt."""
        ws_v, ws_f = self._make_formula_sheet("Sheet1", 42, "=SUM(A1:A10)")
        wb_val, wb_form = _make_workbook([("Sheet1", ws_v, ws_f)])

        with patch.object(_server, "OPENPYXL_AVAILABLE", True), \
             patch("openpyxl.load_workbook", side_effect=_load_wb_side_effect(wb_val, wb_form)):
            result = convert_excel_enhanced(tmp_path / "test.xlsx")

        assert result["success"] is True
        assert "=SUM" not in result["markdown"]
        assert "[=" not in result["markdown"]

    def test_only_formula_cells_annotated(self, tmp_path):
        """Nur Zellen mit '='-Präfix werden annotiert, normale Werte nicht."""
        # Zwei Zeilen: eine mit Formel, eine ohne
        h_v = MagicMock()
        h_v.value = "Value"
        h_f = MagicMock()
        h_f.value = "Value"

        normal_v = MagicMock()
        normal_v.value = "hello"
        normal_f = MagicMock()
        normal_f.value = "hello"  # Kein '='-Präfix

        formula_v = MagicMock()
        formula_v.value = 99
        formula_f = MagicMock()
        formula_f.value = "=A1+B1"

        rows_val = [[h_v], [normal_v], [formula_v]]
        rows_form = [[h_f], [normal_f], [formula_f]]

        ws_v, ws_f = _make_worksheet("Sheet1", rows_val, rows_form)
        wb_val, wb_form = _make_workbook([("Sheet1", ws_v, ws_f)])

        with patch.object(_server, "OPENPYXL_AVAILABLE", True), \
             patch("openpyxl.load_workbook", side_effect=_load_wb_side_effect(wb_val, wb_form)):
            result = convert_excel_enhanced(tmp_path / "test.xlsx", show_formulas=True)

        md = result["markdown"]
        # Formel-Zeile hat Annotation
        assert "[=A1+B1]" in md
        # Normale Zeile hat KEINE Annotation
        assert "hello [=" not in md


# =============================================================================
# TestChartExtraction (AC-007-2)
# =============================================================================

class TestChartExtraction:
    """Testet die Chart-Extraktion aus Excel-Sheets."""

    def _make_chart_mock(self, title: str, series_data: list[dict]):
        """
        Erstellt einen Mock-Chart mit Titel und Datenserien.

        series_data: [{"title": "Name", "values": [1, 2, 3]}, ...]
        """
        chart = MagicMock()
        chart.title = title

        series_mocks = []
        for sd in series_data:
            serie = MagicMock()

            # Serie-Titel
            st = MagicMock()
            st.v = sd["title"]
            st.strRef = None
            serie.title = st

            # Serie-Werte via numRef.numCache.pt
            pts = [MagicMock(v=str(v)) for v in sd["values"]]
            num_cache = MagicMock()
            num_cache.pt = pts

            num_ref = MagicMock()
            num_ref.numCache = num_cache

            val_ref = MagicMock()
            val_ref.numRef = num_ref
            serie.val = val_ref

            series_mocks.append(serie)

        chart.series = series_mocks
        return chart

    def test_chart_section_appears_in_output(self, tmp_path):
        """Wenn ein Chart vorhanden ist, erscheint eine ### Chart: Sektion."""
        chart = self._make_chart_mock("Umsatz 2024", [
            {"title": "Q1", "values": [100, 200]},
        ])

        # Einfaches Sheet mit einem Chart
        h_v = MagicMock()
        h_v.value = "X"
        h_f = MagicMock()
        h_f.value = "X"
        ws_v, ws_f = _make_worksheet("Data", [[h_v]], [[h_f]], charts=[chart])
        wb_val, wb_form = _make_workbook([("Data", ws_v, ws_f)])

        with patch.object(_server, "OPENPYXL_AVAILABLE", True), \
             patch("openpyxl.load_workbook", side_effect=_load_wb_side_effect(wb_val, wb_form)):
            result = convert_excel_enhanced(tmp_path / "test.xlsx")

        assert result["success"] is True
        assert "### Chart:" in result["markdown"]
        assert result["charts_count"] == 1

    def test_chart_data_as_table(self, tmp_path):
        """Chart-Datenserien werden als Markdown-Tabelle dargestellt."""
        chart = self._make_chart_mock("Verkauf", [
            {"title": "Produkt A", "values": [10, 20, 30]},
            {"title": "Produkt B", "values": [15, 25, 35]},
        ])

        h_v = MagicMock()
        h_v.value = "Header"
        h_f = MagicMock()
        h_f.value = "Header"
        ws_v, ws_f = _make_worksheet("Sales", [[h_v]], [[h_f]], charts=[chart])
        wb_val, wb_form = _make_workbook([("Sales", ws_v, ws_f)])

        with patch.object(_server, "OPENPYXL_AVAILABLE", True), \
             patch("openpyxl.load_workbook", side_effect=_load_wb_side_effect(wb_val, wb_form)):
            result = convert_excel_enhanced(tmp_path / "test.xlsx")

        md = result["markdown"]
        assert "| Produkt A | Produkt B |" in md
        assert "10" in md
        assert "15" in md

    def test_no_charts_count_zero(self, tmp_path):
        """Ohne Charts ist charts_count = 0."""
        h_v = MagicMock()
        h_v.value = "Col"
        h_f = MagicMock()
        h_f.value = "Col"
        ws_v, ws_f = _make_worksheet("NoCharts", [[h_v]], [[h_f]])
        wb_val, wb_form = _make_workbook([("NoCharts", ws_v, ws_f)])

        with patch.object(_server, "OPENPYXL_AVAILABLE", True), \
             patch("openpyxl.load_workbook", side_effect=_load_wb_side_effect(wb_val, wb_form)):
            result = convert_excel_enhanced(tmp_path / "test.xlsx")

        assert result["charts_count"] == 0


# =============================================================================
# TestEmptySheet
# =============================================================================

class TestEmptySheet:
    """Testet das graceful Handling von leeren Sheets."""

    def test_empty_sheet_shows_no_content_message(self, tmp_path):
        """Ein Sheet ohne Zeilen zeigt '*Kein Inhalt*'."""
        ws_v = MagicMock()
        ws_v.title = "Empty"
        ws_v.iter_rows = MagicMock(return_value=iter([]))
        merge_mock = MagicMock()
        merge_mock.ranges = []
        ws_v.merged_cells = merge_mock
        ws_v._charts = []

        ws_f = MagicMock()
        ws_f.title = "Empty"
        ws_f.iter_rows = MagicMock(return_value=iter([]))
        ws_f.merged_cells = merge_mock

        wb_val, wb_form = _make_workbook([("Empty", ws_v, ws_f)])

        with patch.object(_server, "OPENPYXL_AVAILABLE", True), \
             patch("openpyxl.load_workbook", side_effect=_load_wb_side_effect(wb_val, wb_form)):
            result = convert_excel_enhanced(tmp_path / "test.xlsx")

        assert result["success"] is True
        assert "## Sheet: Empty" in result["markdown"]
        assert "*Kein Inhalt*" in result["markdown"]

    def test_all_none_cells_shows_no_content(self, tmp_path):
        """Ein Sheet wo alle Zellen None sind zeigt '*Kein Inhalt*'."""
        c1 = MagicMock()
        c1.value = None
        c2 = MagicMock()
        c2.value = None

        ws_v, ws_f = _make_worksheet(
            "AllNone",
            [[c1, c2]],
            [[MagicMock(value=None), MagicMock(value=None)]],
        )
        wb_val, wb_form = _make_workbook([("AllNone", ws_v, ws_f)])

        with patch.object(_server, "OPENPYXL_AVAILABLE", True), \
             patch("openpyxl.load_workbook", side_effect=_load_wb_side_effect(wb_val, wb_form)):
            result = convert_excel_enhanced(tmp_path / "test.xlsx")

        assert result["success"] is True
        assert "*Kein Inhalt*" in result["markdown"]


# =============================================================================
# TestOpenpyxlNotAvailable
# =============================================================================

class TestOpenpyxlNotAvailable:
    """Testet den Fallback wenn openpyxl nicht installiert ist."""

    def test_returns_error_when_openpyxl_missing(self, tmp_path):
        """Wenn OPENPYXL_AVAILABLE=False, wird ein Fehler zurückgegeben."""
        with patch.object(_server, "OPENPYXL_AVAILABLE", False):
            result = convert_excel_enhanced(tmp_path / "test.xlsx")

        assert result["success"] is False
        assert "openpyxl" in result["error"].lower()


# =============================================================================
# TestOpenError
# =============================================================================

class TestOpenError:
    """Testet das Fehler-Handling beim Öffnen einer Excel-Datei."""

    def test_returns_error_on_corrupt_file(self, tmp_path):
        """Bei einem Fehler beim Öffnen der Workbook wird ein Fehler zurückgegeben."""
        with patch.object(_server, "OPENPYXL_AVAILABLE", True), \
             patch("openpyxl.load_workbook", side_effect=Exception("Corrupt XLSX")):
            result = convert_excel_enhanced(tmp_path / "corrupt.xlsx")

        assert result["success"] is False
        assert result["error_code"] == "CONVERSION_FAILED"


# =============================================================================
# TestIntegrationInConvert (AC-007: integration in convert_with_markitdown + convert_auto)
# =============================================================================

class TestIntegrationInConvert:
    """Testet dass xlsx-Dateien über den enhanced Excel-Pfad verarbeitet werden."""

    def _make_simple_workbook(self):
        """Workbook-Mock mit einem Sheet und einfachen Daten."""
        h_v = MagicMock()
        h_v.value = "Col"
        h_f = MagicMock()
        h_f.value = "Col"
        d_v = MagicMock()
        d_v.value = "Val"
        d_f = MagicMock()
        d_f.value = "Val"
        ws_v, ws_f = _make_worksheet("Sheet1", [[h_v], [d_v]], [[h_f], [d_f]])
        wb_val, wb_form = _make_workbook([("Sheet1", ws_v, ws_f)])
        return wb_val, wb_form

    def test_convert_with_markitdown_routes_xlsx(self, tmp_path):
        """convert_with_markitdown leitet .xlsx an convert_excel_enhanced weiter."""
        xlsx_path = tmp_path / "test.xlsx"
        xlsx_path.write_bytes(b"PK fake xlsx")

        wb_val, wb_form = self._make_simple_workbook()

        with patch.object(_server, "OPENPYXL_AVAILABLE", True), \
             patch("openpyxl.load_workbook", side_effect=_load_wb_side_effect(wb_val, wb_form)):
            result = convert_with_markitdown(xlsx_path)

        assert result["success"] is True
        assert "## Sheet: Sheet1" in result["markdown"]
        assert "sheets_count" in result

    def test_convert_with_markitdown_routes_xls(self, tmp_path):
        """convert_with_markitdown leitet .xls an convert_excel_enhanced weiter."""
        xls_path = tmp_path / "test.xls"
        xls_path.write_bytes(b"\xd0\xcf fake xls")

        wb_val, wb_form = self._make_simple_workbook()

        with patch.object(_server, "OPENPYXL_AVAILABLE", True), \
             patch("openpyxl.load_workbook", side_effect=_load_wb_side_effect(wb_val, wb_form)):
            result = convert_with_markitdown(xls_path)

        assert result["success"] is True
        assert "sheets_count" in result

    def test_convert_auto_xlsx_uses_enhanced_path(self, tmp_path):
        """convert_auto schreibt .xlsx in temp-Datei und ruft convert_with_markitdown auf."""
        xlsx_data = b"PK fake xlsx data"

        wb_val, wb_form = self._make_simple_workbook()

        # detect_mimetype_from_bytes muss None zurückgeben damit magic nicht interferiert
        with patch.object(_server, "OPENPYXL_AVAILABLE", True), \
             patch.object(_server, "detect_mimetype_from_bytes", return_value=None), \
             patch("openpyxl.load_workbook", side_effect=_load_wb_side_effect(wb_val, wb_form)):
            response = run_async(convert_auto(
                file_data=xlsx_data,
                filename="report.xlsx",
                source="test",
                source_type="base64",
                input_meta={},
                show_formulas=False,
            ))

        assert response.success is True
        assert "## Sheet: Sheet1" in response.markdown

    def test_show_formulas_passed_through(self, tmp_path):
        """show_formulas=True wird von convert_auto an convert_excel_enhanced weitergereicht."""
        xlsx_data = b"PK fake xlsx data"

        # Sheet mit Formel
        h_v = MagicMock()
        h_v.value = "Result"
        h_f = MagicMock()
        h_f.value = "Result"
        d_v = MagicMock()
        d_v.value = 100
        d_f = MagicMock()
        d_f.value = "=SUM(A2:A5)"

        ws_v, ws_f = _make_worksheet("Sheet1", [[h_v], [d_v]], [[h_f], [d_f]])
        wb_val, wb_form = _make_workbook([("Sheet1", ws_v, ws_f)])

        # detect_mimetype_from_bytes muss None zurückgeben damit magic nicht interferiert
        with patch.object(_server, "OPENPYXL_AVAILABLE", True), \
             patch.object(_server, "detect_mimetype_from_bytes", return_value=None), \
             patch("openpyxl.load_workbook", side_effect=_load_wb_side_effect(wb_val, wb_form)):
            response = run_async(convert_auto(
                file_data=xlsx_data,
                filename="formulas.xlsx",
                source="test",
                source_type="base64",
                input_meta={},
                show_formulas=True,
            ))

        assert response.success is True
        assert "[=SUM(A2:A5)]" in response.markdown
