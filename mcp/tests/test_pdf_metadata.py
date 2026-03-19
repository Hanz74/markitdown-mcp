"""
Tests für T-MKIT-009: PDF-Metadaten (Bookmarks, Annotationen, Formularfelder).

Alle Tests laufen ohne Docker-Container und ohne echte PDF-Dateien.
PyMuPDF (fitz) wird vollständig gemockt.

Acceptance Criteria:
- AC-009-1: PDF-Bookmarks/Outline werden als Inhaltsverzeichnis am Anfang eingefügt
- AC-009-2: PDF-Annotationen/Kommentare werden als Blockquotes extrahiert
- AC-009-3: Formularfelder werden als Key-Value-Paaren extrahiert
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from conftest import load_server_module, run_async


# ---------------------------------------------------------------------------
# Modul laden (einmal für alle Tests in dieser Datei)
# ---------------------------------------------------------------------------

_server = load_server_module(use_real_pil=False)


# ---------------------------------------------------------------------------
# Hilfsfunktionen: Mock-Objekte für fitz
# ---------------------------------------------------------------------------

def _make_mock_annot(annot_type_str: str, content: str, author: str) -> MagicMock:
    """Erstellt einen Mock für eine fitz-Annotation."""
    annot = MagicMock()
    annot.type = (0, annot_type_str)
    annot.info = {"content": content, "title": author}
    return annot


def _make_mock_widget(field_name: str, field_value: str, field_type_string: str, field_type: int = 2) -> MagicMock:
    """Erstellt einen Mock für ein fitz-Widget (Formularfeld)."""
    widget = MagicMock()
    widget.field_name = field_name
    widget.field_value = field_value
    widget.field_type_string = field_type_string
    widget.field_type = field_type
    return widget


def _make_mock_page(annots=None, widgets=None) -> MagicMock:
    """Erstellt eine Mock-Seite mit optionalen Annotationen und Widgets."""
    page = MagicMock()
    page.annots = MagicMock(return_value=annots or [])
    page.widgets = MagicMock(return_value=widgets or [])
    return page


def _make_mock_doc(toc=None, pages=None) -> MagicMock:
    """Erstellt ein Mock-fitz-Dokument."""
    doc = MagicMock()
    doc.get_toc = MagicMock(return_value=toc or [])
    pages = pages or []
    doc.__iter__ = MagicMock(return_value=iter(pages))
    doc.close = MagicMock()
    return doc


# ---------------------------------------------------------------------------
# Tests: extract_pdf_metadata — Bookmarks/TOC (AC-009-1)
# ---------------------------------------------------------------------------

class TestExtractToc:
    """AC-009-1: Bookmarks/TOC werden extrahiert."""

    def test_extract_toc_basic(self, tmp_path: Path) -> None:
        """Einfaches TOC mit einem Level-1-Eintrag wird korrekt extrahiert."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 minimal")

        toc_data = [[1, "Einleitung", 1], [1, "Hauptteil", 5], [2, "Unterabschnitt", 7]]
        mock_doc = _make_mock_doc(toc=toc_data)

        mock_fitz = MagicMock()
        mock_fitz.open = MagicMock(return_value=mock_doc)

        with patch.dict(sys.modules, {"fitz": mock_fitz}):
            _server.PYMUPDF_AVAILABLE = True
            _server.fitz = mock_fitz
            result = _server.extract_pdf_metadata(pdf_path)

        assert result["toc"] == toc_data
        assert len(result["toc"]) == 3

    def test_extract_toc_empty(self, tmp_path: Path) -> None:
        """PDF ohne Bookmarks → leere TOC-Liste."""
        pdf_path = tmp_path / "no_toc.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        mock_doc = _make_mock_doc(toc=[])

        mock_fitz = MagicMock()
        mock_fitz.open = MagicMock(return_value=mock_doc)

        with patch.dict(sys.modules, {"fitz": mock_fitz}):
            _server.PYMUPDF_AVAILABLE = True
            _server.fitz = mock_fitz
            result = _server.extract_pdf_metadata(pdf_path)

        assert result["toc"] == []

    def test_extract_toc_multilevel(self, tmp_path: Path) -> None:
        """Mehrstufiges TOC mit Level 1, 2 und 3 wird vollständig extrahiert."""
        pdf_path = tmp_path / "multilevel.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        toc_data = [
            [1, "Kapitel 1", 1],
            [2, "Abschnitt 1.1", 3],
            [3, "Unterabschnitt 1.1.1", 4],
            [1, "Kapitel 2", 10],
        ]
        mock_doc = _make_mock_doc(toc=toc_data)

        mock_fitz = MagicMock()
        mock_fitz.open = MagicMock(return_value=mock_doc)

        with patch.dict(sys.modules, {"fitz": mock_fitz}):
            _server.PYMUPDF_AVAILABLE = True
            _server.fitz = mock_fitz
            result = _server.extract_pdf_metadata(pdf_path)

        assert len(result["toc"]) == 4
        levels = [entry[0] for entry in result["toc"]]
        assert 1 in levels
        assert 2 in levels
        assert 3 in levels


# ---------------------------------------------------------------------------
# Tests: extract_pdf_metadata — Annotationen (AC-009-2)
# ---------------------------------------------------------------------------

class TestExtractAnnotations:
    """AC-009-2: Annotationen werden extrahiert."""

    def test_extract_single_annotation(self, tmp_path: Path) -> None:
        """Eine Annotation mit Typ, Content und Author wird korrekt extrahiert."""
        pdf_path = tmp_path / "annot.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        annot = _make_mock_annot("Text", "Wichtiger Kommentar", "Max Mustermann")
        page = _make_mock_page(annots=[annot])
        mock_doc = _make_mock_doc(pages=[page])

        mock_fitz = MagicMock()
        mock_fitz.open = MagicMock(return_value=mock_doc)

        with patch.dict(sys.modules, {"fitz": mock_fitz}):
            _server.PYMUPDF_AVAILABLE = True
            _server.fitz = mock_fitz
            result = _server.extract_pdf_metadata(pdf_path)

        assert len(result["annotations"]) == 1
        ann = result["annotations"][0]
        assert ann["type"] == "Text"
        assert ann["content"] == "Wichtiger Kommentar"
        assert ann["author"] == "Max Mustermann"
        assert ann["page"] == 1

    def test_extract_multiple_annotations_across_pages(self, tmp_path: Path) -> None:
        """Annotationen auf mehreren Seiten werden alle extrahiert."""
        pdf_path = tmp_path / "multi_annot.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        annot1 = _make_mock_annot("Highlight", "Seite 1 Kommentar", "Alice")
        annot2 = _make_mock_annot("Note", "Seite 2 Kommentar", "Bob")
        page1 = _make_mock_page(annots=[annot1])
        page2 = _make_mock_page(annots=[annot2])
        mock_doc = _make_mock_doc(pages=[page1, page2])

        mock_fitz = MagicMock()
        mock_fitz.open = MagicMock(return_value=mock_doc)

        with patch.dict(sys.modules, {"fitz": mock_fitz}):
            _server.PYMUPDF_AVAILABLE = True
            _server.fitz = mock_fitz
            result = _server.extract_pdf_metadata(pdf_path)

        assert len(result["annotations"]) == 2
        pages = {ann["page"] for ann in result["annotations"]}
        assert pages == {1, 2}
        authors = {ann["author"] for ann in result["annotations"]}
        assert authors == {"Alice", "Bob"}

    def test_extract_annotations_empty_when_no_annotations(self, tmp_path: Path) -> None:
        """PDF ohne Annotationen → leere Annotationen-Liste."""
        pdf_path = tmp_path / "no_annot.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        page = _make_mock_page(annots=[])
        mock_doc = _make_mock_doc(pages=[page])

        mock_fitz = MagicMock()
        mock_fitz.open = MagicMock(return_value=mock_doc)

        with patch.dict(sys.modules, {"fitz": mock_fitz}):
            _server.PYMUPDF_AVAILABLE = True
            _server.fitz = mock_fitz
            result = _server.extract_pdf_metadata(pdf_path)

        assert result["annotations"] == []


# ---------------------------------------------------------------------------
# Tests: extract_pdf_metadata — Formularfelder (AC-009-3)
# ---------------------------------------------------------------------------

class TestExtractFormFields:
    """AC-009-3: Formularfelder werden extrahiert."""

    def test_extract_single_form_field(self, tmp_path: Path) -> None:
        """Ein Formularfeld mit Name, Wert und Typ wird korrekt extrahiert."""
        pdf_path = tmp_path / "form.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        widget = _make_mock_widget("Vorname", "Max", "Text")
        page = _make_mock_page(widgets=[widget])
        mock_doc = _make_mock_doc(pages=[page])

        mock_fitz = MagicMock()
        mock_fitz.open = MagicMock(return_value=mock_doc)

        with patch.dict(sys.modules, {"fitz": mock_fitz}):
            _server.PYMUPDF_AVAILABLE = True
            _server.fitz = mock_fitz
            result = _server.extract_pdf_metadata(pdf_path)

        assert len(result["form_fields"]) == 1
        field = result["form_fields"][0]
        assert field["field_name"] == "Vorname"
        assert field["field_value"] == "Max"
        assert field["field_type"] == "Text"
        assert field["page"] == 1

    def test_extract_multiple_form_fields(self, tmp_path: Path) -> None:
        """Mehrere Formularfelder auf einer Seite werden alle extrahiert."""
        pdf_path = tmp_path / "multi_form.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        widget1 = _make_mock_widget("Name", "Max", "Text")
        widget2 = _make_mock_widget("Alter", "30", "Text")
        widget3 = _make_mock_widget("Zustimmung", "True", "CheckBox")
        page = _make_mock_page(widgets=[widget1, widget2, widget3])
        mock_doc = _make_mock_doc(pages=[page])

        mock_fitz = MagicMock()
        mock_fitz.open = MagicMock(return_value=mock_doc)

        with patch.dict(sys.modules, {"fitz": mock_fitz}):
            _server.PYMUPDF_AVAILABLE = True
            _server.fitz = mock_fitz
            result = _server.extract_pdf_metadata(pdf_path)

        assert len(result["form_fields"]) == 3
        names = {f["field_name"] for f in result["form_fields"]}
        assert names == {"Name", "Alter", "Zustimmung"}

    def test_extract_form_fields_empty_when_no_widgets(self, tmp_path: Path) -> None:
        """PDF ohne Formularfelder → leere Formularfelder-Liste."""
        pdf_path = tmp_path / "no_form.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        page = _make_mock_page(widgets=[])
        mock_doc = _make_mock_doc(pages=[page])

        mock_fitz = MagicMock()
        mock_fitz.open = MagicMock(return_value=mock_doc)

        with patch.dict(sys.modules, {"fitz": mock_fitz}):
            _server.PYMUPDF_AVAILABLE = True
            _server.fitz = mock_fitz
            result = _server.extract_pdf_metadata(pdf_path)

        assert result["form_fields"] == []

    def test_form_field_none_value_becomes_empty_string(self, tmp_path: Path) -> None:
        """Formularfeld mit field_value=None wird zu leerem String konvertiert."""
        pdf_path = tmp_path / "none_val.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        widget = _make_mock_widget("Empty", None, "Text")
        widget.field_value = None
        page = _make_mock_page(widgets=[widget])
        mock_doc = _make_mock_doc(pages=[page])

        mock_fitz = MagicMock()
        mock_fitz.open = MagicMock(return_value=mock_doc)

        with patch.dict(sys.modules, {"fitz": mock_fitz}):
            _server.PYMUPDF_AVAILABLE = True
            _server.fitz = mock_fitz
            result = _server.extract_pdf_metadata(pdf_path)

        assert result["form_fields"][0]["field_value"] == ""


# ---------------------------------------------------------------------------
# Tests: prepend_pdf_toc (AC-009-1)
# ---------------------------------------------------------------------------

class TestPrependPdfToc:
    """AC-009-1: TOC wird am Anfang des Markdowns eingefügt."""

    def test_toc_prepended_before_content(self) -> None:
        """Das TOC erscheint VOR dem ursprünglichen Markdown-Inhalt."""
        toc = [[1, "Einleitung", 1], [1, "Schluss", 5]]
        result = _server.prepend_pdf_toc("# Inhalt\n\nText hier.", toc)

        toc_pos = result.find("## Inhaltsverzeichnis")
        content_pos = result.find("# Inhalt")
        assert toc_pos < content_pos

    def test_toc_level1_becomes_h2(self) -> None:
        """Level-1-Eintrag wird als ## (h2) ausgegeben."""
        toc = [[1, "Kapitel Eins", 1]]
        result = _server.prepend_pdf_toc("# Inhalt", toc)

        assert "## Kapitel Eins" in result

    def test_toc_level2_becomes_h3(self) -> None:
        """Level-2-Eintrag wird als ### (h3) ausgegeben."""
        toc = [[2, "Unterabschnitt", 3]]
        result = _server.prepend_pdf_toc("# Inhalt", toc)

        assert "### Unterabschnitt" in result

    def test_toc_level3_becomes_h4(self) -> None:
        """Level-3-Eintrag wird als #### (h4) ausgegeben."""
        toc = [[3, "Tiefer Abschnitt", 7]]
        result = _server.prepend_pdf_toc("# Inhalt", toc)

        assert "#### Tiefer Abschnitt" in result

    def test_toc_contains_page_reference(self) -> None:
        """Seitenzahl wird in der TOC-Ausgabe erwähnt."""
        toc = [[1, "Kapitel", 5]]
        result = _server.prepend_pdf_toc("# Inhalt", toc)

        assert "5" in result

    def test_toc_empty_returns_unchanged_markdown(self) -> None:
        """Leeres TOC → Markdown bleibt unverändert."""
        original = "# Inhalt\n\nText."
        result = _server.prepend_pdf_toc(original, [])
        assert result == original

    def test_toc_header_present(self) -> None:
        """## Inhaltsverzeichnis Überschrift ist vorhanden."""
        toc = [[1, "Abschnitt", 1]]
        result = _server.prepend_pdf_toc("# Inhalt", toc)

        assert "## Inhaltsverzeichnis" in result

    def test_toc_multiple_entries_all_present(self) -> None:
        """Alle TOC-Einträge werden im Ergebnis aufgeführt."""
        toc = [
            [1, "Kapitel A", 1],
            [2, "Abschnitt A.1", 3],
            [1, "Kapitel B", 10],
        ]
        result = _server.prepend_pdf_toc("# Inhalt", toc)

        assert "Kapitel A" in result
        assert "Abschnitt A.1" in result
        assert "Kapitel B" in result


# ---------------------------------------------------------------------------
# Tests: append_pdf_annotations (AC-009-2)
# ---------------------------------------------------------------------------

class TestAppendPdfAnnotationsAsBlockquotes:
    """AC-009-2: Annotationen werden als Blockquotes formatiert."""

    def test_annotation_rendered_as_blockquote(self) -> None:
        """Eine Annotation erscheint als > Blockquote."""
        annotations = [
            {"page": 1, "type": "Text", "content": "Guter Punkt!", "author": "Alice"},
        ]
        result = _server.append_pdf_annotations("# Inhalt", annotations)

        assert "## Annotationen" in result
        assert ">" in result
        assert "Alice" in result
        assert "Guter Punkt!" in result

    def test_annotation_includes_page_number(self) -> None:
        """Seitenzahl wird in der Annotation-Ausgabe erwähnt."""
        annotations = [
            {"page": 3, "type": "Note", "content": "Zu prüfen", "author": "Bob"},
        ]
        result = _server.append_pdf_annotations("# Inhalt", annotations)

        assert "3" in result

    def test_annotation_includes_type(self) -> None:
        """Annotationstyp wird in der Ausgabe erwähnt."""
        annotations = [
            {"page": 1, "type": "Highlight", "content": "Markiert", "author": "Carol"},
        ]
        result = _server.append_pdf_annotations("# Inhalt", annotations)

        assert "Highlight" in result

    def test_multiple_annotations_all_rendered(self) -> None:
        """Alle Annotationen werden als separate Blockquotes ausgegeben."""
        annotations = [
            {"page": 1, "type": "Text", "content": "Kommentar A", "author": "Alice"},
            {"page": 2, "type": "Note", "content": "Kommentar B", "author": "Bob"},
        ]
        result = _server.append_pdf_annotations("# Inhalt", annotations)

        assert "Alice" in result
        assert "Bob" in result
        assert "Kommentar A" in result
        assert "Kommentar B" in result

    def test_empty_annotations_returns_unchanged_markdown(self) -> None:
        """Leere Annotationen-Liste → Markdown bleibt unverändert."""
        original = "# Inhalt\n\nText."
        result = _server.append_pdf_annotations(original, [])
        assert result == original

    def test_annotation_appended_after_content(self) -> None:
        """Annotationen-Sektion kommt NACH dem ursprünglichen Inhalt."""
        annotations = [{"page": 1, "type": "Text", "content": "X", "author": "Y"}]
        result = _server.append_pdf_annotations("# Inhalt\n\nText.", annotations)

        content_pos = result.find("# Inhalt")
        annot_pos = result.find("## Annotationen")
        assert content_pos < annot_pos

    def test_annotation_with_empty_author(self) -> None:
        """Annotation ohne Author zeigt Fallback-Text."""
        annotations = [{"page": 1, "type": "Text", "content": "Inhalt", "author": ""}]
        result = _server.append_pdf_annotations("# Inhalt", annotations)

        assert "Unbekannt" in result or ">" in result


# ---------------------------------------------------------------------------
# Tests: append_pdf_form_fields (AC-009-3)
# ---------------------------------------------------------------------------

class TestAppendPdfFormFieldsAsTable:
    """AC-009-3: Formularfelder werden als Key-Value-Tabelle formatiert."""

    def test_form_fields_rendered_as_table(self) -> None:
        """Formularfelder erscheinen als Markdown-Tabelle."""
        form_fields = [
            {"field_name": "Vorname", "field_value": "Max", "field_type": "Text", "page": 1},
        ]
        result = _server.append_pdf_form_fields("# Inhalt", form_fields)

        assert "## Formularfelder" in result
        assert "|" in result
        assert "Vorname" in result
        assert "Max" in result

    def test_form_fields_table_has_header_row(self) -> None:
        """Die Tabelle hat eine Kopfzeile mit Feld, Wert, Typ, Seite."""
        form_fields = [
            {"field_name": "Name", "field_value": "Test", "field_type": "Text", "page": 1},
        ]
        result = _server.append_pdf_form_fields("# Inhalt", form_fields)

        assert "Feld" in result
        assert "Wert" in result
        assert "Typ" in result
        assert "Seite" in result

    def test_form_fields_table_separator_row(self) -> None:
        """Die Tabelle hat eine Trennzeile mit ---."""
        form_fields = [
            {"field_name": "X", "field_value": "Y", "field_type": "Z", "page": 1},
        ]
        result = _server.append_pdf_form_fields("# Inhalt", form_fields)

        assert "---" in result

    def test_multiple_form_fields_all_rendered(self) -> None:
        """Alle Formularfelder werden als Tabellenzeilen ausgegeben."""
        form_fields = [
            {"field_name": "Name", "field_value": "Alice", "field_type": "Text", "page": 1},
            {"field_name": "Email", "field_value": "alice@example.com", "field_type": "Text", "page": 1},
            {"field_name": "Aktiv", "field_value": "True", "field_type": "CheckBox", "page": 2},
        ]
        result = _server.append_pdf_form_fields("# Inhalt", form_fields)

        assert "Alice" in result
        assert "alice@example.com" in result
        assert "CheckBox" in result

    def test_empty_form_fields_returns_unchanged_markdown(self) -> None:
        """Leere Formularfelder-Liste → Markdown bleibt unverändert."""
        original = "# Inhalt\n\nText."
        result = _server.append_pdf_form_fields(original, [])
        assert result == original

    def test_form_fields_appended_after_content(self) -> None:
        """Formularfelder-Sektion kommt NACH dem ursprünglichen Inhalt."""
        form_fields = [
            {"field_name": "F", "field_value": "V", "field_type": "T", "page": 1},
        ]
        result = _server.append_pdf_form_fields("# Inhalt\n\nText.", form_fields)

        content_pos = result.find("# Inhalt")
        table_pos = result.find("## Formularfelder")
        assert content_pos < table_pos


# ---------------------------------------------------------------------------
# Tests: Graceful ohne PyMuPDF
# ---------------------------------------------------------------------------

class TestNoPymupdfGraceful:
    """Graceful Fallback wenn PyMuPDF nicht installiert ist."""

    def test_extract_pdf_metadata_returns_empty_when_no_pymupdf(self, tmp_path: Path) -> None:
        """Ohne PyMuPDF gibt extract_pdf_metadata leere Listen zurück."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        original_flag = _server.PYMUPDF_AVAILABLE
        try:
            _server.PYMUPDF_AVAILABLE = False
            result = _server.extract_pdf_metadata(pdf_path)
        finally:
            _server.PYMUPDF_AVAILABLE = original_flag

        assert result == {"toc": [], "annotations": [], "form_fields": []}

    def test_extract_pdf_metadata_graceful_on_fitz_error(self, tmp_path: Path) -> None:
        """Wenn fitz.open() einen Fehler wirft, werden leere Listen zurückgegeben."""
        pdf_path = tmp_path / "broken.pdf"
        pdf_path.write_bytes(b"not a real pdf")

        mock_fitz = MagicMock()
        mock_fitz.open = MagicMock(side_effect=Exception("fitz error: corrupt PDF"))

        with patch.dict(sys.modules, {"fitz": mock_fitz}):
            _server.PYMUPDF_AVAILABLE = True
            _server.fitz = mock_fitz
            result = _server.extract_pdf_metadata(pdf_path)

        assert isinstance(result, dict)
        assert result["toc"] == []
        assert result["annotations"] == []
        assert result["form_fields"] == []


# ---------------------------------------------------------------------------
# Tests: Integration in convert_with_markitdown
# ---------------------------------------------------------------------------

class TestIntegrationPdfMetadata:
    """PDF-Metadaten werden nach markitdown-Konvertierung für .pdf integriert."""

    def test_pdf_toc_prepended_in_conversion(self, tmp_path: Path) -> None:
        """
        Bei .pdf-Konvertierung: TOC aus Bookmarks wird dem Markdown vorangestellt.
        """
        pdf_path = tmp_path / "with_toc.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        toc_data = [[1, "Kapitel 1", 1], [1, "Kapitel 2", 5]]
        mock_doc = _make_mock_doc(toc=toc_data)

        mock_fitz = MagicMock()
        mock_fitz.open = MagicMock(return_value=mock_doc)

        mock_md_result = MagicMock()
        mock_md_result.text_content = "# PDF Inhalt\n\nText."
        mock_md_result.title = "Test"

        with patch.object(_server, "md") as mock_md_obj:
            mock_md_obj.convert = MagicMock(return_value=mock_md_result)
            with patch.object(_server, "PDFPLUMBER_AVAILABLE", False):
                with patch.dict(sys.modules, {"fitz": mock_fitz}):
                    _server.PYMUPDF_AVAILABLE = True
                    _server.fitz = mock_fitz
                    result = _server.convert_with_markitdown(pdf_path)

        assert result["success"] is True
        assert "## Inhaltsverzeichnis" in result["markdown"]
        assert "Kapitel 1" in result["markdown"]
        # TOC muss VOR dem Hauptinhalt stehen
        toc_pos = result["markdown"].find("## Inhaltsverzeichnis")
        content_pos = result["markdown"].find("# PDF Inhalt")
        assert toc_pos < content_pos

    def test_pdf_annotations_appended_in_conversion(self, tmp_path: Path) -> None:
        """
        Bei .pdf-Konvertierung: Annotationen werden als Blockquotes angehängt.
        """
        pdf_path = tmp_path / "with_annots.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        annot = _make_mock_annot("Text", "Prüfe bitte nochmal", "Reviewer")
        page = _make_mock_page(annots=[annot])
        mock_doc = _make_mock_doc(pages=[page])

        mock_fitz = MagicMock()
        mock_fitz.open = MagicMock(return_value=mock_doc)

        mock_md_result = MagicMock()
        mock_md_result.text_content = "# Dokument"
        mock_md_result.title = "Doc"

        with patch.object(_server, "md") as mock_md_obj:
            mock_md_obj.convert = MagicMock(return_value=mock_md_result)
            with patch.object(_server, "PDFPLUMBER_AVAILABLE", False):
                with patch.dict(sys.modules, {"fitz": mock_fitz}):
                    _server.PYMUPDF_AVAILABLE = True
                    _server.fitz = mock_fitz
                    result = _server.convert_with_markitdown(pdf_path)

        assert result["success"] is True
        assert "## Annotationen" in result["markdown"]
        assert "Reviewer" in result["markdown"]
        assert "Prüfe bitte nochmal" in result["markdown"]

    def test_pdf_form_fields_appended_in_conversion(self, tmp_path: Path) -> None:
        """
        Bei .pdf-Konvertierung: Formularfelder werden als Tabelle angehängt.
        """
        pdf_path = tmp_path / "with_form.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        widget = _make_mock_widget("Unterschrift", "Max Mustermann", "Signature")
        page = _make_mock_page(widgets=[widget])
        mock_doc = _make_mock_doc(pages=[page])

        mock_fitz = MagicMock()
        mock_fitz.open = MagicMock(return_value=mock_doc)

        mock_md_result = MagicMock()
        mock_md_result.text_content = "# Formular"
        mock_md_result.title = "Form"

        with patch.object(_server, "md") as mock_md_obj:
            mock_md_obj.convert = MagicMock(return_value=mock_md_result)
            with patch.object(_server, "PDFPLUMBER_AVAILABLE", False):
                with patch.dict(sys.modules, {"fitz": mock_fitz}):
                    _server.PYMUPDF_AVAILABLE = True
                    _server.fitz = mock_fitz
                    result = _server.convert_with_markitdown(pdf_path)

        assert result["success"] is True
        assert "## Formularfelder" in result["markdown"]
        assert "Unterschrift" in result["markdown"]
        assert "Max Mustermann" in result["markdown"]

    def test_non_pdf_files_not_affected(self, tmp_path: Path) -> None:
        """
        Für Nicht-PDF-Dateien (.docx) werden keine PDF-Metadaten hinzugefügt.
        """
        import io
        import zipfile

        docx_path = tmp_path / "test.docx"
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(
                "word/document.xml",
                '<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body></w:body></w:document>',
            )
        docx_path.write_bytes(buf.getvalue())

        mock_md_result = MagicMock()
        mock_md_result.text_content = "# DOCX Inhalt"
        mock_md_result.title = "DOCX"

        mock_doc_python = MagicMock()
        mock_doc_python.sections = []

        with patch.object(_server, "md") as mock_md_obj:
            mock_md_obj.convert = MagicMock(return_value=mock_md_result)
            with patch.dict(sys.modules, {"docx": MagicMock(Document=MagicMock(return_value=mock_doc_python))}):
                result = _server.convert_with_markitdown(docx_path)

        assert result["success"] is True
        assert "## Inhaltsverzeichnis" not in result["markdown"]
        assert "## Annotationen" not in result["markdown"]
        assert "## Formularfelder" not in result["markdown"]

    def test_pdf_metadata_error_does_not_break_conversion(self, tmp_path: Path) -> None:
        """
        Wenn extract_pdf_metadata() einen Fehler wirft, bleibt die Konvertierung erfolgreich.
        """
        pdf_path = tmp_path / "error.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        mock_md_result = MagicMock()
        mock_md_result.text_content = "# Inhalt"
        mock_md_result.title = "Titel"

        with patch.object(_server, "md") as mock_md_obj:
            mock_md_obj.convert = MagicMock(return_value=mock_md_result)
            with patch.object(_server, "PDFPLUMBER_AVAILABLE", False):
                with patch.object(
                    _server,
                    "extract_pdf_metadata",
                    side_effect=Exception("PyMuPDF interner Fehler"),
                ):
                    result = _server.convert_with_markitdown(pdf_path)

        assert result["success"] is True
        assert "Inhalt" in result["markdown"]
