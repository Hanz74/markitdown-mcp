"""
Tests für T-MKIT-008: Erweiterte DOCX-Extraktion (Kommentare, Header/Footer, Track Changes).

Alle Tests laufen ohne Docker-Container und ohne echte API-Calls.
Alle externen Abhängigkeiten werden per unittest.mock gemockt.
"""

import io
import sys
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from conftest import load_server_module, run_async


# ---------------------------------------------------------------------------
# Hilfsfunktionen: Synthetische DOCX-ZIPs
# ---------------------------------------------------------------------------

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W = "w"


def _make_comments_xml(comments: list[dict]) -> bytes:
    """
    Erstellt word/comments.xml Bytes mit den angegebenen Kommentaren.

    Args:
        comments: Liste von Dicts mit 'author', 'date', 'text'.

    Returns:
        UTF-8 kodierte XML-Bytes.
    """
    parts = [f'<?xml version="1.0" encoding="UTF-8"?>']
    parts.append(f'<w:comments xmlns:w="{W_NS}">')
    for i, c in enumerate(comments):
        author = c.get("author", "Autor")
        date = c.get("date", "2024-01-01T00:00:00Z")
        text = c.get("text", "")
        parts.append(
            f'  <w:comment w:id="{i}" w:author="{author}" w:date="{date}">'
            f'    <w:p><w:r><w:t>{text}</w:t></w:r></w:p>'
            f'  </w:comment>'
        )
    parts.append("</w:comments>")
    return "\n".join(parts).encode("utf-8")


def _make_document_xml_with_track_changes(insertions: list[dict], deletions: list[dict]) -> bytes:
    """
    Erstellt word/document.xml Bytes mit Track-Changes-Elementen.

    Args:
        insertions: Liste von Dicts mit 'author', 'date', 'text'.
        deletions: Liste von Dicts mit 'author', 'date', 'text'.

    Returns:
        UTF-8 kodierte XML-Bytes.
    """
    parts = [f'<?xml version="1.0" encoding="UTF-8"?>']
    parts.append(f'<w:document xmlns:w="{W_NS}"><w:body><w:p>')

    for ins in insertions:
        author = ins.get("author", "Autor")
        date = ins.get("date", "2024-01-01T00:00:00Z")
        text = ins.get("text", "")
        parts.append(
            f'  <w:ins w:id="1" w:author="{author}" w:date="{date}">'
            f'    <w:r><w:t>{text}</w:t></w:r>'
            f'  </w:ins>'
        )

    for d in deletions:
        author = d.get("author", "Autor")
        date = d.get("date", "2024-01-01T00:00:00Z")
        text = d.get("text", "")
        parts.append(
            f'  <w:del w:id="2" w:author="{author}" w:date="{date}">'
            f'    <w:r><w:delText>{text}</w:delText></w:r>'
            f'  </w:del>'
        )

    parts.append("</w:p></w:body></w:document>")
    return "\n".join(parts).encode("utf-8")


def _make_docx_zip_with_comments(comments: list[dict]) -> bytes:
    """Erstellt ein minimales DOCX-ZIP mit comments.xml."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/comments.xml", _make_comments_xml(comments))
        zf.writestr("word/document.xml", _make_document_xml_with_track_changes([], []))
    return buf.getvalue()


def _make_docx_zip_with_track_changes(
    insertions: list[dict],
    deletions: list[dict],
) -> bytes:
    """Erstellt ein minimales DOCX-ZIP mit Track Changes im document.xml."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", _make_document_xml_with_track_changes(insertions, deletions))
    return buf.getvalue()


def _make_empty_docx_zip() -> bytes:
    """Erstellt ein DOCX-ZIP ohne Kommentare und ohne Track Changes."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", _make_document_xml_with_track_changes([], []))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Modul laden (einmal für alle Tests in dieser Datei)
# ---------------------------------------------------------------------------

_server = load_server_module(use_real_pil=False)


# ---------------------------------------------------------------------------
# Tests: extract_docx_extras — Kommentare
# ---------------------------------------------------------------------------

class TestExtractComments:
    """AC-008-1: Kommentare werden extrahiert."""

    def test_extract_comments_basic(self, tmp_path: Path) -> None:
        """Kommentare mit Author, Date und Text werden korrekt extrahiert."""
        comments_data = [
            {"author": "Max Mustermann", "date": "2024-03-01T10:00:00Z", "text": "Sehr guter Punkt!"},
        ]
        docx_bytes = _make_docx_zip_with_comments(comments_data)
        docx_path = tmp_path / "test.docx"
        docx_path.write_bytes(docx_bytes)

        # python-docx mocken (für Header/Footer-Extraktion)
        mock_doc = MagicMock()
        mock_doc.sections = []
        with patch.dict(sys.modules, {"docx": MagicMock(Document=MagicMock(return_value=mock_doc))}):
            extras = _server.extract_docx_extras(docx_path)

        assert len(extras["comments"]) == 1
        comment = extras["comments"][0]
        assert comment["author"] == "Max Mustermann"
        assert comment["date"] == "2024-03-01T10:00:00Z"
        assert comment["text"] == "Sehr guter Punkt!"

    def test_extract_comments_multiple(self, tmp_path: Path) -> None:
        """Mehrere Kommentare werden alle extrahiert."""
        comments_data = [
            {"author": "Alice", "date": "2024-03-01T10:00:00Z", "text": "Kommentar 1"},
            {"author": "Bob", "date": "2024-03-02T11:00:00Z", "text": "Kommentar 2"},
        ]
        docx_bytes = _make_docx_zip_with_comments(comments_data)
        docx_path = tmp_path / "multi_comments.docx"
        docx_path.write_bytes(docx_bytes)

        mock_doc = MagicMock()
        mock_doc.sections = []
        with patch.dict(sys.modules, {"docx": MagicMock(Document=MagicMock(return_value=mock_doc))}):
            extras = _server.extract_docx_extras(docx_path)

        assert len(extras["comments"]) == 2
        authors = {c["author"] for c in extras["comments"]}
        assert authors == {"Alice", "Bob"}

    def test_extract_comments_empty_when_no_comments_xml(self, tmp_path: Path) -> None:
        """Kein comments.xml → leere Kommentar-Liste (kein Crash)."""
        docx_bytes = _make_empty_docx_zip()
        docx_path = tmp_path / "no_comments.docx"
        docx_path.write_bytes(docx_bytes)

        mock_doc = MagicMock()
        mock_doc.sections = []
        with patch.dict(sys.modules, {"docx": MagicMock(Document=MagicMock(return_value=mock_doc))}):
            extras = _server.extract_docx_extras(docx_path)

        assert extras["comments"] == []

    def test_extract_comments_graceful_on_broken_zip(self, tmp_path: Path) -> None:
        """Ungültiges ZIP → leeres Ergebnis ohne Exception."""
        bad_path = tmp_path / "broken.docx"
        bad_path.write_bytes(b"this is not a zip file")

        mock_doc = MagicMock()
        mock_doc.sections = []
        with patch.dict(sys.modules, {"docx": MagicMock(Document=MagicMock(return_value=mock_doc))}):
            extras = _server.extract_docx_extras(bad_path)

        assert isinstance(extras, dict)
        assert extras["comments"] == []
        assert extras["track_changes"] == []


# ---------------------------------------------------------------------------
# Tests: extract_docx_extras — Header/Footer
# ---------------------------------------------------------------------------

class TestExtractHeadersFooters:
    """AC-008-2: Header/Footer werden extrahiert."""

    def test_extract_header(self, tmp_path: Path) -> None:
        """Header-Text wird aus Sektionen extrahiert."""
        docx_bytes = _make_empty_docx_zip()
        docx_path = tmp_path / "header_test.docx"
        docx_path.write_bytes(docx_bytes)

        # Mock-Section mit Header
        mock_header = MagicMock()
        mock_header.text = "Mein Dokumenten-Header"
        mock_footer = MagicMock()
        mock_footer.text = ""
        mock_footer.paragraphs = []
        mock_section = MagicMock()
        mock_section.header = mock_header
        mock_section.footer = mock_footer

        mock_doc = MagicMock()
        mock_doc.sections = [mock_section]
        with patch.dict(sys.modules, {"docx": MagicMock(Document=MagicMock(return_value=mock_doc))}):
            extras = _server.extract_docx_extras(docx_path)

        assert "Mein Dokumenten-Header" in extras["headers"]
        assert extras["footers"] == []

    def test_extract_footer(self, tmp_path: Path) -> None:
        """Footer-Text wird aus Sektionen extrahiert."""
        docx_bytes = _make_empty_docx_zip()
        docx_path = tmp_path / "footer_test.docx"
        docx_path.write_bytes(docx_bytes)

        mock_header = MagicMock()
        mock_header.text = ""
        mock_header.paragraphs = []
        mock_footer = MagicMock()
        mock_footer.text = "Seite 1 von 10"
        mock_section = MagicMock()
        mock_section.header = mock_header
        mock_section.footer = mock_footer

        mock_doc = MagicMock()
        mock_doc.sections = [mock_section]
        with patch.dict(sys.modules, {"docx": MagicMock(Document=MagicMock(return_value=mock_doc))}):
            extras = _server.extract_docx_extras(docx_path)

        assert "Seite 1 von 10" in extras["footers"]
        assert extras["headers"] == []

    def test_extract_headers_no_duplicates(self, tmp_path: Path) -> None:
        """Gleicher Header in mehreren Sektionen wird nur einmal gespeichert."""
        docx_bytes = _make_empty_docx_zip()
        docx_path = tmp_path / "dup_header.docx"
        docx_path.write_bytes(docx_bytes)

        def make_section(header_text: str):
            mock_header = MagicMock()
            mock_header.text = header_text
            mock_footer = MagicMock()
            mock_footer.text = ""
            mock_footer.paragraphs = []
            sec = MagicMock()
            sec.header = mock_header
            sec.footer = mock_footer
            return sec

        mock_doc = MagicMock()
        mock_doc.sections = [make_section("Header Text"), make_section("Header Text")]
        with patch.dict(sys.modules, {"docx": MagicMock(Document=MagicMock(return_value=mock_doc))}):
            extras = _server.extract_docx_extras(docx_path)

        assert extras["headers"].count("Header Text") == 1

    def test_extract_headers_footers_graceful_on_docx_error(self, tmp_path: Path) -> None:
        """Fehler bei python-docx → headers/footers leer, kein Crash."""
        docx_bytes = _make_empty_docx_zip()
        docx_path = tmp_path / "error.docx"
        docx_path.write_bytes(docx_bytes)

        mock_docx_module = MagicMock()
        mock_docx_module.Document = MagicMock(side_effect=Exception("python-docx error"))
        with patch.dict(sys.modules, {"docx": mock_docx_module}):
            extras = _server.extract_docx_extras(docx_path)

        assert extras["headers"] == []
        assert extras["footers"] == []


# ---------------------------------------------------------------------------
# Tests: extract_docx_extras — Track Changes
# ---------------------------------------------------------------------------

class TestExtractTrackChanges:
    """AC-008-3: Track Changes (w:ins, w:del) werden extrahiert."""

    def test_extract_insertion(self, tmp_path: Path) -> None:
        """Einfügungen (w:ins) werden erkannt."""
        insertions = [{"author": "Alice", "date": "2024-03-01T10:00:00Z", "text": "neuer Text"}]
        docx_bytes = _make_docx_zip_with_track_changes(insertions, [])
        docx_path = tmp_path / "ins.docx"
        docx_path.write_bytes(docx_bytes)

        mock_doc = MagicMock()
        mock_doc.sections = []
        with patch.dict(sys.modules, {"docx": MagicMock(Document=MagicMock(return_value=mock_doc))}):
            extras = _server.extract_docx_extras(docx_path)

        tc = extras["track_changes"]
        assert len(tc) == 1
        assert tc[0]["type"] == "insertion"
        assert tc[0]["author"] == "Alice"
        assert tc[0]["text"] == "neuer Text"

    def test_extract_deletion(self, tmp_path: Path) -> None:
        """Löschungen (w:del) werden erkannt."""
        deletions = [{"author": "Bob", "date": "2024-03-02T11:00:00Z", "text": "alter Text"}]
        docx_bytes = _make_docx_zip_with_track_changes([], deletions)
        docx_path = tmp_path / "del.docx"
        docx_path.write_bytes(docx_bytes)

        mock_doc = MagicMock()
        mock_doc.sections = []
        with patch.dict(sys.modules, {"docx": MagicMock(Document=MagicMock(return_value=mock_doc))}):
            extras = _server.extract_docx_extras(docx_path)

        tc = extras["track_changes"]
        assert len(tc) == 1
        assert tc[0]["type"] == "deletion"
        assert tc[0]["author"] == "Bob"
        assert tc[0]["text"] == "alter Text"

    def test_extract_mixed_track_changes(self, tmp_path: Path) -> None:
        """Einfügungen und Löschungen werden gemeinsam extrahiert."""
        insertions = [{"author": "Alice", "date": "2024-03-01T10:00:00Z", "text": "hinzugefügt"}]
        deletions = [{"author": "Bob", "date": "2024-03-02T11:00:00Z", "text": "gelöscht"}]
        docx_bytes = _make_docx_zip_with_track_changes(insertions, deletions)
        docx_path = tmp_path / "mixed.docx"
        docx_path.write_bytes(docx_bytes)

        mock_doc = MagicMock()
        mock_doc.sections = []
        with patch.dict(sys.modules, {"docx": MagicMock(Document=MagicMock(return_value=mock_doc))}):
            extras = _server.extract_docx_extras(docx_path)

        tc = extras["track_changes"]
        assert len(tc) == 2
        types = {c["type"] for c in tc}
        assert types == {"insertion", "deletion"}

    def test_extract_track_changes_empty_when_no_changes(self, tmp_path: Path) -> None:
        """Keine Track Changes → leere Liste."""
        docx_bytes = _make_empty_docx_zip()
        docx_path = tmp_path / "no_changes.docx"
        docx_path.write_bytes(docx_bytes)

        mock_doc = MagicMock()
        mock_doc.sections = []
        with patch.dict(sys.modules, {"docx": MagicMock(Document=MagicMock(return_value=mock_doc))}):
            extras = _server.extract_docx_extras(docx_path)

        assert extras["track_changes"] == []


# ---------------------------------------------------------------------------
# Tests: append_docx_extras_to_markdown
# ---------------------------------------------------------------------------

class TestAppendCommentsAsBlockquotes:
    """AC-008-1: Kommentare werden als Blockquotes formatiert."""

    def test_comments_rendered_as_blockquotes(self) -> None:
        """Kommentare erscheinen als > Blockquotes mit Author und Date."""
        extras = {
            "comments": [
                {"author": "Max", "date": "2024-03-01T10:00:00Z", "text": "Sehr gut!"},
            ],
            "headers": [],
            "footers": [],
            "track_changes": [],
        }
        result = _server.append_docx_extras_to_markdown("# Dokument", extras)

        assert "## Kommentare" in result
        assert "> **Max**" in result
        assert "Sehr gut!" in result
        assert "2024-03-01T10:00:00Z" in result

    def test_multiple_comments_all_rendered(self) -> None:
        """Alle Kommentare werden als separate Blockquotes ausgegeben."""
        extras = {
            "comments": [
                {"author": "Alice", "date": "2024-01-01T00:00:00Z", "text": "Kommentar A"},
                {"author": "Bob", "date": "2024-01-02T00:00:00Z", "text": "Kommentar B"},
            ],
            "headers": [],
            "footers": [],
            "track_changes": [],
        }
        result = _server.append_docx_extras_to_markdown("# Test", extras)

        assert "Alice" in result
        assert "Bob" in result
        assert "Kommentar A" in result
        assert "Kommentar B" in result


class TestAppendHeadersFooters:
    """AC-008-2: Header/Footer werden als eigene Sektion angehängt."""

    def test_header_section_appended(self) -> None:
        """Header-Inhalt erscheint in ## Header und Footer Sektion."""
        extras = {
            "comments": [],
            "headers": ["Firmenname GmbH"],
            "footers": [],
            "track_changes": [],
        }
        result = _server.append_docx_extras_to_markdown("# Inhalt", extras)

        assert "## Header und Footer" in result
        assert "Firmenname GmbH" in result
        assert "**Header:**" in result

    def test_footer_section_appended(self) -> None:
        """Footer-Inhalt erscheint in ## Header und Footer Sektion."""
        extras = {
            "comments": [],
            "headers": [],
            "footers": ["Seite 1 von 5"],
            "track_changes": [],
        }
        result = _server.append_docx_extras_to_markdown("# Inhalt", extras)

        assert "## Header und Footer" in result
        assert "Seite 1 von 5" in result
        assert "**Footer:**" in result

    def test_both_header_and_footer(self) -> None:
        """Beide Header und Footer werden ausgegeben."""
        extras = {
            "comments": [],
            "headers": ["Oben"],
            "footers": ["Unten"],
            "track_changes": [],
        }
        result = _server.append_docx_extras_to_markdown("# Text", extras)

        assert "**Header:**" in result
        assert "**Footer:**" in result
        assert "Oben" in result
        assert "Unten" in result


class TestAppendTrackChanges:
    """AC-008-3: Track Changes werden als Diff-Notation ausgegeben."""

    def test_insertion_rendered_as_diff_plus(self) -> None:
        """Einfügungen werden mit + Prefix im Diff-Block ausgegeben."""
        extras = {
            "comments": [],
            "headers": [],
            "footers": [],
            "track_changes": [
                {"type": "insertion", "author": "Alice", "date": "2024-01-01T00:00:00Z", "text": "neuer Text"},
            ],
        }
        result = _server.append_docx_extras_to_markdown("# Dok", extras)

        assert "## Änderungsverfolgung" in result
        assert "```diff" in result
        assert "+ neuer Text" in result

    def test_deletion_rendered_as_diff_minus(self) -> None:
        """Löschungen werden mit - Prefix im Diff-Block ausgegeben."""
        extras = {
            "comments": [],
            "headers": [],
            "footers": [],
            "track_changes": [
                {"type": "deletion", "author": "Bob", "date": "2024-01-01T00:00:00Z", "text": "alter Text"},
            ],
        }
        result = _server.append_docx_extras_to_markdown("# Dok", extras)

        assert "## Änderungsverfolgung" in result
        assert "```diff" in result
        assert "- alter Text" in result

    def test_diff_block_closed(self) -> None:
        """Der Diff-Code-Block wird korrekt geschlossen."""
        extras = {
            "comments": [],
            "headers": [],
            "footers": [],
            "track_changes": [
                {"type": "insertion", "author": "X", "date": "", "text": "abc"},
            ],
        }
        result = _server.append_docx_extras_to_markdown("# Dok", extras)

        # Prüfe dass ``` sowohl zum Öffnen als auch Schließen vorhanden ist
        fence_count = result.count("```")
        assert fence_count >= 2


class TestAppendEmptyExtras:
    """Kein Anhang wenn alle Extras leer sind."""

    def test_no_section_appended_when_all_empty(self) -> None:
        """Bei leeren Extras bleibt Markdown unverändert."""
        original = "# Mein Dokument\n\nInhalt hier."
        extras = {
            "comments": [],
            "headers": [],
            "footers": [],
            "track_changes": [],
        }
        result = _server.append_docx_extras_to_markdown(original, extras)

        assert result == original

    def test_no_section_appended_when_extras_missing_keys(self) -> None:
        """Bei leerem Dict (fehlende Keys) → keine Sektionen, kein Crash."""
        original = "# Test"
        result = _server.append_docx_extras_to_markdown(original, {})
        assert result == original


# ---------------------------------------------------------------------------
# Tests: Integration in convert_with_markitdown
# ---------------------------------------------------------------------------

class TestIntegrationDocxExtras:
    """Extras werden nach markitdown-Konvertierung für .docx angehängt."""

    def test_docx_extras_appended_after_conversion(self, tmp_path: Path) -> None:
        """
        Bei .docx-Konvertierung: extract_docx_extras + append_docx_extras_to_markdown
        werden aufgerufen und das Ergebnis enthält die Extra-Sektionen.
        """
        docx_path = tmp_path / "test.docx"
        # Minimales echtes DOCX-ZIP mit Kommentar
        comments_data = [{"author": "Reviewer", "date": "2024-01-01T00:00:00Z", "text": "Bitte prüfen"}]
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("word/comments.xml", _make_comments_xml(comments_data))
            zf.writestr("word/document.xml", _make_document_xml_with_track_changes([], []))
        docx_path.write_bytes(buf.getvalue())

        # MarkItDown mocken
        mock_md_result = MagicMock()
        mock_md_result.text_content = "# Dokument\n\nInhalt."
        mock_md_result.title = "Test"

        mock_doc = MagicMock()
        mock_doc.sections = []

        with patch.object(_server, "md") as mock_md_obj:
            mock_md_obj.convert = MagicMock(return_value=mock_md_result)
            with patch.dict(sys.modules, {"docx": MagicMock(Document=MagicMock(return_value=mock_doc))}):
                result = _server.convert_with_markitdown(docx_path)

        assert result["success"] is True
        assert "## Kommentare" in result["markdown"]
        assert "Reviewer" in result["markdown"]
        assert "Bitte prüfen" in result["markdown"]

    def test_non_docx_files_not_affected(self, tmp_path: Path) -> None:
        """
        Für Nicht-DOCX-Dateien (z.B. .pdf) werden keine Extras angehängt.
        """
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 minimal")

        mock_md_result = MagicMock()
        mock_md_result.text_content = "# PDF Inhalt"
        mock_md_result.title = "PDF"

        with patch.object(_server, "md") as mock_md_obj:
            mock_md_obj.convert = MagicMock(return_value=mock_md_result)
            with patch.object(_server, "PDFPLUMBER_AVAILABLE", False):
                result = _server.convert_with_markitdown(pdf_path)

        assert result["success"] is True
        assert "## Kommentare" not in result["markdown"]
        assert "## Änderungsverfolgung" not in result["markdown"]

    def test_extras_error_does_not_break_conversion(self, tmp_path: Path) -> None:
        """
        Wenn extract_docx_extras() einen Fehler wirft, bleibt die Konvertierung erfolgreich.
        """
        docx_path = tmp_path / "test.docx"
        docx_path.write_bytes(_make_empty_docx_zip())

        mock_md_result = MagicMock()
        mock_md_result.text_content = "# Inhalt"
        mock_md_result.title = "Titel"

        with patch.object(_server, "md") as mock_md_obj:
            mock_md_obj.convert = MagicMock(return_value=mock_md_result)
            with patch.object(
                _server,
                "extract_docx_extras",
                side_effect=Exception("Interner Fehler"),
            ):
                result = _server.convert_with_markitdown(docx_path)

        assert result["success"] is True
        assert "Inhalt" in result["markdown"]
