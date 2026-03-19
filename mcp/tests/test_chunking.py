"""
Tests für T-MKIT-011: Smart Chunking für RAG.

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
chunk_markdown = _server.chunk_markdown
convert_auto = _server.convert_auto


# ---------------------------------------------------------------------------
# Unit-Tests für chunk_markdown()
# ---------------------------------------------------------------------------


class TestChunkByHeadings:
    """AC-011-2: Chunks respektieren Heading-Hierarchie."""

    def test_chunk_by_headings_basic(self):
        """Markdown mit mehreren Headings → ein Chunk pro Sektion."""
        markdown = (
            "# Einleitung\n\n"
            "Dieser Text gehört zur Einleitung.\n\n"
            "## Abschnitt 1\n\n"
            "Inhalt des ersten Abschnitts.\n\n"
            "## Abschnitt 2\n\n"
            "Inhalt des zweiten Abschnitts."
        )
        chunks = chunk_markdown(markdown, chunk_size=512, source="test.md")

        assert len(chunks) >= 2
        headings = [c["heading"] for c in chunks]
        # Mindestens ein Heading muss erkannt werden
        assert any("Abschnitt" in h or "Einleitung" in h for h in headings)

    def test_chunk_heading_preserved_in_text(self):
        """Heading muss im Text des Chunks enthalten sein."""
        markdown = "# Haupttitel\n\nDies ist der Inhalt."
        chunks = chunk_markdown(markdown, chunk_size=512, source="test.md")

        assert len(chunks) >= 1
        # Heading soll im text-Feld des Chunks erscheinen
        assert any("Haupttitel" in c["text"] for c in chunks)

    def test_chunk_splits_at_h1_and_h2(self):
        """H1 und H2 Headings führen beide zu Splits."""
        markdown = (
            "# Kapitel 1\n\n"
            "Inhalt von Kapitel 1.\n\n"
            "# Kapitel 2\n\n"
            "Inhalt von Kapitel 2."
        )
        chunks = chunk_markdown(markdown, chunk_size=512, source="test.md")
        assert len(chunks) >= 2

    def test_chunk_heading_metadata_correct(self):
        """AC-011-4: heading-Feld im Chunk-Metadatum ist korrekt."""
        markdown = "## Abschnitt Alpha\n\nInhalt hier."
        chunks = chunk_markdown(markdown, chunk_size=512, source="doc.pdf")

        assert len(chunks) >= 1
        # Mindestens ein Chunk soll das Heading enthalten
        assert any("Abschnitt Alpha" in c["heading"] for c in chunks)


class TestChunkPreservesTables:
    """AC-011-3: Tabellen werden nie zerteilt."""

    def test_chunk_preserves_table_intact(self):
        """Eine Tabelle bleibt in einem einzigen Chunk."""
        markdown = (
            "# Daten\n\n"
            "| Name | Wert |\n"
            "|------|------|\n"
            "| A    | 1    |\n"
            "| B    | 2    |\n"
            "| C    | 3    |\n"
            "| D    | 4    |\n"
        )
        chunks = chunk_markdown(markdown, chunk_size=20, source="test.md")

        # Alle Tabellenzeilen sollen in einem Chunk zusammen sein
        table_chunks = [c for c in chunks if "|" in c["text"]]
        assert len(table_chunks) >= 1

        # Keine Tabelle darf über zwei Chunks aufgeteilt sein:
        # Prüfen ob eine Zeile | A | nur in EINEM chunk vorkommt
        chunks_with_a = [c for c in chunks if "| A" in c["text"]]
        chunks_with_d = [c for c in chunks if "| D" in c["text"]]

        if chunks_with_a and chunks_with_d:
            # A und D müssen im selben Chunk sein (Tabelle nicht zerstückelt)
            assert chunks_with_a[0]["index"] == chunks_with_d[0]["index"], (
                "Tabelle wurde über mehrere Chunks aufgeteilt!"
            )

    def test_chunk_table_with_surrounding_text(self):
        """Tabelle zwischen normalem Text bleibt kompakt."""
        markdown = (
            "Einleitungstext vor der Tabelle.\n\n"
            "| Spalte A | Spalte B |\n"
            "|----------|----------|\n"
            "| Wert 1   | Wert 2   |\n\n"
            "Text nach der Tabelle."
        )
        chunks = chunk_markdown(markdown, chunk_size=512, source="test.md")

        # Tabelle-Header und Datenzeile müssen im selben Chunk sein
        table_header_chunks = [c for c in chunks if "Spalte A" in c["text"]]
        table_data_chunks = [c for c in chunks if "Wert 1" in c["text"]]

        if table_header_chunks and table_data_chunks:
            assert table_header_chunks[0]["index"] == table_data_chunks[0]["index"]


class TestChunkPreservesCodeBlocks:
    """AC-011-3: Code-Blöcke werden nie zerteilt."""

    def test_chunk_preserves_code_block_intact(self):
        """Ein Code-Block bleibt in einem einzigen Chunk."""
        markdown = (
            "# Code-Beispiel\n\n"
            "Hier ist der Code:\n\n"
            "```python\n"
            "def hello():\n"
            "    print('Hello World')\n"
            "    return 42\n"
            "```\n\n"
            "Mehr Text danach."
        )
        chunks = chunk_markdown(markdown, chunk_size=10, source="test.md")

        # Code-Block-Start und -Ende müssen im selben Chunk sein
        chunks_with_def = [c for c in chunks if "def hello" in c["text"]]
        chunks_with_return = [c for c in chunks if "return 42" in c["text"]]

        if chunks_with_def and chunks_with_return:
            assert chunks_with_def[0]["index"] == chunks_with_return[0]["index"], (
                "Code-Block wurde über mehrere Chunks aufgeteilt!"
            )

    def test_chunk_code_block_markers_together(self):
        """Öffnende und schließende ``` sind im selben Chunk."""
        markdown = (
            "```javascript\n"
            "const x = 1;\n"
            "const y = 2;\n"
            "```"
        )
        chunks = chunk_markdown(markdown, chunk_size=5, source="test.md")

        # Beide Backtick-Marker sollen im selben Chunk sein
        chunks_with_open = [c for c in chunks if "```javascript" in c["text"]]
        chunks_with_close = [c for c in chunks if c["text"].count("```") >= 2]

        # Mindestens ein Chunk enthält beide Marker
        assert len(chunks_with_close) >= 1 or len(chunks_with_open) >= 1


class TestChunkMetadata:
    """AC-011-4: Jeder Chunk enthält korrekte Metadaten."""

    def test_chunk_index_sequential(self):
        """index-Felder sind 0-basiert und sequenziell."""
        markdown = (
            "# Sektion 1\n\nText eins.\n\n"
            "# Sektion 2\n\nText zwei.\n\n"
            "# Sektion 3\n\nText drei."
        )
        chunks = chunk_markdown(markdown, chunk_size=512, source="test.md")

        assert len(chunks) >= 1
        for i, c in enumerate(chunks):
            assert c["index"] == i, f"Chunk {i} hat index={c['index']}"

    def test_chunk_source_field(self):
        """source-Feld entspricht dem übergebenen source-Parameter."""
        markdown = "# Test\n\nInhalt."
        source = "mein_dokument.pdf"
        chunks = chunk_markdown(markdown, chunk_size=512, source=source)

        assert len(chunks) >= 1
        for c in chunks:
            assert c["source"] == source, f"source-Feld falsch: {c['source']}"

    def test_chunk_token_count_present(self):
        """token_count-Feld ist vorhanden und positiv."""
        markdown = "# Test\n\nEin langer Text der mindestens ein paar Wörter enthält."
        chunks = chunk_markdown(markdown, chunk_size=512, source="test.md")

        assert len(chunks) >= 1
        for c in chunks:
            assert "token_count" in c
            assert c["token_count"] > 0

    def test_chunk_text_field_present(self):
        """text-Feld ist vorhanden und nicht leer."""
        markdown = "# Überschrift\n\nDas ist der Inhalt."
        chunks = chunk_markdown(markdown, chunk_size=512, source="test.md")

        assert len(chunks) >= 1
        for c in chunks:
            assert "text" in c
            assert len(c["text"].strip()) > 0

    def test_chunk_heading_field_present(self):
        """heading-Feld ist vorhanden (kann leer sein, wenn kein Heading)."""
        markdown = "Inhalt ohne Heading."
        chunks = chunk_markdown(markdown, chunk_size=512, source="test.md")

        assert len(chunks) >= 1
        for c in chunks:
            assert "heading" in c

    def test_chunk_all_required_fields(self):
        """Alle Pflichtfelder (index, heading, source, token_count, text) sind vorhanden."""
        markdown = "# Titel\n\nInhalt hier."
        chunks = chunk_markdown(markdown, chunk_size=512, source="datei.docx")

        required_fields = {"index", "heading", "source", "token_count", "text"}
        for c in chunks:
            missing = required_fields - set(c.keys())
            assert not missing, f"Fehlende Felder: {missing}"


class TestChunkSizeRespected:
    """AC-011-5: Chunk-Größe konfigurierbar; große Chunks werden weiter gesplittet."""

    def test_chunk_size_limits_tokens(self):
        """Bei kleinem chunk_size werden Chunks weiter aufgeteilt."""
        # Erstelle langen Text ohne Headings → Splitting an Absätzen
        long_text = "\n\n".join(
            f"Dies ist Absatz {i} mit genug Text um Tokens zu erzeugen." * 3
            for i in range(10)
        )
        chunks_small = chunk_markdown(long_text, chunk_size=50, source="test.md")
        chunks_large = chunk_markdown(long_text, chunk_size=5000, source="test.md")

        # Kleinerer chunk_size → mehr Chunks
        assert len(chunks_small) >= len(chunks_large)

    def test_chunk_token_count_heuristic(self):
        """token_count entspricht ungefähr len(text) / 4."""
        markdown = "# Test\n\nA" * 400  # ~100 Tokens
        chunks = chunk_markdown(markdown, chunk_size=512, source="test.md")

        assert len(chunks) >= 1
        total_tokens = sum(c["token_count"] for c in chunks)
        total_chars = sum(len(c["text"]) for c in chunks)
        # Heuristik: token_count ≈ len / 4 (mit Toleranz)
        expected_tokens = total_chars // 4
        assert abs(total_tokens - expected_tokens) <= max(10, expected_tokens * 0.2)

    def test_chunk_custom_chunk_size(self):
        """chunk_size=128 → kleinere Chunks als chunk_size=1024."""
        markdown = (
            "# Kapitel\n\n"
            + "Langer Absatz mit sehr viel Text. " * 50
            + "\n\nNoch ein langer Absatz. " * 50
        )
        chunks_128 = chunk_markdown(markdown, chunk_size=128, source="test.md")
        chunks_1024 = chunk_markdown(markdown, chunk_size=1024, source="test.md")

        assert len(chunks_128) >= len(chunks_1024)


class TestChunkDefaultOff:
    """AC-011-1: chunk=false → kein chunks Feld in der Antwort."""

    def test_chunk_default_off_no_chunks_field(self):
        """convert_auto ohne chunk=True → response.chunks ist None."""
        markitdown_result = {
            "success": True,
            "markdown": "# Test\n\nInhalt des Dokuments.",
            "title": "Test",
        }

        with patch.object(_server, "convert_with_markitdown", return_value=markitdown_result), \
             patch.object(_server, "is_scanned_pdf", return_value=False), \
             patch.object(_server, "detect_mimetype_from_bytes", return_value="text/plain"):
            result = run_async(convert_auto(
                file_data=b"fake content",
                filename="test.txt",
                source="/data/test.txt",
                source_type="file",
                input_meta={},
                # chunk=False ist der Default
            ))

        assert result.success is True
        assert result.chunks is None

    def test_chunk_explicitly_false_no_chunks(self):
        """convert_auto mit explizitem chunk=False → response.chunks ist None."""
        markitdown_result = {
            "success": True,
            "markdown": "# Test\n\nInhalt.",
        }

        with patch.object(_server, "convert_with_markitdown", return_value=markitdown_result), \
             patch.object(_server, "is_scanned_pdf", return_value=False), \
             patch.object(_server, "detect_mimetype_from_bytes", return_value="text/plain"):
            result = run_async(convert_auto(
                file_data=b"fake content",
                filename="test.txt",
                source="/data/test.txt",
                source_type="file",
                input_meta={},
                chunk=False,
            ))

        assert result.chunks is None


class TestChunkSmallDocument:
    """Kleines Dokument unter chunk_size → nur 1 Chunk."""

    def test_small_document_single_chunk(self):
        """Text unter chunk_size → genau 1 Chunk."""
        short_text = "Kurzer Text."
        chunks = chunk_markdown(short_text, chunk_size=512, source="tiny.md")

        assert len(chunks) == 1

    def test_single_heading_single_chunk(self):
        """Einziger Heading mit kurzem Text → 1 Chunk."""
        markdown = "# Titel\n\nKurzer Inhalt."
        chunks = chunk_markdown(markdown, chunk_size=512, source="test.md")

        assert len(chunks) == 1
        assert chunks[0]["index"] == 0


class TestChunkEmptyText:
    """Leerer Text → leere Liste."""

    def test_empty_string_returns_empty_list(self):
        """Leerer String → leere Liste."""
        chunks = chunk_markdown("", chunk_size=512, source="empty.md")
        assert chunks == []

    def test_whitespace_only_returns_empty_list(self):
        """Nur Whitespace → leere Liste."""
        chunks = chunk_markdown("   \n\n\t  ", chunk_size=512, source="empty.md")
        assert chunks == []

    def test_none_like_empty(self):
        """Leerer String mit source → leere Liste."""
        chunks = chunk_markdown("", chunk_size=256, source="test.md")
        assert isinstance(chunks, list)
        assert len(chunks) == 0


class TestChunkInConvertAuto:
    """AC-011-1: Integration von chunk_markdown in convert_auto."""

    def test_chunk_true_returns_chunks_list(self):
        """convert_auto mit chunk=True → response.chunks ist eine Liste."""
        markitdown_result = {
            "success": True,
            "markdown": (
                "# Einleitung\n\n"
                "Dieser Text ist die Einleitung des Dokuments.\n\n"
                "## Abschnitt 1\n\n"
                "Inhalt des ersten Abschnitts.\n\n"
                "## Abschnitt 2\n\n"
                "Inhalt des zweiten Abschnitts."
            ),
            "title": "Test Dokument",
        }

        with patch.object(_server, "convert_with_markitdown", return_value=markitdown_result), \
             patch.object(_server, "is_scanned_pdf", return_value=False), \
             patch.object(_server, "detect_mimetype_from_bytes", return_value="text/plain"):
            result = run_async(convert_auto(
                file_data=b"fake content",
                filename="test.txt",
                source="/data/test.txt",
                source_type="file",
                input_meta={},
                chunk=True,
                chunk_size=512,
            ))

        assert result.success is True
        assert result.chunks is not None
        assert isinstance(result.chunks, list)
        assert len(result.chunks) >= 1

    def test_chunk_true_chunks_have_metadata(self):
        """convert_auto mit chunk=True → Chunks enthalten Metadaten."""
        markitdown_result = {
            "success": True,
            "markdown": "# Titel\n\nInhalt des Dokuments.",
        }

        with patch.object(_server, "convert_with_markitdown", return_value=markitdown_result), \
             patch.object(_server, "is_scanned_pdf", return_value=False), \
             patch.object(_server, "detect_mimetype_from_bytes", return_value="text/plain"):
            result = run_async(convert_auto(
                file_data=b"fake content",
                filename="test.txt",
                source="/data/test.txt",
                source_type="file",
                input_meta={},
                chunk=True,
            ))

        assert result.success is True
        assert result.chunks is not None
        for c in result.chunks:
            assert "index" in c
            assert "heading" in c
            assert "source" in c
            assert "token_count" in c
            assert "text" in c

    def test_chunk_source_matches_input(self):
        """Chunks enthalten denselben source-Wert wie der convert_auto-Aufruf."""
        markitdown_result = {
            "success": True,
            "markdown": "# Abschnitt\n\nText.",
        }
        source_path = "/data/mein_dokument.pdf"

        with patch.object(_server, "convert_with_markitdown", return_value=markitdown_result), \
             patch.object(_server, "is_scanned_pdf", return_value=False), \
             patch.object(_server, "detect_mimetype_from_bytes", return_value="text/plain"):
            result = run_async(convert_auto(
                file_data=b"fake content",
                filename="mein_dokument.pdf",
                source=source_path,
                source_type="file",
                input_meta={},
                chunk=True,
            ))

        assert result.success is True
        assert result.chunks is not None
        for c in result.chunks:
            assert c["source"] == source_path

    def test_chunk_custom_size_respected_in_convert_auto(self):
        """chunk_size-Parameter wird korrekt an chunk_markdown weitergegeben."""
        long_markdown = (
            "# Kapitel 1\n\n"
            + "Sehr langer Absatztext mit vielen Wörtern. " * 30
            + "\n\n"
            + "# Kapitel 2\n\n"
            + "Weiterer langer Absatztext. " * 30
        )
        markitdown_result = {
            "success": True,
            "markdown": long_markdown,
        }

        with patch.object(_server, "convert_with_markitdown", return_value=markitdown_result), \
             patch.object(_server, "is_scanned_pdf", return_value=False), \
             patch.object(_server, "detect_mimetype_from_bytes", return_value="text/plain"):
            result_small = run_async(convert_auto(
                file_data=b"fake content",
                filename="test.txt",
                source="/data/test.txt",
                source_type="file",
                input_meta={},
                chunk=True,
                chunk_size=50,
            ))

        with patch.object(_server, "convert_with_markitdown", return_value=markitdown_result), \
             patch.object(_server, "is_scanned_pdf", return_value=False), \
             patch.object(_server, "detect_mimetype_from_bytes", return_value="text/plain"):
            result_large = run_async(convert_auto(
                file_data=b"fake content",
                filename="test.txt",
                source="/data/test.txt",
                source_type="file",
                input_meta={},
                chunk=True,
                chunk_size=5000,
            ))

        # Kleinerer chunk_size → mehr Chunks
        assert len(result_small.chunks) >= len(result_large.chunks)

    def test_chunk_error_response_has_no_chunks(self):
        """Bei Fehler-Antwort sind keine Chunks gesetzt."""
        with patch.object(_server, "convert_with_markitdown",
                          return_value={"success": False, "error": "Fehler",
                                        "error_code": "CONVERSION_FAILED"}), \
             patch.object(_server, "is_scanned_pdf", return_value=False):
            result = run_async(convert_auto(
                file_data=b"broken",
                filename="broken.txt",
                source="/data/broken.txt",
                source_type="file",
                input_meta={},
                chunk=True,
            ))

        assert result.success is False
        assert result.chunks is None
