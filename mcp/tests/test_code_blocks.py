"""
Unit-Tests für Code-Block-Erkennung und Sprach-Fences (FR-MKIT-005).

AC-005-1: Monospace-Fonts in DOCX → Code erkannt
AC-005-2: Indentierte Blöcke in PDFs → Code erkannt
AC-005-3: Sprache automatisch erkannt (Regex-Heuristik)
AC-005-4: Erkannter Code in ```language ... ``` gewrappt

Tests laufen ohne Docker-Container (Mocking via conftest).
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from conftest import load_server_module, run_async


# ---------------------------------------------------------------------------
# Einmalig Server-Modul laden
# ---------------------------------------------------------------------------

_server = load_server_module(use_real_pil=False)
detect_code_language = _server.detect_code_language
detect_and_fence_code_blocks = _server.detect_and_fence_code_blocks


# =============================================================================
# detect_code_language
# =============================================================================

class TestDetectCodeLanguage:
    """Testet die Regex-basierte Spracherkennung."""

    def test_detect_language_python(self):
        """Python-Code wird korrekt erkannt."""
        code = (
            "def hello(self):\n"
            "    import os\n"
            "    class Foo:\n"
            "        pass\n"
        )
        assert detect_code_language(code) == "python"

    def test_detect_language_javascript(self):
        """JavaScript-Code wird korrekt erkannt."""
        code = (
            "const greet = (name) => {\n"
            "    let msg = `Hello ${name}`;\n"
            "    console.log(msg);\n"
            "    function inner() {}\n"
            "};\n"
        )
        assert detect_code_language(code) == "javascript"

    def test_detect_language_sql(self):
        """SQL-Abfragen werden korrekt erkannt."""
        code = (
            "SELECT id, name\n"
            "FROM users\n"
            "WHERE active = 1\n"
            "INSERT INTO log VALUES (1, 'ok');\n"
        )
        assert detect_code_language(code) == "sql"

    def test_detect_language_java(self):
        """Java-Code wird korrekt erkannt."""
        code = (
            "public class Main {\n"
            "    private void run() {\n"
            "        System.out.println('hi');\n"
            "        void helper() {}\n"
            "    }\n"
            "}\n"
        )
        assert detect_code_language(code) == "java"

    def test_detect_language_bash(self):
        """Bash/Shell-Skripte werden korrekt erkannt."""
        code = (
            "#!/bin/bash\n"
            "echo hello\n"
            "if [ -f file ]; then\n"
            "    echo found\n"
            "fi\n"
            "done\n"
        )
        assert detect_code_language(code) == "bash"

    def test_detect_language_go(self):
        """Go-Code wird korrekt erkannt."""
        code = (
            "package main\n"
            "import (\n"
            "    \"fmt\"\n"
            ")\n"
            "func main() {\n"
            "    fmt.Println('hi')\n"
            "}\n"
        )
        assert detect_code_language(code) == "go"

    def test_detect_language_rust(self):
        """Rust-Code wird korrekt erkannt."""
        code = (
            "pub fn main() {\n"
            "    let mut x = 5;\n"
            "    impl Foo {}\n"
            "    fn helper() {}\n"
            "}\n"
        )
        assert detect_code_language(code) == "rust"

    def test_detect_language_cpp(self):
        """C/C++-Code wird korrekt erkannt."""
        code = (
            "#include <iostream>\n"
            "int main() {\n"
            "    printf(\"hello\");\n"
            "    std::cout << \"world\";\n"
            "}\n"
        )
        assert detect_code_language(code) == "cpp"

    def test_detect_language_unknown(self):
        """Unbekannter Text ergibt leeren String."""
        code = "Dies ist normaler Fließtext ohne Code-Indikatoren."
        assert detect_code_language(code) == ""

    def test_detect_language_unknown_single_indicator(self):
        """Ein einzelner Treffer reicht nicht für Sprach-Identifikation."""
        # Nur ein Python-Pattern-Match → unter _MIN_LANG_SCORE=2
        code = "def irgendwas"
        assert detect_code_language(code) == ""

    def test_detect_language_html(self):
        """HTML-Code wird korrekt erkannt."""
        code = (
            "<!DOCTYPE html>\n"
            "<html>\n"
            "<body>\n"
            "<div class='x'></div>\n"
            "</body></html>\n"
        )
        assert detect_code_language(code) == "html"

    def test_detect_language_css(self):
        """CSS-Code wird korrekt erkannt."""
        code = (
            ".box {\n"
            "    margin: 10px;\n"
            "    padding: 5px;\n"
            "    display: flex;\n"
            "    {color: red}\n"
            "}\n"
        )
        assert detect_code_language(code) == "css"


# =============================================================================
# detect_and_fence_code_blocks
# =============================================================================

class TestDetectAndFenceCodeBlocks:
    """Testet das Wrapping von Code-Blöcken in Markdown-Fences."""

    def test_fence_indented_code_block(self):
        """Indentierter Block (4+ Spaces, 3+ Zeilen) wird in Fences gewrappt."""
        markdown = (
            "Einleitungstext\n"
            "\n"
            "    def hello(self):\n"
            "        import os\n"
            "        class Foo:\n"
            "            pass\n"
            "\n"
            "Abschlusstext\n"
        )
        result = detect_and_fence_code_blocks(markdown)
        assert "```" in result
        assert "def hello" in result
        # Kein doppeltes Fencing
        assert result.count("```python") <= 1

    def test_fence_preserves_existing_fences(self):
        """Bereits gefencten Code nicht nochmals wrappen."""
        markdown = (
            "Text davor\n"
            "\n"
            "```python\n"
            "def foo():\n"
            "    pass\n"
            "```\n"
            "\n"
            "Text danach\n"
        )
        result = detect_and_fence_code_blocks(markdown)
        # Nur ein Fence-Paar darf vorhanden sein
        assert result.count("```python") == 1
        assert result.count("```") == 2  # öffnend + schließend

    def test_fence_multiple_blocks(self):
        """Mehrere indentierte Blöcke werden unabhängig gefenct."""
        markdown = (
            "Block 1:\n"
            "\n"
            "    def alpha(self):\n"
            "        import sys\n"
            "        class A:\n"
            "            pass\n"
            "\n"
            "Zwischentext\n"
            "\n"
            "Block 2:\n"
            "\n"
            "    SELECT id\n"
            "    FROM users\n"
            "    WHERE active = 1\n"
            "    INSERT INTO log VALUES (1, 'ok');\n"
            "\n"
        )
        result = detect_and_fence_code_blocks(markdown)
        # Mindestens zwei Fence-Öffnungen
        assert result.count("```") >= 4  # 2x öffnend + 2x schließend
        assert "def alpha" in result
        assert "SELECT id" in result

    def test_fence_short_block_ignored(self):
        """Block mit weniger als 3 Zeilen wird NICHT gefenct (AC-005-2)."""
        markdown = (
            "Text\n"
            "\n"
            "    def foo():\n"
            "        pass\n"
            "\n"
            "Ende\n"
        )
        result = detect_and_fence_code_blocks(markdown)
        # Kein Fence eingefügt (2 Zeilen < Schwellwert 3)
        assert "```" not in result

    def test_fence_empty_string(self):
        """Leerer String wird unverändert zurückgegeben."""
        assert detect_and_fence_code_blocks("") == ""

    def test_fence_no_code_plain_text(self):
        """Normaler Fließtext ohne Einrückung bleibt unverändert."""
        markdown = "Normaler Text.\nNoch eine Zeile.\nUnd eine dritte.\n"
        result = detect_and_fence_code_blocks(markdown)
        assert "```" not in result
        assert result == markdown

    def test_fence_language_detected_in_fence(self):
        """Erkannte Sprache wird in den Fence-Header geschrieben."""
        markdown = (
            "Beispiel:\n"
            "\n"
            "    def main(self):\n"
            "        import os\n"
            "        class Config:\n"
            "            pass\n"
            "\n"
        )
        result = detect_and_fence_code_blocks(markdown)
        assert "```python" in result

    def test_fence_unknown_language_uses_bare_fence(self):
        """Unbekannte Sprache verwendet Fence ohne Sprachkennung."""
        # Text mit 3+ Zeilen Einrückung aber ohne erkennbare Sprache
        markdown = (
            "Block:\n"
            "\n"
            "    Lorem ipsum dolor\n"
            "    sit amet consectetur\n"
            "    adipiscing elit sed\n"
            "\n"
        )
        result = detect_and_fence_code_blocks(markdown)
        # Wenn gefenct → kein Sprachlabel, nur bare fence
        if "```" in result:
            assert "```\n" in result or result.startswith("```\n")


# =============================================================================
# Integration: convert_with_markitdown wendet Code-Fencing an
# =============================================================================

class TestIntegrationInConvert:
    """Testet dass convert_with_markitdown Code-Fencing auf PDF/DOCX anwendet."""

    def _make_md_result(self, text: str) -> MagicMock:
        """Hilfsmethode: Erzeugt ein Mock-MarkItDown-Ergebnis."""
        mock_result = MagicMock()
        mock_result.text_content = text
        mock_result.title = "Test"
        return mock_result

    def test_integration_in_convert_pdf(self):
        """Code-Fencing wird für PDF-Dateien nach der Konvertierung angewendet."""
        server = load_server_module(use_real_pil=False)

        indented_python = (
            "Dokumentation\n"
            "\n"
            "    def run(self):\n"
            "        import sys\n"
            "        class Runner:\n"
            "            pass\n"
            "\n"
        )
        mock_result = self._make_md_result(indented_python)
        server.md.convert = MagicMock(return_value=mock_result)
        server.PDFPLUMBER_AVAILABLE = False  # Tabellen-Extraktion deaktivieren

        result = server.convert_with_markitdown(Path("/tmp/test.pdf"))

        assert result["success"] is True
        assert "```" in result["markdown"]

    def test_integration_in_convert_docx(self):
        """Code-Fencing wird für DOCX-Dateien nach der Konvertierung angewendet."""
        server = load_server_module(use_real_pil=False)

        indented_go = (
            "Anleitung\n"
            "\n"
            "    package main\n"
            "    import (\n"
            "        \"fmt\"\n"
            "    )\n"
            "    func start() {\n"
            "        fmt.Println(\"ok\")\n"
            "    }\n"
            "\n"
        )
        mock_result = self._make_md_result(indented_go)
        server.md.convert = MagicMock(return_value=mock_result)

        result = server.convert_with_markitdown(Path("/tmp/test.docx"))

        assert result["success"] is True
        assert "```" in result["markdown"]

    def test_integration_skipped_for_other_formats(self):
        """Code-Fencing wird für andere Formate (z.B. .txt) NICHT angewendet."""
        server = load_server_module(use_real_pil=False)

        indented = (
            "Bericht\n"
            "\n"
            "    def compute(self):\n"
            "        import math\n"
            "        class Calc:\n"
            "            pass\n"
            "\n"
        )
        mock_result = self._make_md_result(indented)
        server.md.convert = MagicMock(return_value=mock_result)

        result = server.convert_with_markitdown(Path("/tmp/test.txt"))

        assert result["success"] is True
        # Für .txt → kein Fencing → kein ``` im Output
        assert "```" not in result["markdown"]
