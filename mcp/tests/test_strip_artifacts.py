"""
Tests für T-MKIT-016: strip_llm_artifacts() Post-Processing Funktion.

Testet die Bereinigung von LLM-typischen Artefakten:
- Markdown/generische Codeblock-Wrapping
- Deutsche und englische Preamble-Sätze
- Kombination aus Preamble + Codeblock
- Codeblöcke innerhalb des Textes werden NICHT entfernt
"""

from conftest import load_server_module, run_async  # noqa: F401

# Einmal laden; alle Tests in diesem Modul teilen diese Instanz
_server = load_server_module(use_real_pil=False)
strip_llm_artifacts = _server.strip_llm_artifacts


class TestStripMarkdownCodeblock:
    """Entfernt ```markdown ... ``` Wrapping wenn der gesamte Output darin ist."""

    def test_strip_markdown_codeblock(self):
        input_text = "```markdown\ncontent here\n```"
        result = strip_llm_artifacts(input_text)
        assert result == "content here"

    def test_strip_md_alias_codeblock(self):
        input_text = "```md\ncontent here\n```"
        result = strip_llm_artifacts(input_text)
        assert result == "content here"

    def test_strip_markdown_codeblock_multiline(self):
        input_text = "```markdown\n# Heading\n\nSome text\n\n- item 1\n- item 2\n```"
        result = strip_llm_artifacts(input_text)
        assert result == "# Heading\n\nSome text\n\n- item 1\n- item 2"

    def test_strip_markdown_codeblock_case_insensitive(self):
        input_text = "```Markdown\ncontent\n```"
        result = strip_llm_artifacts(input_text)
        assert result == "content"


class TestStripGenericCodeblock:
    """Entfernt ``` ... ``` Wrapping wenn der gesamte Output darin ist."""

    def test_strip_generic_codeblock(self):
        input_text = "```\ncontent here\n```"
        result = strip_llm_artifacts(input_text)
        assert result == "content here"

    def test_strip_generic_codeblock_multiline(self):
        input_text = "```\nline one\nline two\nline three\n```"
        result = strip_llm_artifacts(input_text)
        assert result == "line one\nline two\nline three"


class TestStripPreambleGerman:
    """Entfernt deutsche Preamble-Sätze am Anfang."""

    def test_strip_preamble_german_hier_ist(self):
        input_text = "Hier ist der extrahierte Text:\n\ncontent here"
        result = strip_llm_artifacts(input_text)
        assert result == "content here"

    def test_strip_preamble_german_hier_ist_der(self):
        input_text = (
            "Hier ist der extrahierte und strukturierte Text aus dem gescannten "
            "Laborbericht im Markdown-Format:\n\n# Laborbericht\n\nText"
        )
        result = strip_llm_artifacts(input_text)
        assert "Hier ist" not in result
        assert "# Laborbericht" in result

    def test_strip_preamble_german_im_folgenden(self):
        input_text = "Im Folgenden der Inhalt als Markdown:\n\ncontent here"
        result = strip_llm_artifacts(input_text)
        assert result == "content here"

    def test_strip_preamble_german_nachfolgend(self):
        input_text = "Nachfolgend der extrahierte Text:\n\ncontent here"
        result = strip_llm_artifacts(input_text)
        assert result == "content here"

    def test_strip_preamble_german_gerne(self):
        input_text = "Gerne, hier ist die Auswertung:\n\ncontent here"
        result = strip_llm_artifacts(input_text)
        assert result == "content here"


class TestStripPreambleEnglish:
    """Entfernt englische Preamble-Sätze am Anfang."""

    def test_strip_preamble_english_here_is(self):
        input_text = "Here is the extracted text:\n\ncontent here"
        result = strip_llm_artifacts(input_text)
        assert result == "content here"

    def test_strip_preamble_english_here_is_the(self):
        input_text = "Here is the structured Markdown output:\n\n# Document\n\nText"
        result = strip_llm_artifacts(input_text)
        assert "Here is" not in result
        assert "# Document" in result

    def test_strip_preamble_english_below_is(self):
        input_text = "Below is the extracted content:\n\ncontent here"
        result = strip_llm_artifacts(input_text)
        assert result == "content here"

    def test_strip_preamble_english_the_following(self):
        input_text = "The following is the Markdown representation:\n\ncontent here"
        result = strip_llm_artifacts(input_text)
        assert result == "content here"

    def test_strip_preamble_english_certainly(self):
        input_text = "Certainly! Here is the content:\n\ncontent here"
        result = strip_llm_artifacts(input_text)
        assert result == "content here"

    def test_strip_preamble_english_sure(self):
        input_text = "Sure, here is the output:\n\ncontent here"
        result = strip_llm_artifacts(input_text)
        assert result == "content here"


class TestNoStripCleanText:
    """Sauberer Text bleibt unverändert."""

    def test_no_strip_clean_markdown(self):
        input_text = "# Heading\n\nSome paragraph text.\n\n- item 1\n- item 2"
        result = strip_llm_artifacts(input_text)
        assert result == input_text

    def test_no_strip_plain_text(self):
        input_text = "This is just plain text without any preamble."
        result = strip_llm_artifacts(input_text)
        assert result == input_text

    def test_no_strip_empty_string(self):
        result = strip_llm_artifacts("")
        assert result == ""

    def test_no_strip_whitespace_only(self):
        result = strip_llm_artifacts("   \n  ")
        assert result == ""

    def test_no_strip_german_content_starting_normally(self):
        input_text = "Der Bericht zeigt folgende Ergebnisse:\n\n- Punkt 1\n- Punkt 2"
        result = strip_llm_artifacts(input_text)
        assert result == input_text


class TestStripCombined:
    """Preamble + Codeblock werden gemeinsam entfernt."""

    def test_strip_preamble_then_codeblock(self):
        # Preamble vor einem Codeblock — der Codeblock ist kein outer-wrap
        # da der Preamble davor steht; nach Preamble-Entfernung bleibt
        # nur der Codeblock-Inhalt (ohne Outer-Wrap-Entfernung, da kein Full-Wrap)
        input_text = "Here is the result:\n\n```markdown\n# Content\n\nText\n```"
        result = strip_llm_artifacts(input_text)
        # After preamble strip, the remaining text starts with ```markdown
        # which is then treated as outer code block wrap
        assert "Here is the result" not in result
        assert "# Content" in result

    def test_strip_markdown_codeblock_with_leading_whitespace(self):
        input_text = "  ```markdown\ncontent\n```  "
        result = strip_llm_artifacts(input_text)
        assert result == "content"


class TestStripMermaidPassthrough:
    """Mermaid-Codeblöcke als alleiniger Output werden NICHT gestripped."""

    def test_mermaid_standalone_not_stripped(self):
        input_text = "```mermaid\ngraph TD\n    A --> B\n```"
        result = strip_llm_artifacts(input_text)
        assert result == input_text

    def test_mermaid_standalone_with_whitespace_not_stripped(self):
        input_text = "  ```mermaid\ngraph TD\n    A --> B\n```  "
        result = strip_llm_artifacts(input_text)
        assert result == "```mermaid\ngraph TD\n    A --> B\n```"

    def test_mermaid_complex_not_stripped(self):
        input_text = "```mermaid\nsequenceDiagram\n    Alice ->> Bob: Hello\n    Bob -->> Alice: Hi\n```"
        result = strip_llm_artifacts(input_text)
        assert result == input_text


class TestStripPreservesInternalCodeblocks:
    """Codeblöcke IM Text bleiben erhalten."""

    def test_preserves_internal_codeblock_python(self):
        input_text = (
            "# Documentation\n\n"
            "Some text here.\n\n"
            "```python\ndef hello():\n    print('hello')\n```\n\n"
            "More text."
        )
        result = strip_llm_artifacts(input_text)
        assert "```python" in result
        assert "def hello():" in result
        assert "# Documentation" in result

    def test_preserves_multiple_internal_codeblocks(self):
        input_text = (
            "# Heading\n\n"
            "```bash\necho hello\n```\n\n"
            "Middle text\n\n"
            "```sql\nSELECT * FROM table;\n```"
        )
        result = strip_llm_artifacts(input_text)
        assert "```bash" in result
        assert "```sql" in result
        assert "# Heading" in result

    def test_preserves_mermaid_codeblock_in_text(self):
        input_text = (
            "## Diagramm\n\n"
            "```mermaid\ngraph TD\n    A --> B\n```"
        )
        result = strip_llm_artifacts(input_text)
        assert "```mermaid" in result
        assert "graph TD" in result

    def test_does_not_strip_partial_outer_codeblock(self):
        # A code block that is NOT the full output (has content before/after) stays intact
        input_text = "Some intro\n\n```\ncode\n```\n\nSome outro"
        result = strip_llm_artifacts(input_text)
        assert "```" in result
        assert "code" in result
        assert "Some intro" in result
        assert "Some outro" in result
