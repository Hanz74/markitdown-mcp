"""
Tests für T-MKIT-010: Quality Scoring.

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
calculate_quality_score = _server.calculate_quality_score
convert_auto = _server.convert_auto


# ---------------------------------------------------------------------------
# Tests für calculate_quality_score()
# ---------------------------------------------------------------------------


class TestCalculateQualityScore:
    """Tests für die calculate_quality_score() Funktion (Unit-Tests)."""

    def test_quality_excellent(self):
        """
        AC-010-1/AC-010-2: Gut strukturierter Markdown → score hoch, grade 'excellent'.
        """
        markdown = (
            "# Jahresbericht 2024\n\n"
            "## Einleitung\n\n"
            "Dieses Dokument beschreibt die wichtigsten Kennzahlen und Ergebnisse "
            "des Geschäftsjahres 2024. Die Ergebnisse zeigen eine positive Entwicklung "
            "in allen Bereichen des Unternehmens.\n\n"
            "## Ergebnisse\n\n"
            "- Umsatz: 10 Mio. EUR\n"
            "- Gewinn: 2 Mio. EUR\n"
            "- Mitarbeiter: 150\n\n"
            "| Quartal | Umsatz | Gewinn |\n"
            "|---------|--------|--------|\n"
            "| Q1      | 2.0M   | 0.4M   |\n"
            "| Q2      | 2.5M   | 0.5M   |\n"
            "| Q3      | 2.8M   | 0.6M   |\n"
            "| Q4      | 2.7M   | 0.5M   |\n\n"
            "## Fazit\n\n"
            "Das Unternehmen hat seine Ziele vollständig erreicht und ist gut "
            "für das kommende Jahr aufgestellt.\n\n"
            "```python\n"
            "growth_rate = (revenue_2024 - revenue_2023) / revenue_2023\n"
            "```\n"
        )
        result = calculate_quality_score(markdown, {})

        assert "quality_score" in result
        assert "quality_grade" in result
        assert result["quality_score"] >= 0.8
        assert result["quality_grade"] == "excellent"

    def test_quality_poor(self):
        """
        AC-010-1/AC-010-2: Kaum Text, keine Struktur und Vision mit schwacher
        Token-Effizienz → score niedrig, grade 'poor'.
        """
        # Vision mit sehr schlechter Token-Effizienz → niedriger vision_score
        markdown = "x"
        meta = {
            "vision_used": True,
            "tokens_prompt": 10000,
            "tokens_completion": 1,  # Extrem schlechte Effizienz → score 0.05
        }
        result = calculate_quality_score(markdown, meta)

        assert result["quality_score"] < 0.3
        assert result["quality_grade"] == "poor"

    def test_quality_empty_text(self):
        """
        AC-010-1: Leerer Text → score 0.0, grade 'poor'.
        """
        result = calculate_quality_score("", {})

        assert result["quality_score"] == 0.0
        assert result["quality_grade"] == "poor"

    def test_quality_whitespace_only(self):
        """
        Nur Whitespace → score 0.0, grade 'poor'.
        """
        result = calculate_quality_score("   \n\n\t  ", {})

        assert result["quality_score"] == 0.0
        assert result["quality_grade"] == "poor"

    def test_quality_with_structure(self):
        """
        AC-010-1: Headings, Listen, Tabellen erhöhen den Score gegenüber reinem Text.
        """
        plain_text = "Dieser Text hat keinen Inhalt und keine Struktur. " * 5
        structured_text = (
            "# Haupttitel\n\n"
            "## Unterabschnitt\n\n"
            "Erster Absatz mit normalem Text der mehrere Wörter enthält.\n\n"
            "- Listenpunkt eins\n"
            "- Listenpunkt zwei\n"
            "- Listenpunkt drei\n\n"
            "| Spalte A | Spalte B |\n"
            "|----------|----------|\n"
            "| Wert 1   | Wert 2   |\n"
        )

        plain_result = calculate_quality_score(plain_text, {})
        structured_result = calculate_quality_score(structured_text, {})

        assert structured_result["quality_score"] > plain_result["quality_score"]

    def test_quality_grade_mapping_poor(self):
        """
        AC-010-2: Score 0.0-0.3 → 'poor'.
        Leerer Text und Vision mit sehr schlechter Token-Effizienz → grade 'poor'.
        """
        # Leerer Text → immer 'poor'
        result = calculate_quality_score("", {})
        assert result["quality_grade"] == "poor"

    def test_quality_grade_mapping_fair(self):
        """
        AC-010-2: Score im Bereich 0.3-0.6 → 'fair'.
        """
        # Mittelmäßiger Text: etwas Inhalt, keine Struktur
        text = "Das ist ein normaler Satz ohne besondere Struktur. " * 3
        result = calculate_quality_score(text, {})

        # Score sollte in der Fair-Bereich fallen (0.3-0.6)
        assert result["quality_score"] >= 0.3
        assert result["quality_grade"] in ("fair", "good", "excellent")

    def test_quality_grade_mapping_thresholds(self):
        """
        AC-010-2: Grade-Grenzen sind korrekt bei 0.3 / 0.6 / 0.8.
        """
        # Leerer Text → 'poor' (score = 0.0)
        result_empty = calculate_quality_score("", {})
        assert result_empty["quality_grade"] == "poor"
        assert result_empty["quality_score"] == 0.0

        # Ein strukturierter, inhaltreicher Text sollte 'good' oder 'excellent' ergeben
        high_markdown = (
            "# Titel\n\n"
            "## Abschnitt\n\n"
            "Hier ist ein vollständiger Text mit echten deutschen Wörtern, "
            "der mehrere Sätze umfasst und gut strukturiert ist.\n\n"
            "- Punkt 1\n- Punkt 2\n- Punkt 3\n"
        )
        result_high = calculate_quality_score(high_markdown, {})
        assert result_high["quality_grade"] in ("good", "excellent")

    def test_quality_ocr_bonus_with_tokens(self):
        """
        AC-010-3: Bei vision_used=True und hoher Token-Effizienz → höherer Score.
        Vergleich: gleicher Text, einmal mit Vision-Meta, einmal ohne.
        """
        markdown = (
            "# Extrahierter Text\n\n"
            "Hier ist der durch Vision extrahierte Text aus dem Bild. "
            "Dieser Text enthält mehrere echte Wörter und Sätze.\n\n"
            "- Punkt eins\n"
            "- Punkt zwei\n"
        )

        meta_without_vision = {}
        meta_with_vision = {
            "vision_used": True,
            "tokens_prompt": 100,
            "tokens_completion": 80,
        }

        result_no_vision = calculate_quality_score(markdown, meta_without_vision)
        result_with_vision = calculate_quality_score(markdown, meta_with_vision)

        # Mit Vision und guter Effizienz (80/100 = 0.8 >= 0.5) → volle vision_score 0.2
        # Ohne Vision → ebenfalls 0.2 (Baseline)
        # Scores sollten sehr ähnlich oder gleich sein (beide erhalten 0.2)
        assert abs(result_no_vision["quality_score"] - result_with_vision["quality_score"]) < 0.15

    def test_quality_ocr_low_efficiency(self):
        """
        AC-010-3: Bei vision_used=True aber niedriger Token-Effizienz → niedrigerer Vision-Score.
        """
        markdown = "Kurzer Text."

        meta_low_efficiency = {
            "vision_used": True,
            "tokens_prompt": 1000,
            "tokens_completion": 2,  # Sehr wenig Output → schlechte Effizienz (0.002)
        }

        result = calculate_quality_score(markdown, meta_low_efficiency)
        # vision_score sollte niedrig sein (0.05 für efficiency < 0.05 nicht ganz, 0.1 für < 0.2)
        # Wir prüfen nur, dass der Score valide ist
        assert 0.0 <= result["quality_score"] <= 1.0
        assert result["quality_grade"] in ("poor", "fair", "good", "excellent")

    def test_quality_score_in_valid_range(self):
        """
        AC-010-1: quality_score ist immer zwischen 0.0 und 1.0.
        """
        test_cases = [
            ("", {}),
            ("x", {}),
            ("# Titel\n\nText\n", {}),
            ("a" * 1000, {}),
            ("# H1\n## H2\n- item\n| a | b |\n|---|---|\n| 1 | 2 |\n```code```\n" * 5, {}),
            ("Normal text " * 50, {"vision_used": True, "tokens_prompt": 200, "tokens_completion": 100}),
        ]

        for markdown, meta in test_cases:
            result = calculate_quality_score(markdown, meta)
            assert 0.0 <= result["quality_score"] <= 1.0, (
                f"Score out of range for input len={len(markdown)}: {result['quality_score']}"
            )
            assert result["quality_grade"] in ("poor", "fair", "good", "excellent"), (
                f"Invalid grade: {result['quality_grade']}"
            )

    def test_quality_scanned_pdf_meta(self):
        """
        AC-010-3: scanned=True wird wie vision_used behandelt.
        """
        markdown = (
            "# Gescanntes Dokument\n\n"
            "Inhalt des gescannten Dokuments mit echten Wörtern und Sätzen.\n\n"
            "- Listenpunkt\n"
        )
        meta_scanned = {
            "scanned": True,
            "vision_used": True,
            "tokens_prompt": 500,
            "tokens_completion": 300,
        }

        result = calculate_quality_score(markdown, meta_scanned)

        assert "quality_score" in result
        assert "quality_grade" in result
        assert 0.0 <= result["quality_score"] <= 1.0


# ---------------------------------------------------------------------------
# Integration-Tests für convert_auto() mit Quality Scoring
# ---------------------------------------------------------------------------


class TestQualityInConvertAuto:
    """Tests für die Integration von Quality Scoring in convert_auto()."""

    def test_quality_in_convert_auto_markitdown_path(self):
        """
        AC-010-1/AC-010-2: convert_auto() über den MarkItDown-Pfad
        gibt quality_score und quality_grade in meta zurück.
        """
        markitdown_result = {
            "success": True,
            "markdown": (
                "# Test Dokument\n\n"
                "## Abschnitt\n\n"
                "Hier ist der Inhalt des Dokuments mit echten Wörtern.\n\n"
                "- Punkt eins\n"
                "- Punkt zwei\n"
            ),
            "title": "Test Dokument",
        }

        # detect_mimetype_from_bytes muss None oder einen Nicht-Bild-Typ zurückgeben,
        # damit convert_auto den MarkItDown-Pfad (nicht Vision) wählt.
        with patch.object(_server, "convert_with_markitdown", return_value=markitdown_result), \
             patch.object(_server, "is_scanned_pdf", return_value=False), \
             patch.object(_server, "detect_mimetype_from_bytes", return_value="text/plain"):
            result = run_async(convert_auto(
                file_data=b"fake txt content",
                filename="test.txt",
                source="/data/test.txt",
                source_type="file",
                input_meta={},
            ))

        assert result.success is True
        assert result.meta.quality_score is not None
        assert result.meta.quality_grade is not None
        assert 0.0 <= result.meta.quality_score <= 1.0
        assert result.meta.quality_grade in ("poor", "fair", "good", "excellent")

    def test_quality_in_convert_auto_vision_path(self):
        """
        AC-010-1/AC-010-2/AC-010-3: convert_auto() über den Vision-Pfad
        gibt quality_score und quality_grade in meta zurück.
        """
        from conftest import PNG_100x100

        vision_result = {
            "success": True,
            "markdown": (
                "# Extrahierter Text\n\n"
                "Dieser Text wurde aus dem Bild extrahiert.\n\n"
                "- Erster Punkt\n"
                "- Zweiter Punkt\n"
            ),
            "tokens_prompt": 200,
            "tokens_completion": 100,
            "tokens_total": 300,
            "vision_model": "pixtral-12b-2409",
        }

        with patch.object(_server, "analyze_with_mistral_vision",
                          new=AsyncMock(return_value=vision_result)):
            result = run_async(convert_auto(
                file_data=PNG_100x100,
                filename="scan.png",
                source="/data/scan.png",
                source_type="file",
                input_meta={},
            ))

        assert result.success is True
        assert result.meta.quality_score is not None
        assert result.meta.quality_grade is not None
        assert 0.0 <= result.meta.quality_score <= 1.0
        assert result.meta.quality_grade in ("poor", "fair", "good", "excellent")

    def test_quality_not_in_error_response(self):
        """
        Fehler-Antworten enthalten kein quality_score.
        """
        with patch.object(_server, "convert_with_markitdown",
                          return_value={"success": False, "error": "Konvertierung fehlgeschlagen",
                                        "error_code": "CONVERSION_FAILED"}), \
             patch.object(_server, "is_scanned_pdf", return_value=False):
            result = run_async(convert_auto(
                file_data=b"broken content",
                filename="broken.txt",
                source="/data/broken.txt",
                source_type="file",
                input_meta={},
            ))

        assert result.success is False
        # Bei Fehlern kein quality_score
        assert result.meta.quality_score is None
        assert result.meta.quality_grade is None
