"""
Tests für T-MKIT-003: Embedded Images aus DOCX/PPTX extrahieren und beschreiben.

Alle Tests laufen ohne Docker-Container und ohne echte API-Calls.
Alle externen Abhängigkeiten werden per unittest.mock gemockt.
"""

import asyncio
import io
import sys
import types
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conftest import (
    load_server_module, run_async,
    PNG_100x100 as _PNG_100x100,
    PNG_120x80 as _PNG_120x80,
    PNG_150x150 as _PNG_150x150,
    PNG_200x200 as _PNG_200x200,
    PNG_300x200 as _PNG_300x200,
    PNG_400x300 as _PNG_400x300,
    PNG_200x150 as _PNG_200x150,
    PNG_49x49 as _PNG_49x49,
    PNG_50x50 as _PNG_50x50,
    PNG_30x30 as _PNG_30x30,
    PNG_10x10 as _PNG_10x10,
)


def _make_docx_zip(images: dict[str, bytes]) -> bytes:
    """
    Erstellt ein synthetisches DOCX-ZIP mit Bildern in word/media/.

    Args:
        images: Mapping von Dateinamen zu PNG-Bytes.

    Returns:
        ZIP-Bytes, die ein DOCX mit den angegebenen Bildern simulieren.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in images.items():
            zf.writestr(f"word/media/{name}", data)
    return buf.getvalue()


def _make_pptx_zip(images: dict[str, bytes]) -> bytes:
    """
    Erstellt ein synthetisches PPTX-ZIP mit Bildern in ppt/media/.

    Args:
        images: Mapping von Dateinamen zu PNG-Bytes.

    Returns:
        ZIP-Bytes, die ein PPTX mit den angegebenen Bildern simulieren.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in images.items():
            zf.writestr(f"ppt/media/{name}", data)
    return buf.getvalue()


# Einmal laden; alle Tests teilen diese Instanz
# use_real_pil=True: echtes PIL für Image.open (Größen-Filterung wird getestet)
_server = load_server_module(use_real_pil=True)


# ---------------------------------------------------------------------------
# Tests: extract_images_from_docx
# ---------------------------------------------------------------------------

class TestExtractImagesFromDocx:
    """Tests für extract_images_from_docx."""

    def test_extract_images_from_docx_returns_images(self, tmp_path: Path) -> None:
        """Bilder aus word/media/ werden erkannt und zurückgegeben."""
        png = _PNG_100x100
        docx_bytes = _make_docx_zip({"image1.png": png})
        docx_path = tmp_path / "test.docx"
        docx_path.write_bytes(docx_bytes)

        images = _server.extract_images_from_docx(docx_path)

        assert len(images) == 1
        assert images[0]["name"] == "image1.png"
        assert images[0]["data"] == png
        assert "position_hint" in images[0]

    def test_extract_images_from_docx_multiple_images(self, tmp_path: Path) -> None:
        """Mehrere Bilder werden alle extrahiert."""
        images_data = {
            "image1.png": _PNG_200x200,
            "image2.png": _PNG_150x150,
        }
        docx_bytes = _make_docx_zip(images_data)
        docx_path = tmp_path / "multi.docx"
        docx_path.write_bytes(docx_bytes)

        images = _server.extract_images_from_docx(docx_path)

        assert len(images) == 2
        names = {img["name"] for img in images}
        assert names == {"image1.png", "image2.png"}

    def test_extract_images_from_docx_returns_list_on_error(self, tmp_path: Path) -> None:
        """Bei ungültigem ZIP wird eine leere Liste zurückgegeben (kein Crash)."""
        bad_path = tmp_path / "broken.docx"
        bad_path.write_bytes(b"not a zip file")

        images = _server.extract_images_from_docx(bad_path)

        assert isinstance(images, list)
        assert len(images) == 0


# ---------------------------------------------------------------------------
# Tests: extract_images_from_pptx
# ---------------------------------------------------------------------------

class TestExtractImagesFromPptx:
    """Tests für extract_images_from_pptx."""

    def test_extract_images_from_pptx_returns_images(self, tmp_path: Path) -> None:
        """Bilder aus ppt/media/ werden erkannt und zurückgegeben."""
        png = _PNG_120x80
        pptx_bytes = _make_pptx_zip({"slide_image1.png": png})
        pptx_path = tmp_path / "test.pptx"
        pptx_path.write_bytes(pptx_bytes)

        images = _server.extract_images_from_pptx(pptx_path)

        assert len(images) == 1
        assert images[0]["name"] == "slide_image1.png"
        assert images[0]["data"] == png
        assert "slide_number" in images[0]
        assert images[0]["slide_number"] == 1

    def test_extract_images_from_pptx_multiple_images(self, tmp_path: Path) -> None:
        """Mehrere Bilder in ppt/media/ werden alle extrahiert."""
        images_data = {
            "img1.png": _PNG_300x200,
            "img2.png": _PNG_400x300,
        }
        pptx_bytes = _make_pptx_zip(images_data)
        pptx_path = tmp_path / "multi.pptx"
        pptx_path.write_bytes(pptx_bytes)

        images = _server.extract_images_from_pptx(pptx_path)

        assert len(images) == 2

    def test_extract_images_from_pptx_returns_list_on_error(self, tmp_path: Path) -> None:
        """Bei ungültigem ZIP wird eine leere Liste zurückgegeben."""
        bad_path = tmp_path / "broken.pptx"
        bad_path.write_bytes(b"not a zip")

        images = _server.extract_images_from_pptx(bad_path)

        assert isinstance(images, list)
        assert len(images) == 0


# ---------------------------------------------------------------------------
# Tests: Kleine Bilder überspringen (AC-003-4)
# ---------------------------------------------------------------------------

class TestSkipSmallImages:
    """Bilder unter MIN_IMAGE_SIZE_PX (50x50) werden herausgefiltert."""

    def test_skip_small_images_docx(self, tmp_path: Path) -> None:
        """Bilder < 50x50 aus DOCX werden nicht zurückgegeben."""
        small_png = _PNG_30x30    # zu klein
        large_png = _PNG_100x100  # groß genug

        docx_bytes = _make_docx_zip({
            "small.png": small_png,
            "large.png": large_png,
        })
        docx_path = tmp_path / "mixed.docx"
        docx_path.write_bytes(docx_bytes)

        images = _server.extract_images_from_docx(docx_path)

        names = [img["name"] for img in images]
        assert "large.png" in names
        assert "small.png" not in names

    def test_skip_small_images_pptx(self, tmp_path: Path) -> None:
        """Bilder < 50x50 aus PPTX werden nicht zurückgegeben."""
        tiny_png = _PNG_10x10      # zu klein
        normal_png = _PNG_200x150  # groß genug

        pptx_bytes = _make_pptx_zip({
            "tiny.png": tiny_png,
            "normal.png": normal_png,
        })
        pptx_path = tmp_path / "mixed.pptx"
        pptx_path.write_bytes(pptx_bytes)

        images = _server.extract_images_from_pptx(pptx_path)

        names = [img["name"] for img in images]
        assert "normal.png" in names
        assert "tiny.png" not in names

    def test_exact_threshold_docx(self, tmp_path: Path) -> None:
        """Bilder exakt bei 50x50 werden NICHT übersprungen (Grenzwert-Test)."""
        boundary_png = _PNG_50x50
        docx_bytes = _make_docx_zip({"boundary.png": boundary_png})
        docx_path = tmp_path / "boundary.docx"
        docx_path.write_bytes(docx_bytes)

        images = _server.extract_images_from_docx(docx_path)

        assert len(images) == 1
        assert images[0]["name"] == "boundary.png"

    def test_below_threshold_both_dimensions(self, tmp_path: Path) -> None:
        """49x49 Bild wird übersprungen."""
        small_png = _PNG_49x49
        docx_bytes = _make_docx_zip({"small.png": small_png})
        docx_path = tmp_path / "small.docx"
        docx_path.write_bytes(docx_bytes)

        images = _server.extract_images_from_docx(docx_path)

        assert len(images) == 0


# ---------------------------------------------------------------------------
# Tests: describe_embedded_images
# ---------------------------------------------------------------------------

class TestDescribeEmbeddedImages:
    """Tests für describe_embedded_images — Vision-API wird gemockt."""

    def test_describe_embedded_images_calls_vision(self) -> None:
        """Für jedes Bild wird analyze_with_mistral_vision aufgerufen."""
        mock_result = {
            "success": True,
            "markdown": "Ein Diagramm mit Balken",
            "tokens_prompt": 100,
            "tokens_completion": 50,
            "tokens_total": 150,
        }

        images = [
            {"name": "chart.png", "data": _PNG_100x100},
        ]

        async def _run():
            # classify_image_type wird gemockt, damit analyze_with_mistral_vision
            # genau einmal (für die Beschreibung) aufgerufen wird
            with patch.object(_server, "classify_image_type", new=AsyncMock(return_value="photo")), \
                 patch.object(_server, "analyze_with_mistral_vision", new=AsyncMock(return_value=mock_result)) as mock_vision, \
                 patch.object(_server, "detect_mimetype_from_bytes", return_value="image/png"):
                results = await _server.describe_embedded_images(images, language="de")
            return results, mock_vision

        results, mock_vision = run_async(_run())

        assert len(results) == 1
        assert results[0]["name"] == "chart.png"
        assert results[0]["description"] == "Ein Diagramm mit Balken"
        assert results[0]["tokens"] == 150
        mock_vision.assert_called_once()

    def test_describe_embedded_images_multiple(self) -> None:
        """Mehrere Bilder werden alle beschrieben."""
        mock_result = {
            "success": True,
            "markdown": "Bildbeschreibung",
            "tokens_total": 80,
        }

        images = [
            {"name": "img1.png", "data": _PNG_100x100},
            {"name": "img2.png", "data": _PNG_120x80},
        ]

        async def _run():
            with patch.object(_server, "classify_image_type", new=AsyncMock(return_value="photo")), \
                 patch.object(_server, "analyze_with_mistral_vision", new=AsyncMock(return_value=mock_result)), \
                 patch.object(_server, "detect_mimetype_from_bytes", return_value="image/png"):
                return await _server.describe_embedded_images(images, language="de")

        results = run_async(_run())

        assert len(results) == 2
        assert {r["name"] for r in results} == {"img1.png", "img2.png"}

    def test_describe_embedded_images_handles_api_failure(self) -> None:
        """Bei API-Fehler wird ein Fallback-Text eingefügt, kein Crash."""
        mock_failure = {
            "success": False,
            "error": "API nicht erreichbar",
            "error_code": "API_ERROR",
        }

        images = [{"name": "broken.png", "data": _PNG_100x100}]

        async def _run():
            with patch.object(_server, "classify_image_type", new=AsyncMock(return_value="photo")), \
                 patch.object(_server, "analyze_with_mistral_vision", new=AsyncMock(return_value=mock_failure)), \
                 patch.object(_server, "detect_mimetype_from_bytes", return_value="image/png"):
                return await _server.describe_embedded_images(images)

        results = run_async(_run())

        assert len(results) == 1
        assert results[0]["name"] == "broken.png"
        assert "nicht verfügbar" in results[0]["description"]
        assert results[0]["tokens"] == 0

    def test_describe_embedded_images_passes_language(self) -> None:
        """Die Sprache wird korrekt an analyze_with_mistral_vision übergeben."""
        mock_result = {
            "success": True,
            "markdown": "Image description in English",
            "tokens_total": 60,
        }

        images = [{"name": "img.png", "data": _PNG_100x100}]
        captured_calls = []

        async def fake_vision(data, mimetype, prompt, language):
            captured_calls.append({"language": language})
            return mock_result

        async def _run():
            # classify_image_type wird gemockt um isoliert nur den Vision-Call
            # für die Beschreibung zu erfassen
            with patch.object(_server, "classify_image_type", new=AsyncMock(return_value="photo")), \
                 patch.object(_server, "analyze_with_mistral_vision", new=fake_vision), \
                 patch.object(_server, "detect_mimetype_from_bytes", return_value="image/png"):
                await _server.describe_embedded_images(images, language="en")

        run_async(_run())

        assert len(captured_calls) == 1
        assert captured_calls[0]["language"] == "en"


# ---------------------------------------------------------------------------
# Tests: insert_image_descriptions
# ---------------------------------------------------------------------------

class TestInsertImageDescriptions:
    """Tests für insert_image_descriptions — Platzhalter-Ersetzung."""

    def test_insert_replaces_placeholder(self) -> None:
        """![image](image1.png) wird durch Blockquote ersetzt."""
        markdown = "Hier ist ein Bild:\n\n![image](image1.png)\n\nNoch Text."
        descriptions = [{"name": "image1.png", "description": "Ein Balkendiagramm"}]

        result = _server.insert_image_descriptions(markdown, descriptions)

        assert "![image](image1.png)" not in result
        assert "> **[Bild: image1.png]** Ein Balkendiagramm" in result
        assert "Noch Text." in result

    def test_insert_multiple_placeholders(self) -> None:
        """Mehrere Platzhalter werden alle ersetzt."""
        markdown = "![img](a.png)\n\nText\n\n![img](b.png)"
        descriptions = [
            {"name": "a.png", "description": "Bild A"},
            {"name": "b.png", "description": "Bild B"},
        ]

        result = _server.insert_image_descriptions(markdown, descriptions)

        assert "> **[Bild: a.png]** Bild A" in result
        assert "> **[Bild: b.png]** Bild B" in result

    def test_insert_keeps_unmatched_placeholders(self) -> None:
        """Platzhalter ohne passende Beschreibung bleiben erhalten."""
        markdown = "![img](unknown.png)"
        descriptions = []

        result = _server.insert_image_descriptions(markdown, descriptions)

        assert "![img](unknown.png)" in result

    def test_insert_with_path_in_ref(self) -> None:
        """Bild-Referenzen mit Pfad werden korrekt aufgelöst."""
        markdown = "![img](word/media/image1.png)"
        descriptions = [{"name": "image1.png", "description": "Ein Chart"}]

        result = _server.insert_image_descriptions(markdown, descriptions)

        assert "> **[Bild: image1.png]** Ein Chart" in result

    def test_insert_empty_descriptions(self) -> None:
        """Leere Descriptions-Liste: Markdown bleibt unverändert."""
        markdown = "Kein Bild hier."
        result = _server.insert_image_descriptions(markdown, [])

        assert result == markdown

    def test_insert_preserves_surrounding_text(self) -> None:
        """Text vor und nach dem Platzhalter bleibt erhalten."""
        markdown = "## Abschnitt 1\n\n![img](chart.png)\n\n## Abschnitt 2"
        descriptions = [{"name": "chart.png", "description": "Ein Liniendiagramm"}]

        result = _server.insert_image_descriptions(markdown, descriptions)

        assert "## Abschnitt 1" in result
        assert "## Abschnitt 2" in result
        assert "> **[Bild: chart.png]** Ein Liniendiagramm" in result


# ---------------------------------------------------------------------------
# Tests: describe_images Default-Wert (AC-003-5)
# ---------------------------------------------------------------------------

class TestDescribeImagesDefault:
    """describe_images ist standardmäßig False (wegen API-Kosten)."""

    def test_describe_images_default_false(self) -> None:
        """ConvertRequest.describe_images hat Default-Wert False."""
        models_path = str(Path(__file__).parent.parent)
        if models_path not in sys.path:
            sys.path.insert(0, models_path)
        # Reimport nötig falls models noch nicht geladen
        if "models" in sys.modules:
            del sys.modules["models"]
        import models  # noqa: PLC0415

        req = models.ConvertRequest(path="/data/test.docx")
        assert req.describe_images is False

    def test_describe_images_can_be_enabled(self) -> None:
        """ConvertRequest.describe_images kann auf True gesetzt werden."""
        if "models" in sys.modules:
            del sys.modules["models"]
        import models  # noqa: PLC0415

        req = models.ConvertRequest(path="/data/test.docx", describe_images=True)
        assert req.describe_images is True

    def test_metadata_images_described_field_exists(self) -> None:
        """MetaData hat das Feld images_described."""
        if "models" in sys.modules:
            del sys.modules["models"]
        import models  # noqa: PLC0415

        meta = models.MetaData(images_described=3)
        assert meta.images_described == 3

    def test_metadata_images_described_default_none(self) -> None:
        """MetaData.images_described ist standardmäßig None."""
        if "models" in sys.modules:
            del sys.modules["models"]
        import models  # noqa: PLC0415

        meta = models.MetaData()
        assert meta.images_described is None
