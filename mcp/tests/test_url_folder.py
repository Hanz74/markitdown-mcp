"""
Tests für T-MKIT-022: URL-Pfad + Folder + Extract konsistent durch convert_auto() routen.

- test_url_goes_through_convert_auto: Nicht-HTML-URL nutzt convert_auto
- test_folder_passes_language: language wird durchgereicht
- test_extract_has_accuracy: ExtractRequest hat accuracy
- test_accuracy_high_activates_ocr_correct_for_images
- test_audio_has_accuracy_mode_in_meta
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from conftest import load_server_module, run_async, PNG_100x100


# ---------------------------------------------------------------------------
# Server-Module
#
# Zwei Instanzen:
# 1. _server: Standard-Modul (für convert_auto, convert_folder_contents direkt)
# 2. _server_api: Modul mit app.post=pass-through → api_convert/api_extract sind echte Coroutinen
# ---------------------------------------------------------------------------

_server = load_server_module(use_real_pil=False)
convert_auto = _server.convert_auto
convert_folder_contents = _server.convert_folder_contents


def _load_api_server():
    """Lädt server.py mit app.post/get als Pass-Through-Dekoratoren."""
    fastapi_mock = MagicMock()
    app_mock = MagicMock()
    app_mock.post = lambda *a, **kw: (lambda f: f)
    app_mock.get = lambda *a, **kw: (lambda f: f)
    # exception_handler auch als pass-through
    app_mock.exception_handler = lambda *a, **kw: (lambda f: f)
    fastapi_mock.FastAPI = MagicMock(return_value=app_mock)
    fastapi_mock.HTTPException = Exception
    fastapi_mock.Request = MagicMock()

    return load_server_module(
        use_real_pil=False,
        extra_patches={
            "fastapi": fastapi_mock,
            "fastapi.exceptions": MagicMock(),
            "fastapi.responses": MagicMock(),
        },
    )


_server_api = _load_api_server()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vision_response(content: str, tokens_total: int = 50) -> dict:
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 30,
            "total_tokens": tokens_total,
        },
    }


# ---------------------------------------------------------------------------
# Test: URL-Pfad → convert_auto für Nicht-HTML
# ---------------------------------------------------------------------------

class TestUrlGoesthroughConvertAuto:
    """
    Nicht-HTML-URLs (z.B. PDFs) sollen durch convert_auto() geleitet werden
    um vollständige Pipeline-Features zu erhalten (accuracy, quality_score, pipeline_steps).
    """

    def test_url_non_html_uses_convert_auto(self):
        """
        Bei einer PDF-URL ruft api_convert() intern convert_auto() auf.
        Der Response enthält accuracy_mode und pipeline_steps (kommen nur von convert_auto).
        """
        from models import ConvertRequest

        pdf_bytes = b"%PDF-1.4 test content"

        # HEAD-Antwort: content-type application/pdf
        head_resp = MagicMock()
        head_resp.headers = {"content-type": "application/pdf"}

        # GET-Antwort: PDF-Inhalt
        get_resp = MagicMock()
        get_resp.headers = {"content-type": "application/pdf"}
        get_resp.content = pdf_bytes
        get_resp.raise_for_status = MagicMock()

        client = AsyncMock()
        client.head = AsyncMock(return_value=head_resp)
        client.get = AsyncMock(return_value=get_resp)
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)

        httpx_mock = MagicMock()
        httpx_mock.AsyncClient = MagicMock(return_value=client)

        markitdown_result = {"success": True, "markdown": "# PDF Content"}

        with patch.object(_server_api, "httpx", httpx_mock), \
             patch.object(_server_api, "detect_mimetype_from_bytes", return_value="application/pdf"), \
             patch.object(_server_api, "is_scanned_pdf", return_value=False), \
             patch.object(_server_api, "convert_with_markitdown", return_value=markitdown_result), \
             patch("pathlib.Path.write_bytes", return_value=None), \
             patch("pathlib.Path.unlink", return_value=None):
            request = ConvertRequest(url="https://example.com/doc.pdf")
            result = run_async(_server_api.api_convert(request))

        assert result.success is True
        assert result.markdown == "# PDF Content"
        # convert_auto setzt pipeline_steps — fehlt beim alten URL-Pfad
        assert result.meta.pipeline_steps is not None, \
            "pipeline_steps fehlt — URL nutzt nicht convert_auto"
        assert len(result.meta.pipeline_steps) > 0
        # convert_auto setzt accuracy_mode
        assert result.meta.accuracy_mode == "standard"

    def test_url_html_uses_convert_url_fallback(self):
        """
        HTML-URLs bleiben beim alten convert_url()-Pfad (markitdown ist besser für HTML).
        """
        from models import ConvertRequest

        head_resp = MagicMock()
        head_resp.headers = {"content-type": "text/html; charset=utf-8"}

        client = AsyncMock()
        client.head = AsyncMock(return_value=head_resp)
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)

        httpx_mock = MagicMock()
        httpx_mock.AsyncClient = MagicMock(return_value=client)

        convert_url_result = {"success": True, "markdown": "# HTML Page", "title": "Test"}

        with patch.object(_server_api, "httpx", httpx_mock), \
             patch.object(_server_api, "convert_url", new=AsyncMock(return_value=convert_url_result)):
            request = ConvertRequest(url="https://example.com/page.html")
            result = run_async(_server_api.api_convert(request))

        assert result.success is True
        assert result.markdown == "# HTML Page"
        assert result.meta.title == "Test"

    def test_url_download_failure_returns_error(self):
        """
        Wenn der URL-Download fehlschlägt, gibt api_convert() einen Fehler zurück.
        """
        from models import ConvertRequest

        head_resp = MagicMock()
        head_resp.headers = {"content-type": "application/pdf"}

        client = AsyncMock()
        client.head = AsyncMock(return_value=head_resp)
        client.get = AsyncMock(side_effect=Exception("Connection refused"))
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)

        httpx_mock = MagicMock()
        httpx_mock.AsyncClient = MagicMock(return_value=client)

        with patch.object(_server_api, "httpx", httpx_mock):
            request = ConvertRequest(url="https://example.com/fail.pdf")
            result = run_async(_server_api.api_convert(request))

        assert result.success is False
        assert result.error.code == "CONVERSION_FAILED"


# ---------------------------------------------------------------------------
# Test: Folder → language wird durchgereicht
# ---------------------------------------------------------------------------

class TestFolderPassesLanguage:
    """
    convert_folder_contents() soll language an convert_auto() durchreichen.
    """

    def test_folder_passes_language_to_convert_auto(self, tmp_path):
        """
        Wenn convert_folder_contents mit language='en' aufgerufen wird,
        wird 'en' an convert_auto() durchgereicht.
        """
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"Hello world")

        called_languages = []

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.markdown = "# Test"
        mock_result.meta = MagicMock(tokens_total=None, vision_used=None)
        mock_result.error = None

        async def capturing_convert_auto(**kwargs):
            called_languages.append(kwargs.get("language"))
            return mock_result

        with patch.object(_server, "convert_auto", new=AsyncMock(side_effect=capturing_convert_auto)):
            result = run_async(convert_folder_contents(
                folder_path=tmp_path,
                input_meta={},
                language="en",
            ))

        assert len(called_languages) >= 1
        assert all(lang == "en" for lang in called_languages), \
            f"Erwartet language='en', bekommen: {called_languages}"

    def test_folder_default_language_is_de(self, tmp_path):
        """
        Ohne explizite language-Angabe wird 'de' genutzt (Default).
        """
        test_file = tmp_path / "doc.txt"
        test_file.write_bytes(b"Hallo Welt")

        called_languages = []

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.markdown = "# Hallo"
        mock_result.meta = MagicMock(tokens_total=None, vision_used=None)
        mock_result.error = None

        async def capturing_convert_auto(**kwargs):
            called_languages.append(kwargs.get("language"))
            return mock_result

        with patch.object(_server, "convert_auto", new=AsyncMock(side_effect=capturing_convert_auto)):
            result = run_async(convert_folder_contents(
                folder_path=tmp_path,
                input_meta={},
                # kein language-Parameter → Default 'de'
            ))

        assert all(lang == "de" for lang in called_languages), \
            f"Default language sollte 'de' sein, bekommen: {called_languages}"

    def test_api_convert_folder_passes_language(self, tmp_path):
        """
        api_convert_folder() reicht request.language an convert_folder_contents() durch.
        """
        from models import ConvertFolderRequest

        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"Hello")

        captured_langs = []

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.markdown = "# OK"
        mock_result.meta = MagicMock(tokens_total=None, vision_used=None)
        mock_result.error = None

        async def capturing_convert_auto(**kwargs):
            captured_langs.append(kwargs.get("language"))
            return mock_result

        with patch.object(_server_api, "convert_auto", new=AsyncMock(side_effect=capturing_convert_auto)), \
             patch.object(_server_api, "resolve_path", return_value=tmp_path):
            request = ConvertFolderRequest(path=str(tmp_path), language="en")
            result = run_async(_server_api.api_convert_folder(request))

        assert all(lang == "en" for lang in captured_langs), \
            f"api_convert_folder reicht language='en' nicht durch: {captured_langs}"

    def test_folder_request_has_no_recursive_field(self):
        """
        ConvertFolderRequest hat kein recursive-Feld mehr (toter Code entfernt).
        """
        from models import ConvertFolderRequest

        fields = ConvertFolderRequest.model_fields
        assert "recursive" not in fields, \
            "recursive-Feld wurde nicht aus ConvertFolderRequest entfernt"

    def test_folder_request_valid_without_recursive(self):
        """
        ConvertFolderRequest kann ohne recursive instanziiert werden.
        """
        from models import ConvertFolderRequest

        req = ConvertFolderRequest(path="/data/test", language="en")
        assert req.path == "/data/test"
        assert req.language == "en"


# ---------------------------------------------------------------------------
# Test: ExtractRequest hat accuracy-Feld
# ---------------------------------------------------------------------------

class TestExtractHasAccuracy:
    """
    ExtractRequest soll accuracy, ocr_correct, describe_images, classify haben.
    """

    def test_extract_request_has_accuracy_field(self):
        """ExtractRequest hat accuracy mit Default 'standard'."""
        from models import ExtractRequest

        req = ExtractRequest(path="/data/test.pdf", extract_schema={"type": "object"})
        assert hasattr(req, "accuracy")
        assert req.accuracy == "standard"

    def test_extract_request_has_ocr_correct_field(self):
        """ExtractRequest hat ocr_correct mit Default False."""
        from models import ExtractRequest

        req = ExtractRequest(path="/data/test.pdf", extract_schema={"type": "object"})
        assert hasattr(req, "ocr_correct")
        assert req.ocr_correct is False

    def test_extract_request_has_describe_images_field(self):
        """ExtractRequest hat describe_images mit Default False."""
        from models import ExtractRequest

        req = ExtractRequest(path="/data/test.pdf", extract_schema={"type": "object"})
        assert hasattr(req, "describe_images")
        assert req.describe_images is False

    def test_extract_request_has_classify_field(self):
        """ExtractRequest hat classify mit Default False."""
        from models import ExtractRequest

        req = ExtractRequest(path="/data/test.pdf", extract_schema={"type": "object"})
        assert hasattr(req, "classify")
        assert req.classify is False

    def test_extract_request_accuracy_passed_to_convert(self):
        """
        api_extract() übergibt accuracy/ocr_correct/describe_images/classify
        an den intern erstellten ConvertRequest.
        """
        from models import ExtractRequest

        convert_req_captured = []

        mock_response = MagicMock()
        mock_response.success = True
        mock_response.markdown = "# Test"
        mock_response.meta = MagicMock()
        mock_response.extracted = None
        mock_response.chunks = None

        async def capturing_api_convert(req):
            convert_req_captured.append(req)
            return mock_response

        with patch.object(_server_api, "api_convert", new=AsyncMock(side_effect=capturing_api_convert)):
            request = ExtractRequest(
                path="/data/test.pdf",
                extract_schema={"type": "object", "properties": {"name": {"type": "string"}}},
                accuracy="high",
                ocr_correct=True,
                describe_images=True,
                classify=True,
            )
            result = run_async(_server_api.api_extract(request))

        assert len(convert_req_captured) == 1
        captured = convert_req_captured[0]
        assert captured.accuracy == "high"
        assert captured.ocr_correct is True
        assert captured.describe_images is True
        assert captured.classify is True


# ---------------------------------------------------------------------------
# Test: accuracy="high" aktiviert ocr_correct bei Bildern
# ---------------------------------------------------------------------------

class TestAccuracyHighActivatesOcrCorrectForImages:
    """
    Bei accuracy='high' soll ocr_correct für Bilder automatisch aktiv sein,
    auch wenn ocr_correct=False explizit übergeben wird.
    """

    def test_accuracy_high_activates_ocr_correct_vision(self):
        """
        convert_auto() mit Bild + accuracy='high' ruft correct_ocr_text() auf,
        auch wenn ocr_correct=False.
        """
        vision_resp = _make_vision_response("# OCR Text")
        dual_pass_resp = _make_vision_response("# Korrigiert")
        correction_result = {
            "success": True,
            "corrected_text": "# Bereinigter Text",
            "corrections_count": 3,
        }

        ocr_correct_called = []

        async def capturing_correct_ocr_text(text, language="de"):
            ocr_correct_called.append(True)
            return correction_result

        with patch.object(_server, "MISTRAL_API_KEY", "test-key"), \
             patch.object(_server, "call_mistral_vision_api", new=AsyncMock(
                 side_effect=[vision_resp, dual_pass_resp]
             )), \
             patch.object(_server, "correct_ocr_text", new=AsyncMock(side_effect=capturing_correct_ocr_text)):
            result = run_async(convert_auto(
                file_data=PNG_100x100,
                filename="test.png",
                source="test.png",
                source_type="base64",
                input_meta={},
                language="de",
                accuracy="high",
                ocr_correct=False,  # explizit False — trotzdem soll correct_ocr_text laufen
            ))

        assert result.success is True
        assert len(ocr_correct_called) == 1, \
            "correct_ocr_text() wurde nicht aufgerufen obwohl accuracy='high'"
        assert result.meta.ocr_corrected is True

    def test_accuracy_standard_does_not_activate_ocr_correct_vision(self):
        """
        Im Standard-Modus wird correct_ocr_text NICHT aufgerufen wenn ocr_correct=False.
        """
        vision_resp = _make_vision_response("# Standard Text")

        ocr_correct_called = []

        async def capturing_correct_ocr_text(text, language="de"):
            ocr_correct_called.append(True)
            return {"success": True, "corrected_text": text, "corrections_count": 0}

        with patch.object(_server, "MISTRAL_API_KEY", "test-key"), \
             patch.object(_server, "call_mistral_vision_api", new=AsyncMock(return_value=vision_resp)), \
             patch.object(_server, "correct_ocr_text", new=AsyncMock(side_effect=capturing_correct_ocr_text)):
            result = run_async(convert_auto(
                file_data=PNG_100x100,
                filename="test.png",
                source="test.png",
                source_type="base64",
                input_meta={},
                language="de",
                accuracy="standard",
                ocr_correct=False,
            ))

        assert result.success is True
        assert len(ocr_correct_called) == 0, \
            "correct_ocr_text() wurde aufgerufen obwohl accuracy='standard' und ocr_correct=False"


# ---------------------------------------------------------------------------
# Test: Audio/Video hat accuracy_mode in Meta
# ---------------------------------------------------------------------------

class TestAudioHasAccuracyModeInMeta:
    """
    Audio-Dateien sollen accuracy_mode in der Meta zurückgeben.
    """

    def test_audio_standard_accuracy_mode_in_meta(self):
        """
        Audio-Transkription setzt accuracy_mode='standard' in Meta.
        """
        transcription_result = {
            "success": True,
            "text": "Das ist ein Test.",
            "language": "de",
            "duration": 5.0,
            "model_size": "base",
        }
        # WAV magic bytes → libmagic erkennt als audio
        fake_audio = b"RIFF" + b"\x00" * 40

        with patch.object(_server, "transcribe_audio", return_value=transcription_result), \
             patch.object(_server, "detect_mimetype_from_bytes", return_value="audio/wav"), \
             patch.object(_server, "TEMP_DIR", Path("/tmp/markitdown_test")), \
             patch("pathlib.Path.write_bytes", return_value=None), \
             patch("pathlib.Path.unlink", return_value=None):
            result = run_async(convert_auto(
                file_data=fake_audio,
                filename="test.wav",
                source="test.wav",
                source_type="file",
                input_meta={},
                language="de",
                accuracy="standard",
            ))

        assert result.success is True, f"Erwartet success=True, bekommen: {result}"
        assert result.meta.accuracy_mode == "standard", \
            f"accuracy_mode erwartet 'standard', bekommen: {result.meta.accuracy_mode}"

    def test_audio_high_accuracy_mode_in_meta(self):
        """
        Audio-Transkription mit accuracy='high' setzt accuracy_mode='high' in Meta.
        """
        transcription_result = {
            "success": True,
            "text": "Hohe Genauigkeit.",
            "language": "de",
            "duration": 3.0,
            "model_size": "base",
        }
        fake_audio = b"RIFF" + b"\x00" * 40

        with patch.object(_server, "transcribe_audio", return_value=transcription_result), \
             patch.object(_server, "detect_mimetype_from_bytes", return_value="audio/wav"), \
             patch.object(_server, "TEMP_DIR", Path("/tmp/markitdown_test")), \
             patch("pathlib.Path.write_bytes", return_value=None), \
             patch("pathlib.Path.unlink", return_value=None):
            result = run_async(convert_auto(
                file_data=fake_audio,
                filename="test.wav",
                source="test.wav",
                source_type="file",
                input_meta={},
                language="de",
                accuracy="high",
            ))

        assert result.success is True, f"Erwartet success=True, bekommen: {result}"
        assert result.meta.accuracy_mode == "high", \
            f"accuracy_mode erwartet 'high', bekommen: {result.meta.accuracy_mode}"
