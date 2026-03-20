"""
Tests für T-MKIT-021: MCP-Tools auf REST-API Parität bringen.

Prüft dass alle MCP-Tools dieselben Parameter wie die REST-Endpoints haben
und diese korrekt an die zugrunde liegenden Funktionen durchreichen.

Alle Tests laufen ohne Docker-Container und ohne echte API-Calls.
"""

import inspect
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conftest import load_server_module, run_async, PNG_100x100


# ---------------------------------------------------------------------------
# Loader mit Passthrough-FastMCP
#
# Das Standard-load_server_module() ersetzt fastmcp durch MagicMock, wodurch
# @mcp.tool() die Funktion in einen MagicMock einwickelt statt sie durchzulassen.
# Hier patchen wir fastmcp so, dass @mcp.tool() die dekorierte Funktion unverändert
# zurückgibt — damit inspect.signature() und run_async() funktionieren.
# ---------------------------------------------------------------------------


class _PassthroughMCP:
    """FastMCP-Ersatz: @mcp.tool() gibt Funktion unverändert zurück."""

    def tool(self, name=None, **kwargs):
        def decorator(f):
            return f
        return decorator

    def run(self, *args, **kwargs):
        pass


def _load_server_with_passthrough_mcp():
    """Lädt server.py mit echtem @mcp.tool()-Passthrough."""
    fastmcp_mock = MagicMock()
    fastmcp_mock.FastMCP = MagicMock(return_value=_PassthroughMCP())
    sys.modules["fastmcp"] = fastmcp_mock

    if "server" in sys.modules:
        del sys.modules["server"]

    return load_server_module()


# Einmal laden; alle Tests in diesem Modul teilen diese Instanz
_server = _load_server_with_passthrough_mcp()
mcp_convert = _server.mcp_convert
mcp_extract = _server.mcp_extract
mcp_convert_folder = _server.mcp_convert_folder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_params(func) -> dict:
    """Gibt Parameter-Dict (name → default) einer Funktion zurück."""
    sig = inspect.signature(func)
    return {name: p.default for name, p in sig.parameters.items()}


# ---------------------------------------------------------------------------
# Tests: mcp_convert Signatur
# ---------------------------------------------------------------------------


class TestMcpConvertSignature:
    """Prüft dass mcp_convert alle v2.0-Parameter in seiner Signatur hat."""

    def test_mcp_convert_has_accuracy_param(self):
        params = _get_params(mcp_convert)
        assert "accuracy" in params, "mcp_convert fehlt 'accuracy' Parameter"
        assert params["accuracy"] == "standard", "Default von 'accuracy' muss 'standard' sein"

    def test_mcp_convert_has_classify_param(self):
        params = _get_params(mcp_convert)
        assert "classify" in params, "mcp_convert fehlt 'classify' Parameter"
        assert params["classify"] is False, "Default von 'classify' muss False sein"

    def test_mcp_convert_has_classify_categories_param(self):
        params = _get_params(mcp_convert)
        assert "classify_categories" in params, "mcp_convert fehlt 'classify_categories' Parameter"
        # Optional: Default ist None oder inspect.Parameter.empty
        default = params["classify_categories"]
        assert default is None or default is inspect.Parameter.empty

    def test_mcp_convert_has_describe_images_param(self):
        params = _get_params(mcp_convert)
        assert "describe_images" in params, "mcp_convert fehlt 'describe_images' Parameter"
        assert params["describe_images"] is False

    def test_mcp_convert_has_ocr_correct_param(self):
        params = _get_params(mcp_convert)
        assert "ocr_correct" in params, "mcp_convert fehlt 'ocr_correct' Parameter"
        assert params["ocr_correct"] is False

    def test_mcp_convert_has_show_formulas_param(self):
        params = _get_params(mcp_convert)
        assert "show_formulas" in params, "mcp_convert fehlt 'show_formulas' Parameter"
        assert params["show_formulas"] is False

    def test_mcp_convert_has_chunk_param(self):
        params = _get_params(mcp_convert)
        assert "chunk" in params, "mcp_convert fehlt 'chunk' Parameter"
        assert params["chunk"] is False

    def test_mcp_convert_has_chunk_size_param(self):
        params = _get_params(mcp_convert)
        assert "chunk_size" in params, "mcp_convert fehlt 'chunk_size' Parameter"
        assert params["chunk_size"] == 512

    def test_mcp_convert_has_extract_schema_param(self):
        params = _get_params(mcp_convert)
        assert "extract_schema" in params, "mcp_convert fehlt 'extract_schema' Parameter"

    def test_mcp_convert_has_template_param(self):
        params = _get_params(mcp_convert)
        assert "template" in params, "mcp_convert fehlt 'template' Parameter"

    def test_mcp_convert_has_language_param(self):
        params = _get_params(mcp_convert)
        assert "language" in params, "mcp_convert fehlt 'language' Parameter"
        assert params["language"] == "de"

    def test_mcp_convert_has_prompt_param(self):
        params = _get_params(mcp_convert)
        assert "prompt" in params, "mcp_convert fehlt 'prompt' Parameter"

    def test_mcp_convert_backwards_compatible(self):
        """Alle alten Parameter müssen noch vorhanden sein."""
        params = _get_params(mcp_convert)
        for old_param in ["path", "base64_data", "filename", "url", "meta"]:
            assert old_param in params, f"mcp_convert: rückwärts-inkompatibler Verlust von '{old_param}'"


# ---------------------------------------------------------------------------
# Tests: mcp_convert Parameter-Durchreichung an convert_auto
# ---------------------------------------------------------------------------


class TestMcpConvertPassesParams:
    """Prüft dass mcp_convert alle Parameter korrekt an convert_auto durchreicht."""

    def test_mcp_convert_passes_accuracy_to_convert_auto(self):
        """accuracy wird an convert_auto weitergegeben."""
        with patch.object(_server, "resolve_path") as mock_resolve, \
             patch.object(_server, "convert_auto", new=AsyncMock()) as mock_convert_auto:
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_path.name = "test.txt"
            mock_path.read_bytes.return_value = b"hello"
            mock_resolve.return_value = mock_path

            mock_convert_auto.return_value = MagicMock(
                model_dump_json=MagicMock(return_value='{"success": true}')
            )

            run_async(mcp_convert(path="/data/test.txt", accuracy="high"))

        call_kwargs = mock_convert_auto.call_args.kwargs
        assert call_kwargs.get("accuracy") == "high"

    def test_mcp_convert_passes_classify_to_convert_auto(self):
        """classify wird an convert_auto weitergegeben."""
        with patch.object(_server, "resolve_path") as mock_resolve, \
             patch.object(_server, "convert_auto", new=AsyncMock()) as mock_convert_auto:
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_path.name = "test.pdf"
            mock_path.read_bytes.return_value = b"%PDF-1.4"
            mock_resolve.return_value = mock_path

            mock_convert_auto.return_value = MagicMock(
                model_dump_json=MagicMock(return_value='{"success": true}')
            )

            run_async(mcp_convert(path="/data/test.pdf", classify=True))

        call_kwargs = mock_convert_auto.call_args.kwargs
        assert call_kwargs.get("classify") is True

    def test_mcp_convert_passes_language_to_convert_auto(self):
        """language wird an convert_auto weitergegeben."""
        with patch.object(_server, "resolve_path") as mock_resolve, \
             patch.object(_server, "convert_auto", new=AsyncMock()) as mock_convert_auto:
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_path.name = "test.docx"
            mock_path.read_bytes.return_value = b"PK docx"
            mock_resolve.return_value = mock_path

            mock_convert_auto.return_value = MagicMock(
                model_dump_json=MagicMock(return_value='{"success": true}')
            )

            run_async(mcp_convert(path="/data/test.docx", language="en"))

        call_kwargs = mock_convert_auto.call_args.kwargs
        assert call_kwargs.get("language") == "en"

    def test_mcp_convert_passes_describe_images_to_convert_auto(self):
        """describe_images wird an convert_auto weitergegeben."""
        with patch.object(_server, "resolve_path") as mock_resolve, \
             patch.object(_server, "convert_auto", new=AsyncMock()) as mock_convert_auto:
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_path.name = "test.docx"
            mock_path.read_bytes.return_value = b"PK docx"
            mock_resolve.return_value = mock_path

            mock_convert_auto.return_value = MagicMock(
                model_dump_json=MagicMock(return_value='{"success": true}')
            )

            run_async(mcp_convert(path="/data/test.docx", describe_images=True))

        call_kwargs = mock_convert_auto.call_args.kwargs
        assert call_kwargs.get("describe_images") is True

    def test_mcp_convert_passes_chunk_params_to_convert_auto(self):
        """chunk und chunk_size werden an convert_auto weitergegeben."""
        with patch.object(_server, "resolve_path") as mock_resolve, \
             patch.object(_server, "convert_auto", new=AsyncMock()) as mock_convert_auto:
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_path.name = "test.txt"
            mock_path.read_bytes.return_value = b"hello"
            mock_resolve.return_value = mock_path

            mock_convert_auto.return_value = MagicMock(
                model_dump_json=MagicMock(return_value='{"success": true}')
            )

            run_async(mcp_convert(path="/data/test.txt", chunk=True, chunk_size=256))

        call_kwargs = mock_convert_auto.call_args.kwargs
        assert call_kwargs.get("chunk") is True
        assert call_kwargs.get("chunk_size") == 256

    def test_mcp_convert_passes_extract_schema_to_convert_auto(self):
        """extract_schema wird an convert_auto weitergegeben."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        with patch.object(_server, "resolve_path") as mock_resolve, \
             patch.object(_server, "convert_auto", new=AsyncMock()) as mock_convert_auto:
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_path.name = "test.pdf"
            mock_path.read_bytes.return_value = b"%PDF"
            mock_resolve.return_value = mock_path

            mock_convert_auto.return_value = MagicMock(
                model_dump_json=MagicMock(return_value='{"success": true}')
            )

            run_async(mcp_convert(path="/data/test.pdf", extract_schema=schema))

        call_kwargs = mock_convert_auto.call_args.kwargs
        assert call_kwargs.get("extract_schema") == schema

    def test_mcp_convert_template_resolves_to_schema(self):
        """template wird zu effective_schema aufgelöst und an convert_auto übergeben."""
        fake_schema = {"type": "object", "properties": {"betrag": {"type": "number"}}}

        with patch.object(_server, "EXTRACTION_TEMPLATES", {"invoice": fake_schema}), \
             patch.object(_server, "resolve_path") as mock_resolve, \
             patch.object(_server, "convert_auto", new=AsyncMock()) as mock_convert_auto:
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_path.name = "rechnung.pdf"
            mock_path.read_bytes.return_value = b"%PDF"
            mock_resolve.return_value = mock_path

            mock_convert_auto.return_value = MagicMock(
                model_dump_json=MagicMock(return_value='{"success": true}')
            )

            run_async(mcp_convert(path="/data/rechnung.pdf", template="invoice"))

        call_kwargs = mock_convert_auto.call_args.kwargs
        assert call_kwargs.get("extract_schema") == fake_schema

    def test_mcp_convert_unknown_template_returns_error(self):
        """Unbekanntes template gibt Fehler-JSON zurück (kein Exception)."""
        with patch.object(_server, "EXTRACTION_TEMPLATES", {"invoice": {}}):
            result = run_async(mcp_convert(path="/data/test.pdf", template="nonexistent"))

        data = json.loads(result)
        assert data["success"] is False
        assert "nonexistent" in data["error"]

    def test_mcp_convert_base64_passes_all_params(self):
        """Bei base64-Eingabe werden alle neuen Parameter an convert_auto gereicht."""
        import base64
        b64 = base64.b64encode(PNG_100x100).decode()

        with patch.object(_server, "convert_auto", new=AsyncMock()) as mock_convert_auto:
            mock_convert_auto.return_value = MagicMock(
                model_dump_json=MagicMock(return_value='{"success": true}')
            )

            run_async(mcp_convert(
                base64_data=b64,
                filename="test.png",
                accuracy="high",
                ocr_correct=True,
                language="en",
            ))

        call_kwargs = mock_convert_auto.call_args.kwargs
        assert call_kwargs.get("accuracy") == "high"
        assert call_kwargs.get("ocr_correct") is True
        assert call_kwargs.get("language") == "en"


# ---------------------------------------------------------------------------
# Tests: mcp_convert_folder Signatur
# ---------------------------------------------------------------------------


class TestMcpConvertFolderSignature:
    """Prüft dass mcp_convert_folder den language-Parameter hat."""

    def test_mcp_convert_folder_has_language(self):
        params = _get_params(mcp_convert_folder)
        assert "language" in params, "mcp_convert_folder fehlt 'language' Parameter"
        assert params["language"] == "de"

    def test_mcp_convert_folder_backwards_compatible(self):
        """Alte Parameter müssen noch vorhanden sein."""
        params = _get_params(mcp_convert_folder)
        assert "path" in params
        assert "meta" in params

    def test_mcp_convert_folder_passes_language(self):
        """language wird an convert_folder_contents weitergegeben."""
        with patch.object(_server, "resolve_path") as mock_resolve, \
             patch.object(_server, "convert_folder_contents", new=AsyncMock()) as mock_folder:
            mock_folder.return_value = MagicMock(
                model_dump_json=MagicMock(return_value='{"success": true}')
            )
            mock_resolve.return_value = MagicMock()

            run_async(mcp_convert_folder(path="/data/docs", language="en"))

        call_kwargs = mock_folder.call_args.kwargs
        assert call_kwargs.get("language") == "en"

    def test_mcp_convert_folder_default_language_de(self):
        """Default-Sprache ist 'de'."""
        with patch.object(_server, "resolve_path") as mock_resolve, \
             patch.object(_server, "convert_folder_contents", new=AsyncMock()) as mock_folder:
            mock_folder.return_value = MagicMock(
                model_dump_json=MagicMock(return_value='{"success": true}')
            )
            mock_resolve.return_value = MagicMock()

            run_async(mcp_convert_folder(path="/data/docs"))

        call_kwargs = mock_folder.call_args.kwargs
        assert call_kwargs.get("language") == "de"


# ---------------------------------------------------------------------------
# Tests: mcp_extract Signatur
# ---------------------------------------------------------------------------


class TestMcpExtractSignature:
    """Prüft dass mcp_extract die neuen Parameter hat."""

    def test_mcp_extract_has_accuracy(self):
        params = _get_params(mcp_extract)
        assert "accuracy" in params, "mcp_extract fehlt 'accuracy' Parameter"
        assert params["accuracy"] == "standard"

    def test_mcp_extract_has_ocr_correct(self):
        params = _get_params(mcp_extract)
        assert "ocr_correct" in params, "mcp_extract fehlt 'ocr_correct' Parameter"
        assert params["ocr_correct"] is False

    def test_mcp_extract_has_classify(self):
        params = _get_params(mcp_extract)
        assert "classify" in params, "mcp_extract fehlt 'classify' Parameter"
        assert params["classify"] is False

    def test_mcp_extract_schema_is_optional(self):
        """extract_schema muss Optional sein (None als Default)."""
        params = _get_params(mcp_extract)
        assert "extract_schema" in params
        default = params["extract_schema"]
        assert default is None or default is inspect.Parameter.empty

    def test_mcp_extract_backwards_compatible(self):
        """Alle alten Parameter müssen noch vorhanden sein."""
        params = _get_params(mcp_extract)
        for old_param in ["path", "base64_data", "filename", "url", "template", "language", "meta"]:
            assert old_param in params, f"mcp_extract: rückwärts-inkompatibler Verlust von '{old_param}'"

    def test_mcp_extract_no_schema_no_template_returns_error(self):
        """Ohne extract_schema und ohne template → Fehler-JSON."""
        result = run_async(mcp_extract(path="/data/test.pdf"))
        data = json.loads(result)
        assert data["success"] is False
        assert "extract_schema" in data["error"] or "template" in data["error"]

    def test_mcp_extract_passes_accuracy_to_convert_auto(self):
        """accuracy wird an convert_auto durchgereicht."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        with patch.object(_server, "resolve_path") as mock_resolve, \
             patch.object(_server, "convert_auto", new=AsyncMock()) as mock_convert_auto:
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_path.name = "test.pdf"
            mock_path.read_bytes.return_value = b"%PDF"
            mock_resolve.return_value = mock_path

            mock_convert_auto.return_value = MagicMock(
                model_dump_json=MagicMock(return_value='{"success": true}')
            )

            run_async(mcp_extract(
                extract_schema=schema,
                path="/data/test.pdf",
                accuracy="high",
            ))

        call_kwargs = mock_convert_auto.call_args.kwargs
        assert call_kwargs.get("accuracy") == "high"

    def test_mcp_extract_passes_ocr_correct_to_convert_auto(self):
        """ocr_correct wird an convert_auto durchgereicht."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        with patch.object(_server, "resolve_path") as mock_resolve, \
             patch.object(_server, "convert_auto", new=AsyncMock()) as mock_convert_auto:
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_path.name = "test.pdf"
            mock_path.read_bytes.return_value = b"%PDF"
            mock_resolve.return_value = mock_path

            mock_convert_auto.return_value = MagicMock(
                model_dump_json=MagicMock(return_value='{"success": true}')
            )

            run_async(mcp_extract(
                extract_schema=schema,
                path="/data/test.pdf",
                ocr_correct=True,
            ))

        call_kwargs = mock_convert_auto.call_args.kwargs
        assert call_kwargs.get("ocr_correct") is True

    def test_mcp_extract_passes_classify_to_convert_auto(self):
        """classify wird an convert_auto durchgereicht."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        with patch.object(_server, "resolve_path") as mock_resolve, \
             patch.object(_server, "convert_auto", new=AsyncMock()) as mock_convert_auto:
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_path.name = "test.pdf"
            mock_path.read_bytes.return_value = b"%PDF"
            mock_resolve.return_value = mock_path

            mock_convert_auto.return_value = MagicMock(
                model_dump_json=MagicMock(return_value='{"success": true}')
            )

            run_async(mcp_extract(
                extract_schema=schema,
                path="/data/test.pdf",
                classify=True,
            ))

        call_kwargs = mock_convert_auto.call_args.kwargs
        assert call_kwargs.get("classify") is True
