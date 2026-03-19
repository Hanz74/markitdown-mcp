"""
Tests für T-MKIT-014: Schema-basierte strukturierte Extraktion.

Alle Tests laufen ohne Docker-Container und ohne echte API-Calls.
Alle externen Abhängigkeiten werden per unittest.mock gemockt.
"""

import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conftest import load_server_module, run_async


# Einmal laden; alle Tests in diesem Modul teilen diese Instanz
_server = load_server_module(use_real_pil=False)
extract_structured_data = _server.extract_structured_data
EXTRACTION_TEMPLATES = _server.EXTRACTION_TEMPLATES


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _make_mistral_json_response(data: dict, tokens: int = 20) -> dict:
    """Erzeugt eine minimale Mistral-API-Antwort mit JSON im Content."""
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps(data, ensure_ascii=False)
                }
            }
        ],
        "usage": {
            "prompt_tokens": tokens // 2,
            "completion_tokens": tokens // 2,
            "total_tokens": tokens,
        },
    }


INVOICE_SCHEMA = {
    "type": "object",
    "properties": {
        "invoice_number": {"type": "string"},
        "total_amount": {"type": "number"},
        "vendor": {"type": "string"},
    },
}

INVOICE_MARKDOWN = """
# Rechnung 2024-001

**Lieferant:** ACME GmbH
**Rechnungsnummer:** 2024-001
**Gesamtbetrag:** 1.234,56 EUR
"""


# ---------------------------------------------------------------------------
# Tests: extract_structured_data()
# ---------------------------------------------------------------------------


class TestExtractStructuredData:
    """Tests für die extract_structured_data() Funktion."""

    def test_extract_structured_data_success(self):
        """AC-014-2/3: Erfolgreiche Extraktion gibt extracted-Dict zurück."""
        expected = {"invoice_number": "2024-001", "total_amount": 1234.56, "vendor": "ACME GmbH"}
        api_response = _make_mistral_json_response(expected, tokens=30)

        with patch.object(_server, "MISTRAL_API_KEY", "test-key"), \
             patch.object(_server, "call_mistral_vision_api", new=AsyncMock(return_value=api_response)):
            result = run_async(extract_structured_data(INVOICE_MARKDOWN, INVOICE_SCHEMA))

        assert result["success"] is True
        assert result["extracted"] == expected
        assert result["tokens"] == 30

    def test_extract_with_template_invoice(self):
        """AC-014-4: Invoice-Template enthält alle erwarteten Felder."""
        template = EXTRACTION_TEMPLATES["invoice"]

        assert "invoice_number" in template["properties"]
        assert "total_amount" in template["properties"]
        assert "vendor" in template["properties"]
        assert "line_items" in template["properties"]
        assert "currency" in template["properties"]
        assert "date" in template["properties"]

    def test_extract_with_template_cv(self):
        """AC-014-4: CV-Template enthält alle erwarteten Felder."""
        template = EXTRACTION_TEMPLATES["cv"]

        assert "name" in template["properties"]
        assert "email" in template["properties"]
        assert "skills" in template["properties"]
        assert "experience" in template["properties"]
        assert "education" in template["properties"]

    def test_extract_with_template_contract(self):
        """AC-014-4: Contract-Template enthält alle erwarteten Felder."""
        template = EXTRACTION_TEMPLATES["contract"]

        assert "contract_type" in template["properties"]
        assert "parties" in template["properties"]
        assert "effective_date" in template["properties"]
        assert "key_terms" in template["properties"]

    def test_extract_invalid_json_response(self):
        """Graceful Handling bei ungültigem JSON in der API-Antwort."""
        bad_response = {
            "choices": [
                {
                    "message": {
                        "content": "Das ist definitiv kein JSON und enthält keine geschweiften Klammern."
                    }
                }
            ],
            "usage": {"total_tokens": 5},
        }

        with patch.object(_server, "MISTRAL_API_KEY", "test-key"), \
             patch.object(_server, "call_mistral_vision_api", new=AsyncMock(return_value=bad_response)):
            result = run_async(extract_structured_data(INVOICE_MARKDOWN, INVOICE_SCHEMA))

        assert result["success"] is False
        assert result["extracted"] is None
        assert "error" in result

    def test_extract_validates_against_schema(self):
        """AC-014-7: Schema-Validierung wird durchgeführt (kein Fehler bei valider Antwort)."""
        valid_data = {"invoice_number": "INV-001", "total_amount": 500.0, "vendor": "Test GmbH"}
        api_response = _make_mistral_json_response(valid_data)

        # jsonschema im server-Modul patchen um Validierung zu testen
        mock_jsonschema = MagicMock()
        mock_jsonschema.validate = MagicMock()  # kein Fehler → valide
        mock_jsonschema.ValidationError = Exception

        with patch.object(_server, "MISTRAL_API_KEY", "test-key"), \
             patch.object(_server, "call_mistral_vision_api", new=AsyncMock(return_value=api_response)), \
             patch.object(_server, "JSONSCHEMA_AVAILABLE", True), \
             patch.dict(sys.modules, {"jsonschema": mock_jsonschema}):
            # Direkt jsonschema im server-Modul patchen
            original_jsonschema = getattr(_server, "jsonschema", None)
            _server.jsonschema = mock_jsonschema
            try:
                result = run_async(extract_structured_data(INVOICE_MARKDOWN, INVOICE_SCHEMA))
            finally:
                if original_jsonschema is not None:
                    _server.jsonschema = original_jsonschema
                elif hasattr(_server, "jsonschema"):
                    del _server.jsonschema

        assert result["success"] is True
        assert result["extracted"] == valid_data

    def test_extract_api_error(self):
        """Graceful Handling bei API-Fehler (Exception)."""
        with patch.object(_server, "MISTRAL_API_KEY", "test-key"), \
             patch.object(_server, "call_mistral_vision_api",
                          new=AsyncMock(side_effect=Exception("Connection refused"))):
            result = run_async(extract_structured_data(INVOICE_MARKDOWN, INVOICE_SCHEMA))

        assert result["success"] is False
        assert result["extracted"] is None
        assert "API-Fehler" in result.get("error", "") or result.get("error")

    def test_extract_default_off_no_api_key(self):
        """AC-014-2: Ohne MISTRAL_API_KEY wird kein API-Call durchgeführt."""
        with patch.object(_server, "MISTRAL_API_KEY", ""), \
             patch.object(_server, "call_mistral_vision_api", new=AsyncMock()) as mock_api:
            result = run_async(extract_structured_data(INVOICE_MARKDOWN, INVOICE_SCHEMA))
            mock_api.assert_not_called()

        assert result["success"] is False
        assert result["extracted"] is None

    def test_extract_schema_sent_in_prompt(self):
        """AC-014-2: Das Schema wird im Prompt an die API übermittelt."""
        valid_data = {"invoice_number": "X1"}
        api_response = _make_mistral_json_response(valid_data)

        with patch.object(_server, "MISTRAL_API_KEY", "test-key"), \
             patch.object(_server, "call_mistral_vision_api",
                          new=AsyncMock(return_value=api_response)) as mock_api:
            run_async(extract_structured_data(INVOICE_MARKDOWN, INVOICE_SCHEMA))

        call_payload = mock_api.call_args[0][0]
        user_content = call_payload["messages"][1]["content"]
        # Schema muss im Prompt enthalten sein
        assert "invoice_number" in user_content
        assert "JSON" in user_content or "json" in user_content.lower()

    def test_extract_json_in_markdown_codeblock(self):
        """JSON das in Markdown-Code-Block eingebettet ist, wird korrekt extrahiert."""
        wrapped_content = '```json\n{"invoice_number": "B-2024", "total_amount": 99.0}\n```'
        wrapped_response = {
            "choices": [{"message": {"content": wrapped_content}}],
            "usage": {"total_tokens": 15},
        }

        with patch.object(_server, "MISTRAL_API_KEY", "test-key"), \
             patch.object(_server, "call_mistral_vision_api", new=AsyncMock(return_value=wrapped_response)):
            result = run_async(extract_structured_data(INVOICE_MARKDOWN, INVOICE_SCHEMA))

        assert result["success"] is True
        assert result["extracted"]["invoice_number"] == "B-2024"

    def test_extract_tokens_returned(self):
        """Token-Verbrauch wird korrekt zurückgegeben."""
        api_response = _make_mistral_json_response({"vendor": "Test"}, tokens=42)

        with patch.object(_server, "MISTRAL_API_KEY", "test-key"), \
             patch.object(_server, "call_mistral_vision_api", new=AsyncMock(return_value=api_response)):
            result = run_async(extract_structured_data(INVOICE_MARKDOWN, INVOICE_SCHEMA))

        assert result["tokens"] == 42

    def test_extract_language_en_uses_english_prompt(self):
        """Bei language='en' wird ein englischer Prompt verwendet."""
        api_response = _make_mistral_json_response({"vendor": "Test Corp"})

        with patch.object(_server, "MISTRAL_API_KEY", "test-key"), \
             patch.object(_server, "call_mistral_vision_api",
                          new=AsyncMock(return_value=api_response)) as mock_api:
            run_async(extract_structured_data(INVOICE_MARKDOWN, INVOICE_SCHEMA, language="en"))

        call_payload = mock_api.call_args[0][0]
        user_content = call_payload["messages"][1]["content"]
        # Englischer User-Prompt enthält "JSON schema" oder "Extract"
        assert "Extract" in user_content or "schema" in user_content.lower()


# ---------------------------------------------------------------------------
# Tests: EXTRACTION_TEMPLATES
# ---------------------------------------------------------------------------


class TestExtractionTemplates:
    """Tests für die vordefinierten Extraktions-Templates."""

    def test_templates_all_three_exist(self):
        """AC-014-4: Alle drei vordefinierten Templates existieren."""
        assert "invoice" in EXTRACTION_TEMPLATES
        assert "cv" in EXTRACTION_TEMPLATES
        assert "contract" in EXTRACTION_TEMPLATES

    def test_templates_are_valid_json_schemas(self):
        """Alle Templates haben eine gültige JSON-Schema-Struktur (type + properties)."""
        for name, schema in EXTRACTION_TEMPLATES.items():
            assert "type" in schema, f"Template '{name}' fehlt 'type'"
            assert schema["type"] == "object", f"Template '{name}' sollte type='object' haben"
            assert "properties" in schema, f"Template '{name}' fehlt 'properties'"
            assert isinstance(schema["properties"], dict), f"Template '{name}' properties sollte dict sein"

    def test_invoice_template_line_items_is_array(self):
        """Invoice-Template: line_items ist ein Array."""
        line_items = EXTRACTION_TEMPLATES["invoice"]["properties"]["line_items"]
        assert line_items["type"] == "array"
        assert "items" in line_items

    def test_cv_template_skills_is_array_of_strings(self):
        """CV-Template: skills ist ein Array von Strings."""
        skills = EXTRACTION_TEMPLATES["cv"]["properties"]["skills"]
        assert skills["type"] == "array"
        assert skills["items"]["type"] == "string"

    def test_contract_template_parties_is_array(self):
        """Contract-Template: parties ist ein Array."""
        parties = EXTRACTION_TEMPLATES["contract"]["properties"]["parties"]
        assert parties["type"] == "array"


# ---------------------------------------------------------------------------
# Tests: /v1/templates Endpoint
# ---------------------------------------------------------------------------


class TestTemplatesEndpoint:
    """Tests für den /v1/templates GET-Endpoint (AC-014-4)."""

    def test_templates_endpoint_returns_all_templates(self):
        """AC-014-4: /v1/templates gibt alle drei vordefinierten Templates zurück."""
        # api_templates ist eine async-Funktion; direkt über EXTRACTION_TEMPLATES testen
        # (der Endpoint gibt TemplateResponse(templates=EXTRACTION_TEMPLATES) zurück)
        templates = EXTRACTION_TEMPLATES
        assert "invoice" in templates
        assert "cv" in templates
        assert "contract" in templates

    def test_templates_endpoint_response_model(self):
        """TemplateResponse enthält 'templates' Dict mit Schema-Einträgen."""
        from models import TemplateResponse
        response = TemplateResponse(templates=EXTRACTION_TEMPLATES)
        assert hasattr(response, "templates")
        assert isinstance(response.templates, dict)
        assert "invoice" in response.templates

    def test_templates_endpoint_structure(self):
        """Jedes Template im Response hat die erwartete Schema-Struktur."""
        from models import TemplateResponse
        response = TemplateResponse(templates=EXTRACTION_TEMPLATES)
        for name, schema in response.templates.items():
            assert "type" in schema
            assert "properties" in schema


# ---------------------------------------------------------------------------
# Tests: Integration in convert_auto
# ---------------------------------------------------------------------------


class TestExtractInConvertAuto:
    """Tests für die Integration von extract_schema in convert_auto()."""

    def test_extract_default_off_no_api_call_without_schema(self):
        """AC-014-2: Ohne extract_schema wird extract_structured_data NICHT aufgerufen."""
        with patch.object(_server, "extract_structured_data",
                          new=AsyncMock()) as mock_extract, \
             patch.object(_server, "convert_with_markitdown",
                          return_value={"success": True, "markdown": "# Test"}), \
             patch.object(_server, "detect_mimetype_from_bytes",
                          return_value="text/plain"):
            file_data = b"dummy content for a txt file"
            run_async(_server.convert_auto(
                file_data=file_data,
                filename="test.txt",
                source="test",
                source_type="file",
                input_meta={},
                extract_schema=None,
            ))
            mock_extract.assert_not_called()

    def test_extract_called_when_schema_provided(self):
        """AC-014-2: Mit extract_schema wird extract_structured_data aufgerufen."""
        extracted_data = {"vendor": "TestCo", "total_amount": 100.0}
        with patch.object(_server, "extract_structured_data",
                          new=AsyncMock(return_value={
                              "success": True,
                              "extracted": extracted_data,
                              "tokens": 10,
                          })) as mock_extract, \
             patch.object(_server, "convert_with_markitdown",
                          return_value={"success": True, "markdown": "# Invoice\nVendor: TestCo"}), \
             patch.object(_server, "detect_mimetype_from_bytes",
                          return_value="text/plain"):
            result = run_async(_server.convert_auto(
                file_data=b"dummy txt content",
                filename="invoice.txt",
                source="test",
                source_type="file",
                input_meta={},
                extract_schema=INVOICE_SCHEMA,
            ))

        mock_extract.assert_called_once()
        assert result.extracted == extracted_data

    def test_extract_field_in_response(self):
        """AC-014-3: Response enthält 'extracted' Feld wenn Schema gesetzt."""
        extracted_data = {"invoice_number": "INV-42"}
        with patch.object(_server, "extract_structured_data",
                          new=AsyncMock(return_value={
                              "success": True,
                              "extracted": extracted_data,
                              "tokens": 5,
                          })), \
             patch.object(_server, "convert_with_markitdown",
                          return_value={"success": True, "markdown": "# Invoice"}), \
             patch.object(_server, "detect_mimetype_from_bytes",
                          return_value="text/plain"):
            result = run_async(_server.convert_auto(
                file_data=b"invoice content here",
                filename="invoice.txt",
                source="test",
                source_type="file",
                input_meta={},
                extract_schema=INVOICE_SCHEMA,
            ))

        assert result.success is True
        assert result.extracted is not None
        assert result.extracted["invoice_number"] == "INV-42"

    def test_extract_failure_graceful(self):
        """Wenn Extraktion fehlschlägt, bleibt der convert-Erfolg erhalten (extracted=None)."""
        with patch.object(_server, "extract_structured_data",
                          new=AsyncMock(return_value={
                              "success": False,
                              "error": "API nicht erreichbar",
                              "extracted": None,
                              "tokens": 0,
                          })), \
             patch.object(_server, "convert_with_markitdown",
                          return_value={"success": True, "markdown": "# Invoice"}), \
             patch.object(_server, "detect_mimetype_from_bytes",
                          return_value="text/plain"):
            result = run_async(_server.convert_auto(
                file_data=b"invoice content here",
                filename="invoice.txt",
                source="test",
                source_type="file",
                input_meta={},
                extract_schema=INVOICE_SCHEMA,
            ))

        # Konvertierung war erfolgreich, auch wenn Extraktion fehlschlug
        assert result.success is True
        assert result.markdown is not None
        assert result.extracted is None
