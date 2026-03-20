"""
MarkItDown MCP Server - Datenmodelle und Schemas
"""

from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator


# =============================================================================
# Basis-Schemas
# =============================================================================

class MetaData(BaseModel):
    """
    Metadaten-Container.

    Kombiniert:
    - Pass-through Daten vom Client (beliebige Key-Value-Paare)
    - Angereicherte Daten aus der Verarbeitung
    """
    model_config = ConfigDict(extra="allow")  # Erlaubt beliebige zusätzliche Felder

    # Automatisch hinzugefügte Felder
    source: Optional[str] = Field(None, description="Quelle (Dateipfad, URL, 'base64')")
    source_type: Optional[str] = Field(None, description="Typ: 'file', 'base64', 'url'")
    format: Optional[str] = Field(None, description="Dateiformat (pdf, jpg, docx, ...)")
    size_bytes: Optional[int] = Field(None, description="Dateigröße in Bytes")
    processed_at: Optional[str] = Field(None, description="Verarbeitungszeitpunkt (ISO 8601)")
    duration_ms: Optional[int] = Field(None, description="Verarbeitungsdauer in Millisekunden")

    # Bild-spezifisch
    width: Optional[int] = Field(None, description="Bildbreite in Pixel")
    height: Optional[int] = Field(None, description="Bildhöhe in Pixel")
    resized: Optional[bool] = Field(None, description="Wurde das Bild verkleinert?")
    original_width: Optional[int] = Field(None, description="Ursprüngliche Breite vor Resize")
    original_height: Optional[int] = Field(None, description="Ursprüngliche Höhe vor Resize")

    # Vision-spezifisch
    vision_used: Optional[bool] = Field(None, description="Wurde Mistral Vision verwendet?")
    vision_model: Optional[str] = Field(None, description="Verwendetes Vision-Modell")
    ocr_model: Optional[str] = Field(None, description="Verwendetes OCR-Modell (Mistral OCR 3)")
    tokens_prompt: Optional[int] = Field(None, description="Input-Tokens")
    tokens_completion: Optional[int] = Field(None, description="Output-Tokens")
    tokens_total: Optional[int] = Field(None, description="Gesamt-Tokens")

    # URL-spezifisch
    url: Optional[str] = Field(None, description="Quell-URL")
    content_type: Optional[str] = Field(None, description="Content-Type der URL")

    # PDF-spezifisch
    pages: Optional[int] = Field(None, description="Anzahl Seiten (PDF)")
    author: Optional[str] = Field(None, description="Autor (PDF)")
    title: Optional[str] = Field(None, description="Titel (PDF)")

    # Embedded-Images-spezifisch
    images_described: Optional[int] = Field(None, description="Anzahl beschriebener eingebetteter Bilder")

    # Klassifizierungs-spezifisch
    document_type: Optional[str] = Field(None, description="Erkannter Dokumenttyp")
    document_type_confidence: Optional[float] = Field(None, description="Konfidenz der Klassifizierung (0.0-1.0)")

    # OCR-Korrektur-spezifisch
    ocr_corrected: Optional[bool] = Field(None, description="Wurde OCR-Nachkorrektur via LLM durchgeführt?")
    ocr_corrections_count: Optional[int] = Field(None, description="Anzahl der durch OCR-Korrektur behobenen Fehler")

    # Quality-Scoring-spezifisch (FR-MKIT-010)
    quality_score: Optional[float] = Field(None, description="Qualitäts-Score des konvertierten Textes (0.0-1.0)")
    quality_grade: Optional[str] = Field(None, description="Qualitäts-Bewertung: 'poor', 'fair', 'good', 'excellent'")

    # High-Accuracy-Pipeline-spezifisch (T-MKIT-020)
    accuracy_mode: Optional[str] = Field(None, description="Verwendeter Accuracy-Modus: 'standard' oder 'high'")
    pipeline_steps: Optional[list[str]] = Field(None, description="Liste der durchgeführten Pipeline-Schritte (z.B. ['ocr', 'ocr_correction', 'dual_pass_validation', 'schema_extraction'])")

    # Audio/Video-spezifisch (FR-MKIT-006)
    language: Optional[str] = Field(None, description="Erkannte Sprache (z.B. 'de', 'en') bei Audio/Video-Transkription")
    duration_seconds: Optional[float] = Field(None, description="Dauer der Audio/Video-Datei in Sekunden")
    whisper_model: Optional[str] = Field(None, description="Verwendete Whisper-Modell-Größe (z.B. 'base', 'small')")


class ErrorDetail(BaseModel):
    """Fehlerdetails für einheitliche Fehlerantworten."""
    code: str = Field(..., description="Fehlercode (z.B. 'FILE_NOT_FOUND')")
    message: str = Field(..., description="Menschenlesbare Fehlermeldung")
    details: Optional[dict[str, Any]] = Field(None, description="Zusätzliche Fehlerdetails")


# =============================================================================
# Response-Schemas
# =============================================================================

class ConvertResponse(BaseModel):
    """
    Einheitliche Antwort für alle Konvertierungen.

    Bei Erfolg:
        success=True, markdown=<content>, meta=<enriched>

    Bei Fehler:
        success=False, error=<ErrorDetail>, meta=<pass-through>
    """
    success: bool = Field(..., description="War die Konvertierung erfolgreich?")
    markdown: Optional[str] = Field(None, description="Konvertierter Markdown-Inhalt")
    error: Optional[ErrorDetail] = Field(None, description="Fehlerdetails (nur bei success=False)")
    meta: MetaData = Field(default_factory=MetaData, description="Metadaten")
    extracted: Optional[dict[str, Any]] = Field(None, description="Strukturiert extrahierte Daten (nur wenn extract_schema gesetzt)")
    chunks: Optional[list[dict[str, Any]]] = Field(None, description="RAG-Chunks (nur wenn chunk=True gesetzt, FR-MKIT-011)")


class HealthResponse(BaseModel):
    """Antwort für Health-Check Endpoint."""
    status: str = Field(..., description="Status: 'ok' oder 'error'")
    version: str = Field(..., description="Server-Version")
    meta: dict[str, Any] = Field(default_factory=dict, description="Zusätzliche Infos")


# =============================================================================
# Request-Schemas
# =============================================================================

class ConvertRequest(BaseModel):
    """
    Einheitlicher Request für Konvertierungen.

    Unterstützt drei Eingabe-Modi (genau einer muss gesetzt sein):
    - path: Dateipfad im Container
    - base64 + filename: Base64-kodierte Datei
    - url: URL zu Datei/Webseite
    """
    model_config = ConfigDict(str_strip_whitespace=True)

    # Eingabe-Modi (einer von dreien)
    path: Optional[str] = Field(None, description="Dateipfad (relativ zu /data oder absolut)")
    base64: Optional[str] = Field(None, description="Base64-kodierte Datei")
    filename: Optional[str] = Field(None, description="Dateiname (erforderlich bei base64)")
    url: Optional[str] = Field(None, description="URL zu Datei oder Webseite")

    # Optionen
    password: Optional[str] = Field(None, description="Passwort für geschützte PDFs")  # Reserved for future use
    output_format: str = Field("markdown", description="Ausgabeformat: 'markdown', 'text', 'html'")  # Reserved for future use

    # Vision-spezifisch
    prompt: Optional[str] = Field(None, description="Custom Prompt für Vision-Analyse")
    language: str = Field("de", description="Sprache für Vision-Antwort: 'de', 'en'")

    # Embedded-Images-Option
    describe_images: bool = Field(False, description="Eingebettete Bilder in DOCX/PPTX durch Pixtral beschreiben (Default: false, wegen API-Kosten)")

    # Klassifizierungs-Option
    classify: bool = Field(False, description="Dokumenttyp automatisch klassifizieren")
    classify_categories: Optional[list[str]] = Field(None, description="Custom Klassifizierungs-Kategorien")

    # Extraktion
    extract_schema: Optional[dict[str, Any]] = Field(None, description="JSON-Schema für strukturierte Daten-Extraktion via LLM")
    template: Optional[str] = Field(None, description="Vordefinierter Template-Name: 'invoice', 'cv', 'contract'")

    # OCR-Korrektur
    ocr_correct: bool = Field(False, description="OCR-Nachkorrektur via LLM aktivieren (nur bei Bild/Scanned-PDF-Pfaden, Default: false)")

    # Accuracy-Modus (T-MKIT-020)
    accuracy: str = Field("standard", description="Accuracy-Modus: 'standard' oder 'high'. High aktiviert OCR-Correction und optional Dual-Pass Vision-Validierung")

    @field_validator("accuracy")
    @classmethod
    def validate_accuracy(cls, v: str) -> str:
        if v not in ("standard", "high"):
            raise ValueError(f"accuracy must be 'standard' or 'high', got '{v}'")
        return v

    # Excel-spezifisch (FR-MKIT-007)
    show_formulas: bool = Field(False, description="Excel-Formeln im Output anzeigen (z.B. 42 [=SUM(A1:A10)])")

    # Smart Chunking (FR-MKIT-011)
    chunk: bool = Field(False, description="Aktiviert Smart Chunking für RAG (Default: false)")
    chunk_size: int = Field(512, ge=1, description="Maximale Chunk-Größe in Tokens (Heuristik: len/4), Default: 512")

    # Meta Pass-through
    meta: dict[str, Any] = Field(default_factory=dict, description="Beliebige Metadaten (werden durchgereicht)")


class ExtractRequest(BaseModel):
    """
    Request für Konvertierung + strukturierte Extraktion in einem Schritt.

    Unterstützt drei Eingabe-Modi (genau einer muss gesetzt sein):
    - path: Dateipfad im Container
    - base64 + filename: Base64-kodierte Datei
    - url: URL zu Datei/Webseite

    extract_schema ist required; alternativ kann ein vordefinierter template-Name
    angegeben werden.
    """
    model_config = ConfigDict(str_strip_whitespace=True)

    # Eingabe-Modi (einer von dreien)
    path: Optional[str] = Field(None, description="Dateipfad (relativ zu /data oder absolut)")
    base64: Optional[str] = Field(None, description="Base64-kodierte Datei")
    filename: Optional[str] = Field(None, description="Dateiname (erforderlich bei base64)")
    url: Optional[str] = Field(None, description="URL zu Datei oder Webseite")

    # Extraktion (eines von beiden erforderlich)
    extract_schema: Optional[dict[str, Any]] = Field(None, description="JSON-Schema für strukturierte Daten-Extraktion")
    template: Optional[str] = Field(None, description="Vordefinierter Template-Name: 'invoice', 'cv', 'contract'")

    # Optionen
    language: str = Field("de", description="Sprache für Vision-Antwort: 'de', 'en'")

    # Accuracy-Modus (T-MKIT-022)
    accuracy: str = Field("standard", description="Accuracy-Modus: 'standard' oder 'high'")

    @field_validator("accuracy")
    @classmethod
    def validate_accuracy(cls, v: str) -> str:
        if v not in ("standard", "high"):
            raise ValueError(f"accuracy must be 'standard' or 'high', got '{v}'")
        return v

    # OCR-Korrektur (T-MKIT-022)
    ocr_correct: bool = Field(False, description="OCR-Nachkorrektur via LLM aktivieren")

    # Vision-Optionen (T-MKIT-022)
    describe_images: bool = Field(False, description="Eingebettete Bilder durch Pixtral beschreiben")

    # Klassifizierung (T-MKIT-022)
    classify: bool = Field(False, description="Dokumenttyp automatisch klassifizieren")

    # Meta Pass-through
    meta: dict[str, Any] = Field(default_factory=dict, description="Beliebige Metadaten (werden durchgereicht)")


class TemplateResponse(BaseModel):
    """Antwort für /v1/templates Endpoint."""
    templates: dict[str, dict[str, Any]] = Field(..., description="Vordefinierte Extraktions-Templates")


class ConvertFolderRequest(BaseModel):
    """Request für Ordner-Konvertierung."""
    model_config = ConfigDict(str_strip_whitespace=True)

    path: str = Field(..., description="Ordnerpfad (relativ zu /data oder absolut)")
    language: str = Field(default="de", description="Sprache für Vision-Antwort: 'de', 'en'")
    meta: dict[str, Any] = Field(default_factory=dict, description="Beliebige Metadaten")


class AnalyzeRequest(BaseModel):
    """Request für explizite Vision-Analyse (Bilder)."""
    model_config = ConfigDict(str_strip_whitespace=True)

    # Eingabe-Modi
    path: Optional[str] = Field(None, description="Bildpfad")
    base64: Optional[str] = Field(None, description="Base64-kodiertes Bild")
    filename: Optional[str] = Field(None, description="Dateiname (bei base64)")

    # Vision-Optionen
    prompt: str = Field(
        "Beschreibe dieses Bild detailliert. Erfasse alle wichtigen visuellen Elemente, Text, Diagramme oder Informationen.",
        description="Prompt für die Bildanalyse"
    )
    language: str = Field("de", description="Sprache: 'de' oder 'en'")

    # Meta
    meta: dict[str, Any] = Field(default_factory=dict, description="Beliebige Metadaten")


# =============================================================================
# Error Codes
# =============================================================================

class ErrorCode:
    """Definierte Fehlercodes für einheitliche Fehlerbehandlung."""

    # Input-Fehler
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    INVALID_INPUT = "INVALID_INPUT"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    INVALID_BASE64 = "INVALID_BASE64"

    # Verarbeitungsfehler
    CONVERSION_FAILED = "CONVERSION_FAILED"
    VISION_FAILED = "VISION_FAILED"
    TIMEOUT = "TIMEOUT"

    # API-Fehler
    API_ERROR = "API_ERROR"
    API_RATE_LIMIT = "API_RATE_LIMIT"
    API_KEY_INVALID = "API_KEY_INVALID"

    # Server-Fehler
    INTERNAL_ERROR = "INTERNAL_ERROR"


# =============================================================================
# Helper Functions
# =============================================================================

def create_error_response(
    code: str,
    message: str,
    meta: Optional[dict[str, Any]] = None,
    details: Optional[dict[str, Any]] = None
) -> ConvertResponse:
    """Erstellt eine einheitliche Fehlerantwort."""
    return ConvertResponse(
        success=False,
        error=ErrorDetail(code=code, message=message, details=details),
        meta=MetaData(**(meta or {}))
    )


def create_success_response(
    markdown: str,
    meta: Optional[dict[str, Any]] = None
) -> ConvertResponse:
    """Erstellt eine einheitliche Erfolgsantwort."""
    enriched_meta = meta or {}
    enriched_meta["processed_at"] = datetime.utcnow().isoformat() + "Z"

    return ConvertResponse(
        success=True,
        markdown=markdown,
        meta=MetaData(**enriched_meta)
    )
