"""
Gemeinsame Test-Fixtures und Hilfsfunktionen für alle Test-Module.

WICHTIG: Dieses Modul wird von pytest VOR allen Test-Modulen importiert.
Es sichert echtes PIL (und andere echte Module) bevor irgendein Test-Modul
sie durch MagicMocks ersetzen kann. PNG-Fixtures werden hier einmalig
erzeugt, bevor PIL gemockt werden kann.
"""

import asyncio
import io
import sys
import importlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Echtes PIL sofort beim Modulimport sichern (vor jedem Mocking)
# ---------------------------------------------------------------------------
# conftest.py wird von pytest als ERSTES importiert — noch bevor test_*.py
# Module geladen werden. Hier sichern wir echte PIL-Referenzen und erzeugen
# alle benötigten PNG-Fixtures, bevor PIL durch MagicMocks ersetzt wird.

import PIL as _real_PIL
import PIL.Image as _real_PIL_Image

# Exportierte Referenzen für Test-Module die echtes PIL brauchen
REAL_PIL = _real_PIL
REAL_PIL_IMAGE = _real_PIL_Image


def _make_png_bytes(width: int, height: int, color: tuple = (128, 128, 128)) -> bytes:
    """Erstellt ein minimales In-Memory PNG — MUSS vor jedem PIL-Mocking aufgerufen werden."""
    img = _real_PIL_Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# PNG-Fixtures einmalig beim Modulimport materialisieren (vor jedem Mocking)
PNG_10x10 = _make_png_bytes(10, 10)
PNG_30x30 = _make_png_bytes(30, 30)
PNG_49x49 = _make_png_bytes(49, 49)
PNG_50x50 = _make_png_bytes(50, 50)
PNG_100x100 = _make_png_bytes(100, 100)
PNG_120x80 = _make_png_bytes(120, 80)
PNG_150x150 = _make_png_bytes(150, 150)
PNG_200x200 = _make_png_bytes(200, 200)
PNG_200x150 = _make_png_bytes(200, 150)
PNG_300x200 = _make_png_bytes(300, 200)
PNG_400x300 = _make_png_bytes(400, 300)


# ---------------------------------------------------------------------------
# Shared Helper: Server-Modul isoliert laden
# ---------------------------------------------------------------------------

def load_server_module(use_real_pil: bool = False, extra_patches: dict | None = None):
    """
    Importiert server.py mit gemockten schweren Abhängigkeiten.

    Da server.py beim Import sofort setup_logging() und MarkItDown() ausführt,
    patchen wir alle externen Module vor dem Import.

    Args:
        use_real_pil: Wenn True, wird echtes PIL verwendet (für Bild-Größen-Tests).
                      Wenn False, wird PIL durch MagicMock ersetzt.
        extra_patches: Optionale zusätzliche sys.modules-Einträge.

    Returns:
        Das geladene server-Modul.
    """
    # Stub-Module für alle schweren Abhängigkeiten
    stub_names = [
        "uvloop", "magic", "markitdown", "fastmcp",
        "uvicorn", "httpx", "tenacity",
        "structlog", "fastapi", "fastapi.exceptions", "fastapi.responses",
        "pdfplumber", "pdf2image",
    ]
    # PIL nur mocken wenn use_real_pil=False
    if not use_real_pil:
        stub_names += ["PIL", "PIL.Image"]

    for mod_name in stub_names:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = MagicMock()

    # structlog: get_logger() muss einen Logger zurückgeben
    structlog_mock = MagicMock()
    structlog_mock.get_logger = MagicMock(return_value=MagicMock())
    structlog_mock.configure = MagicMock()
    structlog_mock.stdlib = MagicMock()
    structlog_mock.processors = MagicMock()
    structlog_mock.dev = MagicMock()
    sys.modules["structlog"] = structlog_mock

    # FastAPI muss instanziierbar sein
    fastapi_mock = MagicMock()
    fastapi_mock.FastAPI = MagicMock(return_value=MagicMock())
    fastapi_mock.HTTPException = Exception
    fastapi_mock.Request = MagicMock()
    sys.modules["fastapi"] = fastapi_mock
    sys.modules["fastapi.exceptions"] = MagicMock()
    sys.modules["fastapi.responses"] = MagicMock()

    if use_real_pil:
        # Echtes PIL wiederherstellen (aus den gesicherten Referenzen in conftest)
        sys.modules["PIL"] = REAL_PIL
        sys.modules["PIL.Image"] = REAL_PIL_IMAGE
    else:
        # PIL mocken (für Tests die PIL nicht benötigen)
        pil_mock = MagicMock()
        pil_image_mock = MagicMock()
        pil_mock.Image = pil_image_mock
        sys.modules["PIL"] = pil_mock
        sys.modules["PIL.Image"] = pil_image_mock

    # markitdown.MarkItDown muss instanziierbar sein
    sys.modules["markitdown"].MarkItDown = MagicMock(return_value=MagicMock())

    # tenacity Decorators müssen Funktionen unverändert durchlassen
    tenacity_mock = MagicMock()
    tenacity_mock.retry = lambda **kwargs: (lambda f: f)
    tenacity_mock.stop_after_attempt = MagicMock()
    tenacity_mock.wait_exponential = MagicMock()
    tenacity_mock.retry_if_exception_type = MagicMock()
    sys.modules["tenacity"] = tenacity_mock

    # Extra-Patches anwenden (z.B. für pdf2image)
    if extra_patches:
        for name, mock in extra_patches.items():
            sys.modules[name] = mock

    # Server-Modul neu laden (nicht aus Cache)
    if "server" in sys.modules:
        del sys.modules["server"]

    server_path = str(Path(__file__).parent.parent)
    if server_path not in sys.path:
        sys.path.insert(0, server_path)

    import server  # noqa: PLC0415
    return server


def run_async(coro):
    """Führt eine Coroutine synchron aus (für Pytest ohne asyncio-Plugin)."""
    return asyncio.get_event_loop().run_until_complete(coro)
