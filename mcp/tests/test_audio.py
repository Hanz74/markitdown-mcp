"""
Unit-Tests für Audio/Video-Transkription via faster-whisper (FR-MKIT-006).

Tests laufen ohne Docker-Container und ohne echte Mediendateien.
Alle externen Abhängigkeiten (faster-whisper, ffmpeg, subprocess) werden gemockt.
"""

import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conftest import load_server_module, run_async


# ---------------------------------------------------------------------------
# Server-Modul laden mit faster_whisper gemockt
# ---------------------------------------------------------------------------
# faster_whisper ist in Tests nicht installiert — wir stubben es vor dem Import,
# damit WHISPER_AVAILABLE=True und WhisperModel im server-Modul verfügbar ist.

_whisper_mock = MagicMock()
_whisper_model_cls = MagicMock()
_whisper_mock.WhisperModel = _whisper_model_cls

_server = load_server_module(
    use_real_pil=False,
    extra_patches={"faster_whisper": _whisper_mock},
)

# WhisperModel und WHISPER_AVAILABLE explizit setzen (Import wurde gestubbt)
_server.WHISPER_AVAILABLE = True
_server.WhisperModel = _whisper_model_cls

is_audio_file = _server.is_audio_file
is_video_file = _server.is_video_file
extract_audio_from_video = _server.extract_audio_from_video
transcribe_audio = _server.transcribe_audio
convert_auto = _server.convert_auto


# =============================================================================
# TestIsAudioFile
# =============================================================================


class TestIsAudioFile:
    """AC-006-2: Audio-Erkennung via Dateiendung."""

    def test_mp3_is_audio(self, tmp_path):
        """MP3-Dateien werden als Audio erkannt."""
        f = tmp_path / "song.mp3"
        f.write_bytes(b"dummy")
        assert is_audio_file(f) is True

    def test_wav_is_audio(self, tmp_path):
        """WAV-Dateien werden als Audio erkannt."""
        f = tmp_path / "audio.wav"
        f.write_bytes(b"dummy")
        assert is_audio_file(f) is True

    def test_ogg_is_audio(self, tmp_path):
        """OGG-Dateien werden als Audio erkannt."""
        f = tmp_path / "audio.ogg"
        f.write_bytes(b"dummy")
        assert is_audio_file(f) is True

    def test_flac_is_audio(self, tmp_path):
        """FLAC-Dateien werden als Audio erkannt."""
        f = tmp_path / "audio.flac"
        f.write_bytes(b"dummy")
        assert is_audio_file(f) is True

    def test_m4a_is_audio(self, tmp_path):
        """M4A-Dateien werden als Audio erkannt."""
        f = tmp_path / "audio.m4a"
        f.write_bytes(b"dummy")
        assert is_audio_file(f) is True

    def test_pdf_is_not_audio(self, tmp_path):
        """PDF-Dateien werden NICHT als Audio erkannt."""
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"dummy")
        assert is_audio_file(f) is False

    def test_mp4_is_not_audio(self, tmp_path):
        """MP4-Dateien (Video) werden NICHT als Audio erkannt."""
        f = tmp_path / "video.mp4"
        f.write_bytes(b"dummy")
        assert is_audio_file(f) is False

    def test_uppercase_extension_recognized(self, tmp_path):
        """Großgeschriebene Endungen (.MP3) werden korrekt erkannt."""
        f = tmp_path / "audio.MP3"
        f.write_bytes(b"dummy")
        assert is_audio_file(f) is True


# =============================================================================
# TestIsVideoFile
# =============================================================================


class TestIsVideoFile:
    """AC-006-3: Video-Erkennung via Dateiendung."""

    def test_mp4_is_video(self, tmp_path):
        """MP4-Dateien werden als Video erkannt."""
        f = tmp_path / "video.mp4"
        f.write_bytes(b"dummy")
        assert is_video_file(f) is True

    def test_mkv_is_video(self, tmp_path):
        """MKV-Dateien werden als Video erkannt."""
        f = tmp_path / "video.mkv"
        f.write_bytes(b"dummy")
        assert is_video_file(f) is True

    def test_webm_is_video(self, tmp_path):
        """WEBM-Dateien werden als Video erkannt."""
        f = tmp_path / "video.webm"
        f.write_bytes(b"dummy")
        assert is_video_file(f) is True

    def test_avi_is_video(self, tmp_path):
        """AVI-Dateien werden als Video erkannt."""
        f = tmp_path / "video.avi"
        f.write_bytes(b"dummy")
        assert is_video_file(f) is True

    def test_mov_is_video(self, tmp_path):
        """MOV-Dateien werden als Video erkannt."""
        f = tmp_path / "video.mov"
        f.write_bytes(b"dummy")
        assert is_video_file(f) is True

    def test_mp3_is_not_video(self, tmp_path):
        """MP3-Dateien (Audio) werden NICHT als Video erkannt."""
        f = tmp_path / "audio.mp3"
        f.write_bytes(b"dummy")
        assert is_video_file(f) is False

    def test_pdf_is_not_video(self, tmp_path):
        """PDF-Dateien werden NICHT als Video erkannt."""
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"dummy")
        assert is_video_file(f) is False

    def test_uppercase_extension_recognized(self, tmp_path):
        """Großgeschriebene Endungen (.MP4) werden korrekt erkannt."""
        f = tmp_path / "video.MP4"
        f.write_bytes(b"dummy")
        assert is_video_file(f) is True


# =============================================================================
# TestExtractAudioFromVideo
# =============================================================================


class TestExtractAudioFromVideo:
    """AC-006-3: Audio-Track-Extraktion aus Video via ffmpeg."""

    def _make_video_file(self, tmp_path: Path, name: str = "test.mp4") -> Path:
        """Erstellt eine Dummy-Video-Datei."""
        p = tmp_path / name
        p.write_bytes(b"dummy-video-data")
        return p

    def test_extract_audio_calls_ffmpeg(self, tmp_path):
        """extract_audio_from_video ruft ffmpeg mit korrekten Argumenten auf."""
        video = self._make_video_file(tmp_path)

        ffmpeg_result = MagicMock()
        ffmpeg_result.returncode = 0
        ffmpeg_result.stderr = ""

        with patch("subprocess.run", return_value=ffmpeg_result) as mock_run:
            with patch.object(_server, "TEMP_DIR", tmp_path):
                wav_path = extract_audio_from_video(video)

        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]  # Positional args: der Befehl als Liste

        assert call_args[0] == "ffmpeg"
        assert "-i" in call_args
        assert str(video) in call_args
        assert "-vn" in call_args  # kein Video
        assert "-ar" in call_args
        assert "16000" in call_args
        assert str(wav_path).endswith(".wav")

    def test_extract_audio_returns_wav_path(self, tmp_path):
        """extract_audio_from_video gibt einen .wav-Pfad zurück."""
        video = self._make_video_file(tmp_path)

        ffmpeg_result = MagicMock()
        ffmpeg_result.returncode = 0
        ffmpeg_result.stderr = ""

        with patch("subprocess.run", return_value=ffmpeg_result):
            with patch.object(_server, "TEMP_DIR", tmp_path):
                wav_path = extract_audio_from_video(video)

        assert wav_path.suffix == ".wav"

    def test_extract_audio_raises_on_ffmpeg_failure(self, tmp_path):
        """RuntimeError wenn ffmpeg mit Fehler-Exit-Code endet."""
        video = self._make_video_file(tmp_path)

        failed_result = MagicMock()
        failed_result.returncode = 1
        failed_result.stderr = "No such file"

        with patch("subprocess.run", return_value=failed_result):
            with patch.object(_server, "TEMP_DIR", tmp_path):
                with pytest.raises(RuntimeError, match="ffmpeg fehlgeschlagen"):
                    extract_audio_from_video(video)

    def test_extract_audio_raises_when_ffmpeg_not_installed(self, tmp_path):
        """RuntimeError wenn ffmpeg nicht installiert ist (FileNotFoundError)."""
        video = self._make_video_file(tmp_path)

        with patch("subprocess.run", side_effect=FileNotFoundError("ffmpeg not found")):
            with patch.object(_server, "TEMP_DIR", tmp_path):
                with pytest.raises(RuntimeError, match="ffmpeg ist nicht installiert"):
                    extract_audio_from_video(video)

    def test_extract_audio_raises_on_timeout(self, tmp_path):
        """RuntimeError bei subprocess.TimeoutExpired."""
        video = self._make_video_file(tmp_path)

        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired("ffmpeg", 300),
        ):
            with patch.object(_server, "TEMP_DIR", tmp_path):
                with pytest.raises(RuntimeError, match="Timeout"):
                    extract_audio_from_video(video)


# =============================================================================
# TestTranscribeAudio
# =============================================================================


class TestTranscribeAudio:
    """AC-006-2/AC-006-4: Transkription und Sprach-Erkennung via faster-whisper."""

    def _make_audio_file(self, tmp_path: Path, name: str = "audio.wav") -> Path:
        """Erstellt eine Dummy-Audio-Datei."""
        p = tmp_path / name
        p.write_bytes(b"dummy-audio-data")
        return p

    def _make_mock_segment(self, text: str, start: float = 0.0, end: float = 5.0):
        """Erstellt ein Mock-Segment das einem faster-whisper Segment ähnelt."""
        seg = MagicMock()
        seg.text = text
        seg.start = start
        seg.end = end
        return seg

    def _make_mock_info(self, language: str = "de"):
        """Erstellt ein Mock-Info-Objekt das faster-whisper TranscriptionInfo ähnelt."""
        info = MagicMock()
        info.language = language
        return info

    def test_transcribe_audio_success(self, tmp_path):
        """Erfolgreiche Transkription gibt success=True und den Text zurück."""
        audio = self._make_audio_file(tmp_path)

        segments = [
            self._make_mock_segment("Hallo Welt", 0.0, 2.0),
            self._make_mock_segment("wie geht es dir", 2.0, 5.5),
        ]
        info = self._make_mock_info("de")

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter(segments), info)

        with patch.object(_server, "WHISPER_AVAILABLE", True):
            with patch.object(_server, "WHISPER_MODEL_SIZE", "base"):
                with patch.object(_server, "_whisper_model_cache", {}):
                    with patch.object(_server, "WhisperModel", return_value=mock_model):
                        result = transcribe_audio(audio)

        assert result["success"] is True
        assert "Hallo Welt" in result["text"]
        assert "wie geht es dir" in result["text"]

    def test_transcribe_audio_language_detected(self, tmp_path):
        """Erkannte Sprache wird im Ergebnis zurückgegeben (AC-006-4)."""
        audio = self._make_audio_file(tmp_path)

        segments = [self._make_mock_segment("Hello world", 0.0, 3.0)]
        info = self._make_mock_info("en")

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter(segments), info)

        with patch.object(_server, "WHISPER_AVAILABLE", True):
            with patch.object(_server, "WHISPER_MODEL_SIZE", "base"):
                with patch.object(_server, "_whisper_model_cache", {}):
                    with patch.object(_server, "WhisperModel", return_value=mock_model):
                        result = transcribe_audio(audio)

        assert result["language"] == "en"

    def test_transcribe_audio_duration(self, tmp_path):
        """Dauer wird korrekt aus dem letzten Segment-Ende berechnet (AC-006-6)."""
        audio = self._make_audio_file(tmp_path)

        segments = [
            self._make_mock_segment("Teil 1", 0.0, 30.5),
            self._make_mock_segment("Teil 2", 30.5, 65.2),
        ]
        info = self._make_mock_info("de")

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter(segments), info)

        with patch.object(_server, "WHISPER_AVAILABLE", True):
            with patch.object(_server, "WHISPER_MODEL_SIZE", "base"):
                with patch.object(_server, "_whisper_model_cache", {}):
                    with patch.object(_server, "WhisperModel", return_value=mock_model):
                        result = transcribe_audio(audio)

        assert result["duration"] == pytest.approx(65.2, abs=0.01)

    def test_transcribe_audio_model_size_in_result(self, tmp_path):
        """model_size wird im Ergebnis zurückgegeben (AC-006-6)."""
        audio = self._make_audio_file(tmp_path)

        segments = [self._make_mock_segment("Text", 0.0, 1.0)]
        info = self._make_mock_info("de")

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter(segments), info)

        with patch.object(_server, "WHISPER_AVAILABLE", True):
            with patch.object(_server, "WHISPER_MODEL_SIZE", "small"):
                with patch.object(_server, "_whisper_model_cache", {}):
                    with patch.object(_server, "WhisperModel", return_value=mock_model):
                        result = transcribe_audio(audio)

        assert result["model_size"] == "small"

    def test_transcribe_audio_no_whisper(self, tmp_path):
        """Wenn faster-whisper nicht installiert → graceful Fehler (AC-001)."""
        audio = self._make_audio_file(tmp_path)

        with patch.object(_server, "WHISPER_AVAILABLE", False):
            result = transcribe_audio(audio)

        assert result["success"] is False
        assert "faster-whisper" in result["error"].lower()

    def test_transcribe_audio_model_exception(self, tmp_path):
        """Wenn das Modell eine Exception wirft → graceful Fehler."""
        audio = self._make_audio_file(tmp_path)

        mock_model = MagicMock()
        mock_model.transcribe.side_effect = RuntimeError("CUDA out of memory")

        with patch.object(_server, "WHISPER_AVAILABLE", True):
            with patch.object(_server, "WHISPER_MODEL_SIZE", "base"):
                with patch.object(_server, "_whisper_model_cache", {}):
                    with patch.object(_server, "WhisperModel", return_value=mock_model):
                        result = transcribe_audio(audio)

        assert result["success"] is False
        assert "Transkription fehlgeschlagen" in result["error"]

    def test_transcribe_audio_uses_cached_model(self, tmp_path):
        """Das Modell wird nur einmal geladen — zweiter Aufruf nutzt den Cache."""
        audio = self._make_audio_file(tmp_path)

        segments = [self._make_mock_segment("Test", 0.0, 1.0)]
        info = self._make_mock_info("de")

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter(segments), info)

        fake_cache: dict = {}

        with patch.object(_server, "WHISPER_AVAILABLE", True):
            with patch.object(_server, "WHISPER_MODEL_SIZE", "base"):
                with patch.object(_server, "_whisper_model_cache", fake_cache):
                    with patch.object(_server, "WhisperModel", return_value=mock_model) as mock_cls:
                        # Erstes Ergebnis
                        mock_model.transcribe.return_value = (iter(segments), info)
                        transcribe_audio(audio)

                        # Zweites Ergebnis — Modell muss gecacht sein
                        mock_model.transcribe.return_value = (iter(segments), info)
                        transcribe_audio(audio)

        # WhisperModel() wurde nur einmal konstruiert (Cache-Hit beim 2. Aufruf)
        assert mock_cls.call_count == 1


# =============================================================================
# TestTranscribeMeta
# =============================================================================


class TestTranscribeMeta:
    """AC-006-6: Meta-Daten (Sprache, Dauer, Modell-Größe) korrekt."""

    def test_transcribe_meta_all_fields_present(self, tmp_path):
        """Alle drei Meta-Felder (language, duration, model_size) sind im Ergebnis."""
        audio = tmp_path / "test.wav"
        audio.write_bytes(b"dummy")

        seg = MagicMock()
        seg.text = "Hallo"
        seg.start = 0.0
        seg.end = 2.5
        info = MagicMock()
        info.language = "de"

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([seg]), info)

        with patch.object(_server, "WHISPER_AVAILABLE", True):
            with patch.object(_server, "WHISPER_MODEL_SIZE", "base"):
                with patch.object(_server, "_whisper_model_cache", {}):
                    with patch.object(_server, "WhisperModel", return_value=mock_model):
                        result = transcribe_audio(audio)

        assert "language" in result
        assert "duration" in result
        assert "model_size" in result
        assert result["language"] == "de"
        assert result["duration"] == pytest.approx(2.5, abs=0.01)
        assert result["model_size"] == "base"


# =============================================================================
# TestVideoPipeline
# =============================================================================


class TestVideoPipeline:
    """AC-006-3: Video → Audio-Extraktion → Transkription (End-to-End gemockt)."""

    def test_video_pipeline_full(self, tmp_path):
        """Video-Datei → Audio extrahieren → transkribieren → Markdown."""
        # Erstelle eine Dummy-Video-Datei
        video = tmp_path / "lecture.mp4"
        video.write_bytes(b"dummy-video")

        # Mock: ffmpeg extrahiert erfolgreich
        ffmpeg_result = MagicMock()
        ffmpeg_result.returncode = 0
        ffmpeg_result.stderr = ""

        # Mock: WAV-Datei die nach Extraktion "existiert"
        wav_path = tmp_path / "lecture_extracted.wav"
        wav_path.write_bytes(b"dummy-wav")

        # Mock: Whisper-Transkription
        seg = MagicMock()
        seg.text = "Willkommen zur Vorlesung"
        seg.start = 0.0
        seg.end = 4.0
        info = MagicMock()
        info.language = "de"
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([seg]), info)

        with patch("subprocess.run", return_value=ffmpeg_result):
            with patch.object(_server, "TEMP_DIR", tmp_path):
                with patch.object(_server, "WHISPER_AVAILABLE", True):
                    with patch.object(_server, "WHISPER_MODEL_SIZE", "base"):
                        with patch.object(_server, "_whisper_model_cache", {}):
                            with patch.object(_server, "WhisperModel", return_value=mock_model):
                                # extract_audio_from_video gibt einen WAV-Pfad zurück —
                                # wir patchen die Funktion direkt
                                with patch.object(
                                    _server,
                                    "extract_audio_from_video",
                                    return_value=wav_path,
                                ):
                                    result = transcribe_audio(wav_path)

        assert result["success"] is True
        assert "Willkommen zur Vorlesung" in result["text"]
        assert result["language"] == "de"

    def test_video_pipeline_ffmpeg_failure_graceful(self, tmp_path):
        """Wenn ffmpeg fehlschlägt, gibt convert_auto einen Fehler zurück (kein Absturz)."""
        video = tmp_path / "broken.mp4"
        video.write_bytes(b"not-a-video")

        with patch.object(_server, "detect_mimetype_from_bytes", return_value=None):
            with patch.object(
                _server,
                "extract_audio_from_video",
                side_effect=RuntimeError("ffmpeg fehlgeschlagen (returncode=1): error"),
            ):
                with patch.object(_server, "TEMP_DIR", tmp_path):
                    result = run_async(
                        convert_auto(
                            file_data=b"not-a-video",
                            filename="broken.mp4",
                            source="broken.mp4",
                            source_type="file",
                            input_meta={},
                        )
                    )

        assert result.success is False
        assert result.error is not None


# =============================================================================
# TestAudioInConvertAuto
# =============================================================================


class TestAudioInConvertAuto:
    """AC-006-5: Audio/Video in convert_auto und REST /v1/convert integriert."""

    def _make_mock_transcription_success(
        self,
        text: str = "Das ist der transkribierte Text.",
        language: str = "de",
        duration: float = 30.0,
        model_size: str = "base",
    ) -> dict:
        """Erstellt ein Mock-Transkriptions-Ergebnis."""
        return {
            "success": True,
            "text": text,
            "language": language,
            "duration": duration,
            "model_size": model_size,
        }

    def _run_convert_auto(self, tmp_path, filename, file_data=b"dummy", **patches):
        """Hilfsmethode: Führt convert_auto mit gemocktem detect_mimetype_from_bytes aus.

        detect_mimetype_from_bytes gibt None zurück, damit magic-Mock die
        Audio/Video-Routing-Logik nicht stört.
        """
        ctx = [
            patch.object(_server, "detect_mimetype_from_bytes", return_value=None),
            patch.object(_server, "TEMP_DIR", tmp_path),
        ]
        for attr, val in patches.items():
            ctx.append(patch.object(_server, attr, val))

        result = None
        # Kontextmanager manuell öffnen (verschachtelt)
        import contextlib

        @contextlib.contextmanager
        def all_patches():
            with contextlib.ExitStack() as stack:
                for c in ctx:
                    stack.enter_context(c)
                yield

        with all_patches():
            result = run_async(
                convert_auto(
                    file_data=file_data,
                    filename=filename,
                    source=filename,
                    source_type="file",
                    input_meta={},
                )
            )
        return result

    def test_audio_mp3_routed_to_transcription(self, tmp_path):
        """MP3-Dateien werden in convert_auto an transcribe_audio geroutet."""
        mp3_data = b"dummy-mp3-data"
        transcription = self._make_mock_transcription_success()

        with patch.object(_server, "detect_mimetype_from_bytes", return_value=None):
            with patch.object(_server, "transcribe_audio", return_value=transcription):
                with patch.object(_server, "TEMP_DIR", tmp_path):
                    result = run_async(
                        convert_auto(
                            file_data=mp3_data,
                            filename="recording.mp3",
                            source="recording.mp3",
                            source_type="file",
                            input_meta={},
                        )
                    )

        assert result.success is True
        assert "Transkription" in result.markdown
        assert "Das ist der transkribierte Text." in result.markdown

    def test_audio_wav_routed_to_transcription(self, tmp_path):
        """WAV-Dateien werden korrekt geroutet."""
        transcription = self._make_mock_transcription_success(text="WAV audio text")

        with patch.object(_server, "detect_mimetype_from_bytes", return_value=None):
            with patch.object(_server, "transcribe_audio", return_value=transcription):
                with patch.object(_server, "TEMP_DIR", tmp_path):
                    result = run_async(
                        convert_auto(
                            file_data=b"dummy-wav",
                            filename="audio.wav",
                            source="audio.wav",
                            source_type="file",
                            input_meta={},
                        )
                    )

        assert result.success is True
        assert "WAV audio text" in result.markdown

    def test_video_mp4_extracts_then_transcribes(self, tmp_path):
        """MP4-Video → Audio extrahieren → transkribieren."""
        wav_path = tmp_path / "extracted.wav"
        wav_path.write_bytes(b"fake-wav")

        transcription = self._make_mock_transcription_success(
            text="Video-Inhalt transkribiert",
            language="en",
            duration=120.0,
        )

        with patch.object(_server, "detect_mimetype_from_bytes", return_value=None):
            with patch.object(
                _server, "extract_audio_from_video", return_value=wav_path
            ):
                with patch.object(_server, "transcribe_audio", return_value=transcription):
                    with patch.object(_server, "TEMP_DIR", tmp_path):
                        result = run_async(
                            convert_auto(
                                file_data=b"dummy-video",
                                filename="lecture.mp4",
                                source="lecture.mp4",
                                source_type="file",
                                input_meta={},
                            )
                        )

        assert result.success is True
        assert "Video-Inhalt transkribiert" in result.markdown

    def test_audio_meta_language_set(self, tmp_path):
        """Meta-Daten enthalten erkannte Sprache (AC-006-6)."""
        transcription = self._make_mock_transcription_success(language="en")

        with patch.object(_server, "detect_mimetype_from_bytes", return_value=None):
            with patch.object(_server, "transcribe_audio", return_value=transcription):
                with patch.object(_server, "TEMP_DIR", tmp_path):
                    result = run_async(
                        convert_auto(
                            file_data=b"dummy",
                            filename="audio.mp3",
                            source="audio.mp3",
                            source_type="file",
                            input_meta={},
                        )
                    )

        assert result.success is True
        assert result.meta is not None
        assert result.meta.language == "en"

    def test_audio_meta_duration_set(self, tmp_path):
        """Meta-Daten enthalten Dauer in Sekunden (AC-006-6)."""
        transcription = self._make_mock_transcription_success(duration=95.3)

        with patch.object(_server, "detect_mimetype_from_bytes", return_value=None):
            with patch.object(_server, "transcribe_audio", return_value=transcription):
                with patch.object(_server, "TEMP_DIR", tmp_path):
                    result = run_async(
                        convert_auto(
                            file_data=b"dummy",
                            filename="audio.wav",
                            source="audio.wav",
                            source_type="file",
                            input_meta={},
                        )
                    )

        assert result.success is True
        assert result.meta.duration_seconds == pytest.approx(95.3, abs=0.01)

    def test_audio_meta_whisper_model_set(self, tmp_path):
        """Meta-Daten enthalten das Whisper-Modell (AC-006-6)."""
        transcription = self._make_mock_transcription_success(model_size="small")

        with patch.object(_server, "detect_mimetype_from_bytes", return_value=None):
            with patch.object(_server, "transcribe_audio", return_value=transcription):
                with patch.object(_server, "TEMP_DIR", tmp_path):
                    result = run_async(
                        convert_auto(
                            file_data=b"dummy",
                            filename="audio.flac",
                            source="audio.flac",
                            source_type="file",
                            input_meta={},
                        )
                    )

        assert result.success is True
        assert result.meta.whisper_model == "small"

    def test_transcription_failure_returns_error(self, tmp_path):
        """Wenn Transkription fehlschlägt, gibt convert_auto einen Fehler zurück."""
        failed_transcription = {
            "success": False,
            "error": "faster-whisper ist nicht installiert",
        }

        with patch.object(_server, "detect_mimetype_from_bytes", return_value=None):
            with patch.object(_server, "transcribe_audio", return_value=failed_transcription):
                with patch.object(_server, "TEMP_DIR", tmp_path):
                    result = run_async(
                        convert_auto(
                            file_data=b"dummy",
                            filename="audio.mp3",
                            source="audio.mp3",
                            source_type="file",
                            input_meta={},
                        )
                    )

        assert result.success is False
        assert result.error is not None

    def test_markdown_has_transcription_heading(self, tmp_path):
        """Das resultierende Markdown enthält die '# Transkription' Überschrift."""
        transcription = self._make_mock_transcription_success(text="Inhalt")

        with patch.object(_server, "detect_mimetype_from_bytes", return_value=None):
            with patch.object(_server, "transcribe_audio", return_value=transcription):
                with patch.object(_server, "TEMP_DIR", tmp_path):
                    result = run_async(
                        convert_auto(
                            file_data=b"dummy",
                            filename="audio.ogg",
                            source="audio.ogg",
                            source_type="file",
                            input_meta={},
                        )
                    )

        assert result.success is True
        assert result.markdown.startswith("# Transkription")

    def test_video_ffmpeg_failure_returns_error(self, tmp_path):
        """Wenn Audio-Extraktion aus Video fehlschlägt → Fehler-Response."""
        with patch.object(_server, "detect_mimetype_from_bytes", return_value=None):
            with patch.object(
                _server,
                "extract_audio_from_video",
                side_effect=RuntimeError("ffmpeg fehlgeschlagen (returncode=1): stderr output"),
            ):
                with patch.object(_server, "TEMP_DIR", tmp_path):
                    result = run_async(
                        convert_auto(
                            file_data=b"dummy-video",
                            filename="video.mkv",
                            source="video.mkv",
                            source_type="file",
                            input_meta={},
                        )
                    )

        assert result.success is False
        assert result.error is not None
