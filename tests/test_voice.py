"""Tests for awvoice — both positive assertions (happy path) and failures."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from awvoice import ServiceConfigError, ServiceError, VoiceClient


class TestVoiceClient:
    """Tests for VoiceClient initialization and configuration."""

    def test_init_with_urls(self):
        """Initialization with explicit URLs works."""
        client = VoiceClient(
            stt_url="http://stt.example.com",
            tts_url="http://tts.example.com",
        )
        assert client.stt_url == "http://stt.example.com"
        assert client.tts_url == "http://tts.example.com"

    def test_init_with_env_vars(self, monkeypatch):
        """Initialization reads from environment variables."""
        monkeypatch.setenv("AWVOICE_STT_URL", "http://stt-env.example.com")
        monkeypatch.setenv("AWVOICE_TTS_URL", "http://tts-env.example.com")
        client = VoiceClient()
        assert client.stt_url == "http://stt-env.example.com"
        assert client.tts_url == "http://tts-env.example.com"

    def test_init_explicit_overrides_env(self, monkeypatch):
        """Explicit URLs override environment variables."""
        monkeypatch.setenv("AWVOICE_STT_URL", "http://env.example.com")
        client = VoiceClient(stt_url="http://explicit.example.com")
        assert client.stt_url == "http://explicit.example.com"


class TestTranscribe:
    """Tests for speech-to-text functionality."""

    def test_transcribe_success(self):
        """Successful transcription returns text."""
        # Create a temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_audio = f.name
            f.write(b"fake audio data")

        try:
            client = VoiceClient(stt_url="http://stt.example.com")
            # Mock the urllib call
            mock_response_data = json.dumps({"text": "Hello world"}).encode("utf-8")

            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_response = MagicMock()
                mock_response.status = 200
                mock_response.read.return_value = mock_response_data
                mock_urlopen.return_value.__enter__.return_value = mock_response

                result = client.transcribe(temp_audio)
                assert result == "Hello world"
        finally:
            Path(temp_audio).unlink()

    def test_transcribe_missing_endpoint(self):
        """Transcription fails clearly when endpoint is not configured."""
        client = VoiceClient()
        with pytest.raises(
            ServiceConfigError, match="STT endpoint not configured"
        ):
            client.transcribe("test.wav")

    def test_transcribe_missing_file(self):
        """Transcription fails clearly when audio file does not exist."""
        client = VoiceClient(stt_url="http://stt.example.com")
        with pytest.raises(FileNotFoundError):
            client.transcribe("/nonexistent/file.wav")

    def test_transcribe_service_error(self):
        """Transcription fails when service returns error status."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_audio = f.name
            f.write(b"fake audio data")

        try:
            client = VoiceClient(stt_url="http://stt.example.com")
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_response = MagicMock()
                mock_response.status = 500
                mock_response.reason = "Internal Server Error"
                mock_urlopen.return_value.__enter__.return_value = mock_response

                with pytest.raises(ServiceError, match="returned status 500"):
                    client.transcribe(temp_audio)
        finally:
            Path(temp_audio).unlink()

    def test_transcribe_network_error(self):
        """Transcription fails when network is unreachable."""
        import urllib.error

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_audio = f.name
            f.write(b"fake audio data")

        try:
            client = VoiceClient(stt_url="http://stt.example.com")
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_urlopen.side_effect = urllib.error.URLError(
                    "Connection refused"
                )

                with pytest.raises(ServiceError, match="Cannot reach service"):
                    client.transcribe(temp_audio)
        finally:
            Path(temp_audio).unlink()

    def test_transcribe_invalid_json_response(self):
        """Transcription fails when service returns invalid JSON."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_audio = f.name
            f.write(b"fake audio data")

        try:
            client = VoiceClient(stt_url="http://stt.example.com")
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_response = MagicMock()
                mock_response.status = 200
                mock_response.read.return_value = b"not valid json"
                mock_urlopen.return_value.__enter__.return_value = mock_response

                with pytest.raises(ServiceError, match="invalid JSON"):
                    client.transcribe(temp_audio)
        finally:
            Path(temp_audio).unlink()


class TestSynthesize:
    """Tests for text-to-speech functionality."""

    def test_synthesize_success(self):
        """Successful synthesis returns audio bytes."""
        client = VoiceClient(tts_url="http://tts.example.com")
        fake_audio = b"\xff\xfb\x10\x00fake mp3 data"

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.read.return_value = fake_audio
            mock_urlopen.return_value.__enter__.return_value = mock_response

            result = client.synthesize("Hello world")
            assert result == fake_audio

    def test_synthesize_missing_endpoint(self):
        """Synthesis fails clearly when endpoint is not configured."""
        client = VoiceClient()
        with pytest.raises(
            ServiceConfigError, match="TTS endpoint not configured"
        ):
            client.synthesize("Hello world")

    def test_synthesize_empty_text(self):
        """Synthesis fails when text is empty."""
        client = VoiceClient(tts_url="http://tts.example.com")
        with pytest.raises(ValueError, match="empty"):
            client.synthesize("")

    def test_synthesize_service_error(self):
        """Synthesis fails when service returns error status."""
        client = VoiceClient(tts_url="http://tts.example.com")
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.status = 503
            mock_response.reason = "Service Unavailable"
            mock_urlopen.return_value.__enter__.return_value = mock_response

            with pytest.raises(ServiceError, match="returned status 503"):
                client.synthesize("Hello world")

    def test_synthesize_network_error(self):
        """Synthesis fails when network is unreachable."""
        import urllib.error

        client = VoiceClient(tts_url="http://tts.example.com")
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

            with pytest.raises(ServiceError, match="Cannot reach service"):
                client.synthesize("Hello world")

    def test_synthesize_writes_to_file(self):
        """Synthesize result can be written to a file."""
        client = VoiceClient(tts_url="http://tts.example.com")
        fake_audio = b"\xff\xfb\x10\x00fake mp3 data"

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.read.return_value = fake_audio
            mock_urlopen.return_value.__enter__.return_value = mock_response

            result = client.synthesize("Hello world")

            # Verify we can write it to a file
            with tempfile.NamedTemporaryFile(delete=False) as f:
                temp_file = f.name
                f.write(result)

            try:
                saved = Path(temp_file).read_bytes()
                assert saved == fake_audio
            finally:
                Path(temp_file).unlink()


class TestFailClosed:
    """Tests that verify fail-closed behavior on every error path."""

    def test_transcribe_fails_on_malformed_url(self):
        """Transcription fails on unreachable endpoint."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_audio = f.name
            f.write(b"fake audio data")

        try:
            client = VoiceClient(stt_url="not a valid url")
            with pytest.raises(ServiceError):
                client.transcribe(temp_audio)
        finally:
            Path(temp_audio).unlink()

    def test_synthesis_fails_on_malformed_url(self):
        """Synthesis fails on unreachable endpoint."""
        client = VoiceClient(tts_url="not a valid url")
        with pytest.raises(ServiceError):
            client.synthesize("test")

    def test_no_silent_failures(self):
        """No error path silently succeeds — all raise exceptions."""
        # Missing STT config
        client = VoiceClient()
        with pytest.raises(ServiceConfigError):
            client.transcribe("test.wav")

        # Missing TTS config
        client = VoiceClient()
        with pytest.raises(ServiceConfigError):
            client.synthesize("test")

        # Empty text
        client = VoiceClient(tts_url="http://tts.example.com")
        with pytest.raises(ValueError):
            client.synthesize("")


class TestDeskSay:
    """`awvoice say` -- the desk avatar owns playback; awvoice only asks."""

    def test_payload_trims_caps_and_carries_voice(self):
        from awvoice.desk import MAX_CHARS, speak_payload

        assert speak_payload("  hello  ") == {"text": "hello"}
        assert speak_payload("hi", "nova") == {"text": "hi", "voice": "nova"}
        assert len(speak_payload("x" * (MAX_CHARS + 50))["text"]) == MAX_CHARS

    def test_payload_carries_and_bounds_speed(self):
        from awvoice.desk import speak_payload

        assert speak_payload("hi", speed=1.5) == {"text": "hi", "speed": 1.5}
        with pytest.raises(ValueError):
            speak_payload("hi", speed=9)

    def test_payload_refuses_silence(self):
        from awvoice.desk import speak_payload

        with pytest.raises(ValueError):
            speak_payload("   ")

    def test_desk_url_env_override(self):
        from awvoice.desk import DEFAULT_DESK_URL, desk_url

        assert desk_url({}) == DEFAULT_DESK_URL
        assert desk_url({"AWVOICE_DESK_URL": "http://127.0.0.1:5/"}) == "http://127.0.0.1:5"

    def test_say_posts_to_speak_and_returns_verdict(self):
        from awvoice.desk import say

        seen = {}

        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return b'{"ok": true, "chars": 5, "windows": 1}'

        def opener(req, timeout):
            seen["url"] = req.full_url
            seen["body"] = json.loads(req.data.decode("utf-8"))
            return _Resp()

        verdict = say("hello", "nova", base_url="http://127.0.0.1:1", opener=opener)
        assert seen["url"] == "http://127.0.0.1:1/speak"
        assert seen["body"] == {"text": "hello", "voice": "nova"}
        assert verdict["ok"] is True

    def test_say_raises_when_desk_is_down_or_silent(self):
        import urllib.error

        from awvoice.desk import DeskUnavailableError, say

        def down(req, timeout):
            raise urllib.error.URLError("refused")

        with pytest.raises(DeskUnavailableError):
            say("hello", base_url="http://127.0.0.1:1", opener=down)

        class _Silent:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return b'{"ok": false, "reason": "voice service unavailable"}'

        with pytest.raises(DeskUnavailableError):
            say("hello", base_url="http://127.0.0.1:1", opener=lambda r, timeout: _Silent())
