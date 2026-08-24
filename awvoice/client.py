"""Speech and text service client — configurable endpoints with explicit errors."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


class ServiceConfigError(Exception):
    """Raised when a required service endpoint is not configured."""

    pass


class ServiceError(Exception):
    """Raised when a service call fails."""

    pass


class VoiceClient:
    """Client for speech-to-text and text-to-speech services.

    Both services must be HTTP endpoints you host and configure.
    """

    def __init__(
        self,
        stt_url: str | None = None,
        tts_url: str | None = None,
    ):
        """Initialize the client with service endpoints.

        Args:
            stt_url: HTTP endpoint for speech-to-text (e.g. http://localhost:5000/transcribe)
            tts_url: HTTP endpoint for text-to-speech (e.g. http://localhost:5000/synthesize)

        Both can also be set via environment variables AWVOICE_STT_URL and AWVOICE_TTS_URL.
        """
        import os

        self.stt_url = stt_url or os.environ.get("AWVOICE_STT_URL")
        self.tts_url = tts_url or os.environ.get("AWVOICE_TTS_URL")

    def transcribe(self, audio_path: str | Path) -> str:
        """Convert speech to text.

        Args:
            audio_path: Path to audio file (wav, mp3, ogg, flac, etc.)

        Returns:
            Transcribed text

        Raises:
            ServiceConfigError: If STT endpoint is not configured
            ServiceError: If the service call fails or returns an error
        """
        if not self.stt_url:
            raise ServiceConfigError(
                "STT endpoint not configured. Set AWVOICE_STT_URL environment variable "
                "or pass stt_url to VoiceClient.__init__"
            )

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            with open(audio_path, "rb") as f:
                audio_data = f.read()
        except (IOError, OSError) as exc:
            raise ServiceError(f"Cannot read audio file: {exc}") from exc

        return self._call_service(self.stt_url, audio_data, is_text=False)

    def synthesize(self, text: str) -> bytes:
        """Convert text to speech.

        Args:
            text: Text to convert to speech

        Returns:
            Audio bytes (format depends on the service)

        Raises:
            ServiceConfigError: If TTS endpoint is not configured
            ServiceError: If the service call fails or returns an error
        """
        if not self.tts_url:
            raise ServiceConfigError(
                "TTS endpoint not configured. Set AWVOICE_TTS_URL environment variable "
                "or pass tts_url to VoiceClient.__init__"
            )

        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        return self._call_service(self.tts_url, text.encode("utf-8"), is_text=True)

    def _call_service(self, url: str, data: bytes, is_text: bool) -> str | bytes:
        """Call a voice service and return the result.

        Args:
            url: Service endpoint URL
            data: Data to send (audio bytes or text bytes)
            is_text: True for TTS (text in, audio out), False for STT (audio in, text out)

        Returns:
            Response text (for STT) or bytes (for TTS)

        Raises:
            ServiceError: On network errors, bad responses, or service errors
        """
        try:
            req = urllib.request.Request(url, data=data)
            req.add_header("Content-Type", "application/octet-stream")

            with urllib.request.urlopen(req, timeout=30) as response:
                if response.status not in (200, 201):
                    raise ServiceError(
                        f"Service returned status {response.status}: {response.reason}"
                    )

                response_data = response.read()

                # STT returns JSON with 'text' field
                if not is_text:
                    try:
                        result = json.loads(response_data.decode("utf-8"))
                        if isinstance(result, dict) and "text" in result:
                            return result["text"]
                        return str(result)
                    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                        raise ServiceError(
                            f"Service returned invalid JSON: {exc}"
                        ) from exc

                # TTS returns raw audio bytes
                return response_data

        except urllib.error.URLError as exc:
            raise ServiceError(
                f"Cannot reach service at {url}: {exc.reason}"
            ) from exc
        except urllib.error.HTTPError as exc:
            try:
                error_msg = exc.read().decode("utf-8")
            except (UnicodeDecodeError, AttributeError):
                error_msg = exc.reason

            raise ServiceError(
                f"Service error (HTTP {exc.code}): {error_msg}"
            ) from exc
        except Exception as exc:
            raise ServiceError(f"Unexpected error calling service: {exc}") from exc
