"""awvoice — turn speech into text and text into speech, on a service you host.

    from awvoice import VoiceClient

    client = VoiceClient(
        stt_url="http://localhost:5000/transcribe",
        tts_url="http://localhost:5000/synthesize")

    text = client.transcribe("voice.wav")
    audio = client.synthesize("Hello, world!")

Requires you to host STT and TTS services. Supports any HTTP endpoint that:
- STT: accepts audio bytes, returns JSON with 'text' field
- TTS: accepts text bytes, returns audio bytes

Set AWVOICE_STT_URL and AWVOICE_TTS_URL environment variables to configure
the endpoints, or pass them to VoiceClient.__init__.

Custom voices your workspace built are ids of the form ``custom:<name>``:
``list_custom_voices()`` lists them and ``client.synthesize(text, voice="custom:x")``
speaks through the Aitherium API (bearer from AITHER_API_KEY).

``local:aither`` is the Aither voice on THIS machine: ``client.synthesize(text,
voice="local:aither")`` fetches it once (sha256-pinned) and speaks with no service
(``pip install "awvoice[local]"``).
"""

from .client import ServiceConfigError, ServiceError, VoiceClient
from .genesis import GenesisAuthError, list_custom_voices, say_custom
from .local import say_local

__version__ = "0.3.0"
__all__ = [
    "VoiceClient",
    "ServiceConfigError",
    "ServiceError",
    "GenesisAuthError",
    "list_custom_voices",
    "say_custom",
    "say_local",
]
