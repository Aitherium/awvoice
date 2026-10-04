"""Local voices: speak on THIS machine, with no service running.

A local voice id is ``local:<name>``. Today there is one, ``local:aither``: the Aither
voice (Piper/VITS en-US, MIT), fetched once from its published release into the shared
local model directory, sha256-verified, and then run with onnxruntime + espeak-ng.

    pip install "awvoice[local]"
    awvoice say --voice local:aither "Welcome to Aitherium."      # -> awvoice-say.wav

    from awvoice import VoiceClient
    wav = VoiceClient().synthesize("Hello.", voice="local:aither")

``custom:<name>`` (your workspace's voices) and ``AWVOICE_TTS_URL`` (your own service)
are unchanged: a ``local:`` id never reaches either.

Configuration (environment):
  AITHER_VOICE_DIR      where the voice files live (default: the shared local model
                        directory, the same one ``adk home voice`` uses)
  AITHER_PIPER_THREADS  onnxruntime threads (default 2)
"""

from __future__ import annotations

import sys
from typing import Callable, Optional

from .client import ServiceConfigError, ServiceError

LOCAL_PREFIX = "local:"
VOICES = ("aither",)
EXTRA_HINT = 'pip install "awvoice[local]"'


def split_local(voice: Optional[str]) -> Optional[str]:
    """The local voice NAME for a ``local:<name>`` id, else None."""
    if voice and voice.startswith(LOCAL_PREFIX):
        name = voice[len(LOCAL_PREFIX):].strip()
        return name or None
    return None


def _progress(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def say_local(name: str, text: str, speed: Optional[float] = None,
              progress: Callable[[str], None] = _progress) -> bytes:
    """WAV bytes of ``text`` in the local voice ``name``; fetches the voice on first use.

    Raises:
        ValueError: ``text`` is empty.
        ServiceConfigError: Unknown local voice, or the speech runtime is not installed.
        ServiceError: The voice could not be fetched and verified, or failed to speak.
    """
    if not (text or "").strip():
        raise ValueError("Text cannot be empty")
    if name not in VOICES:
        raise ServiceConfigError(f"no local voice {name!r} (have: "
                                 + ", ".join(LOCAL_PREFIX + v for v in VOICES) + ")")
    from . import aither_voice_runtime as rt

    try:
        return rt.speak(text, speed=float(speed or 1.0), say=progress)
    except rt.RuntimeMissingError as exc:
        raise ServiceConfigError(f"{exc}; install the local voice runtime with: "
                                 f"{EXTRA_HINT}") from exc
    except rt.VoiceError as exc:
        raise ServiceError(str(exc)) from exc
