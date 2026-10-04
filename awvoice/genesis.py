"""Custom voices: the voices your workspace built, spoken through the Aitherium API.

A custom voice id is ``custom:<name>``. Anything else is a stock voice and never
reaches this module -- ``VoiceClient.synthesize`` keeps sending stock text to your
own ``AWVOICE_TTS_URL`` exactly as before.

    from awvoice import list_custom_voices, say_custom

    for v in list_custom_voices():
        print(v["id"], v["reader"])
    wav = say_custom("grandma", "Time for bed.")

Configuration (environment):
  AWVOICE_GENESIS_URL / AITHER_GENESIS_URL  API base, used as given
  AITHER_API_URL / AITHER_GATEWAY_URL       direct Genesis base (same as adk), used as given
  AITHER_PORTAL_URL / AITHER_ELYSIUM_URL    your portal; ``/api/genesis`` is appended
                                            (default https://api.aitherium.com)
  AITHER_API_KEY                            your bearer; sent as ``Authorization: Bearer``

The list is scoped to the caller's workspace server-side. An empty list is a normal
answer (nobody has built a voice yet), not an error.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

from .client import ServiceConfigError, ServiceError

DEFAULT_PORTAL = "https://api.aitherium.com"
GENESIS_PROXY_PREFIX = "/api/genesis"
DEFAULT_API = DEFAULT_PORTAL + GENESIS_PROXY_PREFIX
CUSTOM_PREFIX = "custom:"
MAX_TEXT = 1000
SPEED_MIN, SPEED_MAX = 0.5, 2.0

Opener = Callable[..., Any]


class GenesisAuthError(ServiceError):
    """The API refused the bearer (HTTP 401/403)."""


def split_voice(voice: str | None) -> str | None:
    """Return the custom voice NAME for a ``custom:<name>`` id, else None."""
    if voice and voice.startswith(CUSTOM_PREFIX):
        name = voice[len(CUSTOM_PREFIX):].strip()
        return name or None
    return None


def api_base() -> str:
    """The Genesis API base. Follows adk's control-plane resolution, so a box enrolled
    against its own portal (AITHER_PORTAL_URL) sends the key THERE, not to the default."""
    direct = (os.environ.get("AWVOICE_GENESIS_URL")
              or os.environ.get("AITHER_GENESIS_URL")
              or os.environ.get("AITHER_API_URL")
              or os.environ.get("AITHER_GATEWAY_URL"))
    if direct:
        return direct.rstrip("/")
    portal = (os.environ.get("AITHER_PORTAL_URL")
              or os.environ.get("AITHER_ELYSIUM_URL")
              or DEFAULT_PORTAL)
    return portal.rstrip("/") + GENESIS_PROXY_PREFIX


def _bearer() -> str:
    key = (os.environ.get("AITHER_API_KEY") or "").strip()
    if not key:
        raise ServiceConfigError(
            "Custom voices need your Aitherium API key. Set AITHER_API_KEY."
        )
    return key


def _request(method: str, path: str, body: dict | None, opener: Opener | None) -> dict:
    url = f"{api_base()}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {_bearer()}")
    req.add_header("Accept", "application/json")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    do_open = opener or urllib.request.urlopen
    try:
        with do_open(req, timeout=60) as resp:
            raw = resp.read()
    # HTTPError subclasses URLError: it MUST be caught first or every 4xx/5xx
    # reads as "cannot reach".
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", "replace").strip()[:300]
        except Exception:  # noqa: BLE001
            detail = str(exc.reason)
        if exc.code in (401, 403):
            # Keep the server's words: a portal proxy that does not forward this path
            # also answers 403 ("path not allowed"), and that is not a bad key.
            raise GenesisAuthError(
                f"HTTP {exc.code} from {url}: {detail or exc.reason}. "
                "If this is a key problem, check AITHER_API_KEY belongs to the "
                "workspace that owns the voice."
            ) from exc
        raise ServiceError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ServiceError(f"Cannot reach {url}: {exc.reason}") from exc
    try:
        out = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ServiceError(f"{url} returned invalid JSON: {exc}") from exc
    if not isinstance(out, dict):
        raise ServiceError(f"{url} returned {type(out).__name__}, expected an object")
    return out


def list_custom_voices(opener: Opener | None = None) -> list[dict]:
    """The custom voices the caller's workspace has built. ``[]`` when none."""
    out = _request("GET", "/voice-builds/voices", None, opener)
    voices = out.get("voices") or []
    result = []
    for v in voices:
        if not isinstance(v, dict) or not v.get("name"):
            continue
        name = str(v["name"])
        result.append({
            "id": CUSTOM_PREFIX + name,
            "name": name,
            "reader": v.get("reader"),
            "language": v.get("language"),
            "built_at": v.get("built_at"),
            "gate": v.get("gate"),
        })
    return result


def say_custom(name: str, text: str, speed: float | None = None,
               opener: Opener | None = None) -> bytes:
    """Speak ``text`` in custom voice ``name``; returns WAV bytes."""
    raw = (name or "").strip()
    name = split_voice(raw) if raw.startswith(CUSTOM_PREFIX) else raw
    if not name:
        raise ValueError("Custom voice name cannot be empty")
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")
    body: dict[str, Any] = {"text": text[:MAX_TEXT]}
    if speed is not None:
        if not SPEED_MIN <= float(speed) <= SPEED_MAX:
            raise ValueError(f"speed must be {SPEED_MIN}-{SPEED_MAX} for a custom voice")
        body["speed"] = float(speed)
    path = f"/voice-builds/voices/{urllib.parse.quote(name, safe='')}/say"
    out = _request("POST", path, body, opener)
    audio = out.get("audio_base64")
    if not audio:
        raise ServiceError("Custom voice reply carried no audio_base64")
    try:
        return base64.b64decode(audio, validate=True)
    except (ValueError, TypeError) as exc:
        raise ServiceError(f"Custom voice audio was not valid base64: {exc}") from exc
