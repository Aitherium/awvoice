"""Speak through the desk avatar.

``awvoice say "<text>"`` posts to the local awdesk bridge (``POST /speak``),
which synthesises the line through the voice service and plays it on the
desk with lip-sync. This is how an agent, a routine or a terminal session
gives the platform's own voice a face: awdesk owns playback, so every caller
sounds the same and nothing else has to know how audio reaches a speaker.

The bridge is loopback-only; ``AWVOICE_DESK_URL`` overrides the default
``http://127.0.0.1:47931`` (awdesk reads ``DESK_BRIDGE_PORT`` on its side).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

DEFAULT_DESK_URL = "http://127.0.0.1:47931"
MAX_CHARS = 2000


class DeskUnavailableError(RuntimeError):
    """The desk bridge did not take the line (down, refused, or no avatar)."""


def desk_url(environment: dict[str, str] | None = None) -> str:
    env = os.environ if environment is None else environment
    return (env.get("AWVOICE_DESK_URL") or DEFAULT_DESK_URL).rstrip("/")


def speak_payload(text: str, voice: str | None = None,
                  speed: float | None = None) -> dict[str, object]:
    """The request body for POST /speak (pure; unit-tested). ``speed`` is a
    playback rate, 0.25-4.0; the desk applies its own default when omitted."""
    line = (text or "").strip()
    if not line:
        raise ValueError("nothing to say")
    body: dict[str, object] = {"text": line[:MAX_CHARS]}
    if voice:
        body["voice"] = voice
    if speed is not None:
        if not 0.25 <= float(speed) <= 4.0:
            raise ValueError("speed must be between 0.25 and 4.0")
        body["speed"] = float(speed)
    return body


def say(text: str, voice: str | None = None, *, speed: float | None = None,
        base_url: str | None = None, timeout: float = 30.0,
        opener=urllib.request.urlopen) -> dict:
    """Ask the desk avatar to say ``text``. Returns the bridge's verdict dict.

    Raises ``DeskUnavailableError`` when the bridge cannot be reached or refuses,
    so a caller can fall back (synthesize to a file, print the line) instead
    of believing something was heard.
    """
    url = (base_url or desk_url()).rstrip("/") + "/speak"
    data = json.dumps(speak_payload(text, voice, speed)).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with opener(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:200]
        raise DeskUnavailableError(f"desk refused ({exc.code}): {detail}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise DeskUnavailableError(f"desk not reachable at {url}: {exc}") from exc
    try:
        verdict = json.loads(raw)
    except ValueError as exc:
        raise DeskUnavailableError(f"desk answered non-JSON: {raw[:120]}") from exc
    if not isinstance(verdict, dict) or verdict.get("ok") is not True:
        raise DeskUnavailableError(f"desk did not speak: {verdict}")
    return verdict
