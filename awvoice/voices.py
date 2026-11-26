"""Per-agent voices: ONE map, read by every surface that speaks.

``AitherOS/config/agent_voices.json`` names, for each agent, the engine that speaks
it and the voice that engine uses.  The file is JSON (also valid YAML) so the Python
side, the orchestrator pack and awdesk's JavaScript all parse the same bytes with no
third-party parser and no second copy to drift from.

Two rules that are enforced elsewhere, not here:

* a voice is never a stock Microsoft ``*Neural`` id -- that is a *fallback* only,
  named per agent in ``edge_fallback``, and every use of it is logged by the caller;
* every agent has a DISTINCT voice, so two bots in one channel are told apart by ear.

The path is ``AITHER_AGENT_VOICES``, else the fleet mount ``/etc/aither/agent_voices.json``,
else the repo checkout (a developer running the brick from the source tree).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

ENV_PATH = "AITHER_AGENT_VOICES"
FLEET_PATH = "/etc/aither/agent_voices.json"
_REPO_REL = Path("AitherOS") / "config" / "agent_voices.json"

#: Engines that can actually speak a line. Anything else is a configuration error.
ENGINES = ("piper", "chatterbox", "kokoro")

#: The voices that spoke before the map existed, used ONLY when the map cannot be read
#: at all (missing file, unreadable mount, bad JSON). A missing config file must never
#: mute an agent -- on 2026-11-01 a map that had not been installed left Aither sitting
#: in the voice channel answering nothing, which reads as "the bot is broken", not as
#: "the config is missing". Every use is LOGGED by the caller. Kept equal to the shipped
#: map by ``test_agent_voices.py::test_last_known_good_matches_the_shipped_map``.
LAST_KNOWN_GOOD = {
    "aither": "local:piper",
    "atlas": "chatterbox:atlas",
    "demiurge": "chatterbox:demiurge",
    "hera": "chatterbox:hera",
    "iris": "chatterbox:iris",
    "lyra": "chatterbox:lyra",
    "saga": "chatterbox:saga",
    "vera": "chatterbox:vera",
}
#: Prefixes AitherVoice understands for those engines (``local:piper`` is the one
#: AitherVoice maps to the built Piper model; ``chatterbox:<agent>`` is Chatterbox-Turbo).
CHATTERBOX_PREFIX = "chatterbox:"
KOKORO_PREFIX = "kokoro:"
PIPER_VOICE = "local:piper"


class AgentVoiceError(LookupError):
    """No voice is defined for this agent (or the map itself cannot be read)."""


def config_path(path: Optional[str] = None) -> Path:
    """Where the map lives: explicit, then ``AITHER_AGENT_VOICES``, then the fleet
    mount, then the repo checkout this file was installed from."""
    if path:
        return Path(path)
    env = (os.environ.get(ENV_PATH) or "").strip()
    if env:
        return Path(env)
    fleet = Path(FLEET_PATH)
    if fleet.is_file():
        return fleet
    for parent in Path(__file__).resolve().parents:
        candidate = parent / _REPO_REL
        if candidate.is_file():
            return candidate
    return fleet


def load(path: Optional[str] = None) -> Dict[str, Any]:
    """The whole map, as parsed JSON. Raises AgentVoiceError when it cannot be read."""
    p = config_path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AgentVoiceError(f"agent voice map unreadable at {p}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("agents"), dict):
        raise AgentVoiceError(f"{p} has no 'agents' map")
    return data


def resolve(agent: str, path: Optional[str] = None,
            table: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """One agent's entry: ``{engine, voice, edge_fallback, ...}``.

    An unknown agent raises rather than falling back to another agent's voice: a bot
    that cannot speak as ITSELF must not borrow Aither's voice and pretend.
    """
    name = str(agent or "").strip().lower()
    data = table if table is not None else load(path)
    agents = data.get("agents") or {}
    entry = agents.get(name)
    if not isinstance(entry, dict):
        raise AgentVoiceError(f"no voice defined for agent {name!r} in "
                              f"{config_path(path)} (add it to agents.{name})")
    voice = str(entry.get("voice") or "").strip()
    if not voice:
        raise AgentVoiceError(f"agent {name!r} has an empty voice")
    out = dict(entry)
    out["agent"] = name
    out["voice"] = voice
    return out


def voice_for(agent: str, path: Optional[str] = None) -> str:
    """The string a caller hands AitherVoice's ``/synthesize`` for this agent."""
    return str(resolve(agent, path)["voice"])


def voice_or_last_known_good(agent: str, path: Optional[str] = None) -> Dict[str, str]:
    """This agent's voice, and where it came from. NEVER raises.

    ``source`` is ``"map"`` (the configured voice), ``"last-known-good"`` (the map is
    unreadable, so :data:`LAST_KNOWN_GOOD` speaks) or ``"none"`` (no row anywhere: the
    speaker carries no voice and the service answers in its own default). The caller
    LOGS anything but ``"map"``; silence about a substitution is how a bot ends up
    sounding like somebody else.
    """
    name = str(agent or "").strip().lower()
    detail = ""
    try:
        return {"agent": name, "voice": voice_for(name, path), "source": "map",
                "detail": ""}
    except AgentVoiceError as exc:
        detail = str(exc)[:200]
    known = LAST_KNOWN_GOOD.get(name, "")
    return {"agent": name, "voice": known,
            "source": "last-known-good" if known else "none", "detail": detail}


def default_agent(path: Optional[str] = None) -> str:
    data = load(path)
    return str(data.get("default_agent") or "aither")


def is_stock_voice(voice: str) -> bool:
    """True for a stock Microsoft neural id (``en-GB-SoniaNeural`` and friends).

    Those are the voices the owner rejected: Aither has its own built model and every
    agent has its own speaker, so a stock id is a fallback, never a configured voice.
    """
    v = str(voice or "").strip()
    return bool(v) and v.lower().endswith("neural")


def voice_family(voice: str) -> str:
    """Which engine actually spoke: ``piper``, ``kokoro``, ``edge`` or ``other``.

    Requested ids and SERVED ids are spelled differently by AitherVoice (a request for
    ``local:piper`` is answered with ``piper-local``), so comparing the families is the
    only honest way to tell "spoke the voice I asked for" from "substituted one".
    """
    v = str(voice or "").strip().lower()
    if not v:
        return ""
    if v in (PIPER_VOICE, "piper", "piper-local") or v.startswith("custom:"):
        return "piper"
    if v.startswith(CHATTERBOX_PREFIX) or v.startswith("chatterbox-"):
        return "chatterbox"
    if v.startswith(KOKORO_PREFIX) or v.startswith("kokoro-"):
        return "kokoro"
    if v.endswith("neural"):
        return "edge"
    return "other"


def fallback_log_line(agent: str, requested: str, served: str, extra: str = "") -> str:
    """The one line every caller writes when a fallback voice spoke the line."""
    note = f" [VOICE-FALLBACK] agent={agent} requested={requested or '(none)'} " \
           f"served={served or '(none)'}"
    return note + (f" {extra}" if extra else "")
