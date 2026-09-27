"""Speak an agent's reply through the desk: the Claude Code Stop-hook entrypoint.

``awvoice reply`` reads the Stop hook JSON on stdin (``last_assistant_message``,
``transcript_path``, ``session_id``), reduces the final assistant text to what a person
would say out loud -- no code, tables, paths or markdown; the first line plus the verdict,
at most two sentences / 300 chars -- and POSTs it to the awdesk bridge ``/speak``.

It is a hook, so it never gets in the way: opt-in by ``AITHER_SPEAK_REPLIES=1``, silent
when the desk's ``cast.json`` has ``voice.muted: true``, a 3 s fire-and-forget POST, and
exit 0 on every path (including a crash) so a Stop hook can never block a session.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any, Optional

from .desk import desk_url

MAX_SPOKEN = 300
POST_TIMEOUT_S = 3.0
_VERDICT = re.compile(r"\b(CLOSED|PARTIAL|OPEN)\b")

_FENCE = re.compile(r"```.*?(```|\Z)", re.S)
_INLINE_CODE = re.compile(r"`[^`\n]*`")
_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_URL = re.compile(r"\bhttps?://\S+")
# a Windows path (C:\x, D:/x), a posix-ish path with at least one slash, or a bare file.ext
_WINPATH = re.compile(r"\b[A-Za-z]:[\\/][^\s,;)]*")
_SLASHPATH = re.compile(r"(?<![\w])(?:~|\.{1,2})?(?:[\w.@-]+)?(?:[\\/][\w.@-]+)+[\\/]?")
_FILENAME = re.compile(r"\b[\w-]+\.(?:py|cjs|mjs|js|ts|tsx|json|ya?ml|md|toml|ps1|sh|txt|html|css)\b")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def speakable(text: str, limit: int = MAX_SPOKEN) -> str:
    """Reduce markdown-ish agent output to a short spoken line (pure; unit-tested)."""
    if not text:
        return ""
    body = _FENCE.sub(" ", text)
    lines: list[str] = []
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.startswith("|") or re.fullmatch(r"[-=*_:| ]{3,}", line):
            continue                                   # blank, table row, rule
        line = re.sub(r"^#{1,6}\s*", "", line)          # heading
        line = re.sub(r"^(?:[-*+]|\d+[.)])\s+", "", line)  # bullet / numbered item
        line = re.sub(r"^>\s*", "", line)               # quote
        line = _LINK.sub(r"\1", line)
        line = _INLINE_CODE.sub(" ", line)
        line = _URL.sub(" ", line)
        line = _WINPATH.sub(" ", line)
        line = _SLASHPATH.sub(" ", line)
        line = _FILENAME.sub(" ", line)
        line = re.sub(r"[*_~]{1,3}", "", line)          # emphasis markers
        line = re.sub(r"\s+([,.;:!?])", r"\1", line)
        line = re.sub(r"\s{2,}", " ", line).strip(" -:;,")
        if re.search(r"[A-Za-z]", line):
            lines.append(line)
    if not lines:
        return ""
    picked = [lines[0]]
    verdict = next((ln for ln in lines[1:] if _VERDICT.search(ln)), None)
    if verdict and verdict != lines[0]:
        picked.append(verdict)
    sentences: list[str] = []
    for chunk in picked:
        sentences.extend(s for s in _SENTENCE.split(chunk) if s)
    spoken = " ".join(sentences[:2]).strip()
    if len(spoken) > limit:
        cut = spoken[:limit].rsplit(" ", 1)[0].rstrip(" ,;:-")
        spoken = cut + "."
    return spoken


def last_assistant_text(transcript_path: str) -> str:
    """The text of the last assistant entry in a Claude Code JSONL transcript."""
    last = ""
    try:
        with open(transcript_path, encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                try:
                    entry = json.loads(raw)
                except ValueError:
                    continue
                if not isinstance(entry, dict) or entry.get("type") != "assistant":
                    continue
                content = (entry.get("message") or {}).get("content")
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    text = "\n".join(
                        str(b.get("text", "")) for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                else:
                    text = ""
                if text.strip():
                    last = text
    except OSError:
        return ""
    return last


def cast_path(environment: Optional[dict[str, str]] = None) -> Path:
    env = os.environ if environment is None else environment
    override = env.get("AWDESK_CAST_PATH")
    if override:
        return Path(override)
    base = env.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    return Path(base) / "Desk" / "cast.json"


def desk_muted(path: Path) -> bool:
    """True when the desk's cast says voice.muted. A missing/unreadable cast is NOT muted --
    the desk applies its own builtin default and will refuse there if it must."""
    try:
        cast = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    voice = cast.get("voice") if isinstance(cast, dict) else None
    return isinstance(voice, dict) and voice.get("muted") is True


def post_speak(text: str, base_url: str, timeout: float = POST_TIMEOUT_S,
               opener: Any = urllib.request.urlopen) -> bool:
    """Fire the line at the desk. True when the desk answered 2xx; never raises."""
    body = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(base_url.rstrip("/") + "/speak", data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with opener(req, timeout=timeout) as resp:
            return 200 <= int(getattr(resp, "status", 200)) < 300
    except Exception:  # noqa: BLE001 -- timeout/refused/HTTP error: the hook stays silent
        return False


def run(stdin_text: str, environment: Optional[dict[str, str]] = None,
        opener: Any = urllib.request.urlopen) -> dict:
    """The whole hook, minus process concerns. Returns what it decided (for tests)."""
    env = os.environ if environment is None else environment
    if env.get("AITHER_SPEAK_REPLIES") != "1":
        return {"spoken": False, "reason": "AITHER_SPEAK_REPLIES is not 1"}
    if desk_muted(cast_path(env)):
        return {"spoken": False, "reason": "desk voice.muted"}
    try:
        event = json.loads(stdin_text) if stdin_text.strip() else {}
    except ValueError:
        event = {}
    if not isinstance(event, dict):
        event = {}
    text = event.get("last_assistant_message") or ""
    if not isinstance(text, str) or not text.strip():
        tp = event.get("transcript_path")
        text = last_assistant_text(tp) if isinstance(tp, str) and tp else ""
    line = speakable(text)
    if not line:
        return {"spoken": False, "reason": "nothing speakable"}
    ok = post_speak(line, desk_url(dict(env)), opener=opener)
    return {"spoken": ok, "text": line, "reason": "" if ok else "desk did not take it"}


def main() -> int:
    """Entry for ``awvoice reply``. Always 0: a Stop hook must never block."""
    try:
        data = sys.stdin.read() if not sys.stdin.isatty() else ""
        run(data)
    except Exception:  # noqa: BLE001
        pass
    return 0
