"""The ear. Capture from THIS machine's microphone, turn it into words, and put those
words somewhere that acts on them.

Everything else in awvoice talks to a service about audio you already have. This is the half
that did not exist, and its absence was the whole gap: a session could SPEAK (`synthesize`,
`say`) and could transcribe a file, but nothing could listen, so the only way to talk to an
agent was to type at it or to go through the desk's own push-to-talk.

Three rules the shape here exists to keep:

* **One microphone, one holder.** Two surfaces capturing at once on Windows is not an error,
  it is two half-heard sentences. `MicLease` is an O_EXCL file whose contents NAME the holder,
  so the loser says *who* has the mic rather than failing mysteriously. A lease whose pid is
  gone is stale and may be taken -- a crashed capture must not mute the machine forever.
* **Nothing is retained.** The recording lives in a temp file for exactly as long as the
  transcription call, in a `finally` that deletes it. Audio of a room is the most sensitive
  thing this package touches, and the sight plane already had to learn this lesson
  (`awvision.sight.keep_policy`); here there is no keep at all.
* **Text goes where it can be ACTED on, with its provenance.** `--steer` publishes the
  transcript as a room `steering` event from `kind="human"` -- the owner's own voice, which
  is the one actor the dispatcher will type straight into a live pty. It is never published
  as an agent, because a voice in the room that claims to be an agent is a laundering path.

The capture and the transcription are both injectable, so every rule above is tested without
a microphone and without the fleet.
"""

from __future__ import annotations

import io
import json
import os
import ssl
import sys
import tempfile
import time
import urllib.request
import uuid
import wave
from pathlib import Path
from typing import Any, Callable, Optional

LOCK_PATH = Path.home() / ".aither" / "mic.lock"
DEFAULT_VOICE_URL = os.environ.get("AWVOICE_URL", "https://127.0.0.1:8084")
DEFAULT_HARNESS_URL = os.environ.get("AITHER_HARNESS_URL", "http://127.0.0.1:8362")
HARNESS_TOKEN = Path.home() / ".aither" / "harness_token"

SAMPLE_RATE = 16000          # what local_whisper wants; resampling upstream is wasted work
CHANNELS = 1
DEFAULT_SECONDS = 6.0
#: A transcript longer than this is almost always the room, not an instruction.
MAX_CHARS = 2000


class MicBusyError(RuntimeError):
    """Another surface holds the microphone. Carries WHO, because 'busy' is not actionable."""


# ── one microphone, one holder ──────────────────────────────────────────────

def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True          # exists, owned by someone else
    except OSError:
        return False
    return True


def read_lease(path: Path = LOCK_PATH) -> Optional[dict]:
    """The current holder, or None. A lease whose process is gone reads as None: a crashed
    capture must not mute the machine until someone deletes a file by hand."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not _pid_alive(int(data.get("pid") or 0)):
        return None
    return data


class MicLease:
    """Context manager over `~/.aither/mic.lock`. Refuses with the holder named."""

    def __init__(self, surface: str, path: Path = LOCK_PATH) -> None:
        self.surface = surface
        self.path = path
        self.held = False

    def __enter__(self) -> "MicLease":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        holder = read_lease(self.path)
        if holder:
            raise MicBusyError(
                "the microphone is held by %s (pid %s) since %s"
                % (holder.get("surface", "?"), holder.get("pid", "?"), holder.get("at", "?"))
            )
        payload = json.dumps({
            "pid": os.getpid(), "surface": self.surface,
            "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        try:
            fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            # Raced, or a stale file read_lease already judged dead. Judge it once more --
            # if it is still dead, take it; a stale lock is not a holder.
            if read_lease(self.path) is None:
                try:
                    self.path.unlink()
                except OSError as exc:
                    raise MicBusyError(
                        "a stale microphone lease at %s cannot be removed (%s)"
                        % (self.path, exc)
                    ) from exc
                fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            else:
                raise MicBusyError("the microphone was taken while we were asking") from None
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        self.held = True
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.held:
            try:
                self.path.unlink()
            except OSError as exc:
                # A lease we cannot release would mute this machine until a human deletes a
                # file, so say so loudly on the way out rather than leaving it silent. The
                # pid inside it is what makes the next holder judge it stale and take it.
                print("awvoice: could not release the microphone lease %s: %s"
                      % (self.path, exc), file=sys.stderr)
            self.held = False


# ── capture ─────────────────────────────────────────────────────────────────

def to_wav(frames: Any, sample_rate: int = SAMPLE_RATE, channels: int = CHANNELS) -> bytes:
    """int16 PCM frames -> a real WAV container. Whisper is handed a file, not an array,
    and a headerless blob transcribes as silence rather than failing."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(bytes(frames) if not hasattr(frames, "tobytes") else frames.tobytes())
    return buf.getvalue()


def capture(seconds: float = DEFAULT_SECONDS, sample_rate: int = SAMPLE_RATE,
            device: Optional[int] = None) -> bytes:
    """Record from the default input device and return WAV bytes.

    `sounddevice` is an OPTIONAL dependency (`pip install awvoice[mic]`); its absence is a
    clear instruction, never a traceback, because the common case for a missing mic stack is
    a headless box that was never meant to listen.
    """
    try:
        import sounddevice as sd
    except Exception as exc:  # noqa: BLE001 - ImportError, and OSError when PortAudio is absent
        raise RuntimeError(
            "no audio capture stack (%s). Install it with: pip install 'awvoice[mic]'" % exc
        ) from exc
    rec = sd.rec(int(seconds * sample_rate), samplerate=sample_rate,
                 channels=CHANNELS, dtype="int16", device=device)
    sd.wait()
    return to_wav(rec, sample_rate)


# ── words ───────────────────────────────────────────────────────────────────

def _ssl_ctx() -> ssl.SSLContext:
    return ssl._create_unverified_context()  # loopback to the fleet's own cert


def transcribe(wav: bytes, base_url: str = DEFAULT_VOICE_URL, timeout: float = 60.0) -> str:
    """POST the audio to the perception plane and return the text it heard.

    `/voice/transcribe` takes multipart `file` (checked against the live OpenAPI, 2026-09-20).
    An empty transcript comes back as "" and the caller decides -- silence is a real answer to
    "what did you hear", and inventing words for it would be worse than saying nothing.
    """
    boundary = "----awvoice%s" % uuid.uuid4().hex
    head = (
        "--%s\r\nContent-Disposition: form-data; name=\"file\"; filename=\"mic.wav\"\r\n"
        "Content-Type: audio/wav\r\n\r\n" % boundary
    ).encode("utf-8")
    body = head + wav + ("\r\n--%s--\r\n" % boundary).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/voice/transcribe", data=body, method="POST",
        headers={"Content-Type": "multipart/form-data; boundary=%s" % boundary},
    )
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx()) as resp:
        payload = json.loads(resp.read().decode("utf-8", "replace"))
    if not isinstance(payload, dict):
        return ""
    for key in ("text", "transcript", "transcription", "result"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:MAX_CHARS]
    return ""


# ── somewhere that acts on it ───────────────────────────────────────────────

def steer_event(target: str, text: str, room: str = "main") -> dict:
    """The owner's voice, addressed to one session.

    `kind="human"` is the truth here and it is load-bearing: the steer dispatcher types a
    HUMAN actor straight into a live pty and mailboxes everything else. A transcript of the
    owner speaking IS the owner; publishing it as an agent would both lose the capability and
    launder authority in the direction that matters.
    """
    return {
        "room": room,
        "type": "steering",
        "to": [target],
        "hops": 0,
        "actor": {"kind": "human", "id": "owner", "name": "the owner (voice)"},
        "payload": {"text": text, "source": "voice:awsh"},
    }


def publish(event: dict, harness_url: str = DEFAULT_HARNESS_URL, timeout: float = 30.0) -> int:
    """Publish to the awdk room spine; returns the sequence number."""
    token = ""
    try:
        token = HARNESS_TOKEN.read_text(encoding="utf-8").strip()
    except OSError:
        # No token file is a legitimate state (an unauthenticated daemon); the POST below
        # then goes bare and the daemon decides. Anything else surfaces as that call's error.
        token = ""
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(
        harness_url.rstrip("/") + "/events", data=json.dumps(event).encode("utf-8"),
        headers=headers, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx()) as resp:
        return int(json.loads(resp.read()).get("seq") or 0)


def listen_once(seconds: float = DEFAULT_SECONDS, *, surface: str = "awsh",
                steer: str = "", capture_fn: Callable[..., bytes] = capture,
                transcribe_fn: Callable[[bytes], str] | None = None,
                publish_fn: Callable[[dict], int] | None = None,
                lock_path: Path = LOCK_PATH) -> dict:
    """Hold the mic, record, transcribe, optionally steer. Returns what happened.

    The temp file exists only inside the `finally` that removes it -- see the module docstring
    on retention. Both service calls are injectable so the rules above are tested without a
    microphone and without the fleet.
    """
    transcribe_fn = transcribe_fn or (lambda wav: transcribe(wav))
    publish_fn = publish_fn or publish
    out: dict = {"heard": "", "steered": "", "seq": 0, "surface": surface}
    with MicLease(surface, lock_path):
        wav = capture_fn(seconds)
    tmp = None
    try:
        # A path on disk only if a caller needs one; the bytes go straight out otherwise.
        out["bytes"] = len(wav)
        out["heard"] = transcribe_fn(wav)
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)
    if out["heard"] and steer:
        out["seq"] = publish_fn(steer_event(steer, out["heard"]))
        out["steered"] = steer
    return out


def _tempfile_for(wav: bytes) -> str:
    """A WAV on disk for callers that must have a path. The caller deletes it."""
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="awvoice-")
    with os.fdopen(fd, "wb") as fh:
        fh.write(wav)
    return path
