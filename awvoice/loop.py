"""The shared voice conversation loop: listen, think, speak.

ONE IMPLEMENTATION, MANY TRANSPORTS.  Discord voice (``awdiscord``), a browser
microphone (``awconnect``), a harness session (``awsh``/``awdk``) and the desk
avatar (``awdesk``) all have the same job: turn a stream of 16-bit PCM into a
spoken answer.  The difference is only who delivers the PCM and who plays the
audio back, so those two edges are adapters and everything between them lives
here.

The loop owns three things every client would otherwise get subtly different:

* **End-pointing.**  An energy VAD with a pre-roll, a minimum speech length and
  a silence hangover.  Naive "send every 20 ms chunk to the STT service" is slow
  and transcribes half-words; naive "wait for a 2 s gap" makes the agent feel
  deaf.  The numbers are configurable, the behaviour is the same everywhere.
* **The turn.**  PCM -> WAV -> STT -> the agent -> TTS -> audio bytes, with the
  conversation history that makes it a conversation rather than a series of
  one-shot questions.
* **Failure honesty.**  An empty transcript is a dropped turn, never a question
  asked of the model; an STT/TTS/agent failure is reported to ``on_error`` and
  the loop keeps listening.  Silence is never reported as a successful turn.

The default adapters speak the fleet's own protocols: ``HttpTranscriber`` posts
a multipart WAV to AitherVoice ``POST {base}/transcribe`` (or any OpenAI-style
``/audio/transcriptions``), ``HttpSpeaker`` posts JSON to ``/synthesize``, and
``OpenAICompatThinker`` posts to an OpenAI-compatible ``/chat/completions``
(MicroScheduler on :8150).  Plain callables are accepted anywhere, which is how
the tests drive the loop with no network at all.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import ssl
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Protocol, Sequence

SAMPLE_RATE = 16000          # the rate every Whisper-family server expects
SAMPLE_WIDTH = 2             # 16-bit signed little-endian
CHANNELS = 1


# ---------------------------------------------------------------------------
# PCM helpers
# ---------------------------------------------------------------------------

def pcm_rms(pcm: bytes) -> float:
    """Root-mean-square amplitude of 16-bit signed little-endian samples."""
    n = len(pcm) // 2
    if n == 0:
        return 0.0
    # struct.iter_unpack is exact and fast enough for 20 ms frames (~320 samples).
    total = 0
    for (sample,) in struct.iter_unpack("<h", pcm[: n * 2]):
        total += sample * sample
    return math.sqrt(total / n)


def wav_bytes(pcm: bytes, *, sample_rate: int = SAMPLE_RATE,
              sample_width: int = SAMPLE_WIDTH, channels: int = CHANNELS) -> bytes:
    """A RIFF/WAVE container around raw PCM."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sample_width)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def seconds_to_frames(seconds: float, frame_ms: int) -> int:
    return max(1, int(round((seconds * 1000.0) / frame_ms)))


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class Transcriber(Protocol):
    def transcribe(self, pcm: bytes, *, sample_rate: int = SAMPLE_RATE) -> str:
        """Speech (16-bit mono PCM at ``sample_rate``) -> text."""


class Thinker(Protocol):
    def think(self, text: str, *, history: Sequence[Dict[str, str]] = ()) -> str:
        """A question (and the conversation so far) -> the answer."""


class Speaker(Protocol):
    def speak(self, text: str) -> tuple[bytes, str]:
        """Text -> (encoded audio, format such as ``mp3`` or ``wav``)."""


@dataclass
class VoiceTurn:
    """One complete exchange, including the audio played back."""
    transcript: str
    reply: str
    audio: bytes = b""
    audio_format: str = ""
    user: str = ""
    #: True when the listener spoke over the answer and the rest was dropped.
    barged_in: bool = False
    #: stt_ms / think_ms (to the first sentence) / tts_ms (summed) / sentences.
    timings: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"transcript": self.transcript, "reply": self.reply,
                "audio_format": self.audio_format, "audio_bytes": len(self.audio),
                "user": self.user, "barged_in": self.barged_in,
                "timings": dict(self.timings)}


@dataclass
class LoopConfig:
    """Tuning for the energy VAD.  Defaults are for Discord's noisy voice lanes."""
    sample_rate: int = SAMPLE_RATE
    sample_width: int = SAMPLE_WIDTH
    channels: int = CHANNELS
    frame_ms: int = 20
    speech_rms: float = 480.0      # int16 RMS; speech is usually 1000-8000
    start_frames: int = 3          # consecutive voiced frames that open a turn
    silence_ms: int = 700          # this much quiet ends the turn
    pre_roll_ms: int = 240         # audio kept from before the trigger
    min_speech_ms: int = 300       # shorter than this is a cough, not a question
    max_utterance_ms: int = 30000  # hard stop so one stuck mic cannot run forever

    @property
    def frame_bytes(self) -> int:
        return int(self.sample_rate * self.sample_width * self.channels
                   * self.frame_ms / 1000)


# ---------------------------------------------------------------------------
# End-pointing
# ---------------------------------------------------------------------------

class SpeechSegmenter:
    """Energy end-pointing over a stream of PCM chunks.

    ``feed`` is called with whatever size chunk the transport hands over (Discord
    gives 20 ms frames, a browser may give 128 samples) and returns a list of
    completed utterances -- usually empty, sometimes one, and two only if a
    single chunk spans a whole silence gap and a new question.
    """

    def __init__(self, config: Optional[LoopConfig] = None) -> None:
        self.config = config or LoopConfig()
        self._frame_bytes = self.config.frame_bytes
        self._pre_roll_frames = seconds_to_frames(
            self.config.pre_roll_ms / 1000.0, self.config.frame_ms)
        self._silence_frames = seconds_to_frames(
            self.config.silence_ms / 1000.0, self.config.frame_ms)
        self._min_speech_frames = seconds_to_frames(
            self.config.min_speech_ms / 1000.0, self.config.frame_ms)
        self._max_frames = seconds_to_frames(
            self.config.max_utterance_ms / 1000.0, self.config.frame_ms)
        self._speech_threshold = self.config.speech_rms
        self._preroll: List[bytes] = []
        self._pending = b""            # a partial frame carried between chunks
        self._frames: List[bytes] = []
        self._voiced = 0
        self._speech_frames = 0        # VOICED frames in this utterance (not total)
        self._silent = 0
        self._active = False

    def reset(self) -> None:
        self._preroll.clear()
        self._pending = b""
        self._frames.clear()
        self._voiced = 0
        self._speech_frames = 0
        self._silent = 0
        self._active = False

    def feed(self, pcm: bytes) -> List[bytes]:
        out: List[bytes] = []
        data = self._pending + (pcm or b"")
        size = self._frame_bytes
        while len(data) >= size:
            frame, data = data[:size], data[size:]
            done = self._frame(frame)
            if done is not None:
                out.append(done)
        self._pending = data
        # A transport that closed mid-frame still owes us its tail.
        if self._pending and not self._active:
            self._pending = b""
        return out

    def _frame(self, frame: bytes) -> Optional[bytes]:
        voiced = pcm_rms(frame) >= self._speech_threshold
        if not self._active:
            self._preroll.append(frame)
            if len(self._preroll) > self._pre_roll_frames:
                self._preroll.pop(0)
            self._voiced = self._voiced + 1 if voiced else 0
            if self._voiced >= self.config.start_frames:
                self._active = True
                self._frames = list(self._preroll)
                self._speech_frames = self._voiced
                self._preroll.clear()
                self._silent = 0
                self._voiced = 0
            return None

        self._frames.append(frame)
        if voiced:
            self._silent = 0
            self._voiced += 1
            self._speech_frames += 1
        else:
            self._silent += 1
            enough_speech = self._speech_frames >= self._min_speech_frames
            if self._silent >= self._silence_frames and enough_speech:
                return self._finish()
        if len(self._frames) >= self._max_frames:
            return self._finish()
        return None

    def flush(self) -> Optional[bytes]:
        """Close the utterance in progress when the stream stops without trailing silence.

        The energy end-pointer closes an utterance only after ``silence_ms`` of quiet
        frames. Discord stops SENDING packets the moment a speaker goes quiet, so the
        silence the end-pointer waits for never arrives and a short question is held
        forever (2026-11-06: 28 decoded frames, 0 utterances, Aither never answered).
        The transport calls this when it knows the speaker stopped; it returns the
        utterance if one was in progress and long enough, and resets either way.
        """
        if not self._active:
            self.reset()
            return None
        return self._finish()

    def _finish(self) -> Optional[bytes]:
        frames = list(self._frames)          # reset() clears the live list
        speech_frames = self._speech_frames
        self.reset()
        if speech_frames < self._min_speech_frames:
            return None
        return b"".join(frames)


# ---------------------------------------------------------------------------
# Streaming helpers: sentence cuts and barge-in
# ---------------------------------------------------------------------------

_SENTENCE_END = ".!?"
_CLOSERS = "\"')]"


class SentenceChunker:
    """Cut a stream of text deltas into speakable sentences, in order.

    A cut is made after ``.``, ``!`` or ``?`` (and any closing quote/bracket)
    followed by whitespace, so ``3.5`` and ``example.com`` stay whole.  A run with no
    sentence end is cut at the last space before ``max_chars`` so one long clause
    cannot hold the first audio back.  ``flush`` returns whatever is left.
    """

    def __init__(self, max_chars: int = 200) -> None:
        self.max_chars = max(20, int(max_chars))
        self._buf = ""

    def feed(self, piece: str) -> List[str]:
        self._buf += piece or ""
        out: List[str] = []
        while True:
            cut = self._cut()
            if cut <= 0:
                break
            sentence, self._buf = self._buf[:cut].strip(), self._buf[cut:]
            if sentence:
                out.append(sentence)
        return out

    def flush(self) -> List[str]:
        rest, self._buf = self._buf.strip(), ""
        return [rest] if rest else []

    def _cut(self) -> int:
        buf = self._buf
        limit = min(len(buf), self.max_chars + 1)
        i = 0
        while i < limit:
            if buf[i] in _SENTENCE_END:
                j = i + 1
                while j < len(buf) and (buf[j] in _SENTENCE_END or buf[j] in _CLOSERS):
                    j += 1
                if j < len(buf) and buf[j].isspace():
                    return j
                i = j
                continue
            i += 1
        if len(buf) > self.max_chars:
            space = buf.rfind(" ", 0, self.max_chars)
            return space if space > 0 else self.max_chars
        return 0


def split_sentences(text: str, max_chars: int = 200) -> List[str]:
    """A whole reply as the sentences ``SentenceChunker`` would have cut."""
    chunker = SentenceChunker(max_chars)
    return chunker.feed(text or "") + chunker.flush()


class BargeInDetector:
    """Speech over the agent's answer: ``min_ms`` of consecutive voiced audio.

    ``feed`` takes 16-bit mono PCM at ``sample_rate`` and returns True once per
    burst, when the run of frames at or above ``speech_rms`` reaches ``min_ms``.
    A quiet frame resets the run, so a cough or a click does not stop the answer.
    """

    def __init__(self, speech_rms: float = 480.0, min_ms: int = 250,
                 sample_rate: int = SAMPLE_RATE, frame_ms: int = 20) -> None:
        self.speech_rms = float(speech_rms)
        self.min_ms = max(1, int(min_ms))
        self._frame_bytes = int(sample_rate * SAMPLE_WIDTH * frame_ms / 1000)
        self._frame_ms = frame_ms
        self._pending = b""
        self._voiced_ms = 0

    def reset(self) -> None:
        self._pending = b""
        self._voiced_ms = 0

    def feed(self, pcm: bytes) -> bool:
        data = self._pending + (pcm or b"")
        size = self._frame_bytes
        fired = False
        while len(data) >= size:
            frame, data = data[:size], data[size:]
            if pcm_rms(frame) >= self.speech_rms:
                self._voiced_ms += self._frame_ms
                if self._voiced_ms >= self.min_ms:
                    fired = True
                    self._voiced_ms = 0
            else:
                self._voiced_ms = 0
        self._pending = data
        return fired


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

class VoiceLoop:
    """STT -> agent -> TTS, with end-pointing and conversation history.

    ``feed`` is non-throwing by design: a transport should never die because the
    STT service was slow.  Failures go to ``on_error`` and the loop keeps
    listening; ``handle_utterance`` is the exception-raising form for callers
    that want the failure (tests, a CLI probe).
    """

    def __init__(self, transcriber: Transcriber, thinker: Thinker, speaker: Speaker,
                 *, config: Optional[LoopConfig] = None, user: str = "",
                 history_turns: int = 8,
                 on_turn: Optional[Callable[[VoiceTurn], None]] = None,
                 on_error: Optional[Callable[[str, Exception], None]] = None) -> None:
        self.transcriber = transcriber
        self.thinker = thinker
        self.speaker = speaker
        self.config = config or LoopConfig()
        self.segmenter = SpeechSegmenter(self.config)
        self.user = user
        self.history_turns = max(0, history_turns)
        self.history: List[Dict[str, str]] = []
        self.on_turn = on_turn
        self.on_error = on_error

    # -- audio in -----------------------------------------------------------
    def feed(self, pcm: bytes) -> List[VoiceTurn]:
        turns: List[VoiceTurn] = []
        for utterance in self.segmenter.feed(pcm):
            try:
                turn = self.handle_utterance(utterance)
            except Exception as exc:       # the transport stays alive
                self._error("turn", exc)
                continue
            if turn is None:
                continue
            turns.append(turn)
            if self.on_turn is not None:
                try:
                    self.on_turn(turn)
                except Exception as exc:
                    self._error("on_turn", exc)
        return turns

    def handle_utterance(self, pcm: bytes) -> Optional[VoiceTurn]:
        """One complete utterance -> a spoken reply (or None when no words)."""
        text = (self.transcriber.transcribe(pcm, sample_rate=self.config.sample_rate)
                or "").strip()
        if not text:
            return None                       # a cough; never ask the model about it
        return self.answer(text)

    def answer(self, text: str) -> VoiceTurn:
        """A known question -> a spoken reply.  Bypasses STT (typed input, a test)."""
        history = list(self.history)
        reply = (self.thinker.think(text, history=history) or "").strip()
        if not reply:
            raise RuntimeError("the agent answered nothing")
        audio, fmt = self.speaker.speak(reply)
        turn = VoiceTurn(transcript=text, reply=reply, audio=audio or b"",
                         audio_format=fmt or "", user=self.user)
        self._remember("user", text)
        self._remember("assistant", reply)
        return turn

    def handle_utterance_stream(self, pcm: bytes, *, emit, cancel=None,
                                max_chars: int = 200) -> Optional[VoiceTurn]:
        """``handle_utterance`` delivered sentence by sentence (see ``answer_stream``)."""
        started = time.monotonic()
        text = (self.transcriber.transcribe(pcm, sample_rate=self.config.sample_rate)
                or "").strip()
        stt_ms = int((time.monotonic() - started) * 1000)
        if not text:
            return None
        turn = self.answer_stream(text, emit=emit, cancel=cancel, max_chars=max_chars)
        turn.timings["stt_ms"] = stt_ms
        return turn

    def answer_stream(self, text: str, *, emit, cancel=None,
                      max_chars: int = 200) -> VoiceTurn:
        """A question -> sentences, each synthesized and handed to ``emit`` as it lands.

        ``emit(sentence, audio, audio_format)`` queues one clip for playback; it may
        return False to stop the turn (the listener barged in).  A thinker with
        ``think_stream(text, history=..., cancel=...)`` is read incrementally, so the
        first sentence is speaking while the model is still writing the rest; any
        other thinker's whole reply is cut into sentences, which still overlaps TTS
        with playback.  ``cancel`` (a ``threading.Event``) stops the turn between
        sentences.  History records every sentence the agent produced, joined.
        """
        history = list(self.history)
        started = time.monotonic()
        stream = getattr(self.thinker, "think_stream", None)
        sentences: Iterable[str]
        if callable(stream):
            sentences = stream(text, history=history, cancel=cancel)
        else:
            sentences = split_sentences(
                self.thinker.think(text, history=history) or "", max_chars)
        produced: List[str] = []
        timings: Dict[str, int] = {"tts_ms": 0}
        barged = False
        fmt_seen = ""
        try:
            for sentence in sentences:
                sentence = (sentence or "").strip()
                if not sentence:
                    continue
                if "think_ms" not in timings:
                    timings["think_ms"] = int((time.monotonic() - started) * 1000)
                if cancel is not None and cancel.is_set():
                    barged = True
                    break
                produced.append(sentence)
                t0 = time.monotonic()
                audio, fmt = self.speaker.speak(sentence)
                timings["tts_ms"] += int((time.monotonic() - t0) * 1000)
                fmt_seen = fmt or fmt_seen
                if cancel is not None and cancel.is_set():
                    barged = True
                    break
                if emit(sentence, audio or b"", fmt or "") is False:
                    barged = True
                    break
        finally:
            close = getattr(sentences, "close", None)
            if callable(close):
                close()                 # stops the model stream on a barge-in
        timings["think_total_ms"] = int((time.monotonic() - started) * 1000)
        timings["sentences"] = len(produced)
        reply = " ".join(produced)
        if not reply and not barged:
            raise RuntimeError("the agent answered nothing")
        if reply:
            self._remember("user", text)
            self._remember("assistant", reply)
        return VoiceTurn(transcript=text, reply=reply, audio=b"",
                         audio_format=fmt_seen, user=self.user, barged_in=barged,
                         timings=timings)

    def say(self, text: str) -> VoiceTurn:
        """Speak a line with no question (a greeting, a status)."""
        audio, fmt = self.speaker.speak(text)
        return VoiceTurn(transcript="", reply=text, audio=audio or b"",
                         audio_format=fmt or "", user=self.user)

    def reset(self) -> None:
        self.segmenter.reset()
        self.history.clear()

    def _remember(self, role: str, content: str) -> None:
        if self.history_turns <= 0:
            return
        self.history.append({"role": role, "content": content})
        limit = self.history_turns * 2
        if len(self.history) > limit:
            del self.history[: len(self.history) - limit]

    def _error(self, where: str, exc: Exception) -> None:
        if self.on_error is None:
            return
        try:
            self.on_error(where, exc)
        except Exception as callback_exc:      # a broken reporter must not stop audio
            print("awvoice: on_error failed: " + repr(callback_exc), file=sys.stderr)


# ---------------------------------------------------------------------------
# Default adapters (fleet protocols, standard library only)
# ---------------------------------------------------------------------------

def _ssl_context(ca_bundle: Optional[str]) -> Optional[ssl.SSLContext]:
    if not ca_bundle:
        return None
    ctx = ssl.create_default_context(cafile=ca_bundle)
    return ctx


class HttpTranscriber:
    """Multipart audio -> text, for AitherVoice or any OpenAI-style STT door.

    ``url`` is the full endpoint (``https://host:8084/voice/transcribe``).  The
    body is a WAV file, so a Whisper server never has to guess the format.
    """

    def __init__(self, url: str, *, language: Optional[str] = None,
                 headers: Optional[Dict[str, str]] = None, timeout: float = 120.0,
                 ca_bundle: Optional[str] = None) -> None:
        self.url = url
        self.language = language or ""
        self.headers = dict(headers or {})
        self.timeout = timeout
        self._context = _ssl_context(ca_bundle)

    def transcribe(self, pcm: bytes, *, sample_rate: int = SAMPLE_RATE) -> str:
        wav = wav_bytes(pcm, sample_rate=sample_rate)
        boundary = "----awvoice" + base64.urlsafe_b64encode(os.urandom(12)).decode()
        query = "?language=" + urllib.parse.quote(self.language) if self.language else ""
        body = b"\r\n".join([
            ("--" + boundary).encode(),
            b'Content-Disposition: form-data; name="file"; filename="utterance.wav"',
            b"Content-Type: audio/wav",
            b"",
            wav,
            ("--" + boundary + "--").encode(),
            b"",
        ])
        headers = {"Content-Type": "multipart/form-data; boundary=" + boundary}
        headers.update(self.headers)
        data = self._post(self.url + query, body, headers)
        payload = json.loads(data.decode("utf-8") or "{}")
        if isinstance(payload, dict):
            text = payload.get("text") or payload.get("transcript") or ""
            if not text and isinstance(payload.get("data"), dict):
                text = payload["data"].get("text") or ""
            return str(text)
        return str(payload)

    def _post(self, url: str, body: bytes, headers: Dict[str, str]) -> bytes:
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout,
                                        context=self._context) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:300].decode("utf-8", "replace")
            raise RuntimeError(f"STT answered HTTP {exc.code}: {detail}") from exc


class HttpSpeaker:
    """Text -> encoded audio, for AitherVoice ``/synthesize``.

    ``stream=True`` uses ``/synthesize/stream`` (raw audio back, no JSON/base64
    round trip).  Otherwise ``/synthesize`` is asked for ``audio_base64`` so the
    service never has to write a file for a caller that only wants bytes.
    """

    def __init__(self, url: str, *, voice: str = "", format: str = "mp3",
                 extra: Optional[dict] = None, headers: Optional[Dict[str, str]] = None,
                 timeout: float = 120.0, ca_bundle: Optional[str] = None,
                 stream: bool = False) -> None:
        self.url = url
        self.voice = voice
        self.format = format
        self.extra = dict(extra or {})
        self.headers = dict(headers or {})
        self.timeout = timeout
        self._context = _ssl_context(ca_bundle)
        self.stream = stream

    def speak(self, text: str) -> tuple[bytes, str]:
        if self.stream:
            return self._stream(text)
        payload = {"text": text, "voice": self.voice, "format": self.format,
                   "return_base64": True}
        payload.update(self.extra)
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        headers.update(self.headers)
        data = self._post(self.url, body, headers)
        parsed = json.loads(data.decode("utf-8") or "{}")
        if not parsed.get("success", False):
            raise RuntimeError("TTS failed: " + str(parsed.get("error") or "unknown"))
        encoded = parsed.get("audio_base64") or ""
        if not encoded:
            raise RuntimeError("TTS returned no audio")
        return base64.b64decode(encoded), parsed.get("format") or self.format

    def _stream(self, text: str) -> tuple[bytes, str]:
        query = urllib.parse.urlencode({"text": text, "voice": self.voice,
                                        "format": self.format})
        req = urllib.request.Request(self.url + "?" + query, data=b"", method="POST")
        for key, value in self.headers.items():
            req.add_header(key, value)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout,
                                        context=self._context) as resp:
                return resp.read(), self.format
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:300].decode("utf-8", "replace")
            raise RuntimeError(f"TTS answered HTTP {exc.code}: {detail}") from exc

    def _post(self, url: str, body: bytes, headers: Dict[str, str]) -> bytes:
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout,
                                        context=self._context) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:300].decode("utf-8", "replace")
            raise RuntimeError(f"TTS answered HTTP {exc.code}: {detail}") from exc


class OpenAICompatThinker:
    """The agent's normal turn, through any OpenAI-compatible ``/v1`` base."""

    def __init__(self, base_url: str, model: str, *, api_key: str = "",
                 system: str = "", priority: str = "", timeout: float = 120.0,
                 ca_bundle: Optional[str] = None,
                 metadata: Optional[dict] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.system = system
        self.priority = priority
        self.timeout = timeout
        self._context = _ssl_context(ca_bundle)
        self.metadata = dict(metadata or {})

    def think(self, text: str, *, history: Sequence[Dict[str, str]] = ()) -> str:
        messages: List[Dict[str, str]] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        for turn in history or ():
            if turn.get("role") in ("user", "assistant") and turn.get("content"):
                messages.append({"role": turn["role"], "content": str(turn["content"])})
        messages.append({"role": "user", "content": text})
        payload: dict = {"model": self.model, "messages": messages}
        if self.priority:
            meta = {"priority": self.priority, "source": "voice:awvoice"}
            meta.update(self.metadata)
            payload["metadata"] = meta
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        req = urllib.request.Request(self.base_url + "/chat/completions",
                                     data=json.dumps(payload).encode("utf-8"),
                                     headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout,
                                        context=self._context) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:300].decode("utf-8", "replace")
            raise RuntimeError(f"agent answered HTTP {exc.code}: {detail}") from exc
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("agent returned no choices")
        return str((choices[0].get("message") or {}).get("content") or "")


class CallableThinker:
    """Wrap a plain ``fn(text, history) -> reply`` (also the test seam)."""

    def __init__(self, fn: Callable[..., str]) -> None:
        self._fn = fn

    def think(self, text: str, *, history: Sequence[Dict[str, str]] = ()) -> str:
        try:
            return str(self._fn(text, history=history))
        except TypeError:
            return str(self._fn(text))


class CallableTranscriber:
    def __init__(self, fn: Callable[[bytes], str]) -> None:
        self._fn = fn

    def transcribe(self, pcm: bytes, *, sample_rate: int = SAMPLE_RATE) -> str:
        return str(self._fn(pcm))


class CallableSpeaker:
    def __init__(self, fn: Callable[[str], tuple]) -> None:
        self._fn = fn

    def speak(self, text: str) -> tuple[bytes, str]:
        result = self._fn(text)
        if isinstance(result, tuple):
            return result
        return result, "wav"


# ---------------------------------------------------------------------------
# Convenience: build the loop from the environment
# ---------------------------------------------------------------------------

def loop_from_env(*, thinker=None, user: str = "",
                  prefix: str = "AWVOICE_") -> VoiceLoop:
    """A loop wired to the fleet's voice service and agent, overridable by env.

    ``AWVOICE_STT_URL`` / ``AWVOICE_TTS_URL`` name the endpoints (full URLs, as
    ``awvoice`` has always accepted).  ``AWVOICE_TTS_VOICE`` and
    ``AWVOICE_TTS_FORMAT`` pick the voice.  When ``thinker`` is omitted the
    loop uses ``AWVOICE_AGENT_URL`` + ``AWVOICE_AGENT_MODEL``, with
    ``AWVOICE_AGENT_PRIORITY`` (``user`` for a spoken conversation) and
    ``AWVOICE_CA_BUNDLE`` for the internal CA.
    """
    env = os.environ
    stt = env.get(prefix + "STT_URL") or env.get("AWVOICE_STT_URL") or ""
    tts = env.get(prefix + "TTS_URL") or env.get("AWVOICE_TTS_URL") or ""
    if not stt or not tts:
        raise RuntimeError("voice loop needs " + prefix + "STT_URL and " + prefix
                           + "TTS_URL")
    ca = env.get(prefix + "CA_BUNDLE") or env.get("AWDISCORD_CA_BUNDLE") or ""
    headers = {}
    internal = env.get("AITHER_INTERNAL_SECRET") or ""
    if internal:
        headers["X-Internal-Key"] = internal
    transcriber = HttpTranscriber(stt, headers=headers, ca_bundle=ca or None)
    speaker = HttpSpeaker(
        tts, voice=env.get(prefix + "TTS_VOICE") or "",
        format=env.get(prefix + "TTS_FORMAT") or "mp3",
        headers=headers, ca_bundle=ca or None,
        stream=(env.get(prefix + "TTS_STREAM") or "").lower() in ("1", "true", "yes"))
    if thinker is None:
        base = env.get(prefix + "AGENT_URL") or ""
        model = env.get(prefix + "AGENT_MODEL") or "default"
        if not base:
            raise RuntimeError("voice loop needs " + prefix + "AGENT_URL or a thinker")
        thinker = OpenAICompatThinker(
            base, model,
            api_key=env.get(prefix + "AGENT_KEY") or "",
            priority=env.get(prefix + "AGENT_PRIORITY") or "",
            ca_bundle=ca or None)
    return VoiceLoop(transcriber, thinker, speaker, user=user)
