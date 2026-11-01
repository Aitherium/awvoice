"""The shared voice loop: end-pointing, the turn, and the fleet adapters.

Every test drives real PCM through the same code a Discord channel or a browser
microphone uses.  The only fakes are the three collaborators (STT, agent, TTS),
which is the seam that lets this run with no service and no GPU.
"""
from __future__ import annotations

import io
import json
import math
import struct
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from awvoice.loop import (
    CallableSpeaker,
    CallableThinker,
    CallableTranscriber,
    HttpSpeaker,
    HttpTranscriber,
    LoopConfig,
    SpeechSegmenter,
    VoiceLoop,
    pcm_rms,
    wav_bytes,
)

RATE = 16000
FRAME_MS = 20
FRAME_SAMPLES = RATE * FRAME_MS // 1000


def tone(frames: int, amplitude: int = 8000, freq: float = 220.0) -> bytes:
    """``frames`` 20 ms frames of a sine wave: unambiguously speech energy."""
    out = bytearray()
    for f in range(frames):
        for i in range(FRAME_SAMPLES):
            n = f * FRAME_SAMPLES + i
            out += struct.pack("<h", int(amplitude * math.sin(2 * math.pi * freq * n / RATE)))
    return bytes(out)


def silence(frames: int) -> bytes:
    return b"\x00\x00" * (FRAME_SAMPLES * frames)


def cfg(**kw) -> LoopConfig:
    base = dict(frame_ms=FRAME_MS, speech_rms=480.0, start_frames=3,
                silence_ms=200, pre_roll_ms=100, min_speech_ms=120,
                max_utterance_ms=5000)
    base.update(kw)
    return LoopConfig(**base)


class TestPcmHelpers:
    def test_rms_silence_is_zero(self):
        assert pcm_rms(silence(2)) == 0.0

    def test_rms_of_a_tone_is_above_the_speech_floor(self):
        assert pcm_rms(tone(1)) > 1000

    def test_wav_round_trip(self):
        pcm = tone(2)
        data = wav_bytes(pcm, sample_rate=RATE)
        with wave.open(io.BytesIO(data), "rb") as w:
            assert w.getframerate() == RATE
            assert w.getnchannels() == 1
            assert w.getsampwidth() == 2
            assert w.readframes(w.getnframes()) == pcm


class TestSegmenter:
    def test_silence_alone_never_makes_an_utterance(self):
        seg = SpeechSegmenter(cfg())
        assert seg.feed(silence(50)) == []

    def test_a_short_cough_is_dropped(self):
        seg = SpeechSegmenter(cfg(min_speech_ms=200))
        # 60 ms voiced (< start + min) then silence long enough to close it.
        assert seg.feed(tone(3) + silence(20)) == []

    def test_speech_then_silence_is_one_utterance(self):
        seg = SpeechSegmenter(cfg())
        out = seg.feed(tone(15) + silence(20))
        assert len(out) == 1
        assert len(out[0]) >= 15 * FRAME_SAMPLES * 2

    def test_two_utterances_in_one_chunk(self):
        seg = SpeechSegmenter(cfg())
        out = seg.feed(tone(10) + silence(20) + tone(10) + silence(20))
        assert len(out) == 2

    def test_a_max_length_cut_still_returns_audio(self):
        seg = SpeechSegmenter(cfg(max_utterance_ms=200))
        out = seg.feed(tone(30))
        # A continuous tone is chunked at 200 ms rather than dropped or held.
        assert len(out) >= 1
        assert all(chunk for chunk in out)


class TestVoiceLoop:
    def _loop(self, heard="what is the build doing", reply="it is green"):
        seen = {}

        def transcribe(pcm, *, sample_rate=RATE):
            seen["stt_bytes"] = len(pcm)
            return heard

        def think(text, *, history=()):
            seen["history"] = list(history)
            seen["question"] = text
            return reply

        def speak(text):
            seen["spoken"] = text
            return b"AUDIO", "wav"

        loop = VoiceLoop(CallableTranscriber(transcribe), CallableThinker(think),
                         CallableSpeaker(speak), config=cfg())
        return loop, seen

    def test_answer_runs_the_whole_turn_and_keeps_history(self):
        loop, seen = self._loop()
        turn = loop.answer("are we green?")
        assert turn.transcript == "are we green?"
        assert turn.reply == "it is green"
        assert turn.audio == b"AUDIO" and turn.audio_format == "wav"
        assert seen["spoken"] == "it is green"
        assert loop.history == [{"role": "user", "content": "are we green?"},
                                {"role": "assistant", "content": "it is green"}]

    def test_history_reaches_the_next_turn(self):
        loop, seen = self._loop()
        loop.answer("one")
        loop.answer("two")
        assert [t["content"] for t in seen["history"]] == ["one", "it is green"]
        assert seen["question"] == "two"

    def test_an_empty_transcript_is_not_a_question(self):
        loop, seen = self._loop(heard="   ")
        turn = loop.handle_utterance(tone(10))
        assert turn is None
        assert "question" not in seen

    def test_an_empty_reply_raises(self):
        loop, _ = self._loop(reply="   ")
        with pytest.raises(RuntimeError):
            loop.answer("anything")

    def test_feed_drives_pcm_through_stt_and_playback(self):
        loop, seen = self._loop()
        turns = loop.feed(tone(15) + silence(20))
        assert len(turns) == 1
        assert turns[0].reply == "it is green"
        assert seen["stt_bytes"] > 0

    def test_feed_survives_an_stt_failure(self):
        errors = []

        def broken(pcm, *, sample_rate=RATE):
            raise RuntimeError("stt down")

        loop = VoiceLoop(CallableTranscriber(broken),
                         CallableThinker(lambda text, history=(): "x"),
                         CallableSpeaker(lambda text: (b"A", "wav")), config=cfg(),
                         on_error=lambda where, exc: errors.append(where))
        assert loop.feed(tone(15) + silence(20)) == []
        assert errors == ["turn"]

    def test_on_turn_reports_the_completed_exchange(self):
        turns = []
        loop = VoiceLoop(CallableTranscriber(lambda pcm, sample_rate=RATE: "hi"),
                         CallableThinker(lambda text, history=(): "hello"),
                         CallableSpeaker(lambda text: (b"A", "wav")), config=cfg(),
                         on_turn=turns.append)
        loop.feed(tone(15) + silence(20))
        assert [t.reply for t in turns] == ["hello"]

    def test_say_skips_stt(self):
        loop, seen = self._loop()
        turn = loop.say("welcome")
        assert turn.transcript == ""
        assert seen["spoken"] == "welcome"


# ---------------------------------------------------------------------------
# The HTTP adapters, against a loopback server that records what it was sent
# ---------------------------------------------------------------------------

class _Recorder:
    def __init__(self) -> None:
        self.requests = []
        self.reply_text = "transcribed words"
        self.tts_bytes = b"MP3BYTES"


class _Handler(BaseHTTPRequestHandler):
    recorder: _Recorder

    def log_message(self, *args):
        pass

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n)
        self.recorder.requests.append((self.path, self.headers.get("Content-Type", ""), body))
        if "synthesize" in self.path:
            payload = json.dumps({"success": True, "audio_base64": "TVAzQllURVM=",
                                  "format": "mp3"}).encode()
            ctype = "application/json"
        else:
            payload = json.dumps({"text": self.recorder.reply_text}).encode()
            ctype = "application/json"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture()
def recorder():
    rec = _Recorder()

    class Handler(_Handler):
        recorder = rec

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield rec, "http://127.0.0.1:%d" % server.server_address[1]
    finally:
        server.shutdown()


class TestHttpAdapters:
    def test_transcriber_posts_multipart_wav(self, recorder):
        rec, base = recorder
        stt = HttpTranscriber(base + "/voice/transcribe")
        assert stt.transcribe(tone(2)) == "transcribed words"
        path, ctype, body = rec.requests[-1]
        assert path == "/voice/transcribe"
        assert ctype.startswith("multipart/form-data; boundary=")
        assert b"Content-Type: audio/wav" in body
        assert b"RIFF" in body

    def test_speaker_requests_base64_audio(self, recorder):
        rec, base = recorder
        tts = HttpSpeaker(base + "/voice/synthesize", voice="en-GB-SoniaNeural")
        audio, fmt = tts.speak("hello")
        assert audio == b"MP3BYTES" and fmt == "mp3"
        _, _, body = rec.requests[-1]
        sent = json.loads(body)
        assert sent["text"] == "hello"
        assert sent["return_base64"] is True

    def test_speaker_reports_a_failed_synthesis(self, recorder):
        _, base = recorder
        tts = HttpSpeaker(base + "/voice/synthesize")
        tts._post = lambda *a, **k: json.dumps({"success": False, "error": "no lane"}).encode()
        with pytest.raises(RuntimeError, match="no lane"):
            tts.speak("hello")
