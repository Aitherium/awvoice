"""Streaming replies: sentence cuts, the per-sentence pipeline, barge-in detection.

WHY THESE CASES
---------------
A voice reply used to wait for the whole model answer, then the whole TTS clip. The
stream path cuts the answer into sentences as it arrives and hands each one to TTS and
playback, so the first words play while the rest is still being written. These pin the
order of the sentences, the history (the FULL reply, not the last sentence), and that a
cancel mid-answer stops the remaining sentences and the model stream.
"""

from __future__ import annotations

import struct
import threading

from awvoice.loop import (
    BargeInDetector,
    CallableSpeaker,
    CallableThinker,
    CallableTranscriber,
    SentenceChunker,
    VoiceLoop,
    split_sentences,
)


def _loud(ms: int, amp: int = 3000) -> bytes:
    return struct.pack("<h", amp) * (16 * ms)


def _quiet(ms: int) -> bytes:
    return b"\x00\x00" * (16 * ms)


class TestSentenceChunker:
    def test_deltas_come_out_as_sentences_in_order(self):
        c = SentenceChunker()
        out = []
        for piece in ["Hel", "lo there. How", " are you? I am", " fine! Pi is 3.", "14 ok"]:
            out += c.feed(piece)
        out += c.flush()
        assert out == ["Hello there.", "How are you?", "I am fine!", "Pi is 3.14 ok"]

    def test_a_long_run_is_cut_at_a_space_near_the_limit(self):
        text = ("word " * 80).strip()
        parts = split_sentences(text, max_chars=200)
        assert len(parts) > 1
        assert all(len(p) <= 200 for p in parts)
        assert " ".join(parts) == text

    def test_a_closing_quote_stays_with_its_sentence(self):
        assert split_sentences('He said "go." Then left.') == ['He said "go."', "Then left."]


class TestBargeInDetector:
    def test_fires_after_250ms_of_speech(self):
        d = BargeInDetector(speech_rms=480, min_ms=250)
        assert d.feed(_loud(200)) is False
        assert d.feed(_loud(60)) is True

    def test_a_short_click_does_not_fire(self):
        d = BargeInDetector(speech_rms=480, min_ms=250)
        assert d.feed(_loud(100) + _quiet(40) + _loud(100) + _quiet(40)) is False


def _loop(thinker, spoken):
    def speak(text):
        spoken.append(text)
        return (b"A:" + text.encode(), "wav")
    return VoiceLoop(CallableTranscriber(lambda pcm: "question"), thinker,
                     CallableSpeaker(speak))


class _StreamThinker:
    def __init__(self, sentences):
        self.sentences = sentences
        self.closed = False
        self.history_seen = None

    def think(self, text, *, history=()):
        raise AssertionError("the stream path must not call think()")

    def think_stream(self, text, *, history=(), cancel=None):
        self.history_seen = list(history)
        try:
            for s in self.sentences:
                yield s
        finally:
            self.closed = True


class TestAnswerStream:
    def test_sentences_are_synthesized_and_emitted_in_order(self):
        spoken, emitted = [], []
        thinker = _StreamThinker(["One.", "Two!", "Three?"])
        loop = _loop(thinker, spoken)
        turn = loop.handle_utterance_stream(
            b"\x00\x00" * 100, emit=lambda s, a, f: emitted.append((s, a, f)))
        assert spoken == ["One.", "Two!", "Three?"]
        assert [e[0] for e in emitted] == ["One.", "Two!", "Three?"]
        assert emitted[0][1] == b"A:One." and emitted[0][2] == "wav"
        assert turn.reply == "One. Two! Three?"
        assert turn.barged_in is False
        assert {"stt_ms", "think_ms", "tts_ms"} <= set(turn.timings)

    def test_history_records_the_full_joined_reply(self):
        loop = _loop(_StreamThinker(["First part.", "Second part."]), [])
        loop.answer_stream("q", emit=lambda s, a, f: None)
        assert loop.history == [{"role": "user", "content": "q"},
                                {"role": "assistant", "content": "First part. Second part."}]

    def test_a_plain_thinker_reply_is_split_into_sentences(self):
        spoken = []
        loop = _loop(CallableThinker(lambda t: "Alpha. Beta."), spoken)
        turn = loop.answer_stream("q", emit=lambda s, a, f: None)
        assert spoken == ["Alpha.", "Beta."]
        assert turn.reply == "Alpha. Beta."

    def test_cancel_stops_the_remaining_sentences_and_closes_the_stream(self):
        spoken = []
        cancel = threading.Event()
        thinker = _StreamThinker(["One.", "Two.", "Three."])
        loop = _loop(thinker, spoken)

        def emit(sentence, audio, fmt):
            cancel.set()           # the listener spoke over sentence one
        turn = loop.answer_stream("q", emit=emit, cancel=cancel)
        assert spoken == ["One."]
        assert turn.barged_in is True
        assert thinker.closed is True
        assert loop.history[-1] == {"role": "assistant", "content": "One."}

    def test_emit_returning_false_is_a_barge_in(self):
        spoken = []
        loop = _loop(_StreamThinker(["One.", "Two."]), spoken)
        turn = loop.answer_stream("q", emit=lambda s, a, f: False)
        assert spoken == ["One."] and turn.barged_in is True
