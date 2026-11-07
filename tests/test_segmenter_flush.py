"""SpeechSegmenter.flush: a stream that STOPS (rather than going quiet) still closes.

Discord sends no packets while a speaker is silent, so the trailing silence the
end-pointer waits for never arrives (2026-11-06: 28 decoded frames, no turn)."""

import math
import struct

from awvoice.loop import LoopConfig, SpeechSegmenter


def _tone(ms, amp=3000):
    return b"".join(struct.pack("<h", int(amp * math.sin(2 * math.pi * 220 * i / 16000)))
                    for i in range(int(16 * ms)))


def _feed(seg, pcm):
    out = []
    for i in range(0, len(pcm), 640):
        out += seg.feed(pcm[i:i + 640])
    return out


def test_a_stream_that_just_stops_yields_nothing_until_flushed():
    seg = SpeechSegmenter(LoopConfig())
    assert _feed(seg, _tone(560)) == []
    utterance = seg.flush()
    assert utterance and len(utterance) >= 560 * 32 * 0.9


def test_flush_with_nothing_in_progress_returns_none():
    assert SpeechSegmenter(LoopConfig()).flush() is None


def test_flush_drops_a_blip_shorter_than_min_speech():
    seg = SpeechSegmenter(LoopConfig())
    _feed(seg, _tone(100))
    assert seg.flush() is None


def test_flush_resets_so_the_next_question_starts_clean():
    seg = SpeechSegmenter(LoopConfig())
    _feed(seg, _tone(560))
    assert seg.flush()
    assert seg.flush() is None
    _feed(seg, _tone(560))
    assert seg.flush()


def test_the_normal_silence_path_still_closes_without_flush():
    seg = SpeechSegmenter(LoopConfig())
    assert len(_feed(seg, _tone(560) + _tone(800, amp=0))) == 1
