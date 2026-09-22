"""The ear's three rules: one holder, nothing retained, and the owner's voice stays the owner's."""

# The repo ruff config knows awvoice is first-party and wants its own block; the
# --isolated run CI performs does not. One noqa beats a file red under exactly one gate.
from __future__ import annotations  # noqa: I001

import io
import json
import os
import wave

import pytest
from awvoice import listen as ear


# ── one microphone, one holder ──────────────────────────────────────────────

def test_a_second_surface_is_refused_and_told_who_has_it(tmp_path):
    lock = tmp_path / "mic.lock"
    with ear.MicLease("awsh", lock):
        with pytest.raises(ear.MicBusyError) as exc:
            with ear.MicLease("awdesk", lock):
                pass
    # "busy" alone is not actionable; the refusal names the holder and its pid.
    assert "awsh" in str(exc.value) and str(os.getpid()) in str(exc.value)


def test_the_lease_is_released_even_when_the_body_raises(tmp_path):
    lock = tmp_path / "mic.lock"
    with pytest.raises(ValueError):
        with ear.MicLease("awsh", lock):
            raise ValueError("capture blew up")
    assert not lock.exists()
    with ear.MicLease("awdesk", lock):          # the next surface may take it
        assert lock.exists()


def test_a_lease_whose_process_is_gone_is_stale_and_may_be_taken(tmp_path):
    lock = tmp_path / "mic.lock"
    # pid 2**31-1 is not a live process; a crashed capture must not mute the machine forever.
    lock.write_text(json.dumps({"pid": 2**31 - 1, "surface": "crashed", "at": "then"}),
                    encoding="utf-8")
    assert ear.read_lease(lock) is None
    with ear.MicLease("awsh", lock):
        assert json.loads(lock.read_text(encoding="utf-8"))["surface"] == "awsh"


def test_an_unreadable_lock_is_not_a_holder(tmp_path):
    lock = tmp_path / "mic.lock"
    lock.write_text("{ this is not json", encoding="utf-8")
    assert ear.read_lease(lock) is None


# ── the audio is a real WAV, because whisper is handed a file ───────────────

def test_to_wav_is_a_container_a_reader_can_open():
    pcm = b"\x00\x01" * 1600                 # 0.1 s of int16 at 16 kHz
    blob = ear.to_wav(pcm)
    with wave.open(io.BytesIO(blob), "rb") as w:
        assert w.getframerate() == ear.SAMPLE_RATE
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getnframes() == 1600


# ── the owner's voice stays the owner's ─────────────────────────────────────

def test_a_transcript_is_published_as_the_human_who_spoke():
    ev = ear.steer_event("sess-1", "run the tests")
    assert ev["actor"]["kind"] == "human"     # load-bearing: only human lands on a live pty
    assert ev["to"] == ["sess-1"] and ev["type"] == "steering" and ev["hops"] == 0
    assert ev["payload"]["text"] == "run the tests"
    assert ev["payload"]["source"] == "voice:awsh"


# ── listen_once: the orchestration, with no mic and no fleet ────────────────

def _fake_capture(seconds):
    return ear.to_wav(b"\x00\x01" * 100)


def test_listen_once_records_transcribes_and_steers(tmp_path):
    published = []
    out = ear.listen_once(
        1.0, surface="awsh", steer="sess-1",
        capture_fn=_fake_capture,
        transcribe_fn=lambda wav: "deploy it",
        publish_fn=lambda ev: published.append(ev) or 42,
        lock_path=tmp_path / "mic.lock",
    )
    assert out["heard"] == "deploy it" and out["steered"] == "sess-1" and out["seq"] == 42
    assert published[0]["payload"]["text"] == "deploy it"


def test_silence_is_a_real_answer_and_steers_nobody(tmp_path):
    published = []
    out = ear.listen_once(
        1.0, steer="sess-1", capture_fn=_fake_capture,
        transcribe_fn=lambda wav: "",
        publish_fn=lambda ev: published.append(ev) or 1,
        lock_path=tmp_path / "mic.lock",
    )
    assert out["heard"] == "" and out["steered"] == "" and published == []


def test_the_lease_is_not_held_across_the_network_call(tmp_path):
    """Transcription takes seconds; holding the mic through it would mute the machine for
    the whole round trip, and the audio is already captured by then."""
    lock = tmp_path / "mic.lock"
    seen = {}

    def transcribe(wav):
        seen["held_during_transcribe"] = lock.exists()
        return "hello"

    ear.listen_once(1.0, capture_fn=_fake_capture, transcribe_fn=transcribe,
                  publish_fn=lambda ev: 0, lock_path=lock)
    assert seen["held_during_transcribe"] is False
    assert not lock.exists()


def test_no_recording_is_left_behind(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    before = set(os.listdir(tmp_path))
    ear.listen_once(1.0, capture_fn=_fake_capture, transcribe_fn=lambda w: "x",
                  publish_fn=lambda ev: 0, lock_path=tmp_path / "mic.lock")
    leftover = {f for f in set(os.listdir(tmp_path)) - before if f.endswith(".wav")}
    assert leftover == set()


def test_a_missing_capture_stack_is_an_instruction_not_a_traceback(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "sounddevice", None)
    with pytest.raises(RuntimeError) as exc:
        ear.capture(0.1)
    assert "awvoice[mic]" in str(exc.value)
