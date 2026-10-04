"""``local:aither``: the Aither voice on this machine, with no service running.

Offline: a loopback server stands in for the release, and the speech runtime is
monkeypatched except in the one real-synthesis test, which runs only where
onnxruntime + espeakng-loader AND an installed voice are present.
"""
from __future__ import annotations

import hashlib
import http.server
import importlib.util
import io
import socketserver
import threading
import wave

import pytest
from awvoice import aither_voice_runtime as rt
from awvoice import cli, local
from awvoice.client import ServiceConfigError, VoiceClient


def _wav(seconds: float = 0.25, rate: int = 22050) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\0\0" * int(rate * seconds))
    return buf.getvalue()


def test_split_local():
    assert local.split_local("local:aither") == "aither"
    assert local.split_local("local:") is None
    assert local.split_local("custom:aither") is None
    assert local.split_local(None) is None


def test_client_routes_local_voices_on_device_and_needs_no_url(monkeypatch):
    monkeypatch.delenv("AWVOICE_TTS_URL", raising=False)
    seen = {}

    def _speak(text, **kw):
        seen["text"] = text
        seen["speed"] = kw.get("speed")
        return _wav()

    monkeypatch.setattr(rt, "speak", _speak)
    audio = VoiceClient().synthesize("Hello.", voice="local:aither", speed=1.2)
    assert audio[:4] == b"RIFF" and seen == {"text": "Hello.", "speed": 1.2}


def test_custom_and_url_paths_are_unchanged(monkeypatch):
    """A custom: id still goes to the workspace; no voice still needs AWVOICE_TTS_URL."""
    monkeypatch.delenv("AWVOICE_TTS_URL", raising=False)
    import awvoice.genesis as genesis

    monkeypatch.setattr(genesis, "say_custom",
                        lambda name, text, speed=None: b"custom:" + name.encode())
    monkeypatch.setattr(rt, "speak", lambda *a, **k: pytest.fail("local runtime used"))
    assert VoiceClient().synthesize("hi", voice="custom:grandma") == b"custom:grandma"
    with pytest.raises(ServiceConfigError, match="TTS endpoint not configured"):
        VoiceClient().synthesize("hi")


def test_unknown_local_voice_is_a_config_error():
    with pytest.raises(ServiceConfigError, match="local:aither"):
        local.say_local("nobody", "hi")


def test_missing_runtime_names_the_extra(monkeypatch):
    def _missing(*a, **k):
        raise rt.RuntimeMissingError("onnxruntime and numpy are not installed")

    monkeypatch.setattr(rt, "speak", _missing)
    with pytest.raises(ServiceConfigError, match=r"awvoice\[local\]"):
        local.say_local("aither", "hi")


def test_cli_say_local_writes_a_wav_without_desk_or_service(monkeypatch, tmp_path):
    monkeypatch.delenv("AWVOICE_TTS_URL", raising=False)
    monkeypatch.setenv("AWVOICE_DESK_URL", "http://127.0.0.1:9")  # would fail if used
    monkeypatch.setattr(rt, "speak", lambda text, **kw: _wav())
    out = tmp_path / "say.wav"
    assert cli.main(["say", "--voice", "local:aither", "Welcome to Aitherium.",
                     "-o", str(out)]) == 0
    assert out.read_bytes()[:4] == b"RIFF"


def test_cli_synthesize_accepts_a_local_voice_and_still_refuses_others(monkeypatch, tmp_path):
    monkeypatch.setattr(rt, "speak", lambda text, **kw: _wav())
    out = tmp_path / "s.wav"
    assert cli.main(["synthesize", "hi", "--voice", "local:aither", "-o", str(out)]) == 0
    assert out.is_file()
    assert cli.main(["synthesize", "hi", "--voice", "alloy", "-o", str(out)]) == 2


# ---------------------------------------------------------------- pinned fetch

BLOB = bytes(i % 241 for i in range(3000))


class _H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):  # noqa: D102
        pass

    def do_GET(self):  # noqa: N802
        if self.path != "/v.onnx":
            self.send_response(404)
            self.end_headers()
            return
        start = 0
        rng = self.headers.get("Range", "")
        if rng.startswith("bytes="):
            start = int(rng[6:].split("-")[0])
            self.send_response(206)
        else:
            self.send_response(200)
        self.send_header("Content-Length", str(len(BLOB) - start))
        self.end_headers()
        self.wfile.write(BLOB[start:])


@pytest.fixture()
def base():
    srv = socketserver.TCPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/"
    srv.shutdown()
    srv.server_close()


def test_fetch_is_sha256_pinned(base, tmp_path):
    sha = hashlib.sha256(BLOB).hexdigest()
    good = rt.Pin("v.onnx", len(BLOB), sha, [base + "nope", base + "v.onnx"])
    assert rt.fetch_pinned(good, tmp_path).read_bytes() == BLOB
    bad = rt.Pin("w.onnx", len(BLOB), "f" * 64, [base + "v.onnx"])
    with pytest.raises(rt.VoiceError, match="sha256"):
        rt.fetch_pinned(bad, tmp_path)
    assert not (tmp_path / "w.onnx").exists() and not (tmp_path / "w.onnx.part").exists()
    wrong_size = rt.Pin("x.onnx", len(BLOB) - 1, sha, [base + "v.onnx"])
    with pytest.raises(rt.VoiceError):
        rt.fetch_pinned(wrong_size, tmp_path)
    assert not (tmp_path / "x.onnx").exists()


def test_phoneme_framing():
    ids = {"^": [1], "$": [2], "_": [0], "a": [14], "ɪ": [74]}
    assert rt.phoneme_ids(list("aɪ"), ids) == [1, 14, 0, 74, 0, 2]


_REAL = all(importlib.util.find_spec(m) for m in ("onnxruntime", "espeakng_loader", "numpy"))


@pytest.mark.skipif(not (_REAL and rt.installed()),
                    reason="needs onnxruntime + espeakng-loader and an installed voice")
def test_real_local_synthesis_when_available(tmp_path):
    out = tmp_path / "w.wav"
    assert cli.main(["say", "--voice", "local:aither", "Welcome to Aitherium.",
                     "-o", str(out)]) == 0
    assert 0.5 < rt.wav_seconds(out.read_bytes()) < 6.0
