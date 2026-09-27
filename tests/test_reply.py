"""`awvoice reply`: the Stop hook speaks a short line, or stays silent -- and never blocks."""

from __future__ import annotations  # noqa: I001

import json
import subprocess
import sys
import threading
import time
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from awvoice import listen as ear
from awvoice import reply


class _FakeDesk:
    """A loopback stand-in for the awdesk bridge that records every POST /speak."""

    def __init__(self, delay: float = 0.0, status: int = 200):
        self.bodies: list[dict] = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                n = int(self.headers.get("Content-Length") or 0)
                outer.bodies.append({"path": self.path, "body": json.loads(self.rfile.read(n))})
                if delay:
                    time.sleep(delay)
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"ok": true}')

            def log_message(self, *a):
                pass

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = "http://127.0.0.1:%d" % self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


@pytest.fixture
def desk():
    d = _FakeDesk()
    yield d
    d.close()


def _env(tmp_path, url, **extra):
    env = {"AITHER_SPEAK_REPLIES": "1", "AWVOICE_DESK_URL": url,
           "AWDESK_CAST_PATH": str(tmp_path / "cast.json")}
    env.update(extra)
    return env


REPLY = """**What changed**
- `AitherOS/packages/awvoice/awvoice/reply.py` - new Stop hook.

```python
print("never spoken")
```

| item | verdict |
|---|---|
| x | y |

CLOSED - the reply hook speaks through the desk, see C:\\AitherOS-Fresh\\x.py for details.
"""


def test_speakable_strips_code_tables_paths_and_markdown():
    line = reply.speakable(REPLY)
    assert "never spoken" not in line and "|" not in line
    assert "reply.py" not in line and "AitherOS-Fresh" not in line and "C:" not in line
    assert "**" not in line and "`" not in line
    assert line.startswith("What changed")
    assert "CLOSED" in line                      # first line + the verdict


def test_speakable_caps_at_two_sentences_and_300_chars():
    many = "One thing. Two things. Three things. Four."
    assert reply.speakable(many) == "One thing. Two things."
    long = "word " * 200
    out = reply.speakable(long)
    assert len(out) <= 301 and out.endswith(".")


def test_speaks_last_assistant_message(tmp_path, desk):
    got = reply.run(json.dumps({"last_assistant_message": "Done. Tests pass.",
                                "session_id": "s"}), _env(tmp_path, desk.url))
    assert got["spoken"] is True
    assert desk.bodies == [{"path": "/speak", "body": {"text": "Done. Tests pass."}}]


def test_falls_back_to_transcript(tmp_path, desk):
    tp = tmp_path / "t.jsonl"
    rows = [
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "Old answer."}]}},
        {"type": "user", "message": {"content": "next"}},
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Bash"}]}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "Newest answer."}]}},
        {"type": "assistant", "message": {"content": [{"type": "thinking", "thinking": "x"}]}},
    ]
    tp.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n", encoding="utf-8")
    got = reply.run(json.dumps({"transcript_path": str(tp)}), _env(tmp_path, desk.url))
    assert got["spoken"] is True and desk.bodies[0]["body"]["text"] == "Newest answer."


def test_silent_without_opt_in(tmp_path, desk):
    env = _env(tmp_path, desk.url)
    env.pop("AITHER_SPEAK_REPLIES")
    assert reply.run(json.dumps({"last_assistant_message": "Hi."}), env)["spoken"] is False
    assert desk.bodies == []


def test_silent_when_desk_muted(tmp_path, desk):
    (tmp_path / "cast.json").write_text(json.dumps({"voice": {"muted": True}}), encoding="utf-8")
    got = reply.run(json.dumps({"last_assistant_message": "Hi."}), _env(tmp_path, desk.url))
    assert got == {"spoken": False, "reason": "desk voice.muted"}
    assert desk.bodies == []


def test_unmuted_cast_speaks(tmp_path, desk):
    (tmp_path / "cast.json").write_text(json.dumps({"voice": {"muted": False}}), encoding="utf-8")
    assert reply.run(json.dumps({"last_assistant_message": "Hi."}),
                     _env(tmp_path, desk.url))["spoken"] is True


def test_slow_desk_is_abandoned_within_timeout(tmp_path):
    slow = _FakeDesk(delay=5.0)
    try:
        t0 = time.monotonic()
        assert reply.post_speak("Hi.", slow.url, timeout=0.5) is False
        assert time.monotonic() - t0 < 2.0
    finally:
        slow.close()


def test_dead_desk_and_garbage_stdin_never_raise(tmp_path):
    env = _env(tmp_path, "http://127.0.0.1:9")
    assert reply.run("{not json", env)["spoken"] is False
    assert reply.run(json.dumps({"last_assistant_message": "Hi."}), env)["spoken"] is False


def test_cli_reply_always_exits_zero(tmp_path, desk):
    env = dict(_env(tmp_path, desk.url))
    import os
    full = dict(os.environ)
    full.update(env)
    proc = subprocess.run([sys.executable, "-m", "awvoice.cli", "reply"],
                          input=json.dumps({"last_assistant_message": "Spoken via CLI."}),
                          capture_output=True, text=True, env=full, timeout=30)
    assert proc.returncode == 0, proc.stderr
    assert desk.bodies and desk.bodies[0]["body"]["text"] == "Spoken via CLI."
    full["AWVOICE_DESK_URL"] = "http://127.0.0.1:9"
    proc = subprocess.run([sys.executable, "-m", "awvoice.cli", "reply"], input="garbage",
                          capture_output=True, text=True, env=full, timeout=30)
    assert proc.returncode == 0


# ── dictation: the host shim catches what the fleet voice cannot reach ─────

def test_listen_falls_back_to_shim_only_when_unreachable():
    calls = []

    def fn(wav, url):
        calls.append(url)
        if url == "https://fleet":
            raise urllib.error.URLError("refused")
        return "heard it"

    assert ear.transcribe_with_fallback(b"x", "https://fleet", "http://shim", fn) == "heard it"
    assert calls == ["https://fleet", "http://shim"]


def test_listen_does_not_paper_over_an_http_verdict():
    def fn(wav, url):
        raise urllib.error.HTTPError(url, 500, "boom", {}, None)

    with pytest.raises(urllib.error.HTTPError):
        ear.transcribe_with_fallback(b"x", "https://fleet", "http://shim", fn)
