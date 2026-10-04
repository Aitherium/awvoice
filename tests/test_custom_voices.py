"""Custom voices (custom:<name>) against a mocked /voice-builds endpoint. No network."""

from __future__ import annotations

import base64
import io
import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest
from awvoice import (
    GenesisAuthError,
    ServiceConfigError,
    ServiceError,
    VoiceClient,
    list_custom_voices,
    say_custom,
)
from awvoice import cli as awcli
from awvoice.genesis import DEFAULT_API, api_base, split_voice

WAV = b"RIFF\x24\x00\x00\x00WAVEfmt "


class FakeOpener:
    """Records every Request and answers with a canned JSON body or an exception."""

    def __init__(self, body=None, exc=None):
        self.body = body
        self.exc = exc
        self.requests = []

    def __call__(self, req, timeout=None):
        self.requests.append(req)
        if self.exc is not None:
            raise self.exc
        resp = MagicMock()
        resp.read.return_value = json.dumps(self.body).encode("utf-8")
        cm = MagicMock()
        cm.__enter__.return_value = resp
        cm.__exit__.return_value = False
        return cm


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("AITHER_API_KEY", "test-key")
    monkeypatch.delenv("AWVOICE_GENESIS_URL", raising=False)
    for var in ("AITHER_GENESIS_URL", "AITHER_API_URL", "AITHER_GATEWAY_URL",
                "AITHER_PORTAL_URL", "AITHER_ELYSIUM_URL"):
        monkeypatch.delenv(var, raising=False)


def test_split_voice():
    assert split_voice("custom:grandma") == "grandma"
    assert split_voice("af_heart") is None
    assert split_voice(None) is None
    assert split_voice("custom:") is None


def test_empty_list_is_fine():
    op = FakeOpener({"voices": []})
    assert list_custom_voices(opener=op) == []
    req = op.requests[0]
    assert req.get_method() == "GET"
    assert req.full_url == f"{DEFAULT_API}/voice-builds/voices"
    assert req.get_header("Authorization") == "Bearer test-key"


def test_ids_get_custom_prefix():
    op = FakeOpener({"voices": [{"name": "grandma", "reader": "kokoro", "language": "en",
                                 "built_at": "2026-10-03T00:00:00Z", "gate": {"wer": 0.04}}]})
    voices = list_custom_voices(opener=op)
    assert voices[0]["id"] == "custom:grandma"
    assert voices[0]["reader"] == "kokoro"


def test_base_url_env_override(monkeypatch):
    monkeypatch.setenv("AWVOICE_GENESIS_URL", "https://example.test/api/genesis/")
    op = FakeOpener({"voices": []})
    list_custom_voices(opener=op)
    assert op.requests[0].full_url == "https://example.test/api/genesis/voice-builds/voices"


def test_say_url_body_and_decode():
    op = FakeOpener({"audio_base64": base64.b64encode(WAV).decode(), "format": "wav",
                     "voice": "custom:my voice"})
    out = say_custom("my voice", "x" * 1500, speed=1.25, opener=op)
    assert out == WAV
    req = op.requests[0]
    assert req.get_method() == "POST"
    assert req.full_url == f"{DEFAULT_API}/voice-builds/voices/my%20voice/say"
    body = json.loads(req.data)
    assert body == {"text": "x" * 1000, "speed": 1.25}


def test_speed_out_of_range():
    with pytest.raises(ValueError):
        say_custom("g", "hi", speed=3.0, opener=FakeOpener({}))


def test_401_is_auth_error():
    exc = urllib.error.HTTPError("u", 401, "Unauthorized", {}, io.BytesIO(b"no"))
    with pytest.raises(GenesisAuthError, match="AITHER_API_KEY"):
        list_custom_voices(opener=FakeOpener(exc=exc))


def test_500_is_not_reported_unreachable():
    exc = urllib.error.HTTPError("u", 500, "boom", {}, io.BytesIO(b"kaput"))
    with pytest.raises(ServiceError, match="HTTP 500") as ei:
        list_custom_voices(opener=FakeOpener(exc=exc))
    assert "Cannot reach" not in str(ei.value)


def test_missing_bearer(monkeypatch):
    monkeypatch.delenv("AITHER_API_KEY")
    with pytest.raises(ServiceConfigError, match="AITHER_API_KEY"):
        list_custom_voices(opener=FakeOpener({"voices": []}))


def test_client_routes_custom_voice_through_api():
    payload = {"audio_base64": base64.b64encode(WAV).decode(), "format": "wav"}
    with patch("urllib.request.urlopen", FakeOpener(payload)) as op:
        # tts_url unset: a custom voice must not need it
        assert VoiceClient().synthesize("hi", voice="custom:grandma") == WAV
    assert op.requests[0].full_url.endswith("/voice-builds/voices/grandma/say")


def test_client_stock_voice_still_hits_tts_url():
    with patch("urllib.request.urlopen") as mock_urlopen:
        resp = MagicMock()
        resp.status = 200
        resp.read.return_value = b"audio"
        mock_urlopen.return_value.__enter__.return_value = resp
        out = VoiceClient(tts_url="http://tts.example.com").synthesize("hi", voice="af_heart")
    assert out == b"audio"
    assert mock_urlopen.call_args[0][0].full_url == "http://tts.example.com"


def test_stock_http_error_regression():
    """HTTPError subclasses URLError: a 503 must say HTTP 503, not 'Cannot reach'."""
    exc = urllib.error.HTTPError("http://tts.example.com", 503, "busy", {},
                                 io.BytesIO(b"overloaded"))
    with patch("urllib.request.urlopen", side_effect=exc):
        with pytest.raises(ServiceError, match=r"HTTP 503\): overloaded"):
            VoiceClient(tts_url="http://tts.example.com").synthesize("hi")


def test_cli_voices_empty(capsys):
    with patch("urllib.request.urlopen", FakeOpener({"voices": []})):
        rc = awcli.main(["voices"])
    assert rc == 0
    assert "(no custom voices yet)" in capsys.readouterr().out


def test_cli_voices_json(capsys):
    with patch("urllib.request.urlopen", FakeOpener({"voices": [{"name": "g"}]})):
        rc = awcli.main(["voices", "--json"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["voices"][0]["id"] == "custom:g"


def test_cli_voices_no_key(monkeypatch, capsys):
    monkeypatch.delenv("AITHER_API_KEY")
    assert awcli.main(["voices"]) == 2


def test_cli_synthesize_custom_writes_wav(tmp_path):
    out = tmp_path / "o.wav"
    payload = {"audio_base64": base64.b64encode(WAV).decode()}
    with patch("urllib.request.urlopen", FakeOpener(payload)):
        rc = awcli.main(["synthesize", "hi", "--voice", "custom:g", "-o", str(out)])
    assert rc == 0
    assert out.read_bytes() == WAV


def test_base_follows_adk_portal(monkeypatch):
    """An adk-enrolled box points at its own portal; the key must go THERE."""
    assert api_base() == DEFAULT_API == "https://api.aitherium.com/api/genesis"
    monkeypatch.setenv("AITHER_PORTAL_URL", "https://portal.example/")
    assert api_base() == "https://portal.example/api/genesis"
    monkeypatch.setenv("AITHER_API_URL", "https://genesis.example")
    assert api_base() == "https://genesis.example"


def test_403_keeps_server_detail():
    """The portal proxy answers 403 'path not allowed' for an unlisted path: say so."""
    exc = urllib.error.HTTPError("u", 403, "Forbidden", {},
                                 io.BytesIO(b'{"error":"path not allowed"}'))
    with pytest.raises(GenesisAuthError, match="path not allowed"):
        list_custom_voices(opener=FakeOpener(exc=exc))


def test_say_rejects_bare_prefix():
    with pytest.raises(ValueError):
        say_custom("custom:", "hi", opener=FakeOpener({}))


def test_cli_synthesize_rejects_stock_voice_and_stray_speed(tmp_path, capsys):
    out = tmp_path / "o.wav"
    with patch("urllib.request.urlopen") as mock_urlopen:
        assert awcli.main(["synthesize", "hi", "--voice", "af_heart", "-o", str(out)]) == 2
        assert awcli.main(["synthesize", "hi", "--speed", "1.2", "-o", str(out)]) == 2
    mock_urlopen.assert_not_called()
    assert not out.exists()
