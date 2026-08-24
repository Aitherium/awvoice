"""awvoice CLI — transcribe audio or synthesize speech.

Exit codes:
  0 — success
  1 — service error (unreachable, bad response, etc.)
  2 — configuration error (service not configured) or bad input
  3 — could not complete operation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .client import ServiceConfigError, ServiceError, VoiceClient


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI."""
    # GENERATED doctor intercept (gen_aw_doctor.py) -- do not edit
    _dv = locals().get("argv")
    if (_dv if _dv is not None else __import__("sys").argv[1:])[:1] == ["doctor"]:
        from ._doctor import report
        return report()
    ap = argparse.ArgumentParser(
        prog="awvoice",
        description="Transcribe speech or synthesize text with a service you host.",
    )

    # Global options
    ap.add_argument(
        "--stt-url",
        help="STT service endpoint (or set AWVOICE_STT_URL)",
    )
    ap.add_argument(
        "--tts-url",
        help="TTS service endpoint (or set AWVOICE_TTS_URL)",
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="Run self-tests (no external service required)",
    )

    sub = ap.add_subparsers(dest="cmd", required=False)

    # transcribe subcommand
    transcribe_cmd = sub.add_parser("transcribe", help="Convert speech to text")
    transcribe_cmd.add_argument("audio", help="Path to audio file")
    transcribe_cmd.add_argument(
        "-o", "--output", help="Write output to file (default: stdout)"
    )

    # synthesize subcommand
    synthesize_cmd = sub.add_parser("synthesize", help="Convert text to speech")
    synthesize_cmd.add_argument("text", help="Text to synthesize")
    synthesize_cmd.add_argument(
        "-o", "--output", required=True, help="Output audio file path"
    )

    args = ap.parse_args(argv)

    # Handle self-test
    if args.self_test:
        return _run_self_tests()

    # If no command specified, show help
    if not args.cmd:
        ap.print_help()
        return 0

    # Create client
    client = VoiceClient(stt_url=args.stt_url, tts_url=args.tts_url)

    # Run the requested command
    try:
        if args.cmd == "transcribe":
            return _transcribe(client, args.audio, args.output)
        elif args.cmd == "synthesize":
            return _synthesize(client, args.text, args.output)
    except ServiceConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except ServiceError as exc:
        print(f"Service error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 3


def _transcribe(client: VoiceClient, audio_path: str, output: str | None) -> int:
    """Transcribe an audio file."""
    try:
        text = client.transcribe(audio_path)
        if output:
            Path(output).write_text(text, encoding="utf-8")
            print(f"Transcription written to {output}")
        else:
            print(text)
        return 0
    except FileNotFoundError as exc:
        print(f"File error: {exc}", file=sys.stderr)
        return 2


def _synthesize(client: VoiceClient, text: str, output: str) -> int:
    """Synthesize speech from text."""
    try:
        audio = client.synthesize(text)
        Path(output).write_bytes(audio)
        print(f"Audio written to {output}")
        return 0
    except ValueError as exc:
        print(f"Invalid input: {exc}", file=sys.stderr)
        return 2


def _run_self_tests() -> int:
    """Run self-tests that do not require external services.

    Tests that the client structure is sound and error handling works.
    """
    try:
        # Test 1: Client initialization
        client = VoiceClient(stt_url="http://example.com/stt")
        assert client.stt_url == "http://example.com/stt"

        # Test 2: Missing TTS raises clear error
        client_no_tts = VoiceClient(stt_url="http://example.com/stt")
        try:
            client_no_tts.synthesize("test")
            print("FAIL: Should have raised ServiceConfigError for missing TTS")
            return 1
        except ServiceConfigError as exc:
            assert "TTS endpoint not configured" in str(exc)

        # Test 3: Missing STT raises clear error
        client_no_stt = VoiceClient(tts_url="http://example.com/tts")
        try:
            client_no_stt.transcribe("test.wav")
            print("FAIL: Should have raised ServiceConfigError for missing STT")
            return 1
        except ServiceConfigError as exc:
            assert "STT endpoint not configured" in str(exc)

        # Test 4: Empty text is rejected
        client_with_tts = VoiceClient(tts_url="http://example.com/tts")
        try:
            client_with_tts.synthesize("")
            print("FAIL: Should have raised ValueError for empty text")
            return 1
        except ValueError as exc:
            assert "empty" in str(exc).lower()

        # Test 5: Non-existent audio file is rejected
        client_with_stt = VoiceClient(stt_url="http://example.com/stt")
        file_not_found_raised = False
        try:
            client_with_stt.transcribe("/nonexistent/file.wav")
        except FileNotFoundError:
            file_not_found_raised = True

        if not file_not_found_raised:
            print("FAIL: Should have raised FileNotFoundError")
            return 1

        print("All self-tests passed")
        return 0

    except Exception as exc:
        print(f"Self-test failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
