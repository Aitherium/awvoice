"""First audio sooner (early first clause) and no replies to Whisper's phantoms."""

from awvoice.loop import SentenceChunker, VoiceLoop, is_phantom


def _feed_all(chunker, text, step=3):
    out = []
    for i in range(0, len(text), step):
        out += chunker.feed(text[i:i + step])
    return out + chunker.flush()


def test_the_first_chunk_ends_at_an_early_clause_when_asked():
    text = "Sure, I can check that for you right now, and then I will tell you. Done, ok."
    parts = _feed_all(SentenceChunker(200, first_clause_chars=24), text)
    assert parts[0] == "Sure, I can check that for you right now,"
    # only the FIRST chunk is cut early; the rest are whole sentences
    assert parts[1:] == ["and then I will tell you.", "Done, ok."]


def test_without_first_clause_chars_nothing_changes():
    text = "Sure, I can check that for you right now, and then I will tell you. Done."
    assert _feed_all(SentenceChunker(200), text) == [
        "Sure, I can check that for you right now, and then I will tell you.", "Done."]


def test_a_short_clause_is_not_cut_early():
    parts = _feed_all(SentenceChunker(200, first_clause_chars=24), "Sure, that works. Next.")
    assert parts == ["Sure, that works.", "Next."]


def test_phantoms_are_short_filler_only():
    assert is_phantom("Thank you.", 1.0)
    assert is_phantom("  thanks for watching!", 1.8)
    assert not is_phantom("Thank you.", 4.0)
    assert not is_phantom("Thank you for helping with the car.", 1.5)
    assert not is_phantom("What's up?", 1.0)


class _T:
    def __init__(self, text):
        self.text = text

    def transcribe(self, pcm, sample_rate=16000):
        return self.text


class _Think:
    def __init__(self):
        self.asked = []

    def think(self, text, history=()):
        self.asked.append(text)
        return "ok"


class _Speak:
    def speak(self, text):
        return b"x", "mp3"


def test_a_phantom_utterance_is_never_answered():
    thinker = _Think()
    loop = VoiceLoop(_T("Thank you."), thinker, _Speak())
    one_second = b"\x00\x00" * 16000
    assert loop.handle_utterance(one_second) is None
    assert thinker.asked == [] and loop.phantoms == 1
    assert loop.handle_utterance_stream(one_second, emit=lambda *a: True) is None


def test_real_speech_is_still_answered():
    thinker = _Think()
    loop = VoiceLoop(_T("What's up?"), thinker, _Speak())
    turn = loop.handle_utterance(b"\x00\x00" * 16000)
    assert turn is not None and thinker.asked == ["What's up?"]
