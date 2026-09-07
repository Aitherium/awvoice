# awvoice

Turn speech into text and text into speech, on a service you host.

```bash
pip install awvoice
```

```python
from awvoice import VoiceClient

client = VoiceClient(
    stt_url="http://localhost:5000/transcribe",
    tts_url="http://localhost:5000/synthesize")

text = client.transcribe("voice.wav")
audio = client.synthesize("Hello, world!")
```

```bash
# Set endpoints via environment variables
export AWVOICE_STT_URL=http://localhost:5000/transcribe
export AWVOICE_TTS_URL=http://localhost:5000/synthesize

# Transcribe speech to text
awvoice transcribe voice.wav
awvoice transcribe voice.wav -o output.txt

# Synthesize text to speech
awvoice synthesize "Hello, world!" -o output.wav
```

## How it works

`awvoice` is a client library for speech-to-text (STT) and text-to-speech (TTS) services that **you host and control**.

It does not call any external API or cloud service. Instead, you configure HTTP endpoints to your own services:

```python
client = VoiceClient(
    stt_url="http://your-stt-service.local:8000/transcribe",
    tts_url="http://your-tts-service.local:8000/synthesize")
```

### Service requirements

**Speech-to-text service:**
- Accepts `POST` request with audio bytes
- Returns JSON with a `text` field: `{"text": "transcribed text"}`

**Text-to-speech service:**
- Accepts `POST` request with UTF-8 text bytes
- Returns raw audio bytes (any format your service produces)

### Configuration

Set endpoints via constructor arguments or environment variables:

```python
# Via constructor
client = VoiceClient(
    stt_url="http://localhost:5000/transcribe",
    tts_url="http://localhost:5000/synthesize")

# Via environment variables
# export AWVOICE_STT_URL=http://localhost:5000/transcribe
# export AWVOICE_TTS_URL=http://localhost:5000/synthesize
client = VoiceClient()
```

### Error handling

All errors are explicit and fail closed:

```python
from awvoice import ServiceConfigError, ServiceError

try:
    text = client.transcribe("voice.wav")
except ServiceConfigError as e:
    # Service endpoint not configured
    print(f"Configuration error: {e}")
except ServiceError as e:
    # Service unreachable or returned error
    print(f"Service error: {e}")
except FileNotFoundError as e:
    # Audio file not found
    print(f"File error: {e}")
```

## Testing

```bash
# Run all tests
pip install -e ".[dev]"
pytest

# Run self-tests (no external service required)
awvoice --self-test
```

## The aw family

Standalone tools you can adopt alone. Each replaces something you would otherwise have to trust with something you can check.

| tool | instead of trusting | you check |
|---|---|---|
| [awdk](https://github.com/Aitherium/awdk) | a framework's idea of how your agents should run | one loop you can read |
| [awbac](https://github.com/Aitherium/awbac) | that denied requests are denied | the exact reason for every decision |
| [awvoice](https://github.com/Aitherium/awvoice) | a cloud API for speech | a service you host and control |
| [awkno](https://github.com/Aitherium/awkno) | that you remember the family | the ecosystem in your terminal, offline |

Apache-2.0.
