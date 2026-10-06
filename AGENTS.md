# awvoice for agents

Read this if you are an agent (or a human) editing this package. Short on
purpose: the commands, the traps that cost a session, and where the rest lives.
Nothing here is read at runtime — it is for you.

## What this is

PyPI distribution **`awvoice`** (version in `pyproject.toml`), import package
`awvoice`, Python >= 3.10. Hear and speak — transcribe audio, synthesize a
voice, against a service you host.

This repository is a **synced mirror** of the AitherOS monorepo (lane
`.github/workflows/sync-awvoice.yml`). Hand edits made here are overwritten on
the next sync — change the source and let the lane publish.

## Build, test, verify

```bash
python -m pytest tests -q        # 77 collected at v0.3.0
pip install -e .                 # editable install for developing against it
```

Measured from a source checkout with no prior install: **66 of 77 pass** —
`test_voice.py` + `test_custom_voices.py` (44), `test_local_voice.py` (10),
`test_reply.py` (12). `test_listen.py` (11) needs a LIVE AUDIO DEVICE and
blocks without one; it runs where the desk runs, not on a bare checkout.
The publish lane (`publish-brick.yml`) additionally builds the wheel, installs
it and imports it — a tree that tests green can still ship a broken wheel.

## Rules that keep this useful

- **A silent desk is an error, not silence.** `test_say_raises_when_desk_is_down_or_silent`
  is the contract: speak() against a missing or mute desk RAISES — an agent
  that believes it spoke is worse than one that knows it did not.
- **Voice is data with a shape.** Custom voices and local voices are separate
  tested paths; a change to the voice manifest lands with its test in the same
  commit, or a named voice starts resolving to a different speaker with no
  error anywhere.
- **Never ship a listening path that can hang forever.** The listen tests wait
  on a real device; the product code must bound every capture the same way,
  or one silent room becomes an agent that never returns.
- **The registry drives the public surface.** This repo's README header,
  `llms.txt` and `aither-manifest.json` are generated from the ecosystem
  registry (one yaml in the AitherOS monorepo) and rewritten on every sync.
  Change the registry; do not hand-edit the generated blocks.

## Read next

- `llms.txt` — the install/use card written for an agent to execute
- `README.md` — the human front door
- `docs/` — the generated docs site source
