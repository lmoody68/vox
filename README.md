# 🎙️ VOX — a local-first streaming voice agent (prototype)

A working demonstration of the real voice-agent loop, built fully local + free, reusing the JARVIS stack:

```
mic ──VAD──▶ faster-whisper (STT) ──▶ Groq LLM + tools ──▶ edge-tts / FRIDAY voice (TTS) ──▶ speaker
                              └────────────── barge-in: talk over it and it stops ──────────┘
```

This is the **streaming + barge-in** upgrade from the Voice Agent Build Guide — the piece JARVIS's turn-based
Voice Hub is missing. Same STT engine as **SCRIBE**, your **Groq** key for the LLM, and **FRIDAY's exact voice**
for TTS.

## What it demonstrates
- **The cascaded pipeline** — STT → LLM → TTS as three swappable stages
- **Real tool-calling** — the LLM calls `get_time`, `get_date`, `calculate` mid-conversation (the "do things" layer)
- **VAD turn-taking** — records until you actually stop talking (webrtcvad)
- **Barge-in** — start talking while it's speaking and it stops immediately and listens
- **Conversation memory** — full history passed each turn

## Run it
```bash
pip install faster-whisper edge-tts sounddevice soundfile webrtcvad-wheels imageio-ffmpeg audioop-lts numpy

python vox.py --test          # no-mic self-test: proves STT→LLM(+tools)→TTS  ✅ verified
python vox.py --text "what's 15 times 12 and what day is it?"   # text in → spoken reply
python vox.py                 # LIVE mic conversation (say "goodbye" to exit)
```

## Config (env vars)
| Var | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | read from `API_Keys.txt` | LLM (Groq, OpenAI-compatible) |
| `VOX_LLM` | `openai/gpt-oss-120b` | Groq model (must support tools) |
| `VOX_VOICE` | `en-US-AvaMultilingualNeural` | edge-tts voice (FRIDAY) |
| `VOX_STT` | `base.en` | faster-whisper model |

⚠️ Groq sits behind Cloudflare → the LLM call sends a real `User-Agent` (`curl/8.5.0`) to avoid error 1010.

## How it maps to JARVIS (and the productization path)
VOX is the **streaming loop** JARVIS doesn't have yet. To turn JARVIS into a client-grade voice agent:
1. Adopt this streaming/barge-in loop (or **Pipecat**/**LiveKit** for a hardened version) — keep JARVIS's brain
   routing + `bridge/` tools + FRIDAY voice underneath.
2. Add **Twilio/Vapi** telephony → real phone calls → plugs into the LEGION dialer.
3. Deploy per-client on Proxmox with **GARRISON** in front (input-guard + kill switch) = a sellable product.

Prototype built 2026-09-27. Stack: Python · faster-whisper · Groq · edge-tts · webrtcvad · sounddevice.
Full theory: `../JARVIS/VOICE_AGENT_BUILD_GUIDE.md`.
