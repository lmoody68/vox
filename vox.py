"""
VOX — a local-first streaming voice agent (prototype)
=====================================================
Proves the real voice-agent loop, reusing JARVIS's stack, fully local + free:

    mic ──VAD──▶ faster-whisper (STT) ──▶ Groq LLM + tools ──▶ edge-tts / FRIDAY voice (TTS) ──▶ speaker
                                   └──────────── barge-in: talk over it and it stops ───────────┘

This is the "streaming + barge-in" upgrade from the Voice Agent Build Guide — the piece JARVIS's
turn-based Voice Hub is missing. STT = SCRIBE's Whisper. LLM = your Groq key. TTS = FRIDAY's exact voice.

Modes:
    python vox.py                 # LIVE — real mic conversation with barge-in (say "goodbye" to exit)
    python vox.py --text "hi"     # feed text → LLM(+tools) → speak the reply (no mic)
    python vox.py --test          # full pipeline self-test: TTS a question → STT it → LLM → TTS reply
                                  #   (proves every stage without a mic)

Deps: faster-whisper edge-tts sounddevice soundfile webrtcvad-wheels pydub imageio-ffmpeg numpy
"""
from __future__ import annotations
import argparse, ast, asyncio, datetime, json, os, re, sys, tempfile, threading, time, urllib.request

import numpy as np

# ── config ────────────────────────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
VOICE = os.getenv("VOX_VOICE", "en-GB-SoniaNeural")   # VOX's own voice — British female (distinct from FRIDAY)
STT_MODEL = os.getenv("VOX_STT", "base.en")
LLM_MODEL = os.getenv("VOX_LLM", "openai/gpt-oss-120b")         # Groq model that supports tools + Les's key
LLM_BASE = os.getenv("VOX_LLM_BASE", "https://api.groq.com/openai/v1")
SR = 16000                    # mic sample rate for VAD + Whisper
FRAME_MS = 20
FRAME = SR * FRAME_MS // 1000  # 320 samples/frame
SYSTEM = ("You are Vox, a friendly, concise voice assistant. Your name is Vox — always write and say it as a "
          "single word (Vox), never spell it out. Keep replies to 1-2 sentences — you are being spoken aloud. "
          "Use tools when they help. If the user says goodbye, say a short farewell. "
          "You are part of Leslie Moody's AI team; Leslie is your creator (he/him) — refer to him as he/him. "
          "Friday is the command-line voice assistant — always address "
          "her as Friday. Jarvis is Leslie's main voice assistant — always address him as Jarvis. Use these "
          "names whenever you speak to or about them.")


def _groq_key() -> str:
    k = os.getenv("GROQ_API_KEY")
    if k:
        return k
    try:
        src = open(r"C:\Users\lesli\Documents\API_Keys.txt", encoding="utf-8", errors="ignore").read()
        m = re.search(r"gsk_[A-Za-z0-9]{20,}", src)
        if m:
            return m.group(0)
    except Exception:
        pass
    raise RuntimeError("No Groq key (set GROQ_API_KEY or add a gsk_ key to API_Keys.txt)")


# ── tools (function-calling — the 'do things' layer) ────────────────────────────────────────────────
def tool_get_time(_):
    return datetime.datetime.now().strftime("%-I:%M %p") if os.name != "nt" else datetime.datetime.now().strftime("%I:%M %p").lstrip("0")

def tool_get_date(_):
    return datetime.datetime.now().strftime("%A, %B %d, %Y")

def tool_calculate(args):
    expr = (args or {}).get("expression", "")
    node = ast.parse(expr, mode="eval").body
    def ev(n):
        if isinstance(n, ast.Constant): return n.value
        if isinstance(n, ast.BinOp): return {ast.Add:lambda a,b:a+b, ast.Sub:lambda a,b:a-b, ast.Mult:lambda a,b:a*b, ast.Div:lambda a,b:a/b, ast.Pow:lambda a,b:a**b, ast.Mod:lambda a,b:a%b}[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub): return -ev(n.operand)
        raise ValueError("unsupported")
    return str(ev(node))

TOOLS_IMPL = {"get_time": tool_get_time, "get_date": tool_get_date, "calculate": tool_calculate}
TOOLS_SPEC = [
    {"type": "function", "function": {"name": "get_time", "description": "Current local time.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "get_date", "description": "Today's date.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "calculate", "description": "Evaluate a math expression.",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
]


# ── LLM (Groq, OpenAI-compatible, with tool loop) ───────────────────────────────────────────────────
def _chat(messages):
    payload = json.dumps({"model": LLM_MODEL, "messages": messages, "tools": TOOLS_SPEC,
                          "tool_choice": "auto", "temperature": 0.4, "max_tokens": 300}).encode()
    req = urllib.request.Request(LLM_BASE.rstrip("/") + "/chat/completions", data=payload,
        headers={"Authorization": f"Bearer {_groq_key()}", "Content-Type": "application/json",
                 "User-Agent": "curl/8.5.0"})   # UA avoids Cloudflare 1010 in front of Groq
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())["choices"][0]["message"]

def think(history):
    """Run the tool-calling loop; return (reply_text, updated_history)."""
    for _ in range(4):
        msg = _chat(history)
        history.append(msg)
        calls = msg.get("tool_calls")
        if not calls:
            return (msg.get("content") or "").strip(), history
        for c in calls:
            name = c["function"]["name"]
            try:
                args = json.loads(c["function"].get("arguments") or "{}")
            except Exception:
                args = {}
            try:
                result = TOOLS_IMPL.get(name, lambda a: "unknown tool")(args)
            except Exception as e:
                result = f"error: {e}"
            history.append({"role": "tool", "tool_call_id": c["id"], "content": str(result)})
    return "Sorry, I got tangled up there.", history


# ── STT (faster-whisper) ────────────────────────────────────────────────────────────────────────────
_stt = None
def transcribe(wav_path):
    global _stt
    if _stt is None:
        from faster_whisper import WhisperModel
        _stt = WhisperModel(STT_MODEL, device="cpu", compute_type="int8")
    segs, _ = _stt.transcribe(wav_path, vad_filter=True)
    return " ".join(s.text.strip() for s in segs).strip()


# ── TTS (edge-tts → mp3 → PCM via bundled ffmpeg) ───────────────────────────────────────────────────
def _tts_prep(text):
    """Normalize text so it's spoken naturally: say names as words (not spelled), fix run-together sentences."""
    text = re.sub(r"\bVOX\b", "Vox", text)                     # her name as a word, never "V.O.X."
    text = re.sub(r"([.!?,;:])(?=[A-Za-z])", r"\1 ", text)     # add a space after punctuation if glued
    return text

TTS_SR = 24000   # edge-tts native rate
def synth(text):
    """edge-tts → mp3 → decode to float32 PCM with the bundled ffmpeg. Returns (pcm, sample_rate)."""
    import edge_tts, imageio_ffmpeg, subprocess
    text = _tts_prep(text)
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    mp3 = os.path.join(tempfile.gettempdir(), f"vox_{int(time.time()*1000)}.mp3")
    async def _go():
        await edge_tts.Communicate(text, VOICE).save(mp3)
    asyncio.run(_go())
    cmd = [ff, "-nostdin", "-loglevel", "quiet", "-i", mp3,
           "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(TTS_SR), "pipe:1"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    try:
        os.remove(mp3)
    except OSError:
        pass
    pcm = np.frombuffer(raw, dtype=np.float32).copy()
    return pcm, TTS_SR


# ── audio I/O + barge-in ────────────────────────────────────────────────────────────────────────────
def record_utterance(max_sec=15, start_timeout=10, end_silence_ms=800, device=None):
    """Open the mic, wait for speech, record until `end_silence_ms` of silence. Returns wav path or None."""
    import sounddevice as sd, soundfile as sf, webrtcvad
    vad = webrtcvad.Vad(2)
    voiced, started, silence_run, t0 = [], False, 0, time.time()
    end_frames = end_silence_ms // FRAME_MS
    with sd.RawInputStream(samplerate=SR, blocksize=FRAME, dtype="int16", channels=1, device=device) as stream:
        while True:
            data, _ = stream.read(FRAME)
            is_speech = vad.is_speech(bytes(data), SR)
            if not started:
                if is_speech:
                    started = True; voiced.append(bytes(data))
                elif time.time() - t0 > start_timeout:
                    return None
            else:
                voiced.append(bytes(data))
                silence_run = silence_run + 1 if not is_speech else 0
                if silence_run >= end_frames or len(voiced) * FRAME_MS / 1000 > max_sec:
                    break
    pcm = np.frombuffer(b"".join(voiced), dtype=np.int16)
    wav = os.path.join(tempfile.gettempdir(), f"vox_utt_{int(time.time()*1000)}.wav")
    sf.write(wav, pcm, SR)
    return wav

def play_with_bargein(pcm, sr):
    """Play `pcm`; if the user starts talking (VAD), stop immediately. Returns True if interrupted."""
    import sounddevice as sd, webrtcvad
    interrupted = threading.Event()
    def monitor():
        vad = webrtcvad.Vad(3); run = 0
        try:
            with sd.RawInputStream(samplerate=SR, blocksize=FRAME, dtype="int16", channels=1) as mic:
                while not interrupted.is_set():
                    d, _ = mic.read(FRAME)
                    run = run + 1 if vad.is_speech(bytes(d), SR) else 0
                    if run >= 6:   # ~120ms of speech = barge-in
                        interrupted.set(); return
        except Exception:
            pass
    th = threading.Thread(target=monitor, daemon=True); th.start()
    idx, block = 0, 1024
    with sd.OutputStream(samplerate=sr, channels=1, dtype="float32") as out:
        while idx < len(pcm) and not interrupted.is_set():
            out.write(pcm[idx:idx+block]); idx += block
    interrupted.set(); th.join(timeout=0.3)
    return idx < len(pcm)


# ── modes ───────────────────────────────────────────────────────────────────────────────────────────
def play_simple(pcm, sr):
    """Play audio to the end (no mic monitor — used in push-to-talk so room noise can't trip it)."""
    import sounddevice as sd
    sd.play(pcm, sr); sd.wait()

def say(text, do_play=True, bargein=False, tag="VOX"):
    print(f"🔊 {tag}: {text}")
    if do_play:
        pcm, sr = synth(text)
        if bargein:
            return play_with_bargein(pcm, sr)
        play_simple(pcm, sr)
    return False

def list_devices():
    import sounddevice as sd
    print("Audio INPUT devices (use the number with --device N):\n")
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) > 0:
            mark = "  ← default" if i == sd.default.device[0] else ""
            print(f"  [{i}] {d['name']}{mark}")
    print("\nTip: pick your headset/mic, NOT 'Stereo Mix' or anything that captures system sound.")

def run_text(text):
    hist = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
    print(f"👤 you (text): {text}")
    reply, _ = think(hist)
    say(reply)

def run_test():
    print("🧪 VOX pipeline self-test (no mic) — proving STT → LLM(+tools) → TTS end to end\n")
    question = "What time is it, and what is 15 times 12?"
    print(f"1) TTS: synthesizing the test question in FRIDAY's voice…  \"{question}\"")
    qpcm, qsr = synth(question)
    import soundfile as sf
    qwav = os.path.join(tempfile.gettempdir(), "vox_test_q.wav"); sf.write(qwav, qpcm, qsr)
    print("2) STT: transcribing that audio back with Whisper…")
    heard = transcribe(qwav); os.remove(qwav)
    print(f"   → heard: \"{heard}\"")
    print("3) LLM: sending to Groq with tools (get_time, calculate)…")
    hist = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": heard}]
    reply, hist = think(hist)
    used = [m for m in hist if m.get("role") == "tool"]
    print(f"   → tools used: {len(used)}  |  reply: \"{reply}\"")
    print("4) TTS: synthesizing the reply…")
    rpcm, rsr = synth(reply)
    rwav = os.path.join(HERE, "vox_test_reply.wav"); sf.write(rwav, rpcm, rsr)
    print(f"   → saved reply audio: {rwav}")
    print("\n✅ Full pipeline works: audio→text→LLM+tools→text→audio.")

def run_live(device=None, hands_free=False):
    hist = [{"role": "system", "content": SYSTEM}]
    if hands_free:
        # always-listening (only good in a quiet room — picks up ANY speech, incl. TV/other people)
        print("🎙️  VOX is live (hands-free). Speak anytime; talk over me to interrupt. Say \"goodbye\" to exit.\n")
        say("Hey, I'm VOX. What can I do for you?", bargein=False)
        while True:
            print("… listening")
            wav = record_utterance(device=device)
            if not wav:
                print("… (heard nothing)"); continue
            text = transcribe(wav); os.remove(wav)
            if not text:
                continue
            print(f"👤 you: {text}")
            hist.append({"role": "user", "content": text})
            reply, hist = think(hist)
            say(reply, bargein=True)
            if re.search(r"\b(goodbye|bye|good night)\b", text, re.I):
                say("Talk soon."); break
        return
    # PUSH-TO-TALK (default) — ignores the room; only listens when YOU press Enter
    print("🎙️  VOX is ready (push-to-talk).\n")
    print("   ▶ Press ENTER, then speak your question.")
    print("   ▶ It records until you stop talking, then answers.")
    print("   ▶ Type  q  then ENTER to quit.\n")
    while True:
        try:
            cmd = input("[ Press ENTER to talk — or 'q' to quit ] ")
        except EOFError:
            break
        if cmd.strip().lower() in ("q", "quit", "exit", "goodbye", "bye"):
            say("Talk soon."); break
        print("🎤 recording… speak now (pause when done)")
        wav = record_utterance(device=device, start_timeout=6)
        if not wav:
            print("… (heard nothing — press ENTER and speak clearly)\n"); continue
        text = transcribe(wav); os.remove(wav)
        if not text:
            print("… (couldn't make that out — try again)\n"); continue
        print(f"👤 you: {text}")
        hist.append({"role": "user", "content": text})
        reply, hist = think(hist)
        say(reply, bargein=False)
        print()
        if re.search(r"\b(goodbye|good night|that's all|we're done)\b", text, re.I):
            say("Talk soon."); break


def main():
    ap = argparse.ArgumentParser(description="VOX — local streaming voice agent")
    ap.add_argument("--text", help="feed text instead of the mic")
    ap.add_argument("--test", action="store_true", help="run the no-mic pipeline self-test")
    ap.add_argument("--devices", action="store_true", help="list microphone devices and exit")
    ap.add_argument("--device", type=int, default=None, help="input device number (see --devices)")
    ap.add_argument("--hands-free", action="store_true", help="always-listening mode (quiet rooms only)")
    a = ap.parse_args()
    try:
        if a.devices:
            list_devices()
        elif a.test:
            run_test()
        elif a.text:
            run_text(a.text)
        else:
            run_live(device=a.device, hands_free=a.hands_free)
    except KeyboardInterrupt:
        print("\n👋 VOX stopped.")


if __name__ == "__main__":
    main()
