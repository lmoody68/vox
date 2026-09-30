"""
VOX — a local-first streaming voice agent (prototype)
=====================================================
Proves the real voice-agent loop, reusing JARVIS's stack, fully local + free:

    mic ──VAD──▶ faster-whisper (STT) ──▶ Groq LLM + tools ──▶ edge-tts / FRIDAY voice (TTS) ──▶ speaker
                                   └──────────── barge-in: talk over it and it stops ───────────┘

This is the "streaming + barge-in" upgrade from the Voice Agent Build Guide — the piece JARVIS's
turn-based Voice Hub is missing. STT = SCRIBE's Whisper. LLM = your Groq key. TTS = FRIDAY's exact voice.

Modes:
    python vox.py                 # PUSH-TO-TALK (default) — press ENTER, then speak (Start_VOX.bat)
    python vox.py --wake          # HANDS-FREE wake word — say "Hey Vox" then your question (Start_VOX_HeyVox.bat)
    python vox.py --enroll        # record YOUR voiceprint so wake mode only answers you (ignores TV/others)
    python vox.py --hands-free    # always-listening — responds to ANY speech (quiet rooms only)
    python vox.py --text "hi"     # feed text → LLM(+tools) → speak the reply (no mic)
    python vox.py --test          # full pipeline self-test: TTS a question → STT it → LLM → TTS reply
                                  #   (proves every stage without a mic)

Deps: faster-whisper edge-tts sounddevice soundfile webrtcvad-wheels pydub imageio-ffmpeg numpy
      speechbrain torch torchaudio      # (optional) voiceprint speaker verification for --enroll / --wake
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
WAKE_MIN_RMS = int(os.getenv("VOX_MIN_RMS", "450"))   # ignore background quieter than this (near-field gate)
_last_rms = 0.0               # loudness of the most recent utterance (for the near-field/background gate)
WAKE = os.getenv("VOX_WAKE", "Hey Vox")               # wake phrase for hands-free mode
# Whisper (base.en) often mishears 'Vox' — accept these look-alikes as the wake token.
_WAKE_ALTS = ("vox", "box", "fox", "vaux", "volks", "folks", "walks", "vaults", "vault",
              "vocs", "vex", "bucks", "vaux", "vaughs", "faux", "vox's")

# ── voiceprint (optional speaker verification — only answer the enrolled owner, ignore TV/other people) ──
VOICEPRINT_PATH = os.path.join(HERE, "voiceprint.npy")     # enrolled owner embedding (created by --enroll)
ECAPA_DIR = os.path.join(HERE, "models", "ecapa")          # local cache of the ECAPA-TDNN speaker model
VERIFY_THRESHOLD = float(os.getenv("VOX_VOICEPRINT_THRESHOLD", "0.55"))  # cosine cutoff (same-speaker >0.7, others <0.35)
_encoder = None                                            # cached ECAPA encoder (False = tried and unavailable)

_PERSONA = ("You are Vox, a friendly, concise voice assistant. Your name is Vox — always write and say it as a "
            "single word (Vox), never spell it out. Keep replies to 1-2 sentences — you are being spoken aloud. "
            "Use tools when they help. If the user says goodbye, say a short farewell. ")
_OP_PTT = ("HOW YOU WORK (state this accurately — NEVER invent features): you are in PUSH-TO-TALK mode. The user "
           "presses ENTER, then speaks; you reply when they pause. In THIS mode there is no wake word. There is "
           "also a hands-free WAKE-WORD mode the user can start with Start_VOX_HeyVox.bat, where the wake word is "
           "'Hey Vox'. There is no settings/voice-activation menu. If asked how to talk to you now, say: press "
           "ENTER then speak (type q then ENTER to quit). If you don't know something about your own app, say so. ")
_OP_WAKE = ("HOW YOU WORK: you are in HANDS-FREE mode. The APP detects the wake word 'Hey Vox' and REMOVES it "
            "before the message reaches you — so every message you receive is already a genuine request. Just "
            "ANSWER it directly and helpfully (use tools when relevant). NEVER tell the user to say 'Hey Vox', to "
            "start with a wake word, or to rephrase — that is handled for them automatically; doing so is wrong. "
            "If they ask how they talk to you, explain: say 'Hey Vox' then your question, and the app does the "
            "rest. There is also a push-to-talk mode. If you don't know something about your own app, say so. ")
_NAMES = ("You are part of Leslie Moody's AI team; Leslie is your creator (he/him) — refer to him as he/him. "
          "Friday is the command-line voice assistant — always address her as Friday. Jarvis is Leslie's main "
          "voice assistant — always address him as Jarvis. Use these names whenever you speak to or about them.")
SYSTEM = _PERSONA + _OP_PTT + _NAMES          # push-to-talk / text / test modes
SYSTEM_WAKE = _PERSONA + _OP_WAKE + _NAMES    # hands-free wake-word mode


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


def warm_stt():
    """Load Whisper up front so the FIRST turn isn't slowed by the ~1s cold model load."""
    global _stt
    if _stt is None:
        from faster_whisper import WhisperModel
        _stt = WhisperModel(STT_MODEL, device="cpu", compute_type="int8")


# ── voiceprint speaker verification (ECAPA-TDNN) — only respond to the enrolled owner ─────────────────
def _get_encoder():
    """Load the ECAPA speaker encoder once (offline after the first download). None if unavailable."""
    global _encoder
    if _encoder is not None:
        return _encoder or None
    try:
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy      # COPY: Windows blocks symlinks w/o admin
        src = ECAPA_DIR if os.path.exists(os.path.join(ECAPA_DIR, "hyperparams.yaml")) \
            else "speechbrain/spkrec-ecapa-voxceleb"
        _encoder = EncoderClassifier.from_hparams(source=src, savedir=ECAPA_DIR,
                                                  run_opts={"device": "cpu"}, local_strategy=LocalStrategy.COPY)
    except Exception as e:
        print(f"   (voiceprint model unavailable: {e} — verification skipped)")
        _encoder = False
    return _encoder or None


def embed_wav(wav_path):
    """Return a normalised ECAPA embedding (np.float32) for a 16k-mono wav, or None if it can't be made."""
    enc = _get_encoder()
    if enc is None:
        return None
    try:
        import soundfile as sf, torch
        data, sr = sf.read(wav_path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(1)
        if len(data) < SR // 4:                          # need at least ~0.25s of audio
            return None
        wav = torch.from_numpy(np.ascontiguousarray(data)).unsqueeze(0)
        with torch.no_grad():
            e = enc.encode_batch(wav).squeeze()
        e = e / e.norm()
        return e.cpu().numpy().astype(np.float32)
    except Exception:
        return None


def load_voiceprint():
    """Return the saved owner embedding (normalised), or None if not enrolled."""
    if not os.path.exists(VOICEPRINT_PATH):
        return None
    try:
        vp = np.load(VOICEPRINT_PATH).astype(np.float32)
        n = float(np.linalg.norm(vp))
        return vp / n if n else None
    except Exception:
        return None


def verify_speaker(wav_path):
    """(ok, score). Not enrolled -> (True, -1). Can't embed -> (True, -2) (fail-open so VOX still works)."""
    vp = load_voiceprint()
    if vp is None:
        return True, -1.0
    emb = embed_wav(wav_path)
    if emb is None:
        return True, -2.0
    score = float(np.dot(emb, vp))
    return score >= VERIFY_THRESHOLD, score


def run_enroll(device=None, samples=5):
    """Record several utterances and save YOUR voiceprint, so wake mode ignores the TV and other people."""
    print("\n🔒 VOX voiceprint enrollment — teach VOX your voice so it only listens to you.")
    if _get_encoder() is None:
        print("   Speaker model unavailable — cannot enroll.  (Fix:  pip install speechbrain)\n"); return
    warm_stt()
    phrases = [
        "Hey Vox, what's on my calendar today?",
        "Hey Vox, set a timer for ten minutes please.",
        "The quick brown fox jumps over the lazy dog.",
        "Hey Vox, what's the weather like this afternoon?",
        "I am enrolling my voice so you only listen to me.",
    ]
    embs, i = [], 0
    while len(embs) < samples:
        phrase = phrases[i % len(phrases)]
        i += 1
        print(f'\n   [{len(embs)+1}/{samples}]  Say, clearly:  "{phrase}"')
        try:
            input("   Press ENTER, then speak… ")
        except EOFError:
            pass
        wav = record_utterance(device=device, start_timeout=8, max_sec=8)
        if not wav:
            print("   (heard nothing — let's redo that one)"); continue
        emb = embed_wav(wav); os.remove(wav)
        if emb is None:
            print("   (couldn't read that clearly — redo)"); continue
        embs.append(emb); print("   ✓ got it")
    vp = np.mean(np.stack(embs), axis=0)
    vp = vp / np.linalg.norm(vp)
    sims = [float(np.dot(a, b)) for x, a in enumerate(embs) for b in embs[x + 1:]]
    consistency = sum(sims) / len(sims) if sims else 0.0
    np.save(VOICEPRINT_PATH, vp.astype(np.float32))
    print(f"\n   🔒 Saved your voiceprint → {VOICEPRINT_PATH}")
    print(f"   Sample self-consistency: {consistency:.2f}   ·   match threshold: {VERIFY_THRESHOLD:.2f} (others score <0.35)")
    print("   Wake mode now responds ONLY to your voice. (Delete voiceprint.npy to turn this off.)\n")


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
        last = None
        for _ in range(2):                              # retry once on a network hiccup (reliability)
            try:
                await edge_tts.Communicate(text, VOICE).save(mp3); return
            except Exception as e:
                last = e; await asyncio.sleep(0.4)
        raise last
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
    global _last_rms                                     # loudness, for the near-field/background gate
    _last_rms = float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2))) if pcm.size else 0.0
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
        pcm, sr = synth(text)                         # .save()+decode is the fastest path for short replies
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

def _wake_match(text: str):
    """If the utterance begins with the wake word ('Hey Vox', tolerant of Whisper mishearings),
    return the command that follows it (possibly ''). Otherwise return None so it's ignored."""
    words = re.sub(r"[^a-z0-9' ]", " ", text.lower()).split()
    # Whisper mangles the "Hey" in "Hey Vox" into all sorts of short fillers — accept them as the greeting.
    greet = {"hey", "hay", "he", "hi", " hi", "in", "and", "an", "a", "uh", "um", "oh", "ah",
             "ok", "okay", "ey", "ay", "yo", "hello", "hallo", "they", "i", "eh", "hivox"}
    for i in range(min(4, len(words))):        # only near the start, so mid-sentence look-alikes don't trigger
        if words[i] in _WAKE_ALTS and (i == 0 or words[i - 1] in greet):
            return " ".join(words[i + 1:]).strip()
        # merged form Whisper sometimes emits, e.g. "heyvox" / "hivox" (never real words)
        for g in ("hey", "hi"):
            for v in _WAKE_ALTS:
                if words[i] == g + v:
                    return " ".join(words[i + 1:]).strip()
    return None


def run_wake(device=None):
    """Hands-free WAKE-WORD mode: always listening, but only ACTS after it hears 'Hey Vox'."""
    hist = [{"role": "system", "content": SYSTEM_WAKE}]
    warm_stt()                                    # pre-load Whisper so the first turn isn't slow
    print(f'🎙️  VOX is live (hands-free). Say "{WAKE}" to wake me, then your question.')
    print('   ▶ e.g. "Hey Vox, what time is it?"    ▶ say "Hey Vox, goodbye" to exit.    ▶ Ctrl+C to stop.')
    try:
        import sounddevice as sd
        di = device if device is not None else (sd.default.device[0] if sd.default.device[0] not in (None, -1) else None)
        name = sd.query_devices(di)["name"] if di is not None else sd.query_devices(kind="input")["name"]
        print(f"   (listening on mic: {name}  — if that's wrong, run with  --device N ; see  python vox.py --devices)\n")
    except Exception as e:
        print(f"   (could not read mic device: {e})\n")
    if load_voiceprint() is not None:
        _get_encoder()                       # warm the speaker model so the first verification isn't slow
        print(f"   🔒 Voiceprint ON — I'll only respond to YOUR enrolled voice (match ≥ {VERIFY_THRESHOLD:.2f}).\n")
    else:
        print("   🔓 Voiceprint off — I respond to any voice.  Lock to yours:  python vox.py --enroll\n")
    say(f"Hands-free mode on. Say {WAKE}, then your question.", bargein=False)
    while True:
        print("… listening (say \"Hey Vox …\")")
        wav = record_utterance(device=device, start_timeout=3600)   # wait for any speech
        if not wav:
            continue
        if WAKE_MIN_RMS and _last_rms < WAKE_MIN_RMS:               # too quiet = distant background (TV) — ignore
            os.remove(wav)
            print(f"   (ignored background — volume {int(_last_rms)} below {WAKE_MIN_RMS}; speak toward the mic)")
            continue
        text = transcribe(wav)
        if not text:
            os.remove(wav)
            print("   (heard sound but no words — mic may be too quiet or the wrong device)")
            continue
        cmd = _wake_match(text)
        if cmd is None:
            os.remove(wav)
            print(f'   (heard: "{text}" [vol {int(_last_rms)}] — not the wake word; say "Hey Vox" first)')
            continue
        ok, score = verify_speaker(wav); os.remove(wav)   # wake word matched — but is it the owner's voice?
        if not ok:
            print(f'   (ignored "{text}" — not your enrolled voice; match {score:.2f} < {VERIFY_THRESHOLD:.2f})')
            continue
        print(f'👂 wake heard: "{text}"' + (f'  [you ✓ {score:.2f}]' if score >= 0 else ''))
        if not cmd:                      # they said only the wake word — ask what they need
            say("Yes?", bargein=False)
            wav2 = record_utterance(device=device, start_timeout=8)
            if wav2:
                ok2, _ = verify_speaker(wav2)             # verify the follow-up too, so the TV can't answer "Yes?"
                cmd = transcribe(wav2) if ok2 else ""
                os.remove(wav2)
                if not ok2:
                    print("… (that follow-up wasn't your voice — say the wake word again)\n"); continue
            else:
                cmd = ""
            if not cmd:
                print("… (didn't catch the request — say the wake word again)\n"); continue
        print(f"👤 you: {cmd}")
        hist.append({"role": "user", "content": cmd})
        reply, hist = think(hist)
        # bargein=False: with speakers, an open mic hears VOX's OWN voice and self-interrupts. Replies are
        # short (1-2 sentences), so play them fully. (Barge-in needs a headset to avoid the echo.)
        say(reply, bargein=False)
        print()
        if re.search(r"\b(goodbye|good night|stop listening|that's all|we're done)\b", cmd, re.I):
            say("Talk soon."); break


def run_live(device=None, hands_free=False):
    hist = [{"role": "system", "content": SYSTEM}]
    warm_stt()                                    # pre-load Whisper so the first turn isn't slow
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
            say(reply, bargein=False)   # speakers: an open mic self-interrupts on VOX's own voice
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
    ap.add_argument("--hands-free", action="store_true", help="always-listening mode (responds to any speech)")
    ap.add_argument("--wake", action="store_true", help="hands-free WAKE-WORD mode — say 'Hey Vox' to talk")
    ap.add_argument("--enroll", action="store_true", help="record your voiceprint so wake mode only answers YOU")
    a = ap.parse_args()
    try:
        if a.devices:
            list_devices()
        elif a.enroll:
            run_enroll(device=a.device)
        elif a.test:
            run_test()
        elif a.text:
            run_text(a.text)
        elif a.wake:
            run_wake(device=a.device)
        else:
            run_live(device=a.device, hands_free=a.hands_free)
    except KeyboardInterrupt:
        print("\n👋 VOX stopped.")


if __name__ == "__main__":
    main()
