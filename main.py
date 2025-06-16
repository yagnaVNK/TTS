#!/usr/bin/env python3
"""
ASR → LLM → TTS pipeline
• Mic capture / VAD / ASR / LLM all run at 16 kHz
• XTTS still generates at ~24 kHz but is down-sampled to 8 kHz for playback
"""
import os, re, time, threading, math
from queue import Queue, Empty

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
import torch, onnxruntime
from faster_whisper import WhisperModel
from TTS.api import TTS
from langchain_ollama import ChatOllama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts.chat import (
    ChatPromptTemplate, SystemMessagePromptTemplate,
    MessagesPlaceholder, HumanMessagePromptTemplate,
)

# ─── Helpers ────────────────────────────────────────────────────────────────
def to_8k(x: np.ndarray, src_sr: int) -> np.ndarray:
    """Resample arbitrary SR → 8 kHz (mono)."""
    return resample_poly(x, 8_000, src_sr, axis=0)

# ─── Constants ──────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(__file__)

SR_PROC      = 16_000                      # processing samplerate
SR_PLAY      = 8_000                       # playback samplerate (TTS only)
CHUNK        = SR_PROC // 2               # 0.5 s blocks from mic
SILENCE_SEC  = 0.1
SILENCE_FR   = int(SILENCE_SEC * SR_PROC / CHUNK)

DEFAULT_SPK  = os.path.join(SCRIPT_DIR, "default_audio", "english_audio2_8k.wav")
LANG_SPK     = {
    "ja": os.path.join(SCRIPT_DIR, "default_audio", "japanese_8k.wav"),
    "es": os.path.join(SCRIPT_DIR, "default_audio", "spanish_audio1_8k.wav"),
}

# ─── Silero VAD (expects 16 kHz) ────────────────────────────────────────────
class Vad:
    def __init__(self, threshold: float = .3):
        mp = os.path.join(SCRIPT_DIR, "assets", "silero_vad.onnx")
        so = onnxruntime.SessionOptions() 
        so.log_severity_level = 4
        self.sess = onnxruntime.InferenceSession(mp, sess_options=so,
                                                 providers=["CPUExecutionProvider"])
        self.threshold = threshold
        self.h = np.zeros((2,1,64), np.float32)
        self.c = np.zeros((2,1,64), np.float32)
        self.SR = SR_PROC                   # 16 kHz

    def is_speech(self, buf16k: np.ndarray) -> bool:
        inp = {"input": buf16k.reshape(1, -1),
               "sr": np.array([self.SR], np.int64),
               "h": self.h, "c": self.c}
        out, self.h, self.c = self.sess.run(None, inp)
        return bool(out > self.threshold)

# ─── Models ─────────────────────────────────────────────────────────────────
device   = "cuda" if torch.cuda.is_available() else "cpu"
asr      = WhisperModel("base.en", device=device, compute_type="float16")  # 16 kHz

llm      = ChatOllama(model="gemma3:1b")
memory   = ConversationBufferMemory(memory_key="history", return_messages=True)
sys_msg  = SystemMessagePromptTemplate.from_template(
    "You are an expert assistant who only responds with short and concise answers without special characters and emojis. Use only . , ? !"
)
chat_tpl = ChatPromptTemplate.from_messages([
    sys_msg, MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])
chat     = ConversationChain(llm=llm, memory=memory, prompt=chat_tpl)

tts      = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
TTS_SR   = tts.synthesizer.tts_config.audio["sample_rate"]      # ~24 kHz

# ─── Queues / Threads ───────────────────────────────────────────────────────
utter_q, play_q = Queue(), Queue()
_pb_shutdown = threading.Event()
_tts_cancel  = threading.Event()
playback_thread = None

def playback_worker():
    with sd.OutputStream(samplerate=SR_PLAY, channels=1, dtype='float32') as stream:
        while not _pb_shutdown.is_set():
            chunk = play_q.get()
            if chunk is None:
                break
            stream.write(chunk)

def start_playback():
    global playback_thread
    _pb_shutdown.clear()
    playback_thread = threading.Thread(target=playback_worker, daemon=True)
    playback_thread.start()

def stop_playback():
    _pb_shutdown.set(); play_q.put(None)
    if playback_thread:
        playback_thread.join()
    while not play_q.empty():
        try: play_q.get_nowait()
        except Empty: break

# ─── Worker (ASR → LLM → TTS) ───────────────────────────────────────────────
def inference_worker():
    while True:
        item = utter_q.get()
        if item is None:
            break

        wav16k, _ = item

        # ── ASR ──
        t0 = time.perf_counter()
        segs, info = asr.transcribe(audio=wav16k, without_timestamps=True)
        lang_prob = info.language_probability
        avg_prob  = math.exp(info.average_logprob)
        text      = " ".join(s.text for s in segs).strip()
        print(f"ASR latency: {(time.perf_counter()-t0)*1e3:.1f} ms")
        print("You said:", text,
              f"\n  lang_prob={lang_prob:.3f}  avg_prob={avg_prob:.3f}")

        # ── Decide whether to call LLM ──
        if lang_prob >= 0.75 and avg_prob >= 0.75 and text:
            t1 = time.perf_counter()
            response = chat.predict(input=text)
            print(f"LLM latency: {(time.perf_counter()-t1)*1e3:.1f} ms")
        else:
            response = "Sorry, I couldn't catch that."

        # ── TTS streaming (→ 8 kHz) ──
        lang = getattr(info, "language", "en")
        spk  = LANG_SPK.get(lang, DEFAULT_SPK)

        _tts_cancel.clear()
        t2 = time.perf_counter(); start_playback()
        first_chunk_time = None

        for chunk in tts.synthesizer.tts_stream(text=response,
                                                speaker_wav=spk,
                                                language_name=lang):
            if _tts_cancel.is_set():
                print("TTS cancelled due to new speech.")
                break
            if first_chunk_time is None:
                first_chunk_time = (time.perf_counter() - t2) * 1e3
                print(f"TTS first-chunk latency: {first_chunk_time:.1f} ms")

            chunk8k = to_8k(np.asarray(chunk, dtype=np.float32), TTS_SR)
            play_q.put(chunk8k)

        play_q.put(None)
        if first_chunk_time:
            total = (time.perf_counter() - t2) * 1e3
            print(f"TTS total latency: {total:.1f} ms")
        if device == "cuda":
            torch.cuda.empty_cache()

threading.Thread(target=inference_worker, daemon=True).start()

# ─── Mic capture @ 16 kHz ──────────────────────────────────────────────────
vad = Vad(threshold=.3)
buffer, silence = [], 0

def mic_cb(indata, frames, time_info, status):
    global buffer, silence
    buf = indata.flatten().astype(np.float32)

    #print(type(buf))

    if vad.is_speech(buf):
        # New speech while TTS is speaking → cancel immediately
        if playback_thread and playback_thread.is_alive():
            _tts_cancel.set()
            stop_playback()

        buffer.append(buf)
        silence = 0
    else:
        silence += 1
        if silence > SILENCE_FR and buffer:
            utter_q.put((np.concatenate(buffer), time.perf_counter()))
            buffer.clear(); silence = 0

# ─── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Speak…  Ctrl-C to quit.")
    with sd.InputStream(samplerate=SR_PROC,
                        channels=1,
                        dtype='float32',
                        blocksize=CHUNK,
                        callback=mic_cb):
        try:
            sd.sleep(int(1e9))   # ~11.5 days
        except KeyboardInterrupt:
            utter_q.put(None)
