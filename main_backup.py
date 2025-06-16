import os
import re
import time
import numpy as np
import torch
import sounddevice as sd
from queue import Queue, Empty
import threading

import onnxruntime
from faster_whisper import WhisperModel
from TTS.api import TTS

from langchain_ollama import ChatOllama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
)

# ——— Text cleaning ———
def clean_text(text: str) -> str:
    return re.sub(r"\.", ".", text)

def remove_punctuation_except_specified(text: str) -> str:
    return text

# === Paths & Defaults ===
SCRIPT_DIR = os.path.dirname(__file__)
DEFAULT_SPK = os.path.join(SCRIPT_DIR, "default_audio", "english_audio2.wav")
LANG_SPK = {
    "ja": os.path.join(SCRIPT_DIR, "default_audio", "japanese.wav"),
    "es": os.path.join(SCRIPT_DIR, "default_audio", "spanish_audio1.wav"),
}

# === Silero VAD ===
class Vad:
    def __init__(self, threshold: float = 0.1):
        model_path = os.path.join(SCRIPT_DIR, "assets", "silero_vad.onnx")
        opts = onnxruntime.SessionOptions(); opts.log_severity_level = 4
        self.sess = onnxruntime.InferenceSession(
            model_path, sess_options=opts,
            providers=["CPUExecutionProvider"]
        )
        self.SR = 16000; self.threshold = threshold
        self.h = np.zeros((2,1,64), dtype=np.float32)
        self.c = np.zeros((2,1,64), dtype=np.float32)

    def is_speech(self, audio: np.ndarray) -> bool:
        inp = {"input": audio.reshape(1,-1), "sr": np.array([self.SR], dtype=np.int64), "h": self.h, "c": self.c}
        out, h, c = self.sess.run(None, inp)
        self.h, self.c = h, c
        return bool(out > self.threshold)

# === Settings ===
SAMPLERATE     = 16000
CHUNK          = 8000   # smaller chunks for quicker VAD
SILENCE_SEC    = 0.1
SILENCE_FRAMES = int(SILENCE_SEC * SAMPLERATE / CHUNK)

# === Model Setup ===
device      = "cuda" if torch.cuda.is_available() else "cpu"
asr_model  = WhisperModel("base.en", device=device, compute_type="float16")

llm         = ChatOllama(model="gemma3:1b")
memory      = ConversationBufferMemory(memory_key="history", return_messages=True)
system_tmpl = SystemMessagePromptTemplate.from_template(
    "You are an expert assistant who only responds with short and concise answers without special characters or emojis. Make sure you use only . , ? ! as punctuation."
)
chat_prompt = ChatPromptTemplate.from_messages([
    system_tmpl,
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])
chat_chain = ConversationChain(llm=llm, memory=memory, prompt=chat_prompt)

# TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
sr  = tts.synthesizer.tts_config.audio["sample_rate"]

# === Queues & Threads ===
utterance_queue = Queue()
playback_queue  = Queue()
playback_thread = None
_playback_shutdown = threading.Event()
tts_abort_event    = threading.Event()


def playback_worker():
    stream = sd.OutputStream(samplerate=sr, channels=1, dtype='float32')
    stream.start()
    while not _playback_shutdown.is_set():
        chunk = playback_queue.get()
        if chunk is None: break
        stream.write(chunk)
    stream.stop()
    stream.close()


def start_playback_thread():
    global playback_thread
    _playback_shutdown.clear()
    playback_thread = threading.Thread(target=playback_worker, daemon=True)
    playback_thread.start()


def stop_playback_thread():
    _playback_shutdown.set()
    playback_queue.put(None)
    if playback_thread: playback_thread.join()
    # clear queue
    try:
        while True: playback_queue.get_nowait()
    except Empty:
        pass


def inference_worker():
    while True:
        item = utterance_queue.get()
        if item is None: break
        audio, _ = item
        tts_abort_event.clear()

        # ASR
        t0 = time.perf_counter()
        segments, info = asr_model.transcribe(audio=audio, without_timestamps=True)
        asr_ms = (time.perf_counter() - t0) * 1000
        print(f"ASR latency: {asr_ms:.1f} ms")

        text = " ".join(seg.text for seg in segments).strip()
        print("============================")
        print(info.language_probability, info.average_logprob, info.language, text)

        # LLM
        t1 = time.perf_counter()
        response = chat_chain.predict(input=text)
        llm_ms = (time.perf_counter() - t1) * 1000
        print(f"LLM latency: {llm_ms:.1f} ms")

        # TTS streaming
        lang = getattr(info, "language", "en")
        spk = LANG_SPK.get(lang, DEFAULT_SPK)
        t2 = time.perf_counter()
        print(f"starting TTS… {response}")
        start_playback_thread()
        first = True
        for chunk in tts.synthesizer.tts_stream(text=response, speaker_wav=spk, language_name=lang):
            now = time.perf_counter()
            if first:
                print(f"TTS first-chunk latency: {(now-t2)*1000:.1f} ms")
                first = False
            playback_queue.put(np.array(chunk).astype(np.float32))
        playback_queue.put(None)
        print(f"TTS total latency: {(time.perf_counter()-t2)*1000:.1f} ms")

        if device == "cuda": torch.cuda.empty_cache()

# start thread
t = threading.Thread(target=inference_worker, daemon=True)
t.start()

# === Audio capture ===
vad = Vad(threshold=0.3)
buffer, silence = [], 0

def audio_callback(indata, frames, time_info, status):
    global buffer, silence
    buf = indata.flatten().astype(np.float32)
    if vad.is_speech(buf):
        if playback_thread and playback_thread.is_alive():
            stop_playback_thread(); tts_abort_event.set()
            utt = np.concatenate(buffer + [buf]) if buffer else buf
            utterance_queue.put((utt, time.perf_counter()))
            buffer.clear(); silence=0
            return
        buffer.append(buf); silence=0
    else:
        silence+=1
        if silence>SILENCE_FRAMES and buffer:
            utt=np.concatenate(buffer)
            utterance_queue.put((utt, time.perf_counter()))
            buffer.clear(); silence=0

if __name__ == "__main__":
    print("🔊 Speak… Ctrl+C to stop.")
    with sd.InputStream(samplerate=SAMPLERATE, channels=1, dtype='float32', blocksize=CHUNK, callback=audio_callback):
        try: sd.sleep(int(1e9))
        except KeyboardInterrupt:
            utterance_queue.put(None)
