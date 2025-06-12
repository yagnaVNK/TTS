import os
import re
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
    # surround every period with spaces
    return re.sub(r"\.", ".", text)

def remove_punctuation_except_specified(text):
    return text  # your existing stub

# === Silero VAD ===
class Vad:
    def __init__(self, threshold: float = 0.1):
        current_dir = os.path.dirname(__file__)
        model_path = os.path.join(current_dir, "assets", "silero_vad.onnx")
        opts = onnxruntime.SessionOptions()
        opts.log_severity_level = 4
        self.sess = onnxruntime.InferenceSession(
            model_path, sess_options=opts,
            providers=["CUDAExecutionProvider","CPUExecutionProvider"]
        )
        self.SR = 16000
        self.threshold = threshold
        self.h = np.zeros((2,1,64), dtype=np.float32)
        self.c = np.zeros((2,1,64), dtype=np.float32)

    def is_speech(self, audio: np.ndarray) -> bool:
        inp = {"input": audio.reshape(1, -1),
               "sr": np.array([self.SR], dtype=np.int64),
               "h": self.h, "c": self.c}
        out, h, c = self.sess.run(None, inp)
        self.h, self.c = h, c
        return bool(out > self.threshold)

# === Settings ===
SAMPLERATE     = 16000
CHUNK          = 16000
SILENCE_SEC    = 0.3
SILENCE_FRAMES = int(SILENCE_SEC * SAMPLERATE / CHUNK)

# === Load models once ===
device    = "cuda" if torch.cuda.is_available() else "cpu"
asr_model = WhisperModel("large-v3", device=device, compute_type="float16")

llm        = ChatOllama(model="gemma3:1b")
memory     = ConversationBufferMemory(memory_key="history", return_messages=True)
system_tmpl = SystemMessagePromptTemplate.from_template(
    "You are an expert assistant who knows multiple languages that always checks whether each command makes sense "
    "given the previous context, and only responds if it does. Make responses short and informative. "
    "Use only simple punctuation like . , ! ? and no emojis."
)
chat_prompt = ChatPromptTemplate.from_messages([
    system_tmpl,
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])
chat_chain = ConversationChain(llm=llm, memory=memory, prompt=chat_prompt)

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
sr  = tts.synthesizer.tts_config.audio["sample_rate"]

# === Queues & Threads ===
utterance_queue = Queue()
playback_queue  = Queue()
playback_thread = None
playback_stream = None
_playback_shutdown = threading.Event()
tts_abort_event = threading.Event()

def playback_worker():
    global playback_stream
    playback_stream = sd.OutputStream(samplerate=sr, channels=1, dtype='float32')
    playback_stream.start()
    while not _playback_shutdown.is_set():
        try:
            chunk = playback_queue.get(timeout=0.001)
        except Empty:
            continue
        if chunk is None:
            break
        playback_stream.write(chunk)
    playback_stream.stop()
    playback_stream.close()

def start_playback_thread():
    global playback_thread
    stop_playback_thread()
    _playback_shutdown.clear()
    playback_thread = threading.Thread(target=playback_worker, daemon=True)
    playback_thread.start()

def stop_playback_thread():
    global playback_thread
    if playback_thread and playback_thread.is_alive():
        _playback_shutdown.set()
        playback_queue.put(None)
        playback_thread.join()
    # drain any leftover chunks
    try:
        while True:
            playback_queue.get_nowait()
    except Empty:
        pass

def inference_worker():
    while True:
        utt = utterance_queue.get()
        if utt is None:
            break

        tts_abort_event.clear()

        # 1) ASR
        segments, info = asr_model.transcribe(
            audio=utt, without_timestamps=True, word_timestamps=False
        )
        text = " ".join(seg.text for seg in segments).strip()
        if not text:
            utterance_queue.task_done()
            continue

        lang_prob = info.language_probability
        sent_prob = float(np.exp(info.average_logprob))

        print(f"\n🗣 You said: {text}")
        print(f"Language probability: {lang_prob:.2f}")
        print(f"Sentence average probability: {sent_prob:.2f}")
        response = ""
        # 2) checks
        if lang_prob < 0.7:
            if lang_prob > 0.5:
                response = "Sorry, I cannot understand the language."
                print(f"🤖 Assistant: {response}\n" + "—"*40)
            utterance_queue.task_done()
            continue
        elif sent_prob < 0.5:
            response = "Sorry, I cannot understand what you said. Please try again."
            print(f"🤖 Assistant: {response}\n" + "—"*40)
            utterance_queue.task_done()
            continue
        else:
            response = chat_chain.predict(input=text)
            if not response or response.isspace():
                utterance_queue.task_done()
                continue
            response = remove_punctuation_except_specified(response)
            response = clean_text(response)
            print(f"🤖 Assistant: {response}\n" + "—"*40)

        print("starting TTS streaming… with response:", response)
        # 4) playback setup
        start_playback_thread()

        # 5) TTS streaming → playback queue
        lang = getattr(info, "language", "en")
        spk = {'ja':"default_audio/japanese.wav",
               'es':"default_audio/spanish_audio1.wav"}.get(lang, "default_audio/audio1.wav")

        for chunk in tts.synthesizer.tts_stream(
            text=response, speaker_wav=spk, language_name=lang
        ):
            if tts_abort_event.is_set():
                break
            arr = chunk if isinstance(chunk, np.ndarray) else np.array(chunk)
            playback_queue.put(arr.astype(np.float32))

        playback_queue.put(None)
        if device=="cuda":
            torch.cuda.empty_cache()
        utterance_queue.task_done()

# launch threads
threading.Thread(target=inference_worker, daemon=True).start()

# === Audio callback ===
vad = Vad(threshold=0.3)
audio_buffer = []
silence_cnt  = 0

def audio_callback(indata, frames, time, status):
    global silence_cnt
    buf = indata.flatten().astype(np.float32)

    # Speech detected
    if vad.is_speech(buf):
        # 1) if we're in the middle of playing back, interrupt everything
        if playback_thread and playback_thread.is_alive():
            stop_playback_thread()
            tts_abort_event.set()

            # 2) immediately send whatever we have plus this chunk to ASR
            if audio_buffer:
                utt = np.concatenate(audio_buffer + [buf])
            else:
                utt = buf
            utterance_queue.put(utt)

            # 3) clear buffer and reset
            audio_buffer.clear()
            silence_cnt = 0
            return

        # Normal buffering when not interrupting
        audio_buffer.append(buf)
        silence_cnt = 0

    # Silence detected → end of utterance
    else:
        silence_cnt += 1
        if silence_cnt > SILENCE_FRAMES and audio_buffer:
            utt = np.concatenate(audio_buffer)
            audio_buffer.clear()
            silence_cnt = 0
            utterance_queue.put(utt)


if __name__ == "__main__":
    print("🔊 Real-time ASR → LLM → TTS (with cleaning). Speak… Ctrl+C to stop.\n")
    with sd.InputStream(
        samplerate=SAMPLERATE, channels=1,
        dtype="float32", blocksize=CHUNK,
        callback=audio_callback
    ):
        try:
            while True:
                sd.sleep(100)
        except KeyboardInterrupt:
            utterance_queue.put(None)
            stop_playback_thread()
            print("\n🛑 Shutting down.")
