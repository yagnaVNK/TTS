#!/usr/bin/env python3
"""
FastAPI server for ASR → LLM → TTS pipeline with WebSocket streaming
TTS streaming runs in a separate thread and can be preempted by new speech.
"""
import os, re, time, threading, math, asyncio, json
from queue import Queue, Empty
from typing import Optional
import uuid

import numpy as np
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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

# ─── Helpers ────────────────────────────────────────────────────────────────
def to_8k(x: np.ndarray, src_sr: int) -> np.ndarray:
    """Resample arbitrary SR → 8 kHz (mono)."""
    return resample_poly(x, 8_000, src_sr, axis=0)

def audio_to_base64(audio_data: np.ndarray) -> str:
    """Convert audio array to base64 for web transmission."""
    import base64
    audio_int16 = (audio_data * 32767).astype(np.int16)
    return base64.b64encode(audio_int16.tobytes()).decode('utf-8')

# ─── Constants ──────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(__file__)

SR_PROC      = 16_000
SR_PLAY      = 8_000
CHUNK        = SR_PROC // 2
SILENCE_SEC  = 0.2
SILENCE_FR   = int(SILENCE_SEC * SR_PROC / CHUNK)

DEFAULT_SPK  = os.path.join(SCRIPT_DIR, "default_audio", "english_audio2_8k.wav")
LANG_SPK     = {
    "ja": os.path.join(SCRIPT_DIR, "default_audio", "japanese_8k.wav"),
    "es": os.path.join(SCRIPT_DIR, "default_audio", "spanish_audio1_8k.wav"),
}

# ─── Silero VAD ──────────────────────────────────────────────────────────────
class Vad:
    def __init__(self, threshold: float = 0.3):
        mp = os.path.join(SCRIPT_DIR, "assets", "silero_vad.onnx")
        so = onnxruntime.SessionOptions()
        so.log_severity_level = 4
        self.sess = onnxruntime.InferenceSession(mp, sess_options=so,
                                                 providers=["CPUExecutionProvider"])
        self.threshold = threshold
        self.h = np.zeros((2,1,64), np.float32)
        self.c = np.zeros((2,1,64), np.float32)
        self.SR = SR_PROC

    def is_speech(self, buf16k: np.ndarray) -> bool:
        inp = {"input": buf16k.reshape(1, -1),
               "sr": np.array([self.SR], np.int64),
               "h": self.h, "c": self.c}
        out, self.h, self.c = self.sess.run(None, inp)
        return bool(out > self.threshold)

# ─── Audio Pipeline Session ────────────────────────────────────────────────
class AudioSession:
    def __init__(self, session_id: str, websocket: WebSocket):
        self.session_id = session_id
        self.websocket = websocket
        self.vad = Vad(threshold=0.3)
        self.buffer = []
        self.silence = 0
        self.current_response_id = None
        self.is_processing = False

        # --- TTS thread/cancellation ---
        self.tts_thread = None
        self.tts_cancel_event = threading.Event()
        self.tts_lock = threading.Lock()
        self.last_tts_response = None

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    async def send_message(self, message_type: str, data: dict):
        message = {
            "type": message_type,
            "timestamp": time.time(),
            "session_id": self.session_id,
            **data
        }
        try:
            await self.websocket.send_text(json.dumps(message))
        except Exception as e:
            print(f"Error sending message: {e}")

    async def process_audio_chunk(self, audio_data: np.ndarray):
        if self.vad.is_speech(audio_data):
            # --- Cancel any running TTS thread if new speech is detected ---
            if self.tts_thread and self.tts_thread.is_alive():
                print("Cancelling TTS thread (new speech detected)")
                self.tts_cancel_event.set()
                await self.send_message("response_cancelled", {
                    "reason": "new_speech_detected"
                })
                self.tts_thread.join()
                self.tts_cancel_event.clear()
            self.buffer.append(audio_data)
            self.silence = 0
            await self.send_message("speech_detected", {
                "buffer_length": len(self.buffer)
            })
        else:
            self.silence += 1
            if self.silence > SILENCE_FR and self.buffer:
                utterance = np.concatenate(self.buffer)
                self.buffer.clear()
                self.silence = 0
                await self.send_message("speech_ended", {
                    "utterance_length": len(utterance)
                })
                asyncio.create_task(self.process_utterance(utterance))

    async def process_utterance(self, wav16k: np.ndarray):
        if self.is_processing:
            return
        self.is_processing = True
        response_id = str(uuid.uuid4())
        self.current_response_id = response_id
        try:
            # --- ASR ---
            t0 = time.perf_counter()
            segs, info = asr.transcribe(audio=wav16k, without_timestamps=True)
            avg_prob = math.exp(info.average_logprob)
            lang_prob = info.language_probability
            text = " ".join(s.text for s in segs).strip()
            asr_latency = (time.perf_counter() - t0) * 1e3
            await self.send_message("asr_result", {
                "text": text,
                "language_probability": lang_prob,
                "average_probability": avg_prob,
                "latency_ms": asr_latency,
                "response_id": response_id
            })

            # --- LLM ---
            if lang_prob >= 0.2 and avg_prob >= 0.2 and text:
                t1 = time.perf_counter()
                response = chat.predict(input=text)
                llm_latency = (time.perf_counter() - t1) * 1e3
                await self.send_message("llm_result", {
                    "response": response,
                    "latency_ms": llm_latency,
                    "response_id": response_id
                })
            else:
                response = "Sorry, I couldn't catch that."
                await self.send_message("llm_result", {
                    "response": response,
                    "latency_ms": 0,
                    "response_id": response_id,
                    "low_confidence": True
                })

            # --- Start TTS streaming in a new thread ---
            lang = getattr(info, "language", "en")
            spk = LANG_SPK.get(lang, DEFAULT_SPK)
            self.last_tts_response = (response, spk, lang, response_id)
            self.tts_cancel_event.clear()
            tts_thread = threading.Thread(
                target=self._tts_stream_worker,
                args=(response, spk, lang, response_id)
            )
            self.tts_thread = tts_thread
            tts_thread.start()
        except Exception as e:
            await self.send_message("error", {
                "message": str(e),
                "response_id": response_id
            })
        finally:
            self.is_processing = False

    def _tts_stream_worker(self, response, spk, lang, response_id):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            coro = self._tts_stream(response, spk, lang, response_id)
            loop.run_until_complete(coro)
        finally:
            loop.close()

    async def _tts_stream(self, response, spk, lang, response_id):
        try:
            t2 = time.perf_counter()
            first_chunk_time = None
            await self.send_message("tts_start", {
                "text": response,
                "language": lang,
                "response_id": response_id
            })
            for chunk in tts.synthesizer.tts_stream(
                    text=response,
                    speaker_wav=spk,
                    language_name=lang):
                if self.tts_cancel_event.is_set():
                    await self.send_message("tts_cancelled", {
                        "response_id": response_id
                    })
                    print("TTS cancelled (worker thread)")
                    break

                if first_chunk_time is None:
                    first_chunk_time = (time.perf_counter() - t2) * 1e3
                    await self.send_message("tts_first_chunk", {
                        "latency_ms": first_chunk_time,
                        "response_id": response_id
                    })

                chunk8k = to_8k(np.asarray(chunk, dtype=np.float32), TTS_SR)
                audio_b64 = audio_to_base64(chunk8k)
                await self.send_message("audio_chunk", {
                    "audio_data": audio_b64,
                    "sample_rate": SR_PLAY,
                    "response_id": response_id
                })
            if not self.tts_cancel_event.is_set():
                total_latency = (time.perf_counter() - t2) * 1e3
                await self.send_message("tts_complete", {
                    "total_latency_ms": total_latency,
                    "response_id": response_id
                })
            if self.device == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            await self.send_message("error", {
                "message": str(e),
                "response_id": response_id
            })

# ─── Initialize Models ─────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
asr = WhisperModel("large-v3", device=device, compute_type="float16")
llm = ChatOllama(model="gemma3:1b")
memory = ConversationBufferMemory(memory_key="history", return_messages=True)
sys_msg = SystemMessagePromptTemplate.from_template(
    "You are an expert assistant who only responds with short and concise answers without special characters and emojis. Use only . , ? !"
)
chat_tpl = ChatPromptTemplate.from_messages([
    sys_msg, MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])
chat = ConversationChain(llm=llm, memory=memory, prompt=chat_tpl)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
TTS_SR = tts.synthesizer.tts_config.audio["sample_rate"]

# ─── FastAPI App ───────────────────────────────────────────────────────────
app = FastAPI(title="Audio Pipeline API")
sessions = {}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    session = AudioSession(session_id, websocket)
    sessions[session_id] = session

    await session.send_message("connected", {
        "message": "Connected to audio pipeline",
        "sample_rate": SR_PROC
    })

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            if message["type"] == "audio":
                import base64
                audio_bytes = base64.b64decode(message["data"])
                audio_f32 = np.frombuffer(audio_bytes, dtype=np.float32)
                sr_client = 48000
                if sr_client != SR_PROC:
                    audio_f32 = resample_poly(audio_f32, SR_PROC, sr_client)
                if audio_f32.ndim > 1:
                    audio_f32 = audio_f32.mean(axis=1)
                await session.process_audio_chunk(audio_f32)
    except WebSocketDisconnect:
        print(f"Session {session_id} disconnected")
    except Exception as e:
        print(f"Error in session {session_id}: {e}")
    finally:
        sessions.pop(session_id, None)


if __name__ == "__main__":
    print("Starting Audio Pipeline Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)