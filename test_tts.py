import os
import asyncio
import time
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import torch
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
import librosa

# --- App setup ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Paths & Defaults ---
SCRIPT_DIR = os.path.dirname(__file__)
DEFAULT_SPK = os.path.join(SCRIPT_DIR, "default_audio", "english_audio2_8k.wav")
LANG_SPK = {
    "ja": os.path.join(SCRIPT_DIR, "default_audio", "japanese_8k.wav"),
    "es": os.path.join(SCRIPT_DIR, "default_audio", "spanish_audio1_8k.wav"),
}

TARGET_SAMPLE_RATE = 8000

# --- Load models once ---
device = "cuda" if torch.cuda.is_available() else "cpu"
asr_model = WhisperModel("base", device=device, compute_type="float16")
llm = ChatOllama(model="gemma3:1b")
memory = ConversationBufferMemory(memory_key="history", return_messages=True)
system = SystemMessagePromptTemplate.from_template(
    "You are concise. Use only [. , ? !] as punctuation. Do not use emojis. Only respond in the language that the question is asked."
)
chat_prompt = ChatPromptTemplate.from_messages([
    system,
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}"),
])
chat_chain = ConversationChain(llm=llm, memory=memory, prompt=chat_prompt)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def simple_resample(audio, orig_sr, target_sr):
    if orig_sr == target_sr:
        return audio
    ratio = target_sr / orig_sr
    indices = np.arange(0, len(audio), 1/ratio)
    indices = np.clip(indices, 0, len(audio) - 1)
    return np.interp(np.arange(len(indices)), indices, audio[indices.astype(int)])

async def process_and_stream(
    ws: WebSocket,
    audio: np.ndarray,
    abort_event: asyncio.Event
):
    try:
        # Total pipeline timer
        t_start_total = time.perf_counter()

        # 1) ASR
        t0 = time.perf_counter()
        if abort_event.is_set(): 
            return
        segments, info = asr_model.transcribe(audio=audio, without_timestamps=True)
        text = " ".join(seg.text for seg in segments).strip()
        lang = getattr(info, "language", "en")
        t1 = time.perf_counter()
        asr_time = t1 - t0
        print(f"[Timing] ASR took {asr_time:.3f}s")

        if not text:
            return
        await ws.send_json({"stage": "asr", "text": text, "language": lang})

        # 2) LLM
        t2 = time.perf_counter()
        if abort_event.is_set():
            return
        reply = chat_chain.predict(input=text)
        t3 = time.perf_counter()
        llm_time = t3 - t2
        print(f"[Timing] LLM took {llm_time:.3f}s")

        await ws.send_json({"stage": "chat", "reply": reply})

        # 3) TTS streaming
        t4 = time.perf_counter()
        if abort_event.is_set():
            return
        spk = LANG_SPK.get(lang, DEFAULT_SPK)
        if not os.path.exists(spk):
            spk = DEFAULT_SPK

        chunk_count = 0
        tts_sample_rate = tts.synthesizer.output_sample_rate

        for i,chunk in enumerate(tts.synthesizer.tts_stream(
            text=reply,
            speaker_wav=spk,
            language_name=lang
        )):
            if abort_event.is_set():
                await ws.send_json({"stage": "cancel"})
                return

            chunk_array = np.array(chunk, dtype=np.float32)
            if i == 0:
                t5 = time.perf_counter()
                tts_time = t5 - t4
                print(f"[Timing] Time for initial tts chunk took {tts_time:.3f}s")
            if len(chunk_array) < 50:
                continue

            # Resample if needed
            if tts_sample_rate != TARGET_SAMPLE_RATE:
                try:
                    chunk_array = librosa.resample(
                        chunk_array, orig_sr=tts_sample_rate, target_sr=TARGET_SAMPLE_RATE
                    )
                except ImportError:
                    chunk_array = simple_resample(
                        chunk_array, tts_sample_rate, TARGET_SAMPLE_RATE
                    )

            chunk_array = chunk_array.astype(np.float32)
            if np.max(np.abs(chunk_array)) > 1.0:
                chunk_array /= np.max(np.abs(chunk_array))

            await ws.send_bytes(chunk_array.tobytes())
            chunk_count += 1
            await asyncio.sleep(0.005)

        
        

        if chunk_count > 0:
            await ws.send_json({"stage": "tts_complete"})

        # Total time
        total_time = time.perf_counter() - t_start_total
        print(f"[Timing] Total pipeline took {total_time:.3f}s")

    except Exception as e:
        print(f"Processing error: {e}")
        await ws.send_json({"stage": "error", "message": str(e)})

@app.websocket("/ws/pipeline")
async def ws_pipeline(ws: WebSocket):
    await ws.accept()
    current_task: Optional[asyncio.Task] = None
    current_abort: Optional[asyncio.Event] = None

    try:
        while True:
            data = await ws.receive_bytes()
            audio = np.frombuffer(data, dtype=np.float32)
            if len(audio) == 0:
                continue

            await ws.send_json({"stage": "cancel"})
            if current_abort and not current_abort.is_set():
                current_abort.set()
            if current_task and not current_task.done():
                current_task.cancel()
                try: await current_task
                except asyncio.CancelledError: pass

            memory.clear()
            current_abort = asyncio.Event()
            current_task = asyncio.create_task(
                process_and_stream(ws, audio, current_abort)
            )

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        if current_abort and not current_abort.is_set():
            current_abort.set()
        if current_task and not current_task.done():
            current_task.cancel()
            try: await current_task
            except asyncio.CancelledError: pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
