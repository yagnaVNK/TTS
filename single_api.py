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

@app.get("/")
async def get_index():
    return HTMLResponse("""
 <!DOCTYPE html>
    <html>
    <head>
        <title>Audio Pipeline Interface</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: #333;
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: rgba(255, 255, 255, 0.95);
                border-radius: 20px;
                padding: 30px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                backdrop-filter: blur(10px);
            }
            
            h1 {
                text-align: center;
                color: #4a5568;
                margin-bottom: 30px;
                font-size: 2.5em;
                font-weight: 300;
            }
            
            .controls {
                display: flex;
                justify-content: center;
                gap: 20px;
                margin-bottom: 30px;
            }
            
            .btn {
                padding: 15px 30px;
                border: none;
                border-radius: 50px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            
            .btn-primary {
                background: linear-gradient(45deg, #667eea, #764ba2);
                color: white;
            }
            
            .btn-primary:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
            }
            
            .btn-danger {
                background: linear-gradient(45deg, #ff6b6b, #ee5a24);
                color: white;
            }
            
            .btn-danger:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(255, 107, 107, 0.3);
            }
            
            .btn:disabled {
                opacity: 0.5;
                cursor: not-allowed;
                transform: none;
            }
            
            .status {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            
            .status-card {
                background: white;
                padding: 20px;
                border-radius: 15px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                border-left: 4px solid #667eea;
            }
            
            .status-card h3 {
                margin: 0 0 10px 0;
                color: #4a5568;
                font-size: 1.1em;
            }
            
            .status-value {
                font-size: 1.5em;
                font-weight: 600;
                color: #667eea;
            }
            
            .conversation {
                background: white;
                border-radius: 15px;
                padding: 20px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                max-height: 400px;
                overflow-y: auto;
            }
            
            .message {
                margin: 15px 0;
                padding: 15px;
                border-radius: 15px;
                animation: fadeIn 0.3s ease;
            }
            
            .message.user {
                background: linear-gradient(45deg, #667eea, #764ba2);
                color: white;
                margin-left: 20%;
            }
            
            .message.assistant {
                background: #f7fafc;
                border: 1px solid #e2e8f0;
                margin-right: 20%;
            }
            
            .message.system {
                background: #fff5f5;
                border: 1px solid #fed7d7;
                text-align: center;
                font-style: italic;
                color: #c53030;
            }
            
            .timing-info {
                font-size: 0.8em;
                opacity: 0.7;
                margin-top: 5px;
            }
            
            .recording-indicator {
                display: none;
                position: fixed;
                top: 20px;
                right: 20px;
                background: #ff4757;
                color: white;
                padding: 10px 20px;
                border-radius: 25px;
                animation: pulse 1s infinite;
                z-index: 1000;
            }
            
            .recording-indicator.active {
                display: block;
            }
            
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            @keyframes pulse {
                0% { transform: scale(1); }
                50% { transform: scale(1.05); }
                100% { transform: scale(1); }
            }
            
            .progress-bar {
                width: 100%;
                height: 4px;
                background: #e2e8f0;
                border-radius: 2px;
                overflow: hidden;
                margin: 10px 0;
            }
            
            .progress-fill {
                height: 100%;
                background: linear-gradient(90deg, #667eea, #764ba2);
                width: 0%;
                transition: width 0.3s ease;
            }
        </style>
    </head>
<body>
    <div class="container">
        <h1>🎤 Audio Pipeline Interface</h1>
        <div class="controls">
            <button id="startBtn" class="btn btn-primary">Start Listening</button>
            <button id="stopBtn" class="btn btn-danger" disabled>Stop Listening</button>
        </div>
        <div class="status">
            <div class="status-card">
                <h3>Connection Status</h3>
                <div id="connectionStatus" class="status-value">Disconnected</div>
            </div>
            <div class="status-card">
                <h3>ASR Latency</h3>
                <div id="asrLatency" class="status-value">-</div>
            </div>
            <div class="status-card">
                <h3>LLM Latency</h3>
                <div id="llmLatency" class="status-value">-</div>
            </div>
            <div class="status-card">
                <h3>TTS Latency</h3>
                <div id="ttsLatency" class="status-value">-</div>
            </div>
        </div>
        <div class="conversation" id="conversation">
            <div class="message system">
                Welcome! Click "Start Listening" to begin voice conversation.
            </div>
        </div>
    </div>
    <div id="recordingIndicator" class="recording-indicator">
        🎤 Recording...
    </div>
      <script>
        let ws = null;
        let audioContext = null;
        let isRecording = false;
        let currentAudio = null;
        let playbackSources = []; // TRACK ALL SOURCES

        // Audio queue variables
        let audioQueue = [];
        let isPlayingQueue = false;
        let nextPlayTime = 0;

        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const connectionStatus = document.getElementById('connectionStatus');
        const conversation = document.getElementById('conversation');
        const recordingIndicator = document.getElementById('recordingIndicator');

        function addMessage(type, content, timing = null) {
            const message = document.createElement('div');
            message.className = `message ${type}`;
            message.innerHTML = content;
            if (timing) {
                const timingDiv = document.createElement('div');
                timingDiv.className = 'timing-info';
                timingDiv.textContent = timing;
                message.appendChild(timingDiv);
            }
            conversation.appendChild(message);
            conversation.scrollTop = conversation.scrollHeight;
        }

        function updateStatus(elementId, value) {
            document.getElementById(elementId).textContent = value;
        }

        function stopAllAudioPlayback() {
            // Stop and clear all currently playing AudioBufferSourceNodes
            playbackSources.forEach(src => {
                try { src.stop(); } catch(e) {}
            });
            playbackSources = [];
            
            // Clear the audio queue
            audioQueue = [];
            isPlayingQueue = false;
            nextPlayTime = 0;
            
            if (currentAudio) {
                try { currentAudio.pause(); } catch(e) {}
                currentAudio = null;
            }
        }

        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

            ws.onopen = function() {
                updateStatus('connectionStatus', 'Connected');
                startBtn.disabled = false;
            };

            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                handleMessage(data);
            };

            ws.onclose = function() {
                updateStatus('connectionStatus', 'Disconnected');
                startBtn.disabled = true;
                stopBtn.disabled = true;
                stopAllAudioPlayback();
            };

            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
                updateStatus('connectionStatus', 'Error');
                stopAllAudioPlayback();
            };
        }

        function handleMessage(data) {
            switch(data.type) {
                case 'connected':
                    addMessage('system', data.message);
                    break;

                case 'speech_detected':
                    recordingIndicator.classList.add('active');
                    stopAllAudioPlayback(); // Stop any TTS playback immediately!
                    break;

                case 'speech_ended':
                    recordingIndicator.classList.remove('active');
                    break;

                case 'asr_result':
                    updateStatus('asrLatency', `${data.latency_ms.toFixed(1)}ms`);
                    if (data.text) {
                        addMessage('user', data.text, 
                            `Confidence: ${(data.average_probability * 100).toFixed(1)}%`);
                    }
                    break;

                case 'llm_result':
                    updateStatus('llmLatency', `${data.latency_ms.toFixed(1)}ms`);
                    if (!data.low_confidence) {
                        addMessage('assistant', data.response);
                    }
                    break;

                case 'tts_first_chunk':
                    updateStatus('ttsLatency', `${data.latency_ms.toFixed(1)}ms`);
                    // Initialize queue for new TTS response
                    if (!isPlayingQueue) {
                        initializeAudioQueue();
                    }
                    break;

                case 'audio_chunk':
                    queueAudioChunk(data.audio_data, data.sample_rate);
                    break;

                case 'response_cancelled':
                    addMessage('system', 'Response cancelled - new speech detected');
                    stopAllAudioPlayback();
                    break;

                case 'tts_cancelled':
                    stopAllAudioPlayback();
                    break;

                case 'error':
                    addMessage('system', `Error: ${data.message}`);
                    stopAllAudioPlayback();
                    break;
            }
        }

        function initializeAudioQueue() {
            if (!audioContext) {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
            }
            
            // Reset queue state
            audioQueue = [];
            isPlayingQueue = true;
            nextPlayTime = audioContext.currentTime;
        }

        function queueAudioChunk(audioData, sampleRate) {
            try {
                // Convert base64 to audio buffer
                const binaryString = atob(audioData);
                const bytes = new Uint8Array(binaryString.length);
                for (let i = 0; i < binaryString.length; i++) {
                    bytes[i] = binaryString.charCodeAt(i);
                }
                const int16Array = new Int16Array(bytes.buffer);
                
                if (!audioContext) {
                    audioContext = new (window.AudioContext || window.webkitAudioContext)();
                }
                
                const audioBuffer = audioContext.createBuffer(1, int16Array.length, sampleRate);
                const channelData = audioBuffer.getChannelData(0);
                for (let i = 0; i < int16Array.length; i++) {
                    channelData[i] = int16Array[i] / 32768.0;
                }

                // Add to queue instead of playing immediately
                audioQueue.push({
                    buffer: audioBuffer,
                    duration: audioBuffer.duration
                });

                // Start playing if this is the first chunk
                if (audioQueue.length === 1 && isPlayingQueue) {
                    playNextInQueue();
                }

            } catch (error) {
                console.error('Error processing audio chunk:', error);
            }
        }

        function playNextInQueue() {
            if (!isPlayingQueue || audioQueue.length === 0) {
                return;
            }

            const audioItem = audioQueue.shift();
            const source = audioContext.createBufferSource();
            source.buffer = audioItem.buffer;
            source.connect(audioContext.destination);

            // Calculate when to start this chunk
            const startTime = Math.max(audioContext.currentTime, nextPlayTime);
            source.start(startTime);
            
            // Update next play time
            nextPlayTime = startTime + audioItem.duration;

            // Track the source for cleanup
            playbackSources.push(source);

            // Set up to play next chunk when this one ends
            source.onended = () => {
                playbackSources = playbackSources.filter(s => s !== source);
                
                // Play next chunk if queue is still active
                if (isPlayingQueue) {
                    playNextInQueue();
                } else if (audioQueue.length === 0) {
                    // Queue is finished
                    isPlayingQueue = false;
                    nextPlayTime = 0;
                }
            };

            // Handle case where source is stopped before ending naturally
            const originalStop = source.stop.bind(source);
            source.stop = function(...args) {
                playbackSources = playbackSources.filter(s => s !== source);
                originalStop(...args);
            };

        }

        async function startRecording() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        sampleRate: 16000,
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true
                    } 
                });

                if (!audioContext) {
                    audioContext = new (window.AudioContext || window.webkitAudioContext)();
                }
                const source = audioContext.createMediaStreamSource(stream);
                const processor = audioContext.createScriptProcessor(8192, 1, 1);

                processor.onaudioprocess = function(e) {
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        const inputData = e.inputBuffer.getChannelData(0);
                        const audioData = new Float32Array(inputData);
                        // Convert to base64
                        const buffer = new ArrayBuffer(audioData.length * 4);
                        const view = new Float32Array(buffer);
                        view.set(audioData);
                        const base64 = btoa(String.fromCharCode(...new Uint8Array(buffer)));
                        ws.send(JSON.stringify({
                            type: 'audio',
                            data: base64
                        }));
                    }
                };

                source.connect(processor);
                processor.connect(audioContext.destination);

                isRecording = true;
                startBtn.disabled = true;
                stopBtn.disabled = false;

                // Store references for cleanup
                window.audioStream = stream;
                window.audioProcessor = processor;
                window.audioSource = source;

            } catch (error) {
                console.error('Error starting recording:', error);
                addMessage('system', 'Error: Could not access microphone');
            }
        }

        function stopRecording() {
            if (window.audioStream) {
                window.audioStream.getTracks().forEach(track => track.stop());
            }
            if (window.audioProcessor) {
                window.audioProcessor.disconnect();
            }
            if (window.audioSource) {
                window.audioSource.disconnect();
            }
            isRecording = false;
            startBtn.disabled = false;
            stopBtn.disabled = true;
            recordingIndicator.classList.remove('active');
        }

        startBtn.addEventListener('click', startRecording);
        stopBtn.addEventListener('click', stopRecording);

        // Initialize connection
        connectWebSocket();
    </script>      
    </body>
    </html>
    """)

if __name__ == "__main__":
    print("Starting Audio Pipeline Server...")
    print("Open http://localhost:8000 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=8000)