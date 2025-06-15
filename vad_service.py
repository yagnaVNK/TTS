import os
import numpy as np
import onnxruntime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

class VADRequest(BaseModel):
    audio: List[float]   # downsampled or raw floats from frontend
    sample_rate: int     # e.g. 48000
    threshold: float     # VAD detection threshold

class VADResponse(BaseModel):
    is_speech: bool

app = FastAPI(title="VAD Microservice (48kHz)")
# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Silero VAD (expects 16kHz)
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "assets", "silero_vad.onnx")
opts = onnxruntime.SessionOptions(); opts.log_severity_level = 4
sess = onnxruntime.InferenceSession(model_path, sess_options=opts,
                                     providers=["CUDAExecutionProvider","CPUExecutionProvider"])
# initial hidden states
h = np.zeros((2,1,64), dtype=np.float32)
c = np.zeros((2,1,64), dtype=np.float32)
MODEL_RATE = 16000

@app.post("/api/vad", response_model=VADResponse)
def vad_endpoint(req: VADRequest):
    audio = np.array(req.audio, dtype=np.float32)
    # resample to model rate if needed
    if req.sample_rate != MODEL_RATE:
        src = np.linspace(0, len(audio), num=len(audio), endpoint=False)
        dst = np.linspace(0, len(audio), num=int(len(audio)*MODEL_RATE/req.sample_rate), endpoint=False)
        audio = np.interp(dst, src, audio).astype(np.float32)
    # run VAD ONNX
    global h, c
    probs, (h_new, c_new) = sess.run(None, {"input": audio[np.newaxis,:], "h": h, "c": c})
    h, c = h_new, c_new
    # speech = probability of index 1 > threshold
    is_speech = float(probs[0][1]) > req.threshold
    return VADResponse(is_speech=is_speech)