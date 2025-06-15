import os
import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from faster_whisper import WhisperModel
from fastapi.middleware.cors import CORSMiddleware

class ASRRequest(BaseModel):
    audio: list[float]
    chunk: int
    lang_threshold: float = None   # optional
    sent_threshold: float = None   # optional

class ASRResponse(BaseModel):
    text: str
    lang_prob: float
    sent_prob: float
    is_confident: bool = None
    language: str

app = FastAPI(title="ASR Microservice")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Whisper once
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = os.getenv("ASR_MODEL", "large-v3")
asr_model = WhisperModel(model_name, device=device, compute_type="float16")

@app.post("/api/asr", response_model=ASRResponse)
async def asr_endpoint(req: ASRRequest):
    audio_np = np.array(req.audio, dtype=np.float32)
    segments, info = asr_model.transcribe(
        audio=audio_np,
        without_timestamps=True,
        word_timestamps=False
    )
    text = " ".join(seg.text for seg in segments).strip()
    lang_prob = info.language_probability
    sent_prob = float(np.exp(info.average_logprob))
    language_detected = info.language

    # optional threshold check
    is_confident = None
    if req.lang_threshold is not None and req.sent_threshold is not None:
        is_confident = (
            lang_prob >= req.lang_threshold
            and sent_prob >= req.sent_threshold
        )

    return {
        "text": text,
        "lang_prob": lang_prob,
        "sent_prob": sent_prob,
        "is_confident": is_confident,
        "language": language_detected
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "asr_service:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8002)),
        reload=True,
    )
