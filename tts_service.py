import numpy as np
import torch
from fastapi import FastAPI, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from TTS.api import TTS
from threading import Event
import os

app = FastAPI(title="TTS Microservice")

# Allow requests from any origin (your static frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "OPTIONS", "POST"],
    allow_headers=["*"],
)

# Load TTS model onto GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
sr = tts.synthesizer.tts_config.audio["sample_rate"]

# Global event to signal when TTS should stop streaming
tts_stop_event = Event()


def wav_header(channels: int, sample_rate: int, bits_per_sample: int) -> bytes:
    """Build a WAV header with 'infinite' chunk sizes so the browser streams continuously."""
    hdr = b"RIFF" + (0xFFFFFFFF).to_bytes(4, "little") + b"WAVE"
    hdr += b"fmt " + (16).to_bytes(4, "little")            # Subchunk1Size
    hdr += (1).to_bytes(2, "little")                       # AudioFormat=PCM
    hdr += (channels).to_bytes(2, "little")                # NumChannels
    hdr += (sample_rate).to_bytes(4, "little")             # SampleRate
    byte_rate = sample_rate * channels * bits_per_sample // 8
    hdr += (byte_rate).to_bytes(4, "little")               # ByteRate
    block_align = channels * bits_per_sample // 8
    hdr += (block_align).to_bytes(2, "little")             # BlockAlign
    hdr += (bits_per_sample).to_bytes(2, "little")         # BitsPerSample
    hdr += b"data" + (0xFFFFFFFF).to_bytes(4, "little")    # Data subchunk with infinite size
    return hdr


def generate_audio_stream(text: str, language: str):
    try:
        tts_stop_event.clear()  # Reset stop signal before new stream
        yield wav_header(channels=1, sample_rate=sr, bits_per_sample=16)

        # Choose speaker audio file based on language
        speaker_file = {
            "ja": "default_audio/japanese.wav",
            "en": ["default_audio/english_audio2.wav"],
        }.get(language, "default_audio/spanish_audio1.wav")

        with torch.no_grad():
            for chunk in tts.synthesizer.tts_stream(
                text=text,
                speaker_wav=speaker_file,
                language_name=language,
            ):
                if tts_stop_event.is_set():
                    print("TTS stream interrupted.")
                    break

                # Convert chunk to numpy
                arr = (
                    chunk if isinstance(chunk, np.ndarray)
                    else chunk.cpu().numpy() if hasattr(chunk, 'cpu')
                    else np.array(chunk, dtype=np.float32)
                )

                FRAME = 8192
                for start in range(0, arr.shape[0], FRAME):
                    if tts_stop_event.is_set():
                        break
                    seg = arr[start:start + FRAME]
                    pcm16 = (seg * 32767).astype(np.int16)
                    yield pcm16.tobytes()
    finally:
        pass


@app.get("/synthesize")
def synthesize(
    background_tasks: BackgroundTasks,
    text: str = Query(..., description="Text to synthesize"),
    language: str = Query(..., description="Language code (e.g. 'ja', 'en', 'es')"),
):
    """Streams back a WAV audio file using TTS."""
    #background_tasks.add_task(torch.cuda.empty_cache)
    if language not in ['ja', 'en', 'es']:
        language = "en"
    return StreamingResponse(
        generate_audio_stream(text, language),
        media_type="audio/wav",
        background=background_tasks
    )


@app.post("/stop")
def stop_tts():
    print("Received TTS stop request.")
    tts_stop_event.set()
    return {"status": "TTS stream stop requested."}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "tts_service:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8004)),
        reload=True,
    )
