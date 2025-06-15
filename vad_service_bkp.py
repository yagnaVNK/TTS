import os
import numpy as np
import onnxruntime
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

class VADRequest(BaseModel):
    audio: list[float]        # one float per sample
    threshold: float          # detection threshold

class VADResponse(BaseModel):
    is_speech: bool

app = FastAPI(title="VAD Microservice")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load Silero VAD once
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "assets", "silero_vad.onnx")
opts = onnxruntime.SessionOptions()
opts.log_severity_level = 4
sess = onnxruntime.InferenceSession(
    model_path, sess_options=opts,
    providers=["CUDAExecutionProvider","CPUExecutionProvider"]
)
# hidden state
h = np.zeros((2,1,64), dtype=np.float32)
c = np.zeros((2,1,64), dtype=np.float32)

@app.post("/api/vad", response_model=VADResponse)
async def vad_endpoint(req: VADRequest):
    global h, c
    audio_np = np.array(req.audio, dtype=np.float32).reshape(1, -1)
    inputs = {
        "input": audio_np,
        "sr": np.array([16000], dtype=np.int64),
        "h": h,
        "c": c
    }
    out, h_new, c_new = sess.run(None, inputs)
    h, c = h_new, c_new
    print(out > req.threshold)
    return {"is_speech": bool(out > req.threshold)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("vad_service:app", host="0.0.0.0", port=int(os.getenv("PORT", 8001)), reload=True)
