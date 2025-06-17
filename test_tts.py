import os
import torch
import time
import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
from queue import Queue, Empty
import threading

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir
from TTS.utils.manage import ModelManager

# ─── Model Setup ───────────────────────────────────────────────
torch.set_num_threads(int(os.environ.get("NUM_THREADS", os.cpu_count())))
device = torch.device("cuda" if os.environ.get("USE_CPU", "0") == "0" else "cpu")
if not torch.cuda.is_available() and device == "cuda":
    raise RuntimeError("CUDA device unavailable, please use Dockerfile.cpu instead.")

custom_model_path = os.environ.get("CUSTOM_MODEL_PATH", "/app/tts_models")

if os.path.exists(custom_model_path) and os.path.isfile(custom_model_path + "/config.json"):
    model_path = custom_model_path
    print("Loading custom model from", model_path, flush=True)
else:
    print("Loading default model", flush=True)
    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    print("Downloading XTTS Model:", model_name, flush=True)
    ModelManager().download_model(model_name)
    model_path = os.path.join(get_user_data_dir("tts"), model_name.replace("/", "--"))
    print("XTTS Model downloaded", flush=True)

print("Loading XTTS", flush=True)
config = XttsConfig()
config.load_json(os.path.join(model_path, "config.json"))
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=model_path, eval=True, use_deepspeed=True if device == "cuda" else False)
model.to(device)
print("XTTS Loaded.\n", flush=True)

# ─── Speaker Embedding (Simple: use built-in or clone) ──────────────
def get_default_speaker():
    if hasattr(model, "speaker_manager") and hasattr(model.speaker_manager, "speakers"):
        speakers = list(model.speaker_manager.speakers.keys())
        print("Available studio speakers:", speakers)
        default = speakers[12]
        speaker_embedding = model.speaker_manager.speakers[default]["speaker_embedding"].cpu().squeeze().float()
        gpt_cond_latent = model.speaker_manager.speakers[default]["gpt_cond_latent"].cpu().squeeze().float()
        return speaker_embedding, gpt_cond_latent
    else:
        # If no studio speakers, require reference audio
        wav_path = input("Enter path to a short WAV file for cloning: ").strip()
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(wav_path)
        return speaker_embedding.cpu().squeeze().float(), gpt_cond_latent.cpu().squeeze().float()

speaker_embedding, gpt_cond_latent = get_default_speaker()
print("Speaker ready.\n")

# ─── Audio Playback with Queue ─────────────────────────────────────────────
def postprocess(wav):
    if isinstance(wav, list):
        wav = torch.cat(wav, dim=0)
    wav = wav.clone().detach().cpu().numpy()
    wav = np.clip(wav, -1, 1)
    wav = (wav * 32767).astype(np.int16)
    return wav

def playback_worker(audio_queue, sr=24000, blocksize=4096):
    """Continuously play audio chunks from the queue."""
    try:
        # Prime the output stream
        with sd.OutputStream(samplerate=sr, channels=1, dtype='int16', blocksize=blocksize) as stream:
            while True:
                chunk = audio_queue.get()
                if chunk is None:
                    break
                stream.write(chunk)
    except Exception as e:
        print("Playback thread error:", e)

def trim_trailing_silence(wav, threshold=500):
    idx = np.where(np.abs(wav) > threshold)[0]
    if idx.size == 0:
        return wav
    return wav[:idx[-1]+1]

# ─── Main Loop ────────────────────────────────────────────────────
while True:
    text = input("Enter text to synthesize (or just Enter to exit): ").strip()
    if not text:
        print("Exiting.")
        break

    language = input("Language code (default 'en'): ").strip() or "en"

    # Prepare tensors (always float32)
    emb = speaker_embedding.unsqueeze(0).unsqueeze(-1).float()
    cond_latent = gpt_cond_latent.reshape((-1, 1024)).unsqueeze(0).float()

    print("\nSynthesizing and streaming audio...")

    audio_queue = Queue(maxsize=8)
    playback_thread = threading.Thread(target=playback_worker, args=(audio_queue,))
    playback_thread.start()

    start_time = time.time()
    first_chunk_time = None
    played = False

    for i, chunk in enumerate(model.inference_stream(
        text,
        language,
        cond_latent,
        emb,
        stream_chunk_size=5,
        overlap_wav_len=4096,
        temperature=0.75,
        length_penalty=1.0,
        repetition_penalty=10.0,
        top_k=50,
        top_p=0.85,
        do_sample=True,
        speed=1.1,
        enable_text_splitting=True,
    )):
        audio_np = postprocess(chunk)
        if not played:
            first_chunk_time = time.time()
            print(f"Time to first audio chunk: {first_chunk_time - start_time:.2f} sec")
            played = True
        audio_queue.put(audio_np)

    audio_queue.put(None)

    playback_thread.join()

    print("Done.\n")
