import sounddevice as sd
import numpy as np
import requests

ASR_API_URL = "http://135.181.71.42:8002/api/asr"  # Change if deployed remotely

SAMPLE_RATE = 16000     # Whisper expects 16kHz
DURATION = 5            # Duration in seconds to record
CHUNK = 512             # Chunk size to pass in request

def record_audio(duration_sec, sample_rate):
    print(f"\nRecording for {duration_sec} seconds...")
    audio = sd.rec(int(duration_sec * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is done
    return audio.flatten()

def send_to_asr(audio_data):
    payload = {
        "audio": audio_data.tolist(),
        "chunk": CHUNK,
        "lang_threshold": 0.7,
        "sent_threshold": 0.5
    }
    response = requests.post(ASR_API_URL, json=payload)
    if response.status_code == 200:
        result = response.json()
        print("\n=== ASR RESULT ===")
        print(f"Text        : {result['text']}")
        print(f"Language    : {result['language']}")
        print(f"Lang Prob   : {result['lang_prob']:.2f}")
        print(f"Sent Prob   : {result['sent_prob']:.2f}")
        print(f"Confident?  : {result['is_confident']}")
    else:
        print(f"Error {response.status_code}: {response.text}")

if __name__ == "__main__":
    audio_clip = record_audio(DURATION, SAMPLE_RATE)
    send_to_asr(audio_clip)
