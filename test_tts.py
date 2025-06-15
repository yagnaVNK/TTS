import requests
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)

def test_tts_api(ip="135.181.71.42", port=8004):
    base_url = f"http://{ip}:{port}"
    endpoint = "/synthesize"
    full_url = f"{base_url}{endpoint}"
    
    # Sample query parameters
    params = {
        "text": "Hello, this is a test.",
        "language": "en"
    }

    logging.info(f"Testing FastAPI TTS endpoint at {full_url}")
    logging.info(f"Query params: {params}")

    try:
        logging.info("Sending GET request to the API...")
        response = requests.get(full_url, params=params, stream=True, timeout=15)

        logging.info(f"Received HTTP status code: {response.status_code}")
        if response.status_code == 200:
            logging.info("API is accessible ✅")
            
            # Read first few bytes of the stream
            logging.info("Reading stream to verify audio content...")
            audio_chunk = next(response.iter_content(chunk_size=1024), None)
            if audio_chunk:
                logging.info(f"Received {len(audio_chunk)} bytes of audio data.")
            else:
                logging.warning("No audio content received from stream ❗")
        else:
            logging.error(f"Non-200 response: {response.status_code}")
            logging.error(f"Response content: {response.text}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {e}")

if __name__ == "__main__":
    start_time = time.time()
    test_tts_api()
    logging.info(f"Test completed in {time.time() - start_time:.2f} seconds")
