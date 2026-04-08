import subprocess
import time
import threading
import riva.client

SERVER = "192.168.4.62:50051"
INPUT_MP3 = "/home/re_nikitav/parakeet-asr-multilingual/audio_maria/maria1.mp3"
TEMP_WAV = "/tmp/maria1_stream.wav"

SAMPLE_RATE = 16000
CHUNK_MS = 500
CHUNK_BYTES = SAMPLE_RATE * 2 * CHUNK_MS // 1000
LANGUAGE = "es-US"
MAX_TEST_AUDIO_SEC = 10
MAX_WALL_TIME_SEC = 60

print("Converting mp3 -> wav...")

subprocess.run(
    [
        "ffmpeg",
        "-y",
        "-i",
        INPUT_MP3,
        "-t", str(MAX_TEST_AUDIO_SEC),
        "-ar", "16000",
        "-ac", "1",
        "-sample_fmt", "s16",
        TEMP_WAV
    ],
    check=True
)

with open(TEMP_WAV, "rb") as f:
    wav_bytes = f.read()

pcm = wav_bytes[44:]
print(f"PCM bytes = {len(pcm)}")

def audio_stream():
    offset = 0
    chunk_num = 0
    while offset < len(pcm):
        chunk = pcm[offset: offset + CHUNK_BYTES]
        offset += CHUNK_BYTES
        chunk_num += 1
        print(f"SENDING CHUNK {chunk_num} | bytes={len(chunk)}")
        yield chunk

auth = riva.client.Auth(uri=SERVER)
asr_service = riva.client.ASRService(auth)

config = riva.client.StreamingRecognitionConfig(
    config=riva.client.RecognitionConfig(
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        sample_rate_hertz=SAMPLE_RATE,
        language_code=LANGUAGE,
        max_alternatives=1,
        enable_automatic_punctuation=False
    ),
    interim_results=True
)

responses = asr_service.streaming_response_generator(
    audio_chunks=audio_stream(),
    streaming_config=config
)

stop_flag = {"stop": False}

def killer():
    time.sleep(MAX_WALL_TIME_SEC)
    stop_flag["stop"] = True
    print(f"\nSTOPPING: exceeded {MAX_WALL_TIME_SEC} sec wall time\n")

threading.Thread(target=killer, daemon=True).start()

got_text = False
start = time.time()

try:
    for response in responses:
        if stop_flag["stop"]:
            break

        print("RAW RESPONSE RECEIVED")

        if not response.results:
            print("NO RESULTS")
            continue

        for result in response.results:
            text = result.alternatives[0].transcript if result.alternatives else ""
            print("TEXT:", repr(text))
            print("FINAL:", result.is_final)

            if text.strip():
                got_text = True

        if got_text:
            break

except Exception as e:
    print("ERROR:", e)

print(f"Done in {time.time() - start:.2f} sec")
