import subprocess
import time
import riva.client

SERVER = "192.168.4.62:50051"
INPUT_MP3 = "/home/re_nikitav/audio_maria/maria1.mp3"
TEMP_WAV = "/tmp/maria1_stream.wav"

SAMPLE_RATE = 16000

# 160 ms chunk
CHUNK_MS = 160
CHUNK_BYTES = SAMPLE_RATE * 2 * CHUNK_MS // 1000

# FAST SEND SPEED
SEND_DELAY_MS = 20

LANGUAGE = "es-US"

print("Converting mp3 -> wav...")

subprocess.run(
    [
        "ffmpeg",
        "-y",
        "-i",
        INPUT_MP3,
        "-t", "30",
        "-ar", "16000",
        "-ac", "1",
        "-sample_fmt", "s16",
        TEMP_WAV
    ],
    check=True
)

print("Reading wav...")

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

        print(
            f"SENDING CHUNK {chunk_num} | "
            f"bytes={len(chunk)}"
        )

        yield chunk

        # FASTER THAN REAL TIME
        time.sleep(SEND_DELAY_MS / 1000.0)


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

print("Starting streaming gRPC...")

start_time = time.time()
first_response_time = None
response_num = 0

responses = asr_service.streaming_response_generator(
    audio_chunks=audio_stream(),
    streaming_config=config
)

for response in responses:
    now = time.time()

    print("RAW RESPONSE RECEIVED")

    if not response.results:
        print("NO RESULTS")
        continue

    response_num += 1

    latency_ms = (now - start_time) * 1000

    if first_response_time is None:
        first_response_time = latency_ms
        print(
            f"TTFT = {first_response_time:.2f} ms"
        )

    for result in response.results:
        text = ""

        if result.alternatives:
            text = result.alternatives[0].transcript

        print(
            f"RESP {response_num} | "
            f"LAT={latency_ms:.2f} ms"
        )
        print(f"TEXT: {text}")
        print(f"FINAL: {result.is_final}")
        print("-" * 80)

print("COMPLETED")
