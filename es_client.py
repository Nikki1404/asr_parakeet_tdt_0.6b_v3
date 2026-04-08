import subprocess
import wave
import riva.client

SERVER = "192.168.4.62:50051"
INPUT_MP3 = "/home/re_nikitav/audio_maria/maria1.mp3"
TEMP_WAV = "/tmp/maria1_test.wav"
SAMPLE_RATE = 16000
CHUNK_SEC = 30
CHUNK_BYTES = SAMPLE_RATE * 2 * CHUNK_SEC

print("Converting mp3 -> wav...")

subprocess.run(
    [
        "ffmpeg",
        "-y",
        "-i",
        INPUT_MP3,
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

# skip WAV header
pcm = wav_bytes[44:]

print(f"PCM bytes = {len(pcm)}")

def audio_stream():
    offset = 0
    chunk_num = 0
    while offset < len(pcm):
        chunk = pcm[offset: offset + CHUNK_BYTES]
        offset += CHUNK_BYTES
        chunk_num += 1
        print(f"SENDING CHUNK {chunk_num}")
        yield chunk

auth = riva.client.Auth(uri=SERVER)
asr_service = riva.client.ASRService(auth)

config = riva.client.StreamingRecognitionConfig(
    config=riva.client.RecognitionConfig(
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        sample_rate_hertz=SAMPLE_RATE,
        language_code="es-US",
        max_alternatives=1,
        enable_automatic_punctuation=True
    ),
    interim_results=False
)

print("Starting gRPC stream...")

responses = asr_service.streaming_response_generator(
    audio_chunks=audio_stream(),
    streaming_config=config
)

for response in responses:
    print("RAW RESPONSE RECEIVED")
    if not response.results:
        print("No results in response")
        continue

    for result in response.results:
        if result.alternatives:
            print("TEXT:", result.alternatives[0].transcript)
            print("FINAL:", result.is_final)



import subprocess
import riva.client

SERVER = "192.168.4.62:50051"
INPUT_MP3 = "/home/re_nikitav/parakeet-asr-multilingual/audio_maria/maria1.mp3"
TEMP_WAV = "/tmp/maria1_30s.wav"

print("Converting first 30 sec to wav...")

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

with open(TEMP_WAV, "rb") as f:
    wav_bytes = f.read()

pcm = wav_bytes[44:]

auth = riva.client.Auth(uri=SERVER)
asr_service = riva.client.ASRService(auth)

for lang in ["es-US", "es-ES", "es", "en-US"]:
    print(f"\nTESTING LANGUAGE = {lang}")

    config = riva.client.RecognitionConfig(
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        sample_rate_hertz=16000,
        language_code=lang,
        max_alternatives=1,
        enable_automatic_punctuation=False
    )

    try:
        response = asr_service.offline_recognize(
            audio_bytes=pcm,
            config=config
        )

        if not response.results:
            print("NO RESULTS")
            continue

        for i, result in enumerate(response.results, start=1):
            if result.alternatives:
                print(f"RESULT {i}: {result.alternatives[0].transcript}")

    except Exception as e:
        print("ERROR:", e)
