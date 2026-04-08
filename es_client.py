import riva.client
import librosa
import numpy as np

SERVER = "192.168.4.62:50051"
AUDIO_FILE = "/home/re_nikitav/parakeet-asr-multilingual/audio_maria/maria1.mp3"
SAMPLE_RATE = 16000
CHUNK_SEC = 30
CHUNK_BYTES = SAMPLE_RATE * 2 * CHUNK_SEC

print("Loading audio...")

audio, sr = librosa.load(
    AUDIO_FILE,
    sr=SAMPLE_RATE,
    mono=True
)

pcm = (
    np.clip(audio, -1.0, 1.0) * 32767
).astype(np.int16).tobytes()

print(f"Loaded | bytes={len(pcm)}")


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

print("STARTING STREAM")

responses = asr_service.streaming_response_generator(
    audio_chunks=audio_stream(),
    streaming_config=config
)

for response in responses:
    print("RAW RESPONSE RECEIVED")

    if not response.results:
        print("NO RESULTS")
        continue

    for result in response.results:
        if result.alternatives:
            text = result.alternatives[0].transcript
            print(f"TRANSCRIPT -> {text}")
            print(f"FINAL -> {result.is_final}")
