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



getting this 
video:0kB audio:28195kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 0.000270%
Reading wav...
PCM bytes = 28871818
Starting gRPC stream...
SENDING CHUNK 1
SENDING CHUNK 2
SENDING CHUNK 3
SENDING CHUNK 4
SENDING CHUNK 5
SENDING CHUNK 6
SENDING CHUNK 7
SENDING CHUNK 8
SENDING CHUNK 9
SENDING CHUNK 10
SENDING CHUNK 11
SENDING CHUNK 12
SENDING CHUNK 13
SENDING CHUNK 14
SENDING CHUNK 15
SENDING CHUNK 16
SENDING CHUNK 17
SENDING CHUNK 18
SENDING CHUNK 19
SENDING CHUNK 20
SENDING CHUNK 21
SENDING CHUNK 22
SENDING CHUNK 23
SENDING CHUNK 24
SENDING CHUNK 25
SENDING CHUNK 26
SENDING CHUNK 27
SENDING CHUNK 28
SENDING CHUNK 29
SENDING CHUNK 30
SENDING CHUNK 31
RAW RESPONSE RECEIVED
No results in response
RAW RESPONSE RECEIVED
No results in response
RAW RESPONSE RECEIVED
No results in response
RAW RESPONSE RECEIVED
No results in response
RAW RESPONSE RECEIVED
No results in response
RAW RESPONSE RECEIVED
No results in response
RAW RESPONSE RECEIVED
No results in response
RAW RESPONSE RECEIVED
No results in response
RAW RESPONSE RECEIVED
No results in response
RAW RESPONSE RECEIVED
No results in response
RAW RESPONSE RECEIVED
No results in response
RAW RESPONSE RECEIVED
No results in response
RAW RESPONSE RECEIVED
No results in response
RAW RESPONSE RECEIVED
No results in response
RAW RESPONSE RECEIVED
No results in response
RAW RESPONSE RECEIVED
No results in response
